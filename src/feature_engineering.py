"""
Feature engineering with strict point-in-time alignment.

Timeline for decision week T (a Friday):
  * transport features: latest observation whose `available` date < T
  * transforms (WoW, MA4, YoY, seasonal anomaly) are computed on each series'
    own observation calendar using ONLY earlier observations, then aligned
  * market features: futures returns up to and including the Friday-T close
  * target: return from snapshot T to snapshot T+h (h = 1, 2, 4 weeks)

Seasonal anomaly (the key variable for separating seasonality from news):
  anomaly_t = log(MA4_t) - mean_{k=1..3}( log(MA4) in the same week-of-year,
                                          k years earlier, +/- 1 week )
  requires at least 2 prior years. z-score = anomaly / expanding std of past
  anomalies (min 52 obs). Only past data enters both pieces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_cleaning import build_raw_weekly, futures_weekly_usda, futures_weekly

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# series whose natural transform is a difference (can be <=0 or are indices/levels)
LEVEL_SERIES = {"basis_ia_corn", "basis_ia_soy", "basis_gulf_corn", "basis_gulf_soy",
                "sales_corn_net_sales", "sales_soy_net_sales"}

STALE_LIMIT_DAYS = 21  # never carry a value forward more than 3 weeks


def _seasonal_anomaly(s: pd.Series, years=3, min_years=2) -> pd.Series:
    """s indexed by obs_date (weekly). For each date, mean of s at the same
    calendar point 1..`years` years earlier (nearest obs within +/-10 days)."""
    idx = s.index
    vals = []
    s_sorted = s.dropna()
    for d in idx:
        past = []
        for k in range(1, years + 1):
            tgt = d - pd.DateOffset(years=k)
            lo, hi = tgt - pd.Timedelta(days=10), tgt + pd.Timedelta(days=10)
            w = s_sorted[(s_sorted.index >= lo) & (s_sorted.index <= hi)]
            if len(w):
                past.append(w.mean())
        vals.append(np.mean(past) if len(past) >= min_years else np.nan)
    return s - pd.Series(vals, index=idx)


def _yoy(s: pd.Series) -> pd.Series:
    """s_t - s_{t-1yr} (nearest obs within +/-4 days)."""
    cur = pd.DataFrame({"d": s.index, "x": s.values}).sort_values("d")
    prev = pd.DataFrame({"d": s.index + pd.DateOffset(years=1), "p": s.values})
    prev = prev.drop_duplicates("d").sort_values("d")
    al = pd.merge_asof(cur, prev, on="d", direction="nearest", tolerance=pd.Timedelta(days=4))
    return pd.Series(al.x.values - al.p.values, index=s.index)


def transform_series(df: pd.DataFrame) -> pd.DataFrame:
    """df: [obs_date, <name>, available] -> adds transformed columns."""
    name = df.columns[1]
    s = df.set_index("obs_date")[name].astype(float)
    s = s[~s.index.duplicated(keep="last")]
    out = pd.DataFrame(index=s.index)
    if name in LEVEL_SERIES:
        base = s
        out[f"{name}__lvl"] = base
        out[f"{name}__wow"] = base.diff()
        ma4 = base.rolling(4, min_periods=4).mean()
        out[f"{name}__ma4chg"] = ma4.diff(4)
    else:
        base = np.log(s.where(s > 0))
        out[f"{name}__wow"] = base.diff()
        ma4 = np.log(s.where(s > 0).rolling(4, min_periods=4).mean())
        out[f"{name}__ma4chg"] = ma4.diff(4)
        out[f"{name}__rel52"] = ma4 - ma4.rolling(52, min_periods=40).mean()  # vs trailing-year avg
    out[f"{name}__yoy"] = _yoy(ma4)
    anom = _seasonal_anomaly(ma4)
    out[f"{name}__anom"] = anom
    sd = anom.expanding(min_periods=52).std().shift(1)
    out[f"{name}__anomz"] = anom / sd
    out["available"] = df.drop_duplicates("obs_date", keep="last").set_index("obs_date")["available"].reindex(out.index)
    out["obs_date"] = out.index
    return out.reset_index(drop=True)


def align_point_in_time(trans: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """For each decision Friday T take the row with the latest available < T."""
    t = trans.dropna(subset=["available"]).sort_values("available")
    t = t.assign(avail_key=t.available + pd.Timedelta(days=1))  # strict: available <= T-1
    left = pd.DataFrame({"week": weeks}).sort_values("week")
    m = pd.merge_asof(left, t, left_on="week", right_on="avail_key", direction="backward")
    stale = (m.week - m.obs_date).dt.days > STALE_LIMIT_DAYS
    cols = [c for c in m.columns if "__" in c]
    m.loc[stale, cols] = np.nan
    return m.set_index("week")[cols]


def market_features(sym: str, source: str = "usda", live: bool = False) -> pd.DataFrame:
    """source='usda' (PRIMARY): USDA AMS weekly reference-contract prints,
    validated (see data_cleaning.futures_weekly_usda).
    source='yahoo' (robustness): CBOT front month, Friday close, with
    roll/delivery-window weeks removed.
    source='bloomberg': contract-level CBOT settlements from a Bloomberg
    export, rolled 5 business days before first notice (bloomberg.py)."""
    y = futures_weekly(sym).set_index("week")
    u = futures_weekly_usda(sym).set_index("week")
    roll = None
    if source == "bloomberg":
        # contract-level settlements, roll-correct (see bloomberg.py); USDA as the alternative
        import bloomberg
        b = bloomberg.weekly_continuous(bloomberg.load_any(), sym).set_index("week")
        price, r, alt, roll = b["close"] / 100, b["ret"], u["ret"], b["roll"]
    elif source == "yahoo":
        price, r, alt = y["close"] / 100, y["ret_clean"], u["ret"]
    else:
        price, r, alt = u["close"], u["ret"], y["ret_clean"]
    idx = price.index[price.index >= "2005-01-01"]
    if live:
        # add the current week's Friday so the live forecast uses this week's
        # transport releases even before this week's futures price is published
        this_fri = pd.Timestamp.today().normalize().to_period("W-FRI").end_time.normalize()
        if this_fri not in idx:
            idx = idx.append(pd.DatetimeIndex([this_fri]))
    price, r = price.reindex(idx), r.reindex(idx)
    out = pd.DataFrame(index=idx)
    out["price"] = price.ffill()  # only used to convert tick costs to % (never for returns)
    out["ret"] = r
    out["mom_1w"] = r
    out["mom_4w"] = r.rolling(4, min_periods=3).sum()
    out["mom_12w"] = r.rolling(12, min_periods=8).sum()
    out["vol_12w"] = r.rolling(12, min_periods=8).std() * np.sqrt(52)
    out["vol_26w"] = r.rolling(26, min_periods=16).std() * np.sqrt(52)
    if live:
        out[["vol_12w", "vol_26w", "price"]] = out[["vol_12w", "vol_26w", "price"]].ffill()
    for h in (1, 2, 4):
        out[f"fwd_{h}"] = sum(r.shift(-k) for k in range(1, h + 1))  # NaN if any leg excluded
    out["fwd_1_alt"] = alt.reindex(idx).shift(-1)   # other price source, robustness
    if roll is not None:
        out["roll"] = roll.reindex(idx).fillna(False).astype(bool)   # contract changes at this close
    return out


def seasonality(idx: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.DataFrame(index=idx)
    s["month"] = idx.month
    s["woy"] = idx.isocalendar().week.values.astype(int)
    s["harvest"] = idx.month.isin([9, 10, 11]).astype(int)
    s["planting"] = idx.month.isin([4, 5]).astype(int)
    s["growing"] = idx.month.isin([6, 7, 8]).astype(int)
    return s


COMMODITY_SPECIFIC = {
    "zc": ["barge_corn_tons", "insp_corn_mt", "sales_corn_exports", "sales_corn_net_sales",
           "basis_ia_corn", "basis_gulf_corn"],
    "zs": ["barge_soy_tons", "insp_soy_mt", "sales_soy_exports", "sales_soy_net_sales",
           "basis_ia_soy", "basis_gulf_soy"],
}
SHARED = ["ia_railcars", "us_railcars", "aar_grain_carloads", "nola_unloads", "cost_truck",
          "cost_shuttle_train", "cost_barge", "cost_gulf_vessel", "barge_rate_stl",
          "ia_rural_interstate_traffic"]


def build_dataset(save=True, source="usda", live=False) -> dict[str, pd.DataFrame]:
    raw = build_raw_weekly()
    trans = {r.columns[1]: transform_series(r) for r in raw}
    out = {}
    for sym in ("zc", "zs"):
        mkt = market_features(sym, source, live)
        weeks = mkt.index
        feats = [align_point_in_time(trans[n], weeks) for n in SHARED + COMMODITY_SPECIFIC[sym] if n in trans]
        df = pd.concat([mkt, seasonality(weeks)] + feats, axis=1)
        # commodity-neutral names for commodity-specific series
        ren = {}
        for c in df.columns:
            base = c.split("__")[0]
            nb = base.replace("_corn", "_com").replace("_soy", "_com")
            if nb != base:
                ren[c] = c.replace(base, nb)
        df = df.rename(columns=ren)
        out[sym] = df
        if save:
            df.to_csv(PROC / (f"features_{sym}.csv" if source == "usda" else f"features_{sym}_{source}.csv"))
    return out


if __name__ == "__main__":
    d = build_dataset()
    for k, v in d.items():
        print(k, v.shape, v.index.min().date(), v.index.max().date())
        print(v.filter(like="__anomz").notna().sum().to_string())
