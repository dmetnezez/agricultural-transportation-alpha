"""
Cleaning: turn raw downloads into weekly series, each carrying the date the
observation could first have been known by a trader ("available").

Point-in-time rule used everywhere downstream:
    a value may be used for a trade decision on Friday T only if
    available < T  (strictly earlier day).

Publication-lag assumptions (conservative; all documented in README):
    rail cars loaded & billed (STB/EP724) : submission_date + 1 day (GTR Thursday)
    AAR rail carloadings                   : submission_date + 1 day
    barge lock tonnages (USACE via GTR)    : week ending Sat + 5 days (Thursday GTR)
    NOLA barge unloads                     : week ending Sat + 5 days
    FGIS export inspections                : week ending Thu + 4 days (Monday release)
    FAS export sales                       : week ending Thu + 7 days (Thursday release)
    transport cost indicators              : index date (Wed) + 1 day
    barge rates                            : quote date (Tue) + 2 days
    Iowa/IL/Gulf basis                     : quote date (Fri) + 6 days (next GTR)
    grain vessel loading (Gulf/PNW)        : reading date (Thu) + 7 days (next GTR)
    FHWA TMAS traffic                      : published months later -> NOT tradable in
                                             real time; used only as an ex-post diagnostic.

Missing observations stay missing. Nothing is interpolated.
Note on revisions: AgTransport serves the CURRENT vintage. Original first-release
values are not archived there, so any later revisions are an unavoidable
(small) look-ahead risk; flagged in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def _read(name, dates=("date",)):
    df = pd.read_csv(RAW / f"{name}.csv")
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize(None).dt.normalize()
    return df


def _series(df, value, lag_days, name, avail_col=None):
    """Return DataFrame[obs_date, available, <name>] sorted by obs_date."""
    out = df[["date", value]].rename(columns={"date": "obs_date", value: name}).copy()
    if avail_col is not None:
        out["available"] = df[avail_col].values + pd.Timedelta(days=lag_days)
    else:
        out["available"] = out["obs_date"] + pd.Timedelta(days=lag_days)
    return out.sort_values("obs_date").reset_index(drop=True)


# --------------------------------------------------------------------------
# transportation series
# --------------------------------------------------------------------------
IOWA_CORE_RAILROADS = {"BNSF", "UP"}   # ~75% of Iowa grain cars; must both report


def rail_cars():
    """Grain rail cars loaded & billed. Iowa total and all-US total.
    The source omits carrier-state rows with no loadings, so a missing small
    carrier (CN, CP, NS, CSX) is indistinguishable from zero and is summed as 0.
    A week is set to NaN if BNSF or UP (the dominant Iowa carriers) is absent,
    so that reporting gaps do not masquerade as activity changes."""
    r = _read("rail_cars_by_state", ("date", "submission_date"))
    r["railroad"] = r["railroad"].replace({"CPKC": "CP"})
    r["all"] = pd.to_numeric(r["all"], errors="coerce")
    sub = r.groupby("date")["submission_date"].max()

    ia = r[r.state == "IA"]
    ia_tot = ia.groupby("date")["all"].sum()
    ia_rep = ia.groupby("date")["railroad"].apply(lambda s: IOWA_CORE_RAILROADS.issubset(set(s)))
    ia_tot[~ia_rep.reindex(ia_tot.index, fill_value=False)] = np.nan

    us_tot = r.groupby("date")["all"].sum()
    us_rep = r.groupby("date")["railroad"].nunique()
    us_tot[us_rep < us_rep.rolling(9, center=True, min_periods=1).median()] = np.nan

    df = pd.DataFrame({"ia_railcars": ia_tot, "us_railcars": us_tot, "sub": sub}).reset_index()
    a = _series(df, "ia_railcars", 1, "ia_railcars", "sub")
    b = _series(df, "us_railcars", 1, "us_railcars", "sub")
    return [a, b]


def aar_grain():
    r = _read("rail_carloadings_grain", ("date", "submission_date"))
    r = r[r.type == "Originated"]
    r["carloads"] = pd.to_numeric(r.carloads, errors="coerce")
    g = r.groupby("date").agg(aar_grain_carloads=("carloads", "sum"),
                              n=("railroad", "nunique"), sub=("submission_date", "max")).reset_index()
    g.loc[g.n < g.n.max(), "aar_grain_carloads"] = np.nan
    return [_series(g, "aar_grain_carloads", 1, "aar_grain_carloads", "sub")]


def barge():
    """Downbound grain barge tons through the 7 locks reported since 2013."""
    b = _read("barge_grain_tons")
    b["tons"] = pd.to_numeric(b.tons, errors="coerce")
    b = b[b.date >= "2013-01-01"]
    out = []
    for com, nm in [("Corn", "barge_corn_tons"), ("Soybeans", "barge_soy_tons")]:
        g = b[b.commodity == com].groupby("date").agg(v=("tons", "sum"), n=("lock", "nunique")).reset_index()
        g.loc[g.n < 7, "v"] = np.nan
        out.append(_series(g, "v", 5, nm))
    u = _read("nola_barge_unloads")
    u["unloads"] = pd.to_numeric(u.unloads, errors="coerce")
    out.append(_series(u, "unloads", 5, "nola_unloads"))
    return out


def inspections():
    """Daily FGIS certificate totals -> weeks ending Thursday."""
    d = pd.read_csv(RAW / "export_inspections_daily.csv")
    d["day"] = pd.to_datetime(d["day"]).dt.normalize()
    d["mt"] = pd.to_numeric(d.mt, errors="coerce")
    out = []
    for g, nm in [("CORN", "insp_corn_mt"), ("SOYBEANS", "insp_soy_mt")]:
        s = d[d.grain == g].set_index("day")["mt"].resample("W-THU").sum(min_count=1)
        df = s.rename("v").reset_index().rename(columns={"day": "date"})
        out.append(_series(df, "v", 4, nm))
    return out


def export_sales():
    e = _read("export_sales_weekly")
    out = []
    for com, tag in [("Corn", "corn"), ("Soybeans", "soy")]:
        x = e[e.commodity == com].copy()
        for c in ["exports", "net_sales"]:
            x[c] = pd.to_numeric(x[c], errors="coerce")
            out.append(_series(x, c, 7, f"sales_{tag}_{c}"))
    return out


def costs():
    c = _read("transport_cost_indicators")
    out = []
    for col in ["truck", "shuttle_train", "barge", "gulf_vessel"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
        out.append(_series(c, col, 1, f"cost_{col}"))
    br = _read("barge_rates")
    br["rate"] = pd.to_numeric(br.rate, errors="coerce")
    s = br[br.location == "St. Louis"]
    out.append(_series(s, "rate", 2, "barge_rate_stl"))
    return out


def basis():
    b = _read("grain_basis_iowa")
    b["basis"] = pd.to_numeric(b.basis, errors="coerce")
    out = []
    for mk, com, nm in [("Iowa", "Corn", "basis_ia_corn"), ("Iowa", "Soybeans", "basis_ia_soy"),
                        ("Gulf - Louisiana", "Corn", "basis_gulf_corn"),
                        ("Gulf - Louisiana", "Soybeans", "basis_gulf_soy")]:
        s = b[(b.market_name == mk) & (b.commodity == com)].drop_duplicates("date")
        out.append(_series(s, "basis", 6, nm))
    return out


def vessels():
    """Ocean grain vessels at US export ports (GTR table, Thursday dates since 1995).
    Gulf: vessels in port, loaded in the last 7 days, due in the next 10 days.
    PNW: vessels in port only. A Thursday reading is published in the FOLLOWING
    Thursday's report (verified: latest row lags the latest report by a week),
    so availability = date + 7 days."""
    v = _read("vessel_loading")
    out = []
    for port, col, nm in [("Gulf", "loaded_7_days", "vessel_gulf_loaded"),
                          ("Gulf", "due_10_days", "vessel_gulf_due"),
                          ("Gulf", "in_port", "vessel_gulf_inport"),
                          ("PNW", "in_port", "vessel_pnw_inport")]:
        x = v[v.port == port].drop_duplicates("date").copy()
        x[col] = pd.to_numeric(x[col], errors="coerce")
        out.append(_series(x, col, 7, nm))
    return out


def tmas_iowa():
    """Iowa rural-interstate (f_system 1R) daily vehicle volumes, summed by week
    (Sat-Fri). Uses only station-directions present on every day of the week to
    avoid composition noise; then a 'same-station' growth index is NOT built -
    we use the plain total of complete stations and record the station count.
    Not available in real time (published with long lags)."""
    d = RAW / "tmas_iowa"
    files = sorted(d.glob("*.csv")) if d.exists() else []
    if not files:
        return []
    t = pd.concat(pd.read_csv(f, dtype={"station_id": str, "f_system": str}) for f in files)
    t["date"] = pd.to_datetime(t["date"])
    # rural interstate: "1R" (TMG legacy) or "1"/"01" (numeric HPMS codes, rural 1-7,
    # urban 11-17, used from 2025)
    t = t[t.f_system.astype(str).str.strip().str.upper().isin(["1R", "01", "1"])]
    t = t.groupby(["station_id", "date"])["volume"].sum(min_count=1).reset_index()
    t["wk"] = t.date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    full = t.groupby(["station_id", "wk"]).volume.count()
    full = full[full == 7].reset_index()[["station_id", "wk"]]
    t = t.merge(full, on=["station_id", "wk"])
    # average volume per complete station: robust to the number of stations reporting
    w = t.groupby("wk").agg(v=("volume", "sum"), n=("station_id", "nunique")).reset_index()
    w["v"] = w.v / w.n
    w.loc[w.n < 10, "v"] = np.nan
    w = w.rename(columns={"wk": "date"})
    # HYPOTHETICAL real-time availability (+6 days). In reality FHWA posts these
    # files months later, so this series is an ex-post diagnostic only and is
    # excluded from every tradable strategy.
    s = _series(w, "v", 6, "ia_rural_interstate_traffic")
    return [s]


# --------------------------------------------------------------------------
# futures
# --------------------------------------------------------------------------
CONTRACT_MONTHS = {"zc": [3, 5, 7, 9, 12], "zs": [1, 3, 5, 7, 8, 9, 11]}


def futures_weekly(sym: str) -> pd.DataFrame:
    """Weekly (Friday or last trading day) front-month closes from Yahoo.

    Yahoo's ZC=F/ZS=F roll to the next contract on the first trading day on or
    after the 15th of each contract month (the day after last trade), producing
    artificial jumps (up to -17% on Jul->Sep corn). Between first notice and
    expiry the front contract is also illiquid. Because expired contracts are
    not available to build a proper back-adjusted series, any week whose return
    window overlaps the period [first position day (FND - 2 business days), roll day] is flagged NON-TRADABLE. Returns in those weeks are excluded from
    every test and the strategy is flat in them.
    """
    f = _read(f"futures_{sym}")
    f = f.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    f["wk"] = f.date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    w = f.groupby("wk").agg(close=("close", "last"), last_day=("date", "last"),
                            volume=("volume", "sum")).reset_index().rename(columns={"wk": "week"})

    # blocked calendar days
    days = f.date
    blocked = pd.Series(False, index=days.values)
    for y in range(days.dt.year.min() - 1, days.dt.year.max() + 2):
        for m in CONTRACT_MONTHS[sym]:
            roll = days[(days >= pd.Timestamp(y, m, 15))]
            if roll.empty:
                continue
            roll = roll.iloc[0]
            # first notice day = last business day of the prior month; a long
            # must be out 2 business days before that (first position day)
            fnd = pd.Timestamp(y, m, 1) - pd.offsets.BDay(1)
            start = fnd - pd.offsets.BDay(2)
            blocked[(blocked.index >= start) & (blocked.index <= roll)] = True
    blocked_days = set(blocked[blocked].index)
    # a weekly return from week t-1 close to week t close is clean if no day in
    # (last_day[t-1], last_day[t]] is blocked
    prev = w.last_day.shift(1)
    clean = []
    for p, cur in zip(prev, w.last_day):
        if pd.isna(p):
            clean.append(False)
            continue
        rng = days[(days > p) & (days <= cur)]
        clean.append(not any(d in blocked_days for d in rng))
    w["ret"] = np.log(w.close).diff()
    w["clean"] = clean
    w["ret_clean"] = w.ret.where(w.clean)
    return w


def futures_weekly_usda(sym: str) -> pd.DataFrame:
    """PRIMARY return series (Yahoo front month is the robustness check).
    The AgTransport basis table records, every Friday,
    the CBOT futures price used as the basis reference, together with the
    contract month. USDA rolls to the next contract roughly 2-3 weeks before
    first notice day, which mimics what a speculator would hold. A weekly
    return is computed only between two Fridays quoting the SAME contract, so
    no labelled roll gap enters a return. CAVEAT found during validation: the
    contract label is not always reliable (e.g. soybeans Oct-Dec 2025 labelled
    'Dec', a month CBOT soybeans do not trade; corn Jul-Aug 2024 labelled 'Sep'
    but priced ~19c above CBOT Sep). Unlabelled contract switches can therefore
    leak into returns. Validation (below) removes every print that can be checked
    against Yahoo and fails, plus invalid labels; unverifiable weeks remain a
    documented residual risk. Yahoo is not used as primary because removing its
    roll/delivery weeks discards 35-50% of the sample."""
    com = {"zc": "Corn", "zs": "Soybeans"}[sym]
    b = _read("grain_basis_iowa")
    b = b[(b.market_name == "Iowa") & (b.commodity == com)].sort_values("date").drop_duplicates("date")
    b["futures_price"] = pd.to_numeric(b.futures_price, errors="coerce")
    w = b[["date", "contract_month", "futures_price"]].rename(columns={"date": "week", "futures_price": "close"})
    w["week"] = w.week.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    w = w.drop_duplicates("week", keep="last").reset_index(drop=True)
    # data-quality filter (transcription errors, e.g. corn 2026-01-16 = 5.246
    # while CBOT March corn settled ~4.25). Two rules; flagged prints are set to
    # NaN (never replaced) and the affected returns simply drop out:
    #  (1) cross-check vs Yahoo: when Yahoo's front month IS the same contract,
    #      a USDA print more than 3% away from Yahoo's same-day close is bad;
    #  (2) isolated spike: >10% away from both neighbours (same contract) in
    #      opposite directions while the neighbours agree within 4%.
    y = _read(f"futures_{sym}").dropna(subset=["close"]).set_index("date")["close"] / 100
    mon = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                       "Sep", "Oct", "Nov", "Dec"], 1)}

    def yahoo_front_month(d):
        for m in CONTRACT_MONTHS[sym] + [CONTRACT_MONTHS[sym][0] + 12]:
            yr, mm = (d.year, m) if m <= 12 else (d.year + 1, m - 12)
            if d < pd.Timestamp(yr, mm, 15):
                return mm
        return None

    yc = y.reindex(w.week).values
    same_c = np.array([mon.get(cm) == yahoo_front_month(d) for cm, d in zip(w.contract_month, w.week)])
    dev = np.abs(np.log(w.close.values / yc))
    rule1 = same_c & (dev > 0.03)
    c = w.close
    up, dn = np.log(c / c.shift(1)), np.log(c.shift(-1) / c)
    nb = np.log(c.shift(-1) / c.shift(1)).abs()
    samec = (w.contract_month == w.contract_month.shift(1)) & (w.contract_month == w.contract_month.shift(-1))
    rule2 = samec & (up.abs() > 0.10) & (dn.abs() > 0.10) & (np.sign(up) != np.sign(dn)) & (nb < 0.04)
    # (3) contract labels CBOT does not list for this commodity (e.g. soybean "Dec")
    rule3 = ~w.contract_month.map(mon).isin(CONTRACT_MONTHS[sym]).values
    bad = rule1 | rule2.values | rule3
    w["bad_print"] = bad
    w.loc[bad, "close"] = np.nan
    same = (w.contract_month == w.contract_month.shift()) & ((w.week - w.week.shift()).dt.days == 7)
    w["ret"] = np.log(w.close).diff().where(same)
    w["clean"] = same
    w["ret_clean"] = w["ret"]
    return w


def build_raw_weekly():
    series = []
    for fn in (rail_cars, aar_grain, barge, inspections, export_sales, costs, basis, vessels, tmas_iowa):
        try:
            series += fn()
        except FileNotFoundError as e:
            print(f"  missing raw file for {fn.__name__}: {e}")
    return series


if __name__ == "__main__":
    s = build_raw_weekly()
    for x in s:
        c = x.columns[1]
        print(f"{c:30s} {x.obs_date.min().date()} -> {x.obs_date.max().date()}  n={x[c].notna().sum()}  missing={x[c].isna().sum()}")
    for sym in ("zc", "zs"):
        w = futures_weekly(sym)
        u = futures_weekly_usda(sym)
        print(sym, "yahoo clean share", round(w.clean.mean(), 3), "| usda clean share", round(u.clean.mean(), 3))
        m = u.merge(w[["week", "ret", "clean"]], on="week", suffixes=("", "_y"))
        m = m[m.clean & m.clean_y]
        print("   corr(usda, yahoo) on weeks both clean:", round(m.ret.corr(m.ret_y), 4),
              " mean abs diff:", round((m.ret - m.ret_y).abs().mean(), 5), "n", len(m))
