"""
Non-transport factors, all from free public sources, all point-in-time.

Each loader returns DataFrames [obs_date, <feature columns '__'>, available]
whose feature columns are computed ONLY from past observations (trailing
windows), so they can be aligned with feature_engineering.align_point_in_time.

Sources and publication lags
  CFTC Disaggregated Commitments of Traders (futures only), corn 002602 / soy 005602
     Tuesday positions, released Friday 3:30pm ET (after the CBOT close)
     -> available = Tuesday + 4 days (first usable decision: the FOLLOWING Friday).
     Caveat: releases were delayed during US government shutdowns (2013, 2018-19,
     Oct-Nov 2025); a fixed lag slightly overstates availability in those weeks.
  EIA weekly fuel-ethanol production (week ending Friday), released Wednesday
     -> available = week end + 5 days.  Since 2010-06.
  US Drought Monitor, % area in D1-D4 (cumulative) for IA, IL, NE, MN, IN
     map valid Tuesday, released Thursday 8:30am ET -> available = map date + 2.
  USGS daily stage, Mississippi River at St. Louis (07010000)
     near real-time; weekly mean Fri-Thu -> available = Thursday + 1 day... we use
     week ending Thursday with available = obs + 0 (strict rule makes it usable
     on Friday).
  Board crush margin from Yahoo front-month ZM, ZL, ZS (Thursday close)
     available = obs (Thursday close; strict rule -> usable Friday).
     Caveat: front-month level jumps at rolls add noise to the level.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "factors"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "agtransport-research/1.0"}
CORN_BELT = {"IA": 19, "IL": 17, "NE": 31, "MN": 27, "IN": 18}


def _get(url, retries=5, **kw):
    import time
    for i in range(retries):
        r = requests.get(url, headers=UA, timeout=180, **kw)
        if r.status_code == 429 and i < retries - 1:   # rate limited: back off
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        return r


def _trailing_z(s: pd.Series, win=156, minp=104) -> pd.Series:
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd


def _frame(idx, cols: dict, lag_days):
    out = pd.DataFrame(cols, index=idx)
    out["obs_date"] = idx
    out["available"] = idx + pd.Timedelta(days=lag_days)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
def download_all():
    # CFTC
    frames = []
    # 2006-2016 is published as one combined file, later years one file each
    urls = ["https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2006_2016.zip"] + \
           [f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip"
            for y in range(2017, pd.Timestamp.today().year + 1)]
    for y in urls:
        try:
            z = zipfile.ZipFile(io.BytesIO(_get(y).content))
            df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
            frames.append(df)
        except Exception as e:
            print(f"  CFTC {y} failed: {e}")
    for f in frames:
        f.columns = [c.strip() for c in f.columns]
    cot = pd.concat(frames)
    code_col = [c for c in cot.columns if c.startswith("CFTC_Contract_Market_Code")][0]
    cot[code_col] = cot[code_col].astype(str).str.strip().str.zfill(6)
    keep = cot[cot[code_col].isin(["002602", "005602"])]
    keep.to_csv(RAW / "cftc_cot_disagg_corn_soy.csv", index=False)
    print(f"  CFTC COT rows: {len(keep)}")
    # EIA ethanol
    b = _get("https://www.eia.gov/dnav/pet/hist_xls/W_EPOOXE_YOP_NUS_MBBLDw.xls").content
    e = pd.read_excel(io.BytesIO(b), sheet_name="Data 1", skiprows=2)
    e.columns = ["date", "ethanol_kbd"]
    e.to_csv(RAW / "eia_ethanol_production.csv", index=False)
    print(f"  EIA ethanol rows: {len(e)}")
    # Drought monitor
    d = []
    end = pd.Timestamp.today().strftime("%m/%d/%Y")
    for st, fips in CORN_BELT.items():
        txt = _get("https://usdmdataservices.unl.edu/api/StateStatistics/GetDroughtSeverityStatisticsByAreaPercent",
                   params={"aoi": fips, "startdate": "1/1/2004", "enddate": end, "statisticsType": 1}).text
        d.append(pd.read_csv(io.StringIO(txt)))
    pd.concat(d).to_csv(RAW / "usdm_cornbelt.csv", index=False)
    print(f"  Drought rows: {sum(len(x) for x in d)}")
    # USGS river stage
    txt = _get("https://waterservices.usgs.gov/nwis/dv/", params={
        "format": "rdb", "sites": "07010000", "parameterCd": "00065", "startDT": "2004-01-01"}).text
    lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
    rv = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t")
    val = [c for c in rv.columns if c.endswith("_00065_00003") or (c.endswith("00065_30800"))]
    val = val[0] if val else [c for c in rv.columns if "00065" in c and not c.endswith("_cd")][0]
    rv = rv[["datetime", val]].rename(columns={"datetime": "date", val: "stage_ft"})
    rv.to_csv(RAW / "usgs_stlouis_stage.csv", index=False)
    print(f"  River rows: {len(rv)}")
    # crush inputs
    p1, p2 = int(pd.Timestamp("2004-01-01").timestamp()), int(pd.Timestamp.utcnow().timestamp())
    import time
    for t in ("ZM=F", "ZL=F", "ZS=F"):
        time.sleep(3)
        js = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}",
                  params={"period1": p1, "period2": p2, "interval": "1d"}).json()["chart"]["result"][0]
        pd.DataFrame({"date": pd.to_datetime(js["timestamp"], unit="s").normalize(),
                      "close": js["indicators"]["quote"][0]["close"]}).dropna().to_csv(
            RAW / f"yahoo_{t[:2].lower()}.csv", index=False)


# --------------------------------------------------------------------------
def cot_features(sym):
    code = {"zc": "002602", "zs": "005602"}[sym]
    c = pd.read_csv(RAW / "cftc_cot_disagg_corn_soy.csv", low_memory=False)
    code_col = [x for x in c.columns if x.startswith("CFTC_Contract_Market_Code")][0]
    c = c[c[code_col].astype(str).str.zfill(6) == code]
    date_col = [x for x in c.columns if x.startswith("Report_Date_as_YYYY")][0]
    c["d"] = pd.to_datetime(c[date_col])
    c = c.sort_values("d").drop_duplicates("d").set_index("d")
    oi = c["Open_Interest_All"].astype(float)
    mm = (c["M_Money_Positions_Long_All"].astype(float) - c["M_Money_Positions_Short_All"].astype(float)) / oi
    pm = (c["Prod_Merc_Positions_Long_All"].astype(float) - c["Prod_Merc_Positions_Short_All"].astype(float)) / oi
    return _frame(c.index, {"cot_mm_net__lvl": mm, "cot_mm_net__z": _trailing_z(mm),
                            "cot_mm_net__chg4": mm.diff(4), "cot_pm_net__z": _trailing_z(pm),
                            "cot_oi__chg4": np.log(oi).diff(4)}, lag_days=4)


def ethanol_features():
    import feature_engineering as fe
    e = pd.read_csv(RAW / "eia_ethanol_production.csv", parse_dates=["date"])
    df = pd.DataFrame({"obs_date": e.date, "ethanol": e.ethanol_kbd, "available": e.date + pd.Timedelta(days=5)})
    t = fe.transform_series(df)
    return t[["obs_date", "available", "ethanol__anomz", "ethanol__yoy", "ethanol__ma4chg"]]


def drought_features():
    d = pd.read_csv(RAW / "usdm_cornbelt.csv")
    d["d"] = pd.to_datetime(d.MapDate.astype(str), format="%Y%m%d")
    w = d.groupby("d").D1.mean()            # simple average of the 5 states, % area D1-D4
    w = w.sort_index()
    return _frame(w.index, {"drought_cb__lvl": w, "drought_cb__chg4": w.diff(4),
                            "drought_cb__chg1": w.diff(1)}, lag_days=2)


def river_features():
    import feature_engineering as fe
    r = pd.read_csv(RAW / "usgs_stlouis_stage.csv", parse_dates=["date"])
    r["stage_ft"] = pd.to_numeric(r.stage_ft, errors="coerce")
    wk = r.set_index("date").stage_ft.resample("W-THU").mean()
    fe.LEVEL_SERIES.add("river_stl")
    df = pd.DataFrame({"obs_date": wk.index, "river_stl": wk.values, "available": wk.index})
    t = fe.transform_series(df)
    return t[["obs_date", "available", "river_stl__lvl", "river_stl__anomz", "river_stl__ma4chg"]]


def crush_features():
    px = {}
    for t in ("zm", "zl", "zs"):
        s = pd.read_csv(RAW / f"yahoo_{t}.csv", parse_dates=["date"]).set_index("date").close
        px[t] = s
    df = pd.DataFrame(px).dropna()
    crush = df.zm * 0.022 + df.zl * 0.11 - df.zs / 100        # $/bu board crush
    wk = crush.resample("W-THU").last()
    return _frame(wk.index, {"crush__z52": (wk - wk.rolling(52, min_periods=40).mean()) / wk.rolling(52, min_periods=40).std(),
                             "crush__chg4": wk.diff(4)}, lag_days=0)


def all_factor_frames(sym):
    frames = [cot_features(sym), ethanol_features(), drought_features(), river_features()]
    if sym == "zs":
        frames.append(crush_features())
    return frames
