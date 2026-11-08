"""
Data collection for the Agricultural Transportation Alpha project.

Every dataset is downloaded from its original public source and saved,
unmodified apart from server-side aggregation, to data/raw/. Nothing here
fills, interpolates or estimates missing values.

Sources
-------
USDA AMS AgTransport (Socrata open-data portal behind the Grain
Transportation Report, https://agtransport.usda.gov). Dataset ids:
    27k8-utc2  Grain Rail Cars Loaded and Billed, by railroad and STATE (weekly)
    tb7q-kn5i  AAR Rail Carloadings by commodity (weekly, national)
    n4pw-9ygw  Downbound Barge Grain Movements (tons) by lock (weekly)
    spd3-ku7f  Grain Barge Unloads in New Orleans Region (weekly)
    sruw-w49i  FGIS Grain Inspections for export (certificate level)
    wnn7-29tu  FAS Export Sales (weekly, by country)
    8uye-ieij  Grain Transportation Cost Indicators (weekly index)
    deqi-uken  Downbound Grain Barge Rates (weekly, % of tariff)
    v85y-3hep  Grain Basis (weekly, incl. Iowa cash market)
    vh5k-mnyv  Grain Vessel Loading Activity, Gulf & PNW (weekly, since 1995)

Yahoo Finance chart API: ZC=F and ZS=F, CBOT front-month continuous futures
(daily OHLCV). Expired single contracts are NOT available from Yahoo, which
constrains the roll methodology (see data_cleaning.py).

FHWA Travel Monitoring Analysis System (TMAS), monthly continuous-count-station
hourly VOLUME files (https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/).
These contain ALL vehicles, not trucks only; public files do not include the
vehicle-classification counts. Optional, large download (~25 MB per month).
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOCRATA = "https://agtransport.usda.gov/resource/{id}.json"
UA = {"User-Agent": "agtransport-research/1.0"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 180):
    """GET with exponential back-off. Raises after the last failure."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))


def socrata_query(dataset_id: str, soql: str, page: int = 50000) -> pd.DataFrame:
    """Run a SoQL query against AgTransport, paging until exhausted."""
    out, offset = [], 0
    while True:
        q = f"{soql} LIMIT {page} OFFSET {offset}"
        rows = _get(SOCRATA.format(id=dataset_id), {"$query": q}).json()
        out.extend(rows)
        if len(rows) < page:
            break
        offset += page
    return pd.DataFrame(out)


def _save(df: pd.DataFrame, name: str, meta: dict):
    path = RAW / f"{name}.csv"
    df.to_csv(path, index=False)
    meta = {**meta, "rows": int(len(df)), "downloaded_utc": pd.Timestamp.utcnow().isoformat()}
    (RAW / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved {path.name}: {len(df):,} rows")


# --------------------------------------------------------------------------
# USDA AgTransport
# --------------------------------------------------------------------------
AGT_QUERIES = {
    # Iowa-specific AND all-state rail cars (we keep every state; IA is filtered later)
    "rail_cars_by_state": ("27k8-utc2",
        "SELECT date, submission_date, railroad, state, `all`, dedicated_or_shuttle, other "
        "ORDER BY date, railroad, state"),
    "rail_carloadings_grain": ("tb7q-kn5i",
        "SELECT date, submission_date, railroad, type, carloads "
        "WHERE commodity = 'Grain' ORDER BY date, railroad"),
    "barge_grain_tons": ("n4pw-9ygw",
        "SELECT date, commodity, lock, tons WHERE commodity in ('Corn','Soybeans') "
        "ORDER BY date, lock"),
    "nola_barge_unloads": ("spd3-ku7f", "SELECT date, unloads ORDER BY date"),
    # inspections aggregated server-side to daily totals (raw table is ~600k rows)
    "export_inspections_daily": ("sruw-w49i",
        "SELECT date_trunc_ymd(cert_date) AS day, grain, sum(mt) AS mt "
        "WHERE grain in ('CORN','SOYBEANS') AND cert_date >= '2005-01-01' "
        "GROUP BY day, grain ORDER BY day"),
    "export_sales_weekly": ("wnn7-29tu",
        "SELECT date, commodity, sum(wkexportscmy) AS exports, sum(net_sales_total) AS net_sales, "
        "sum(outstanding_sales_total) AS outstanding WHERE commodity in ('Corn','Soybeans') "
        "AND date >= '2005-01-01' GROUP BY date, commodity ORDER BY date"),
    # grain vessels at export ports (Gulf: in port, loaded last 7 days, due next 10 days)
    "vessel_loading": ("vh5k-mnyv",
        "SELECT date, port, in_port, loaded_7_days, due_10_days ORDER BY date, port"),
    "transport_cost_indicators": ("8uye-ieij",
        "SELECT date, truck, shuttle_train, barge, gulf_vessel, pacific_vessel ORDER BY date"),
    "barge_rates": ("deqi-uken", "SELECT date, location, rate ORDER BY date"),
    "grain_basis_iowa": ("v85y-3hep",
        "SELECT date, market_name, market_type, commodity, bid, contract_month, futures_price, basis "
        "WHERE market_name in ('Iowa','Illinois','Gulf - Louisiana') AND commodity in ('Corn','Soybeans') "
        "ORDER BY date"),
}


def collect_agtransport():
    print("USDA AgTransport ...")
    for name, (ds, soql) in AGT_QUERIES.items():
        df = socrata_query(ds, soql)
        _save(df, name, {"source": "USDA AMS AgTransport", "dataset_id": ds,
                         "url": f"https://agtransport.usda.gov/d/{ds}", "soql": soql})


# --------------------------------------------------------------------------
# Futures (Yahoo Finance chart endpoint)
# --------------------------------------------------------------------------
def collect_futures(tickers=("ZC=F", "ZS=F"), start="2000-01-01"):
    print("Yahoo Finance futures ...")
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp.utcnow().timestamp())
    for t in tickers:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
        js = _get(url, {"period1": p1, "period2": p2, "interval": "1d"}).json()
        res = js["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
            "open": q["open"], "high": q["high"], "low": q["low"],
            "close": q["close"], "volume": q["volume"],
        }).dropna(subset=["close"])
        name = "futures_" + t.replace("=F", "").lower()
        _save(df, name, {"source": "Yahoo Finance chart API", "ticker": t, "url": url,
                         "note": "front-month continuous, NOT roll-adjusted, prices in US cents/bu"})


# --------------------------------------------------------------------------
# FHWA TMAS Iowa traffic volumes (optional, ~3 GB download)
# --------------------------------------------------------------------------
TMAS_INDEX = "https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/"


def collect_tmas_iowa(first_year=2014, last_year=2025):
    """Download monthly TMAS volume zips, keep only Iowa (state_code 19)
    and aggregate hourly counts to daily totals per station/direction.
    Each zip is discarded after extraction to keep disk use small."""
    print("FHWA TMAS (Iowa) ...")
    html = _get(TMAS_INDEX).text
    links = sorted(set(re.findall(r'href="(20\d\d/[^"]+\.zip)"', html)))
    links = [l for l in links if "station" not in l and first_year <= int(l[:4]) <= last_year]
    out_dir = RAW / "tmas_iowa"
    out_dir.mkdir(exist_ok=True)
    for link in links:
        dest = out_dir / (Path(link).stem + "_IA_daily.csv")
        if dest.exists():
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(_get(TMAS_INDEX + link, timeout=600).content))
        except Exception as e:  # record the gap rather than hiding it
            print(f"  FAILED {link}: {e}")
            continue
        # naming varies by year: "IA_JAN_2015 (TMAS).VOL", "IA0519.VOL"
        members = [m for m in z.namelist() if re.match(r"IA[_0-9]", Path(m).name.upper())]
        if not members:
            print(f"  no Iowa file in {link}")
            continue
        frames = []
        for m in members:
            raw = z.read(m).decode("latin-1")
            df = _parse_tmas_volume(raw)
            if df is not None:
                frames.append(df)
        if frames:
            pd.concat(frames).to_csv(dest, index=False)
            print(f"  {link}: {sum(len(f) for f in frames):,} station-days")


def _parse_tmas_volume(text: str) -> pd.DataFrame | None:
    """TMAS .VOL files are either pipe-delimited with a header (2021+)
    or the legacy fixed-width 3-card format (FHWA TMG)."""
    first = text.splitlines()[0] if text else ""
    if "|" in first:
        df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]   # header case varies by year
        hours = [c for c in df.columns if c.startswith("hour_")]
        df[hours] = df[hours].apply(pd.to_numeric, errors="coerce")
        df["volume"] = df[hours].sum(axis=1, min_count=24)  # require all 24 hours
        yr = df.year_record.astype(int)
        yr = yr.where(yr >= 100, yr + 2000)        # some files use 2-digit years
        df["date"] = pd.to_datetime(dict(year=yr,
                                         month=df.month_record.astype(int),
                                         day=df.day_record.astype(int)), errors="coerce")
        return df[["station_id", "f_system", "travel_dir", "travel_lane", "date", "volume"]]
    # legacy fixed-width (FHWA TMG "3" card): rec(1) state(2) fclass(2) station(6)
    # dir(1) lane(1) yy(2) mm(2) dd(2) dow(1) then 24 x 5-digit hourly volumes
    rows = []
    for line in text.splitlines():
        line = line.rstrip("\r")
        if len(line) < 140 or line[0] != "3":
            continue
        try:
            hours = [int(line[20 + 5 * i: 25 + 5 * i]) for i in range(24)]
            yy = int(line[13:15])
            date = pd.Timestamp(2000 + yy, int(line[15:17]), int(line[17:19]))
        except ValueError:
            continue  # malformed record: skipped, never imputed
        rows.append({"station_id": line[5:11], "f_system": line[3:5], "travel_dir": line[11],
                     "travel_lane": line[12], "date": date, "volume": sum(hours)})
    return pd.DataFrame(rows) if rows else None


if __name__ == "__main__":
    import sys
    collect_agtransport()
    collect_futures()
    if "--tmas" in sys.argv:
        collect_tmas_iowa()
