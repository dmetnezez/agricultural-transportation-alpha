"""
Bloomberg contract-level futures data (via an Excel export, no API needed).

Workflow
--------
1. `python -m src.bloomberg template`  -> writes bloomberg/bloomberg_futures_template.xlsx
2. On the Bloomberg terminal PC: open it in Excel (Bloomberg add-in loaded),
   wait until every block has filled (Bloomberg ribbon -> Refresh Workbook if
   needed), then File -> Save As -> keep .xlsx, name it
   bloomberg_futures_export.xlsx.
3. Copy that file into data/raw/bloomberg/ in this repo.
4. `python main.py --prices bloomberg` (research) or `python live.py` (signals).

Only prices come from Bloomberg. Transportation data stays on the free USDA
API (it is the same underlying data). Bloomberg's licence does not allow
redistributing its data, so data/raw/bloomberg/ is git-ignored: publish code
and results, never the export.

Ticker convention
-----------------
Bloomberg names a specific CBOT contract  <root><month code><year> Comdty,
root C = corn, S = soybeans. Contracts that have already expired use a
2-digit year (e.g. "CZ19 Comdty" = Dec-2019 corn); currently listed ones use
a 1-digit year ("CZ6 Comdty" = Dec-2026). The template applies that rule as of
the day it is generated. If a block shows #N/A on the terminal, type the other
year form into row 1 of that block. The loader reads the contract identity from
row 2 (written by this module), never from the ticker text, so either form works.

Roll rule (continuous series)
-----------------------------
Hold the nearest contract in the trading cycle whose roll date is still more
than one week away; roll date = first notice day (last business day of the
month before the contract month) minus 5 business days. A weekly return is
always computed on ONE contract, from Friday settle to Friday settle, so no
roll gap is ever booked and no week is lost. Each roll is flagged so the
backtest charges a two-sided roll cost.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BBG_DIR = ROOT / "data" / "raw" / "bloomberg"
TEMPLATE_DIR = ROOT / "bloomberg"
EXPORT_NAME = "bloomberg_futures_export.xlsx"

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
ROOTS = {"zc": "C", "zs": "S"}
# months held by the strategy (soybean Aug/Sep are thin and usually skipped by specs)
CYCLE = {"zc": [3, 5, 7, 9, 12], "zs": [1, 3, 5, 7, 11]}
FIELDS = ["PX_LAST", "PX_VOLUME", "OPEN_INT"]
BLOCK = len(FIELDS) + 2   # date + fields + one spacer column
START_YEAR = 2006


# --------------------------------------------------------------------------
# contract calendar
# --------------------------------------------------------------------------
def first_notice_day(year: int, month: int) -> pd.Timestamp:
    """CBOT grains: last business day of the month preceding the contract month."""
    return pd.Timestamp(year, month, 1) - pd.offsets.BDay(1)


def roll_date(year: int, month: int, bdays_before_fnd: int = 5) -> pd.Timestamp:
    return first_notice_day(year, month) - pd.offsets.BDay(bdays_before_fnd)


def contracts(sym: str, first_year=START_YEAR, last_year=None):
    last_year = last_year or dt.date.today().year + 1
    return [(y, m) for y in range(first_year, last_year + 1) for m in CYCLE[sym]]


def bbg_ticker(sym: str, year: int, month: int, today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    expired = pd.Timestamp(year, month, 15) < pd.Timestamp(today)
    yr = f"{year % 100:02d}" if expired else f"{year % 10}"
    return f"{ROOTS[sym]}{MONTH_CODES[month]}{yr} Comdty"


# --------------------------------------------------------------------------
# Excel template
# --------------------------------------------------------------------------
def make_template(path: Path | None = None) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    path = path or TEMPLATE_DIR / "bloomberg_futures_template.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    info = wb.active
    info.title = "README"
    lines = [
        "Agricultural Transportation Alpha - Bloomberg export template",
        "1. Open on a Bloomberg terminal PC with the Excel add-in loaded.",
        "2. Wait for every block to fill (Bloomberg ribbon > Refresh Workbook if needed).",
        "3. Any #N/A block: edit the ticker in row 1 (try 1-digit vs 2-digit year).",
        "4. Save As .xlsx named bloomberg_futures_export.xlsx; copy to data/raw/bloomberg/.",
        "Row 1 = Bloomberg ticker, row 2 = contract id read by the loader (do not edit), row 3 = BDH formula.",
        f"Fields: {', '.join(FIELDS)}  (PX_LAST on a futures contract = daily settlement).",
        "Bloomberg data may not be redistributed: do not commit the filled file to GitHub.",
    ]
    for i, t in enumerate(lines, 1):
        info.cell(i, 1, t).font = Font(bold=(i == 1))
    for sym in ("zc", "zs"):
        ws = wb.create_sheet(sym)
        for j, (y, m) in enumerate(contracts(sym)):
            col = 1 + j * BLOCK
            tk = bbg_ticker(sym, y, m)
            start = f"{y - 2}-01-01"
            end = (first_notice_day(y, m) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
            ws.cell(1, col, tk).font = Font(bold=True)
            ws.cell(2, col, f"{sym}|{y}|{m}")
            # header-less vertical history: date, PX_LAST, PX_VOLUME, OPEN_INT (trading days only)
            ws.cell(3, col, f'=BDH("{tk}","{",".join(FIELDS)}","{start}","{end}",'
                            f'"Dir=V","Dts=S","Days=T","Sort=A")')
    wb.save(path)
    return path


# --------------------------------------------------------------------------
# loading the filled export
# --------------------------------------------------------------------------
def _to_date(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return pd.Timestamp(v)
    if isinstance(v, (int, float)) and 20000 < v < 80000:      # Excel serial date
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))
    try:
        return pd.Timestamp(v)
    except Exception:
        return pd.NaT


def load_export(path: Path | None = None) -> pd.DataFrame:
    """Return long DataFrame [sym, year, month, date, settle, volume, oi].
    Prices are in cents/bu as Bloomberg reports them."""
    from openpyxl import load_workbook

    path = path or BBG_DIR / EXPORT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see src/bloomberg.py for the export steps")
    wb = load_workbook(path, data_only=True, read_only=True)
    rows, problems = [], []
    for sym in ("zc", "zs"):
        if sym not in wb.sheetnames:
            continue
        grid = list(wb[sym].iter_rows(values_only=True))
        if len(grid) < 3:
            continue
        width = max(len(r) for r in grid)
        for col in range(0, width, BLOCK):
            meta = grid[1][col] if col < len(grid[1]) else None
            if not meta or "|" not in str(meta):
                continue
            s, y, m = str(meta).split("|")
            got = 0
            for r in grid[2:]:
                if col >= len(r) or r[col] is None:
                    continue
                d = _to_date(r[col])
                px = r[col + 1] if col + 1 < len(r) else None
                if pd.isna(d) or not isinstance(px, (int, float)):
                    continue
                vol = r[col + 2] if col + 2 < len(r) else None
                oi = r[col + 3] if col + 3 < len(r) else None
                rows.append((s, int(y), int(m), d, float(px),
                             vol if isinstance(vol, (int, float)) else np.nan,
                             oi if isinstance(oi, (int, float)) else np.nan))
                got += 1
            if got == 0:
                problems.append(f"{s} {y}-{m:0>2} ({grid[0][col]})")
    if problems:
        print(f"  Bloomberg export: {len(problems)} contract blocks empty/#N/A, e.g. {problems[:5]}")
    df = pd.DataFrame(rows, columns=["sym", "year", "month", "date", "settle", "volume", "oi"])
    return df.sort_values(["sym", "year", "month", "date"]).reset_index(drop=True)


def from_long(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a CSV export with the same long columns (handy for testing)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------
# roll-correct weekly series
# --------------------------------------------------------------------------
def weekly_continuous(long: pd.DataFrame, sym: str, roll_bdays: int = 5) -> pd.DataFrame:
    """Weekly frame indexed by Friday 'week':
        close      settle of the contract held over the coming week (cents)
        contract   'YYYY-MM' of that contract
        ret        log return of the contract held over the PREVIOUS week,
                   measured on that same contract (Friday settle -> Friday settle)
        roll       True if the held contract changes at this week's close
    """
    d = long[long.sym == sym].copy()
    if d.empty:
        raise ValueError(f"no Bloomberg data for {sym}")
    d["key"] = list(zip(d.year, d.month))
    d["week"] = d.date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    # last settle of each contract in each week
    wk = d.sort_values("date").groupby(["key", "week"]).settle.last().unstack(0)
    keys = sorted(wk.columns)
    rolls = {k: roll_date(*k, roll_bdays) for k in keys}
    out = []
    held_prev = None
    for w in wk.index:
        # hold the nearest contract whose roll date is more than a week after w
        cand = [k for k in keys if rolls[k] > w + pd.Timedelta(days=7) and pd.notna(wk.at[w, k])]
        held = cand[0] if cand else None
        ret = np.nan
        if held_prev is not None and pd.notna(wk.at[w, held_prev]):
            prev_w = out[-1]["week"]
            p0 = wk.at[prev_w, held_prev]
            if pd.notna(p0) and (w - prev_w).days == 7:
                ret = np.log(wk.at[w, held_prev] / p0)
        out.append(dict(week=w, close=wk.at[w, held] if held else np.nan,
                        contract=f"{held[0]}-{held[1]:02d}" if held else None,
                        ret=ret, roll=held is not None and held_prev is not None and held != held_prev))
        held_prev = held
    return pd.DataFrame(out)


def available() -> bool:
    return (BBG_DIR / EXPORT_NAME).exists() or (BBG_DIR / "bloomberg_futures_long.csv").exists()


def load_any() -> pd.DataFrame:
    csv = BBG_DIR / "bloomberg_futures_long.csv"
    return from_long(pd.read_csv(csv)) if csv.exists() else load_export()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "template":
        print("written", make_template())
    else:
        long = load_any()
        for s in ("zc", "zs"):
            w = weekly_continuous(long, s)
            print(s, w.week.min().date(), "->", w.week.max().date(), "weeks with return:", int(w.ret.notna().sum()),
                  "rolls:", int(w.roll.sum()))
