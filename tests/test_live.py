"""Tests for the Bloomberg loader, roll logic and paper-order planning (no network, no broker)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bloomberg as BB  # noqa: E402
import live_signal as L  # noqa: E402
import paper_trader as PT  # noqa: E402


def _fake_long(sym="zc"):
    """Two contracts with a deliberate 10% price gap between them."""
    rows = []
    for (y, m), base in [((2024, 3), 400.0), ((2024, 5), 440.0), ((2024, 7), 480.0)]:
        for d in pd.bdate_range("2023-10-01", BB.first_notice_day(y, m) + pd.Timedelta(days=20)):
            rows.append((sym, y, m, d, base + 0.0 * d.dayofyear, 1000, 5000))
    return pd.DataFrame(rows, columns=["sym", "year", "month", "date", "settle", "volume", "oi"])


def test_continuous_never_books_roll_gap():
    w = BB.weekly_continuous(_fake_long(), "zc")
    # prices are flat inside each contract, so every return must be exactly 0
    assert np.allclose(w.ret.dropna(), 0.0)
    assert w.roll.sum() >= 1
    # the held contract always rolls before first notice
    for _, r in w.dropna(subset=["contract"]).iterrows():
        y, m = map(int, r.contract.split("-"))
        assert r.week + pd.Timedelta(days=7) < BB.roll_date(y, m)


def test_template_and_export_roundtrip(tmp_path):
    from openpyxl import Workbook, load_workbook
    t = BB.make_template(tmp_path / "t.xlsx")
    wb = load_workbook(t)
    assert wb["zc"].cell(3, 1).value.startswith('=BDH("CH06 Comdty"')
    assert wb["zc"].cell(2, 1).value == "zc|2006|3"
    # simulate a filled export: replace the formula by values
    out = Workbook(); ws = out.active; ws.title = "zc"
    ws.cell(1, 1, "CH24 Comdty"); ws.cell(2, 1, "zc|2024|3")
    for i, d in enumerate(pd.bdate_range("2024-01-02", periods=5)):
        ws.cell(3 + i, 1, d.to_pydatetime()); ws.cell(3 + i, 2, 450 + i); ws.cell(3 + i, 3, 100); ws.cell(3 + i, 4, 900)
    out.save(tmp_path / "e.xlsx")
    df = BB.load_export(tmp_path / "e.xlsx")
    assert len(df) == 5 and df.settle.tolist() == [450, 451, 452, 453, 454]


def test_ticker_year_format():
    import datetime as dt
    assert BB.bbg_ticker("zc", 2019, 12, dt.date(2026, 11, 8)) == "CZ19 Comdty"
    assert BB.bbg_ticker("zs", 2027, 1, dt.date(2026, 11, 8)) == "SF7 Comdty"


def test_contract_sizing_and_cap():
    assert L.contracts_for(0.5, 100_000, 10.0) == 1          # $50k / $50k notional
    assert L.contracts_for(-3.0, 1_000_000, 4.0) == -L.LIVE_CONFIG["max_contracts"]
    assert L.contracts_for(np.nan, 100_000, 4.0) == 0


def test_plan_orders_rolls_and_respects_disabled():
    month = PT.contract_month_to_trade("zs")
    sig = {"signals": {
        "zs": {"enabled": True, "target_weight": 1.0, "ref_price_usd_bu": 10.0, "z": 1.2},
        "zc": {"enabled": False, "target_weight": 1.0, "ref_price_usd_bu": 4.0, "z": 2.0}}}
    pos = {"zs": {"200001": 1}, "zc": {PT.contract_month_to_trade("zc"): 2}}
    orders = PT.plan_orders(sig, pos, 100_000, {"zs": 10.0})
    o = {(x["sym"], x["month"]): x for x in orders}
    assert o[("zs", "200001")]["action"] == "SELL"            # old contract closed
    assert o[("zs", month)]["action"] == "BUY" and o[("zs", month)]["qty"] == 2
    assert o[("zc", PT.contract_month_to_trade("zc"))]["action"] == "SELL"   # disabled -> flatten
