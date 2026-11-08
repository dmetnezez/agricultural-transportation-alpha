"""Sanity tests for look-ahead bias. Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import backtesting as B  # noqa: E402
import data_cleaning as dc  # noqa: E402
import feature_engineering as fe  # noqa: E402
import statistical_models as S  # noqa: E402


def test_transport_features_published_before_decision():
    weeks = pd.date_range("2016-01-01", "2026-10-30", freq="W-FRI")
    for r in dc.build_raw_weekly():
        t = fe.transform_series(r).dropna(subset=["available"]).sort_values("available")
        t = t.assign(avail_key=t.available + pd.Timedelta(days=1))
        m = pd.merge_asof(pd.DataFrame({"week": weeks}), t[["avail_key", "available"]],
                          left_on="week", right_on="avail_key").dropna()
        assert (m.available < m.week).all(), r.columns[1]


def test_seasonal_anomaly_uses_only_past():
    idx = pd.date_range("2010-01-01", periods=400, freq="W-FRI")
    s = pd.Series(np.random.default_rng(0).normal(size=400), index=idx)
    a1 = fe._seasonal_anomaly(s)
    s2 = s.copy(); s2.iloc[300:] += 100          # change the future only
    a2 = fe._seasonal_anomaly(s2)
    pd.testing.assert_series_equal(a1.iloc[:300], a2.iloc[:300])


def test_walk_forward_ignores_future_targets():
    idx = pd.date_range("2010-01-01", periods=400, freq="W-FRI")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=400), "fwd_1": rng.normal(size=400)}, index=idx)
    p1, _ = S.WalkForward().run(df, ["x"], 1, "ols")
    df2 = df.copy(); df2.iloc[250:, df.columns.get_loc("fwd_1")] = 50.0   # corrupt future targets
    p2, _ = S.WalkForward().run(df2, ["x"], 1, "ols")
    # forecast at row 250 may use targets up to row 249 only
    pd.testing.assert_series_equal(p1.iloc[:251], p2.iloc[:251])


def test_backtest_flat_when_return_excluded():
    idx = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    df = pd.DataFrame({"fwd_1": [0.01, np.nan, 0.02, 0.0, -0.01, 0.01], "price": 4.0}, index=idx)
    bt = B.run_backtest(df, pd.Series(1.0, index=idx))
    assert bt.pos.iloc[1] == 0 and bt.gross.iloc[1] == 0
