"""
Tests non-transport factors with the same point-in-time, walk-forward,
net-of-cost framework as main.py.

    python factor_research.py --download   # fetch CFTC / EIA / Drought / USGS / Yahoo
    python factor_research.py              # reuse data/raw/factors

Writes results/performance/factor_tests.csv and factor_tests.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import backtesting as B  # noqa: E402
import extra_factors as X  # noqa: E402
import performance_metrics as P  # noqa: E402
import statistical_models as S  # noqa: E402
from feature_engineering import align_point_in_time, build_dataset  # noqa: E402

OUT = ROOT / "results" / "performance"
COMMON_START = "2016-01-01"     # every factor is live by then -> like-for-like comparison
WF = S.WalkForward(min_train=156, refit_every=4)

FACTORS = {
    "Momentum (fitted)": ["mom_4w", "mom_12w", "mom_26w", "mom_52w"],
    "CFTC positioning": ["cot_mm_net__z", "cot_mm_net__chg4", "cot_pm_net__z", "cot_oi__chg4"],
    "Ethanol (EIA)": ["ethanol__anomz", "ethanol__yoy"],
    "Drought (Corn Belt)": ["drought_cb__lvl", "drought_cb__chg4", "drought_cb__chg1"],
    "River level (St. Louis)": ["river_stl__anomz", "river_stl__ma4chg"],
    "Crush margin (soy only)": ["crush__z52", "crush__chg4"],
    "Transport (best earlier)": ["ia_railcars__anomz", "vessel_gulf_due__anomz"],
}


def strat(df, pred, k=0.5, demean=False):
    p = pred.dropna()
    pos = B.positions_from_forecast(p, df["vol_26w"].reindex(p.index), k=k, demean=demean)
    return B.run_backtest(df, pos)


def timing_sharpe(df, pred):
    """Sharpe since COMMON_START of the pure-timing (de-meaned forecast) version."""
    return window(strat(df, pred, demean=True), COMMON_START).get("sharpe")


def window(bt, start=None):
    b = bt.loc[start:] if start else bt
    return P.summarize(b.pnl, b.pos)


def main():
    if "--download" in sys.argv:
        X.download_all()
    data = build_dataset(save=False, source="usda")
    rows, ens_inputs = [], {}
    for sym, df in data.items():
        r = df["ret"]
        df["mom_26w"] = r.rolling(26, min_periods=17).sum()
        df["mom_52w"] = r.rolling(52, min_periods=35).sum()
        for f in X.all_factor_frames(sym):
            al = align_point_in_time(f, df.index)
            for c in al.columns:
                df[c] = al[c]
        hm, _ = WF.run(df, ["mom_1w"], 1, "hist_mean")
        ens_inputs[sym] = {}
        for group, feats in FACTORS.items():
            for feat in feats:
                if feat not in df or df[feat].notna().sum() < 300:
                    continue
                pred, _ = WF.run(df, [feat], 1, "ols")
                st = S.oos_stats(df["fwd_1"], pred, hm.reindex(pred.index))
                bt = strat(df, pred)
                full, com = window(bt), window(bt, COMMON_START)
                lo, hi = P.block_bootstrap_sharpe(bt.loc[COMMON_START:].pnl)
                rows.append(dict(commodity=sym, group=group, feature=feat, model="walk-forward OLS",
                                 oos_start=str(pred.first_valid_index().date()) if pred.notna().any() else None,
                                 sharpe_full=full.get("sharpe"), sharpe_2016=com.get("sharpe"),
                                 sharpe_2016_lo=lo, sharpe_2016_hi=hi, ann_ret_2016=com.get("ann_return"),
                                 maxdd_2016=com.get("max_drawdown"), oos_r2=st.get("oos_r2_vs_mean"),
                                 cw_p=st.get("cw_p"), hit=st.get("hit_rate"),
                                 timing_sharpe_2016=timing_sharpe(df, pred),
                                 long_share_2016=(bt.loc[COMMON_START:].pos > 0).mean()))
                ens_inputs[sym][feat] = pred
        # rule-based time-series momentum (no fitting at all)
        vol = df["vol_26w"]
        for lb in ("mom_12w", "mom_26w", "mom_52w"):
            pos = (np.sign(df[lb]) * (0.10 / vol).clip(upper=3)).fillna(0)
            bt = B.run_backtest(df, pos.loc["2010-01-01":])
            full, com = window(bt), window(bt, COMMON_START)
            lo, hi = P.block_bootstrap_sharpe(bt.loc[COMMON_START:].pnl)
            rows.append(dict(commodity=sym, group="Momentum (rule: sign of past return)", feature=lb,
                             model="rule", sharpe_full=full.get("sharpe"), sharpe_2016=com.get("sharpe"),
                             sharpe_2016_lo=lo, sharpe_2016_hi=hi, ann_ret_2016=com.get("ann_return"),
                             maxdd_2016=com.get("max_drawdown")))
        # combinations
        feats_all = [f for f in ens_inputs[sym]]
        z = pd.DataFrame({f: p / p.rolling(52, min_periods=26).std().shift(1) for f, p in ens_inputs[sym].items()})
        ens = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1) >= 5)
        for name, pred in [("Equal-weight ensemble of all factors", ens)]:
            bt = strat(df, pred)
            full, com = window(bt), window(bt, COMMON_START)
            lo, hi = P.block_bootstrap_sharpe(bt.loc[COMMON_START:].pnl)
            rows.append(dict(commodity=sym, group="COMBINED", feature=name, model="average of OOS forecasts",
                             timing_sharpe_2016=timing_sharpe(df, pred),
                             long_share_2016=(bt.loc[COMMON_START:].pos > 0).mean(),
                             sharpe_full=full.get("sharpe"), sharpe_2016=com.get("sharpe"), sharpe_2016_lo=lo,
                             sharpe_2016_hi=hi, ann_ret_2016=com.get("ann_return"), maxdd_2016=com.get("max_drawdown")))
        core = [f for f in feats_all if not f.startswith("ethanol")]   # ethanol starts late
        for model in ("ridge", "lasso"):
            pred, _ = WF.run(df, core, 1, model, start="2007-01-01")
            st = S.oos_stats(df["fwd_1"], pred, hm.reindex(pred.index))
            bt = strat(df, pred)
            full, com = window(bt), window(bt, COMMON_START)
            lo, hi = P.block_bootstrap_sharpe(bt.loc[COMMON_START:].pnl)
            rows.append(dict(commodity=sym, group="COMBINED", feature=f"{model} on all factors", model=model,
                             timing_sharpe_2016=timing_sharpe(df, pred),
                             long_share_2016=(bt.loc[COMMON_START:].pos > 0).mean(),
                             sharpe_full=full.get("sharpe"), sharpe_2016=com.get("sharpe"), sharpe_2016_lo=lo,
                             sharpe_2016_hi=hi, ann_ret_2016=com.get("ann_return"), maxdd_2016=com.get("max_drawdown"),
                             oos_r2=st.get("oos_r2_vs_mean"), cw_p=st.get("cw_p")))
        # buy & hold reference
        bt = B.run_backtest(df, ((0.10 / vol).clip(upper=3)).fillna(0).loc["2010-01-01":])
        com = window(bt, COMMON_START)
        rows.append(dict(commodity=sym, group="BENCHMARK", feature="long only, vol-targeted", model="rule",
                         sharpe_2016=com.get("sharpe"), ann_ret_2016=com.get("ann_return"), maxdd_2016=com.get("max_drawdown")))
    t = pd.DataFrame(rows)
    ok = t.cw_p.notna()
    t.loc[ok, "cw_p_bh"] = multipletests(t.loc[ok, "cw_p"], method="fdr_bh")[1]
    t.to_csv(OUT / "factor_tests.csv", index=False)
    cols = ["commodity", "group", "feature", "sharpe_2016", "sharpe_2016_lo", "sharpe_2016_hi", "timing_sharpe_2016",
            "long_share_2016", "sharpe_full",
            "ann_ret_2016", "maxdd_2016", "oos_r2", "cw_p", "cw_p_bh"]
    md = t[cols].round(3).to_markdown(index=False) if hasattr(t, "to_markdown") else t[cols].round(3).to_string()
    (OUT / "factor_tests.md").write_text("# Factor tests (OOS, net of costs)\n\n" + md)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(t[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
