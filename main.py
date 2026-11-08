"""
Agricultural Transportation Alpha - full research pipeline.

    python main.py              # use data already in data/raw
    python main.py --download   # re-download USDA + Yahoo data first
    python main.py --download --tmas   # also FHWA Iowa traffic (~3 GB, slow)

Outputs go to data/processed and results/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import backtesting as B  # noqa: E402
import performance_metrics as P  # noqa: E402
import statistical_models as S  # noqa: E402
import visualization as V  # noqa: E402
from feature_engineering import build_dataset  # noqa: E402

RES = ROOT / "results"
REG, PERF = RES / "regression_results", RES / "performance"
for d in (REG, PERF):
    d.mkdir(parents=True, exist_ok=True)

NAMES = {"zc": "Corn", "zs": "Soybeans"}
PRIMARY_FEATURE = "ia_railcars__anomz"      # pre-specified: the Iowa hypothesis
IOWA_START = "2016-10-01"                   # first week the Iowa anomaly is defined
LONG_START = "2009-01-01"
K_DEFAULT, TV = 0.5, 0.10
WF = S.WalkForward(min_train=156, refit_every=4)

IOWA_UNIVERSE = ["ia_railcars__anomz", "us_railcars__anomz", "barge_com_tons__anomz", "nola_unloads__anomz",
                 "insp_com_mt__anomz", "sales_com_exports__anomz", "sales_com_net_sales__anomz",
                 "cost_truck__anomz", "cost_barge__anomz", "cost_gulf_vessel__anomz", "barge_rate_stl__anomz",
                 "basis_ia_com__anomz", "basis_gulf_com__anomz"]
LONG_UNIVERSE = ["nola_unloads__anomz", "insp_com_mt__anomz", "sales_com_exports__anomz",
                 "sales_com_net_sales__anomz", "cost_truck__anomz", "cost_barge__anomz",
                 "barge_rate_stl__anomz", "basis_ia_com__anomz", "basis_gulf_com__anomz"]
CTRL = ["mom_4w", "mom_12w", "vol_12w", "harvest", "planting", "growing"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strat(df, pred, k=K_DEFAULT, cost_mult=1.0, ticks=1.0):
    p = pred.dropna()
    pos = B.positions_from_forecast(p, df["vol_26w"].reindex(p.index), k=k, target_vol=TV)
    return B.run_backtest(df, pos, cost_mult=cost_mult, ticks=ticks)


def eval_window(bt, start=None, end=None):
    b = bt.loc[start:end]
    return P.summarize(b.pnl, b.pos)


def main():
    if "--download" in sys.argv:
        import data_collection as DC
        DC.collect_agtransport()
        DC.collect_futures()
        if "--tmas" in sys.argv:
            DC.collect_tmas_iowa()

    log("building point-in-time features")
    data = build_dataset(source="usda")           # primary returns (validated USDA prints)
    data_alt = build_dataset(source="yahoo")      # robustness returns (Yahoo front month)
    summary = {}

    # ---------------------------------------------------------------- in-sample
    log("in-sample statistics")
    uni_tabs = {}
    for sym, df in data.items():
        t = S.univariate_table(df, S.PRIMARY)
        t.to_csv(REG / f"univariate_primary_{sym}.csv", index=False)
        uni_tabs[sym] = t
        ti = S.univariate_table(df, S.PRIMARY, start=IOWA_START)
        ti.to_csv(REG / f"univariate_primary_{sym}_iowa_sample.csv", index=False)
        # all transforms (secondary, much larger multiple-testing burden)
        allf = [c for c in df.columns if "__" in c and not c.startswith("ia_rural")]
        S.univariate_table(df, allf).to_csv(REG / f"univariate_all_transforms_{sym}.csv", index=False)
        # alternative target from Yahoo front-month
        dy = df.copy(); dy["fwd_1"] = dy["fwd_1_alt"]
        S.univariate_table(dy, S.PRIMARY, horizons=(1,)).to_csv(REG / f"univariate_primary_{sym}_yahoo_target.csv", index=False)
        # FHWA Iowa traffic: ex-post diagnostic only
        tm = [c for c in df.columns if c.startswith("ia_rural")]
        S.univariate_table(df, tm).to_csv(REG / f"univariate_iowa_traffic_diagnostic_{sym}.csv", index=False)

        with open(REG / f"multiple_regression_{sym}.txt", "w") as fh:
            for lab, feats, st in [("Iowa universe + controls (h=1)", IOWA_UNIVERSE, IOWA_START),
                                   ("Long-history national universe + controls (h=1)", LONG_UNIVERSE, LONG_START),
                                   ("Controls only, Iowa sample (h=1)", [], IOWA_START)]:
                res, m = S.multiple_regression(df, feats, 1, CTRL, start=st)
                fh.write(f"===== {NAMES[sym]}: {lab} =====\n{res.summary()}\n")
                fh.write(f"Joint F-test transport coefficients = 0: ")
                if feats:
                    names = [f for f in feats if f in m.columns]
                    R = np.zeros((len(names), len(res.params)))
                    for i, f in enumerate(names):
                        R[i, list(res.params.index).index(f)] = 1
                    w = res.wald_test(R, scalar=True)
                    fh.write(f"stat={float(w.statistic):.3f}, p={float(w.pvalue):.4f}\n\n")
                else:
                    fh.write("n/a\n\n")
        S.adf_table(df, S.PRIMARY).to_csv(REG / f"adf_{sym}.csv", index=False)
        S.granger_table(df, S.PRIMARY).to_csv(REG / f"granger_{sym}.csv", index=False)
        ac, lb = S.autocorr_returns(df)
        ac.to_csv(REG / f"return_autocorr_{sym}.csv", index=False)
        lb.to_csv(REG / f"return_ljungbox_{sym}.csv")
        summary[sym] = {"n_weeks": int(df.fwd_1.notna().sum()),
                        "in_sample_bh_significant_h1": t[(t.horizon == 1) & (t.p_bh < 0.05)].feature.tolist(),
                        "in_sample_bh_significant_any": t[t.p_bh < 0.05][["feature", "horizon"]].values.tolist()}

    # ---------------------------------------------------------------- OOS univariate
    log("walk-forward univariate forecasts")
    oos_rows, preds = [], {s: {} for s in data}
    for sym, df in data.items():
        hm, _ = WF.run(df, ["mom_1w"], 1, "hist_mean")
        preds[sym]["hist_mean"] = hm
        for f in S.PRIMARY + ["mom_4w", "mom_12w"]:
            for h in (1, 2, 4):
                p, _ = WF.run(df, [f], h, "ols")
                bench = hm if h == 1 else WF.run(df, ["mom_1w"], h, "hist_mean")[0]
                st = S.oos_stats(df[f"fwd_{h}"], p, bench.reindex(p.index))
                st.update(commodity=sym, feature=f, horizon=h, model="ols_univariate")
                if h == 1:
                    preds[sym][f] = p
                    bt = strat(df, p)
                    m = P.summarize(bt.pnl, bt.pos)
                    st.update(strategy_sharpe=m.get("sharpe"), strategy_ann_return=m.get("ann_return"))
                oos_rows.append(st)
    oos_uni = pd.DataFrame(oos_rows)
    oos_uni["cw_p_bh"] = np.nan
    ok = oos_uni.cw_p.notna()
    from statsmodels.stats.multitest import multipletests
    oos_uni.loc[ok, "cw_p_bh"] = multipletests(oos_uni.loc[ok, "cw_p"], method="fdr_bh")[1]
    oos_uni.to_csv(PERF / "oos_univariate.csv", index=False)

    # ---------------------------------------------------------------- OOS multivariate / ML
    log("walk-forward multivariate & ML forecasts")
    mv_rows = []
    specs = [("iowa", IOWA_UNIVERSE, IOWA_START), ("long", LONG_UNIVERSE, LONG_START)]
    for sym, df in data.items():
        for uname, feats, st in specs:
            hm = WF.run(df.loc[st:], ["mom_1w"], 1, "hist_mean")[0]
            for model, fs in [("seasonal_only_ols", ["harvest", "planting", "growing"]),
                              ("controls_only_ridge", CTRL),
                              ("kitchen_sink_ols", feats + CTRL),
                              ("ridge", feats + CTRL), ("lasso", feats + CTRL),
                              ("rf", feats + CTRL), ("best_uni", feats)]:
                mname = "ols" if model.endswith("_ols") else model.replace("controls_only_", "")
                p, info = WF.run(df, fs, 1, mname, start=st)
                s_ = S.oos_stats(df["fwd_1"], p, hm.reindex(p.index))
                bt = strat(df, p)
                m = P.summarize(bt.pnl, bt.pos)
                s_.update(commodity=sym, universe=uname, model=model, strategy_sharpe=m.get("sharpe"),
                          strategy_ann_return=m.get("ann_return"), max_dd=m.get("max_drawdown"))
                if model == "best_uni":
                    s_["features_picked"] = info.dropna().value_counts().to_dict().__repr__()
                if model == "lasso":
                    s_["features_picked"] = info.dropna().str.split(",").explode().value_counts().head(5).to_dict().__repr__()
                mv_rows.append(s_)
                preds[sym][f"{uname}_{model}"] = p
    oos_mv = pd.DataFrame(mv_rows)
    oos_mv.to_csv(PERF / "oos_multivariate.csv", index=False)

    # ---------------------------------------------------------------- strategies & benchmarks
    log("backtests")
    metrics, curves, bts = [], {}, {}
    for sym, df in data.items():
        p_primary = preds[sym][PRIMARY_FEATURE]
        main_bt = {"STRAT iowa-rail": strat(df, p_primary),
                   "STRAT ridge (iowa univ.)": strat(df, preds[sym]["iowa_ridge"]),
                   "STRAT best-univariate (iowa univ.)": strat(df, preds[sym]["iowa_best_uni"])}
        # common evaluation window = OOS period of the primary strategy once its z-score is live
        start = main_bt["STRAT iowa-rail"].pos.ne(0).idxmax()
        idx = df.loc[start:].index[df.loc[start:].index <= df.fwd_1.dropna().index.max()]
        bench = {k: B.run_backtest(df, v) for k, v in B.benchmark_positions(df, idx, TV).items()}
        allbt = {**main_bt, **bench}
        bts[sym] = allbt
        curves[sym] = {}
        for k, bt in allbt.items():
            b = bt.loc[idx[0]: idx[-1]]
            m = P.summarize(b.pnl, b.pos)
            lo, hi = P.block_bootstrap_sharpe(b.pnl)
            m.update(commodity=sym, strategy=k, sharpe_ci_lo=lo, sharpe_ci_hi=hi,
                     start=str(idx[0].date()), end=str(idx[-1].date()))
            metrics.append(m)
            if k != "cash":
                curves[sym][k] = b.pnl
        summary[sym]["oos_window"] = [str(idx[0].date()), str(idx[-1].date())]
    met = pd.DataFrame(metrics)
    met.to_csv(PERF / "strategy_metrics.csv", index=False)

    # ---------------------------------------------------------------- robustness
    log("robustness")
    rob = []
    for sym, df in data.items():
        p = preds[sym][PRIMARY_FEATURE]
        pr = preds[sym]["iowa_ridge"]
        win = summary[sym]["oos_window"]
        for lab, pp in [("iowa-rail", p), ("ridge", pr)]:
            for k in (0.0, 0.25, 0.5, 1.0, 1.5):
                m = eval_window(strat(df, pp, k=k), *win); rob.append(dict(commodity=sym, series=lab, test="threshold", value=k, **m))
            for cm in (0, 1, 2, 4):
                m = eval_window(strat(df, pp, cost_mult=cm), *win); rob.append(dict(commodity=sym, series=lab, test="cost_mult", value=cm, **m))
            bt = strat(df, pp)
            mid = pd.Timestamp(win[0]) + (pd.Timestamp(win[1]) - pd.Timestamp(win[0])) / 2
            for nm, a, b in [("first_half", win[0], mid), ("second_half", mid, win[1])]:
                m = eval_window(bt, a, b); rob.append(dict(commodity=sym, series=lab, test="subperiod", value=nm, **m))
            for yr, g in bt.loc[win[0]:win[1]].groupby(bt.loc[win[0]:win[1]].index.year):
                m = P.summarize(g.pnl, g.pos); rob.append(dict(commodity=sym, series=lab, test="calendar_year", value=yr, **m))
            sea = df["harvest"].reindex(bt.index) == 1
            for nm, mask in [("harvest_SepNov", sea), ("non_harvest", ~sea)]:
                b = bt.loc[win[0]:win[1]]; mk = mask.loc[b.index]
                m = P.summarize(b.pnl[mk], b.pos[mk]); rob.append(dict(commodity=sym, series=lab, test="season", value=nm, **m))
        # holding period / forecast horizon (primary feature)
        for h in (1, 2, 4):
            ph, _ = WF.run(df, [PRIMARY_FEATURE], h, "ols")
            m = eval_window(strat(df, ph), *win); rob.append(dict(commodity=sym, series="iowa-rail", test="forecast_horizon", value=h, **m))
        # training window length
        for mt in (104, 156, 260):
            pw, _ = S.WalkForward(min_train=mt).run(df, [PRIMARY_FEATURE], 1, "ols")
            m = eval_window(strat(df, pw), *win); rob.append(dict(commodity=sym, series="iowa-rail", test="min_train_weeks", value=mt, **m))
        # alternative transforms of the Iowa and US rail series (seasonal vs non-seasonal)
        for base in ("ia_railcars", "us_railcars"):
            for tr in ("wow", "ma4chg", "rel52", "yoy", "anom", "anomz"):
                f = f"{base}__{tr}"
                pf, _ = WF.run(df, [f], 1, "ols")
                m = eval_window(strat(df, pf), *win); rob.append(dict(commodity=sym, series=base, test="transform", value=tr, **m))
        # every primary variable as a stand-alone strategy, same window
        for f in S.PRIMARY:
            if f in preds[sym]:
                m = eval_window(strat(df, preds[sym][f]), *win); rob.append(dict(commodity=sym, series=f, test="variable", value=f, **m))
        # alternative price source (Yahoo front month) for the primary strategy
        du = data_alt[sym]
        pu, _ = WF.run(du, [PRIMARY_FEATURE], 1, "ols")
        hmu, _ = WF.run(du, ["mom_1w"], 1, "hist_mean")
        stu = S.oos_stats(du["fwd_1"], pu, hmu.reindex(pu.index))
        m = eval_window(strat(du, pu), *win)
        rob.append(dict(commodity=sym, series="iowa-rail", test="price_source", value="yahoo_front",
                        oos_r2=stu.get("oos_r2_vs_mean"), cw_p=stu.get("cw_p"), **m))
        stp = S.oos_stats(df["fwd_1"], p, preds[sym]["hist_mean"].reindex(p.index))
        m = eval_window(strat(df, p), *win)
        rob.append(dict(commodity=sym, series="iowa-rail", test="price_source", value="usda_weekly (primary)",
                        oos_r2=stp.get("oos_r2_vs_mean"), cw_p=stp.get("cw_p"), **m))
        # execution delay: trade one week later than the signal
        pdl = p.shift(1)
        m = eval_window(strat(df, pdl), *win); rob.append(dict(commodity=sym, series="iowa-rail", test="execution_delay_weeks", value=1, **m))
    robdf = pd.DataFrame(rob)
    robdf.to_csv(PERF / "robustness.csv", index=False)

    # ---------------------------------------------------------------- permutation test
    log("permutation test (circular shifts of the Iowa rail feature)")
    null, actual = {}, {}
    rng = np.random.default_rng(42)
    for sym, df in data.items():
        win = summary[sym]["oos_window"]
        actual[sym] = eval_window(bts[sym]["STRAT iowa-rail"], *win)["sharpe"]
        x = df[PRIMARY_FEATURE]
        valid = x.dropna()
        vals = []
        for _ in range(300):
            k = int(rng.integers(52, len(valid) - 52))
            shifted = pd.Series(np.roll(valid.values, k), index=valid.index).reindex(df.index)
            d2 = df.copy(); d2[PRIMARY_FEATURE] = shifted
            pp, _ = WF.run(d2, [PRIMARY_FEATURE], 1, "ols")
            vals.append(eval_window(strat(d2, pp), *win).get("sharpe", np.nan))
        null[sym] = [v for v in vals if pd.notna(v)]
        summary[sym]["permutation_p"] = float((np.array(null[sym]) >= actual[sym]).mean())
    pd.DataFrame({s: pd.Series(v) for s, v in null.items()}).to_csv(PERF / "permutation_null_sharpes.csv", index=False)

    # ---------------------------------------------------------------- charts
    log("charts")
    import data_cleaning as DC
    V.transport_volumes(pd.concat([pd.DataFrame({r.columns[1]: r.set_index("obs_date")[r.columns[1]]})
                                   for r in DC.build_raw_weekly()
                                   if r.columns[1] in ("ia_railcars", "us_railcars", "barge_corn_tons", "insp_corn_mt")],
                                  axis=1).rename(columns={"barge_corn_tons": "barge_com_tons", "insp_corn_mt": "insp_com_mt"}))
    for i, sym in enumerate(("zc", "zs")):
        y = pd.read_csv(ROOT / "data/raw" / f"futures_{sym}.csv", parse_dates=["date"])
        V.price_chart(data[sym], y[y.date >= "2007-01-01"], sym, NAMES[sym], 2 + i)
    V.scatter_feature(data, PRIMARY_FEATURE, "Iowa rail cars, seasonal anomaly (z)")
    V.rolling_corr_chart({s: {f.replace("__anomz", ""): S.rolling_corr(d, f) for f in
                              ["ia_railcars__anomz", "us_railcars__anomz", "insp_com_mt__anomz", "basis_ia_com__anomz"]}
                          for s, d in data.items()})
    V.coef_chart(uni_tabs)
    V.actual_vs_pred({s: (data[s]["fwd_1"], preds[s][PRIMARY_FEATURE]) for s in data}, "Iowa rail anomaly model")
    V.cumulative(curves)
    V.drawdown({s: {k: v for k, v in c.items() if k in ("STRAT iowa-rail", "STRAT ridge (iowa univ.)", "long_voltarget", "momentum_12w")}
                for s, c in curves.items()})
    monthly = {}
    for s in data:
        win = summary[s]["oos_window"]
        b = bts[s]["STRAT iowa-rail"].loc[win[0]:win[1]]
        monthly[s] = b.pnl.groupby(b.index.month).mean().reindex(range(1, 13))
    raw_ia = [r for r in DC.build_raw_weekly() if r.columns[1] == "ia_railcars"][0].set_index("obs_date")["ia_railcars"]
    prof = pd.DataFrame({yr: g.groupby(g.index.isocalendar().week.values).mean()
                         for yr, g in raw_ia.groupby(raw_ia.index.year) if yr in (2016, 2018, 2020, 2022, 2024, 2025)})
    V.seasonal(monthly, prof)
    V.permutation_hist(null, actual)
    V.cost_sensitivity(robdf[robdf.test == "cost_mult"].assign(series=lambda d: d.commodity.map(NAMES) + " " + d.series))

    (PERF / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("done")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(met[["commodity", "strategy", "ann_return", "ann_vol", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
                   "max_drawdown", "n_trades", "exposure"]].round(3).to_string())
        print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
