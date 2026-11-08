"""Research charts (matplotlib). Palette: fixed categorical order, single y-axis per panel."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "results" / "charts"
CH.mkdir(parents=True, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(CH / name, dpi=160)
    plt.close(fig)


def transport_volumes(df):
    cols = [("ia_railcars", "Iowa grain rail cars loaded (cars/wk)  [IOWA]"),
            ("us_railcars", "All-US grain rail cars loaded (cars/wk)  [NATIONAL]"),
            ("barge_com_tons", "Downbound corn barge tons, 7 locks  [NATIONAL]"),
            ("insp_com_mt", "Corn export inspections (metric t/wk)  [NATIONAL]")]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, (c, t), col in zip(axes, cols, C):
        s = df[c].dropna()
        ax.plot(s.index, s.values, color=col, lw=0.9, alpha=0.55)
        ax.plot(s.index, s.rolling(4).mean(), color=col, lw=1.8, label="4-wk avg")
        ax.set_title(t, loc="left", fontsize=10)
    axes[0].legend(loc="upper left")
    fig.suptitle("Fig 1. Weekly grain transportation activity (by observation week)", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, "01_transport_volumes.png")


def price_chart(feat, yahoo, sym, name, fig_no):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(yahoo.date, yahoo.close / 100, color=C[0], lw=1.0, label="CBOT front month, daily close (Yahoo; robustness, rolls at expiry)")
    ax.plot(feat.index, feat.price, color=C[1], lw=1.0, alpha=0.8, label="USDA AMS weekly reference-contract print (primary)")
    ax.set_ylabel("$/bushel")
    ax.set_title(f"Fig {fig_no}. CBOT {name} futures ({sym.upper()})", loc="left")
    ax.legend(loc="upper left")
    _save(fig, f"0{fig_no}_{name.lower()}_price.png")


def scatter_feature(dfs, feature, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (sym, df), col in zip(axes, dfs.items(), C):
        m = df[[feature, "fwd_1"]].dropna()
        ax.scatter(m[feature], m.fwd_1 * 100, s=14, color=col, alpha=0.5, edgecolor="#fcfcfb", lw=0.5)
        b = np.polyfit(m[feature], m.fwd_1 * 100, 1)
        xs = np.linspace(m[feature].min(), m[feature].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=INK, lw=1.5)
        ax.set_title(f"{'Corn' if sym == 'zc' else 'Soybeans'}  (slope {b[0]:.2f}%/sd, n={len(m)})", loc="left", fontsize=10)
        ax.set_xlabel(label)
    axes[0].set_ylabel("Next-week futures return (%)")
    fig.suptitle(f"Fig 4. Transportation anomaly vs next-week return (in-sample)", x=0.01, ha="left", fontweight="bold")
    _save(fig, "04_transport_vs_returns.png")


def rolling_corr_chart(rc: dict):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, (sym, d) in zip(axes, rc.items()):
        for (k, s), col in zip(d.items(), C):
            ax.plot(s.index, s.values, color=col, label=k)
        ax.axhline(0, color=INK2, lw=0.8)
        for lvl in (0.196, -0.196):
            ax.axhline(lvl, color=INK2, lw=0.6, ls=":")
        ax.set_title("Corn" if sym == "zc" else "Soybeans", loc="left", fontsize=10)
        ax.set_ylabel("104-wk corr with next-wk return")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle("Fig 5. Rolling 2-year correlations (dotted: approx. 5% significance band)", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, "05_rolling_correlations.png")


def coef_chart(tabs: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, (sym, t) in zip(axes, tabs.items()):
        t = t[t.horizon == 1].set_index("feature").sort_values("t_hac")
        y = np.arange(len(t))
        sig = t.p_bh < 0.05
        ax.barh(y, t.t_hac, color=[C[0] if s else "#b9b8b3" for s in sig], height=0.6)
        ax.axvline(1.96, color=INK2, ls=":", lw=0.8); ax.axvline(-1.96, color=INK2, ls=":", lw=0.8)
        ax.set_yticks(y, [s.replace("__anomz", "") for s in t.index], fontsize=8)
        ax.set_title(("Corn" if sym == "zc" else "Soybeans") + ": HAC t-stat, 1-wk horizon", loc="left", fontsize=10)
    fig.suptitle("Fig 6. Univariate predictive regressions (blue = survives Benjamini-Hochberg 5% FDR)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, "06_regression_tstats.png")


def actual_vs_pred(preds: dict, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, (sym, (y, p)) in enumerate(preds.items()):
        m = pd.concat([y, p], axis=1, keys=["y", "p"]).dropna()
        a = axes[row, 0]
        a.plot(m.index, m.y * 100, color="#b9b8b3", lw=0.8, label="actual")
        a.plot(m.index, m.p * 100, color=C[row], lw=1.6, label="OOS forecast")
        a.set_title(("Corn" if sym == "zc" else "Soybeans") + " - weekly return (%)", loc="left", fontsize=10)
        a.legend(loc="upper left", fontsize=8)
        b = axes[row, 1]
        b.scatter(m.p * 100, m.y * 100, s=12, color=C[row], alpha=0.5)
        b.axhline(0, color=INK2, lw=0.6); b.axvline(0, color=INK2, lw=0.6)
        b.set_xlabel("forecast (%)"); b.set_ylabel("actual (%)")
        b.set_title(f"corr = {m.p.corr(m.y):.3f}, n = {len(m)}", loc="left", fontsize=10)
    fig.suptitle(f"Fig 7. Actual vs predicted (walk-forward, out-of-sample): {title}", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, "07_actual_vs_predicted.png")


def cumulative(curves: dict, fname="08_cumulative_returns.png", title="Fig 8. Cumulative net returns (out-of-sample)"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (sym, d) in zip(axes, curves.items()):
        for (k, s), col in zip(d.items(), C):
            eq = np.expm1(s.cumsum()) * 100
            ax.plot(eq.index, eq.values, color=col, label=k, lw=2.0 if k.startswith("STRAT") else 1.3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title("Corn" if sym == "zc" else "Soybeans", loc="left", fontsize=10)
        ax.set_ylabel("cumulative return (%)")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    _save(fig, fname)


def drawdown(curves: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (sym, d) in zip(axes, curves.items()):
        for (k, s), col in zip(d.items(), C):
            eq = s.cumsum()
            dd = np.expm1(eq - eq.cummax()) * 100
            ax.plot(dd.index, dd.values, color=col, label=k, lw=1.3)
        ax.set_title("Corn" if sym == "zc" else "Soybeans", loc="left", fontsize=10)
        ax.set_ylabel("drawdown (%)")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Fig 9. Strategy drawdowns (out-of-sample, net of costs)", x=0.01, ha="left", fontweight="bold")
    _save(fig, "09_drawdowns.png")


def seasonal(monthly: dict, season_profile: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, (sym, s) in enumerate(monthly.items()):
        ax = axes[i]
        ax.bar(s.index, s.values * 100, color=[C[i] if v >= 0 else "#b9b8b3" for v in s.values], width=0.7)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax.set_title(("Corn" if sym == "zc" else "Soybeans") + ": avg weekly strategy P&L by month (%)",
                     loc="left", fontsize=10)
    ax = axes[2]
    for (k, s), col in zip(season_profile.items(), C):
        ax.plot(s.index, s.values, color=col, label=k)
    ax.set_xlabel("week of year"); ax.set_title("Seasonal profile: Iowa rail cars/wk by year", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Fig 10. Seasonality of strategy performance and of the underlying transport data", x=0.01,
                 ha="left", fontweight="bold")
    _save(fig, "10_seasonal_performance.png")


def permutation_hist(null: dict, actual: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sym, col in zip(axes, null, C):
        ax.hist(null[sym], bins=40, color="#b9b8b3")
        ax.axvline(actual[sym], color=col, lw=2.2, label=f"actual Sharpe {actual[sym]:.2f}")
        p = (np.array(null[sym]) >= actual[sym]).mean()
        ax.set_title(("Corn" if sym == "zc" else "Soybeans") + f": permutation p = {p:.3f}", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("Sharpe under null (feature circularly shifted)")
    fig.suptitle("Fig 11. Is the primary strategy's Sharpe distinguishable from luck?", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, "11_permutation_test.png")


def cost_sensitivity(tab: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    for (k, g), col in zip(tab.groupby("series"), C):
        ax.plot(g.value.astype(float), g.sharpe, marker="o", ms=6, color=col, label=k)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("cost multiple (1 = 1 tick slippage + $2.50/side)"); ax.set_ylabel("OOS Sharpe")
    ax.legend(fontsize=8)
    ax.set_title("Fig 12. Sensitivity of net Sharpe to transaction costs", loc="left")
    _save(fig, "12_cost_sensitivity.png")
