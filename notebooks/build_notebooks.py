"""Generates the five walkthrough notebooks. They call the same src/ code as
main.py and read its saved outputs, so they stay consistent with the report.
Run:  python notebooks/build_notebooks.py && jupyter nbconvert --execute --inplace notebooks/0*.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd, numpy as np
from IPython.display import Image, display
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
R = ROOT / 'results'"""

NB = {
    "01_data_exploration.ipynb": [
        ("md", "# 01 - Data exploration\nRaw downloads, coverage, gaps and publication lags. "
               "Every file in `data/raw` has a `.meta.json` with its source URL and query."),
        ("code", SETUP),
        ("code", "for m in sorted((ROOT/'data/raw').glob('*.meta.json')):\n"
                 "    j = json.loads(m.read_text()); print(f\"{m.stem.replace('.meta',''):28s} rows={j['rows']:>7,}  {j.get('url','')}\")"),
        ("code", "import data_cleaning as dc\nfor s in dc.build_raw_weekly():\n"
                 "    c = s.columns[1]; lag = (s.available - s.obs_date).dt.days.median()\n"
                 "    print(f'{c:30s} {s.obs_date.min().date()} -> {s.obs_date.max().date()}  n={s[c].notna().sum():4d}  missing={s[c].isna().sum():3d}  median lag={lag:.0f}d')"),
        ("code", "for s in ('zc','zs'):\n    y = dc.futures_weekly(s); u = dc.futures_weekly_usda(s)\n"
                 "    print(s, 'Yahoo usable weekly returns:', round(y[y.week>='2007'].clean.mean(),3), '| USDA bad prints removed:', int(u.bad_print.sum()))"),
        ("code", "display(Image(R/'charts/01_transport_volumes.png'))\ndisplay(Image(R/'charts/02_corn_price.png'))\ndisplay(Image(R/'charts/03_soybeans_price.png'))"),
    ],
    "02_feature_engineering.ipynb": [
        ("md", "# 02 - Feature engineering\nPoint-in-time alignment and the seasonal-anomaly transform."),
        ("code", SETUP),
        ("code", "zc = pd.read_csv(ROOT/'data/processed/features_zc.csv', index_col=0, parse_dates=True)\n"
                 "zc.filter(like='ia_railcars').dropna().tail(8)"),
        ("md", "**Leakage check**: for every decision week, the transport observation used must have been published before that Friday."),
        ("code", "import feature_engineering as fe, data_cleaning as dc\n"
                 "raw = {r.columns[1]: r for r in dc.build_raw_weekly()}\n"
                 "for name in ['ia_railcars','insp_corn_mt','sales_corn_net_sales','barge_corn_tons']:\n"
                 "    t = fe.transform_series(raw[name]); t = t.dropna(subset=['available']).sort_values('available')\n"
                 "    t = t.assign(avail_key=t.available + pd.Timedelta(days=1))\n"
                 "    m = pd.merge_asof(pd.DataFrame({'week': zc.index}), t[['avail_key','available','obs_date']], left_on='week', right_on='avail_key')\n"
                 "    m = m.dropna(); print(f\"{name:22s} violations (available >= decision Friday): {(m.available >= m.week).sum()}   median age of data used: {(m.week-m.obs_date).dt.days.median():.0f} days\")"),
        ("code", "display(Image(R/'charts/10_seasonal_performance.png'))"),
    ],
    "03_statistical_analysis.ipynb": [
        ("md", "# 03 - Statistical analysis (in-sample)\nHAC regressions, multiple-testing control, ADF, Granger, rolling correlations. "
               "In-sample significance is *not* evidence of tradability."),
        ("code", SETUP),
        ("code", "for s in ('zc','zs'):\n    t = pd.read_csv(R/f'regression_results/univariate_primary_{s}.csv')\n"
                 "    print(s); display(t.sort_values('p_hac').head(8)[['feature','horizon','n','start','beta','t_hac','p_hac','p_bh','r2']].round(4))"),
        ("code", "print(open(R/'regression_results/multiple_regression_zs.txt').read()[:4000])"),
        ("code", "for s in ('zc','zs'):\n    g = pd.read_csv(R/f'regression_results/granger_{s}.csv'); print(s); display(g.sort_values('p_value').head(6).round(4))"),
        ("code", "display(Image(R/'charts/04_transport_vs_returns.png'))\ndisplay(Image(R/'charts/05_rolling_correlations.png'))\ndisplay(Image(R/'charts/06_regression_tstats.png'))"),
    ],
    "04_model_development.ipynb": [
        ("md", "# 04 - Model development (walk-forward, out-of-sample)"),
        ("code", SETUP),
        ("code", "o = pd.read_csv(R/'performance/oos_univariate.csv')\n"
                 "o[o.horizon==1][['commodity','feature','n_oos','oos_r2_vs_mean','cw_p','cw_p_bh','hit_rate','strategy_sharpe']].round(4)"),
        ("code", "m = pd.read_csv(R/'performance/oos_multivariate.csv')\n"
                 "m[['commodity','universe','model','n_oos','oos_r2_vs_mean','cw_p','hit_rate','strategy_sharpe']].round(4)"),
        ("code", "display(Image(R/'charts/07_actual_vs_predicted.png'))"),
    ],
    "05_backtesting.ipynb": [
        ("md", "# 05 - Backtesting & robustness"),
        ("code", SETUP),
        ("code", "s = pd.read_csv(R/'performance/strategy_metrics.csv')\n"
                 "s[['commodity','strategy','start','weeks','ann_return','ann_vol','sharpe','sharpe_ci_lo','sharpe_ci_hi','max_drawdown','n_trades','win_rate','profit_factor','exposure','turnover_ann']].round(3)"),
        ("code", "r = pd.read_csv(R/'performance/robustness.csv')\n"
                 "r[r.test.isin(['threshold','cost_mult','subperiod','season','price_source','execution_delay_weeks'])][['commodity','series','test','value','weeks','ann_return','sharpe','max_drawdown']].round(3)"),
        ("code", "print(json.dumps(json.loads((R/'performance/summary.json').read_text()), indent=1))"),
        ("code", "for f in ['08_cumulative_returns','09_drawdowns','11_permutation_test','12_cost_sensitivity']:\n    display(Image(R/f'charts/{f}.png'))"),
    ],
}

for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if t == "md" else nbf.v4.new_code_cell(c) for t, c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
print("notebooks written")
