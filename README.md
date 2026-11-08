# Agricultural Transportation Alpha

**Do weekly grain-transportation data predict CBOT corn (ZC) and soybean (ZS) futures?**

This repo tests that hypothesis with point-in-time public data, walk-forward out-of-sample forecasts and a rules-based backtest that includes trading costs.

## Bottom line

**No transportation variable gives a robust, tradable signal.** Some variables are significant in-sample. Those results do not hold up out-of-sample once multiple testing is accounted for.

| | Corn | Soybeans |
|---|---|---|
| Pre-specified strategy (Iowa rail-car anomaly), OOS net Sharpe | **0.03** (95% CI −0.78 to 0.86) | **0.25** (95% CI −0.63 to 0.92) |
| Permutation p-value (Sharpe vs. shifted-signal null) | 0.40 | 0.13 |
| OOS R² vs. historical mean (Iowa rail) | −4.0% | +2.3% (Clark-West p = 0.024; BH-adjusted p = 0.53) |
| First half / second half of OOS (Sharpe) | 0.88 / −0.82 | 0.83 / −0.31 |
| Same signal, Yahoo front-month prices | −0.50 | −0.68 |
| Buy-and-hold over the same window (Sharpe) | −0.10 | 0.09 |

The out-of-sample window runs from January 2022 to October 2026 (about 240 weeks). Out of 15 transportation variables × 2 commodities, **none** survives Benjamini-Hochberg correction out-of-sample. The multivariate models (OLS, ridge, lasso, random forest) do not meaningfully beat a historical-mean forecast. The kitchen-sink OLS has a strongly significant in-sample joint F-test but an OOS R² of −54% for corn, which is a textbook overfit.

The only pattern worth more work: rail grain loadings above their seasonal norm (Iowa, all-US and AAR) tend to come before *higher* soybean returns. The sign is consistent across these three measures, but the effect is small and concentrated in 2022. → [Research report](docs/research_report.md)

## Data (all public, all documented in `data/raw/*.meta.json`)

| Series | Geography | Source | Weekly since | Lag assumed |
|---|---|---|---|---|
| Grain rail cars loaded & billed | **Iowa** + US | USDA AMS AgTransport `27k8-utc2` (STB EP 724) | 2014-10 | submission + 1 day |
| AAR grain carloads | US | AgTransport `tb7q-kn5i` | 2017-03 | submission + 1 day |
| Downbound grain barge tons (7 locks) | US rivers | AgTransport `n4pw-9ygw` (USACE) | 2013 | +5 days |
| NOLA grain barge unloads | Gulf | AgTransport `spd3-ku7f` | 2005 | +5 days |
| Export inspections (FGIS) | US ports | AgTransport `sruw-w49i` | 1983 (used 2005+) | +4 days |
| Export sales (FAS) | US | AgTransport `wnn7-29tu` | 1999 (used 2005+) | +7 days |
| Grain transport cost indicators, barge rates | US | AgTransport `8uye-ieij`, `deqi-uken` | 2002–04 | +1–2 days |
| Cash basis | **Iowa**, Gulf | AgTransport `v85y-3hep` | 2007 | +6 days |
| Rural-interstate traffic (all vehicles) | **Iowa** | FHWA TMAS monthly files | 2014–2024, with gaps | not real-time (diagnostic only) |
| Futures (primary) | CBOT | USDA AMS weekly reference-contract print (in `v85y-3hep`) | 2007 | Friday |
| Futures (robustness) | CBOT | Yahoo Finance `ZC=F`, `ZS=F` daily | 2000 | Friday close |

**Iowa caveat.** Rail cars, the cash basis and FHWA traffic are the only Iowa-specific series. Everything else is national. Iowa DOT does **not** publish downloadable weekly truck counts. Its vehicle-classification data is only available on request, and the public FHWA files have volume only, with no truck class. There is therefore no truck-specific Iowa series in this study.

**Futures caveat.** Free data does not include expired individual contracts.
- Yahoo's continuous series rolls at expiry, with jumps of up to −17%, and holds illiquid delivery-month contracts. Dropping those weeks removes 35–50% of the sample.
- The USDA weekly print names its contract and rolls before delivery, so it is the primary series. Validation against Yahoo found mislabeled and mistyped prints, for example soybean "Dec" (a month CBOT soybeans do not trade) and corn 5.25 vs. an actual 4.25. Those prints are set to missing, never replaced (32 corn / 34 soybean weeks).
- Exchange-grade contract data (e.g. CME DataMine) is the most important upgrade for any follow-up.

## Method in one paragraph

Each transportation series is turned into a **seasonal anomaly z-score**: the log 4-week average minus the same calendar week's average over the prior 1–3 years, scaled by the expanding standard deviation of past anomalies. This separates normal harvest seasonality from unusual activity. A value is used for the Friday-T decision only if it was published **before** T. Forecasts come from expanding-window walk-forward models: 156-week minimum training, refit every 4 weeks, and an h-week embargo. A forecast is standardized by the standard deviation of past forecasts and traded only if |z| > 0.5. Positions are sized to 10% annual volatility (max 3× leverage), with costs of 1 tick + $2.50 per side. The strategy is flat in weeks with no measurable return (contract rolls). Tests run in `tests/` check that no future data leaks into features, models or the backtest.

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python main.py --download
```

```bash
python -m pytest -q tests
```

```bash
python notebooks/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

`main.py --download` refreshes the USDA and Yahoo data and runs the full pipeline (about 3 minutes). Use `python main.py` alone to reuse the files in `data/raw`. Adding `--tmas` also downloads the FHWA Iowa traffic files, which is about 3 GB and slow. All data downloads are automated; nothing requires manual download.

## Repository

```
data/raw/            original downloads + .meta.json (source URL, query, timestamp)
data/processed/      point-in-time weekly feature sets (features_zc.csv, features_zs.csv, *_yahoo.csv)
src/
  data_collection.py     USDA AgTransport (Socrata), Yahoo, FHWA TMAS downloaders
  data_cleaning.py       weekly series + publication dates; futures roll handling & price validation
  feature_engineering.py transforms, seasonal anomaly, point-in-time alignment, targets
  statistical_models.py  HAC regressions, BH/Bonferroni, ADF, Granger, walk-forward models, Clark-West
  backtesting.py         signal -> position -> net P&L
  performance_metrics.py Sharpe/Sortino/drawdown/trade stats, block-bootstrap CI
  visualization.py       12 research charts
notebooks/           01-05 walkthroughs (generated by build_notebooks.py, executed)
results/
  charts/            figures 1-12
  regression_results/ in-sample tables (univariate, multiple, ADF, Granger, autocorrelation)
  performance/       OOS forecast stats, strategy metrics, robustness grid, permutation null
tests/               look-ahead / leakage tests
docs/research_report.md
```

## Resume bullet

> Built a point-in-time research pipeline in Python (pandas, statsmodels, scikit-learn) testing whether weekly USDA grain-transportation data (rail, barge, export inspections, Iowa rail loadings) predict CBOT corn and soybean futures. Used walk-forward models with publication-lag alignment, transaction costs, a contract-roll-aware backtest and permutation tests. Found that in-sample significance (HAC t up to 3.6) did not survive out-of-sample testing or multiple-testing control (best OOS Sharpe 0.25, CI includes 0), and documented the data-quality issues in the free futures sources.
