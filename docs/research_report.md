# Weekly Grain Transportation Data and CBOT Corn/Soybean Futures: A Point-in-Time Test

*Research note · data through October 2026 · all numbers come from `results/` and can be regenerated with `python main.py`*

## 1. Executive summary

**Question.** Does weekly grain-transportation activity (rail, barge, export inspections, export sales, transport costs, and Iowa-specific rail loadings and basis) predict 1–4-week returns of CBOT corn and soybean futures?

**Hypothesis.** Unusual changes in grain movement, especially out of Iowa, are early signals of supply-chain conditions and commodity demand. The direction of the effect was not assumed; it was left for the data to decide.

**Finding.** The evidence does not support a tradable signal.
- In-sample, rail-loading anomalies are positively related to next-week returns. AAR grain carloads have a HAC t of about 3.6 for both crops. Iowa rail cars have t = 3.1 for soybeans and survive Benjamini-Hochberg (BH) correction across 45 tests.
- Out-of-sample (2022–2026), the pre-specified Iowa-rail strategy earns a net Sharpe of **0.03 in corn** and **0.25 in soybeans**. Both 95% confidence intervals include zero.
- The permutation p-values are 0.40 and 0.13. Both strategies are negative in the second half of the out-of-sample period, and the soybean result turns negative when Yahoo prices are used instead of USDA prices.
- None of 30 univariate out-of-sample tests survives BH correction, and no multivariate or machine-learning model reliably beats a historical-mean forecast.

**Verdict.** Not deployable. One narrow follow-up is reasonable: soybeans vs. rail-loading anomalies, re-tested on exchange-grade contract data with a longer sample.

## 2. Data

### 2.1 Sources, coverage, reliability

All transportation data comes from the USDA AMS **AgTransport** portal, which is the database behind the weekly *Grain Transportation Report*. It is pulled through its Socrata API with the exact queries saved alongside each file.

| Variable | Iowa-specific? | Weekly obs. | Notes |
|---|---|---|---|
| Grain rail cars loaded & billed (STB EP 724) | **Yes** (state field) + all-US | 2014-10 → | BNSF + UP must both report or the week is missing (3 weeks). Small carriers' zero weeks are absent from the source and treated as 0. |
| AAR grain carloads originated | No | 2017-03 → | Weeks with an incomplete set of railroads are set to missing (78). |
| Downbound grain barge tons, 7 locks | No | 2013 → | Locks 15/25/26/La Grange only start in 2013, so the sample is restricted to keep the lock set constant. |
| NOLA barge unloads, FGIS export inspections, FAS export sales | No | 2005 → | Inspections are summed from daily certificate totals into weeks ending Thursday. |
| Truck / rail / barge / ocean cost indicators, St. Louis barge rate | No | 2002 → | Shuttle-train index has 384 missing weeks, all left missing. |
| Elevator cash basis | **Yes** (Iowa) + Gulf | 2007 → | |
| FHWA TMAS hourly volumes, rural interstates | **Yes** | 2014–2024 with gaps | All vehicles, **not trucks**. Too few stations 2016–18, numbering changes in 2025, and published months late. Diagnostic only. |

**What could not be obtained.** Iowa DOT publishes annual average daily traffic and a map viewer, but no downloadable weekly or daily truck-classification counts; continuous-count data is available on request only. The public FHWA TMAS files contain volume but not vehicle class. A weekly *truck-specific* Iowa series therefore cannot be rebuilt from public sources. The nearest available proxies (Iowa grain rail cars and Iowa rural-interstate total volume) were used instead and are labeled as such.

**Revisions.** AgTransport serves only the latest version of each series. If later revisions were made, the first-release values cannot be recovered, which is a small, unavoidable source of look-ahead.

### 2.2 Futures prices

Free sources do not have expired individual CBOT contracts. Two series were built and cross-checked:

1. **USDA AMS weekly reference print (primary).** Every Friday, the basis table records the futures price and contract month used for the basis quote. USDA rolls to the next contract about 2–3 weeks before first notice day. Returns are computed only between consecutive Fridays that quote the same contract.
2. **Yahoo `ZC=F` / `ZS=F` (robustness).** This series rolls on the first trading day on or after the 15th of the contract month. That creates jumps (e.g. −17.4% on 2021-07-15) and holds the contract through its illiquid delivery period. Weeks between first position day and the roll are removed, which drops 36% of corn weeks and 50% of soybean weeks.

The USDA print matches Yahoo to within a median of 0.25¢ when both refer to the same contract. The two series correlate at 0.88 for corn and 0.94 for soybeans on overlapping clean weeks. The USDA print appears to be taken during the session rather than at settlement. Validation also found label errors: soybean "Dec", a month CBOT soybeans do not list, and corn "Sep" in July–August 2024 priced about 19¢ above CBOT September. It also found typos, such as corn at 5.246 on 2026-01-16 when CBOT was about 4.25. Prints failing the checks are set to missing (32 corn and 34 soybean weeks), never replaced. Hidden contract switches that cannot be checked against Yahoo remain a documented residual risk. **Using contract-level exchange data is the single most valuable upgrade.**

## 3. Methodology

### 3.1 Point-in-time construction
- **Decision time:** each Friday T. A transportation value is used only if its estimated publication date is earlier than T. Publication lags are conservative: rail data at submission + 1 day, barge +5, inspections +4, export sales +7, costs +1–2, basis +6. A unit test confirms that no feature violates this rule.
- **Targets:** log returns from T to T+h, for h = 1, 2 and 4 weeks. A target is missing if any week inside it crosses a roll or a removed print.

### 3.2 Features
For each series the pipeline computes the week-over-week log change, the 4-week change of the 4-week mean, the gap to the trailing 52-week mean, the year-over-year change and the **seasonal anomaly**: the log 4-week mean minus its average in the same calendar week 1–3 years earlier. The anomaly needs at least 2 prior years and is scaled to a z-score using the expanding standard deviation of past anomalies. The anomaly z-score was **pre-specified as the primary transform** because it is the only one that separates "harvest happened" from "this harvest is unusual" (see Fig 10, right panel).

Controls: 4- and 12-week momentum, 12-week realized volatility, and harvest, planting and growing-season dummies.

### 3.3 Models
1. Pearson/Spearman correlations.
2. Univariate predictive OLS with Newey-West standard errors (h + 2 lags).
3. Multiple regression of transportation variables plus controls, with a joint Wald test.
4. Time-series checks: ADF (all anomaly series are stationary at 5%), Ljung-Box on returns, Granger F-tests, rolling 104-week correlations.
5. Walk-forward out-of-sample models: expanding window, 156-week minimum, refit every 4 weeks, h-week embargo. Variants are univariate OLS, seasonal-only, controls-only, kitchen-sink OLS, ridge, lasso (time-series cross-validation), a shallow random forest (depth 3, at least 20 observations per leaf), and a *best-univariate* model that re-selects the top in-sample variable inside each training window to mimic real-time data mining.

Out-of-sample evaluation uses OOS R² against the expanding historical mean, the Clark-West test, hit rate and a binomial test, with BH correction across the whole family of tests.

### 3.4 Backtest
- Position: sign(z) when |z| > k (k = 0.5), where z is the forecast divided by the standard deviation of the previous 52 forecasts. Size is 10% target volatility divided by trailing 26-week volatility, capped at 3×.
- Entry and exit at the weekly print. The strategy is flat whenever next week's return cannot be measured.
- Costs: 1 tick of slippage (0.25¢/bu) plus $2.50 commission per side per 5,000-bu contract, about 7 bp for corn and 3 bp for soybeans per side.
- Benchmarks over the same window: cash, 1× long, volatility-targeted long, and 4- and 12-week momentum.
- The **pre-specified primary strategy** is the univariate Iowa-rail anomaly. It was chosen because it is the user's hypothesis, not because of the results.

## 4. Results

### 4.1 In-sample (Figs 4–6)
| Feature (anomaly z) | Crop | h | n | β per 1 sd | 95% CI | HAC t | BH p |
|---|---|---|---|---|---|---|---|
| AAR grain carloads | Corn | 1 | 213 | +0.60% | [0.28, 0.93] | 3.60 | 0.014 |
| AAR grain carloads | Soy | 1 | 226 | +0.44% | [0.20, 0.68] | 3.62 | 0.007 |
| Iowa rail cars | Soy | 1 | 374 | +0.38% | [0.14, 0.62] | 3.12 | 0.027 |
| Iowa rail cars | Soy | 4 | 226 | +1.35% | [0.45, 2.25] | 2.94 | 0.037 |
| All-US rail cars | Corn | 1 | 360 | +0.42% | [0.10, 0.74] | 2.60 | 0.106 |

- All rail measures have the **same positive sign**: heavier-than-seasonal grain loadings come before higher prices. The economic reading is that rail loadings reflect demand, mainly export and processor pull, more than supply, so strong shipments signal tight demand.
- Barge, inspection, export-sales, cost and basis anomalies are not significant after correction.
- In the multiple regression, the transportation block is jointly significant in the Iowa sample (Wald p = 0.001 corn, p < 0.001 soybeans), with R² of 0.15 in corn against 0.04 for controls only.
- Granger tests agree for soybeans/Iowa rail (p = 0.009 raw) but not after correction (0.24).
- On Yahoo prices, Iowa rail → soybeans weakens to t = 2.0.

These in-sample significant variables come from **short samples**. AAR anomalies start only in April 2020 and the Iowa anomaly in December 2017, so the samples are dominated by the 2020–2022 China-demand bull market, when both shipments and prices rose. That is the central confound.

### 4.2 Out-of-sample forecasts
- **Univariate, h = 1:**
  - Iowa rail: OOS R² −4.0% for corn (CW p = 0.42) and +2.3% for soybeans (CW p = 0.024).
  - All-US rail: −2.4% corn, +1.4% soybeans (CW p = 0.027).
  - AAR: +6.8% corn (CW p = 0.004), but on only 50 OOS weeks.
  - Of the remaining 25 variable × crop combinations, soybean AAR is −3.9% and the rest lie between −1.4% and +0.4%.
  - After BH correction across 30 tests, the smallest adjusted p-value is **0.18**.
- **Multivariate / machine learning:** every ridge, random-forest, kitchen-sink and best-univariate model has a negative OOS R². Lasso is at best +0.05% (soybeans, CW p ≥ 0.37) and negative for corn. Kitchen-sink OLS reaches −54% in the Iowa corn sample, which is the in-sample joint test overfitting. Seasonal-only and controls-only models are also negative, so seasonality and momentum add nothing on their own either.

### 4.3 Backtests (OOS windows: corn 2022-01-21 → 2026-10-23, soybeans 2021-12-31 → 2026-08-07; Figs 8–9)

| Strategy | Crop | Weeks | Ann. return | Ann. vol | Sharpe [95% CI] | Sortino | Max DD | Trades | Win % | Profit factor | Exposure | Turnover/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Iowa rail (primary)** | Corn | 246 | +0.3% | 9.0% | **0.03** [−0.78, 0.86] | 0.04 | −16.8% | 33 | 52% | 1.06 | 60% | 8.1× |
| Ridge, Iowa universe | Corn | 117 | +3.1% | 9.2% | 0.33 [−0.85, 1.66] | 0.40 | −10.4% | 17 | 53% | 1.45 | 44% | 10.1× |
| Best-univariate | Corn | 176 | +4.8% | 7.9% | 0.59 [−0.26, 1.76] | 0.82 | −8.5% | 21 | 62% | 1.73 | 52% | 7.8× |
| Long, vol-targeted | Corn | 246 | −0.2% | 10.5% | −0.02 | — | −20.3% | — | — | — | 80% | — |
| Momentum 12w | Corn | 246 | −1.2% | 10.2% | −0.12 | — | −28.6% | 53 | 42% | 0.93 | 70% | 13.2× |
| **Iowa rail (primary)** | Soy | 238 | +2.1% | 8.4% | **0.25** [−0.63, 0.92] | 0.31 | −11.1% | 33 | 61% | 1.39 | 56% | 9.0× |
| Ridge, Iowa universe | Soy | 147 | −4.3% | 7.8% | −0.56 | −0.60 | −12.8% | 15 | 27% | 0.43 | 55% | 7.6× |
| Best-univariate | Soy | 170 | −0.3% | 7.3% | −0.04 | −0.05 | −6.8% | 20 | 45% | 0.97 | 38% | 8.2× |
| Long, vol-targeted | Soy | 238 | −1.5% | 9.5% | −0.16 | — | −23.1% | — | — | — | 76% | — |
| Momentum 12w | Soy | 238 | −7.4% | 9.4% | −0.81 | — | −36.4% | 55 | 29% | 0.57 | 76% | 14.2× |

(Buy-and-hold 1×: corn −1.9%/yr with Sharpe −0.10; soybeans +1.5%/yr with Sharpe 0.09. The model-based strategies have fewer weeks because they need more complete history before trading.)

**What matters most.** With about 240 weekly observations, a Sharpe ratio's standard error is roughly 1/√4.6 years ≈ 0.47. A real Sharpe of 0.25 is indistinguishable from zero over this horizon; showing it at 95% confidence would take about 60 years of data. The confidence interval, the permutation test and the stability across sub-periods are therefore the metrics to trust, not the point Sharpe. The corn best-univariate Sharpe of 0.59 looks best, but it comes from a procedure that searched 13 variables, has a CI that includes zero, and is negative for soybeans.

### 4.4 Robustness (Figs 11–12, `results/performance/robustness.csv`)

| Test | Corn (Iowa rail Sharpe) | Soybeans (Iowa rail Sharpe) |
|---|---|---|
| Threshold k = 0 / 0.25 / 0.5 / 1.0 / 1.5 | 0.31 / 0.29 / 0.03 / −0.41 / −0.15 | 0.35 / 0.34 / 0.25 / 0.54 / 0.53 |
| Costs ×0 / ×1 / ×2 / ×4 | 0.09 / 0.03 / −0.02 / −0.14 | 0.28 / 0.25 / 0.22 / 0.17 |
| First / second half of OOS | 0.88 / −0.82 | 0.83 / −0.31 |
| Harvest (Sep–Nov) / rest of year | −2.20 / 0.43 | 0.23 / 0.26 |
| Forecast horizon 1 / 2 / 4 weeks | 0.03 / −0.10 / 0.01 | 0.25 / 0.04 / −0.27 |
| One-week execution delay | −0.16 | 0.12 |
| Yahoo instead of USDA prices | −0.50 | −0.68 |
| Permutation p (300 circular shifts, full walk-forward refit) | 0.40 | 0.13 |

Other transforms of Iowa and all-US rail (week-over-week, year-over-year, 52-week relative, raw anomaly) give OOS Sharpes between −1.1 and +0.6, with no consistent winner.

**Interpretation.**
- The soybean signal holds up better than corn across thresholds and costs, because soybean costs are low relative to price.
- It **fails the tests that matter**: it decays over time (2022 strong, 2023–26 flat to negative), its significance depends on the price source, and it cannot be told apart from shifted-signal noise at conventional levels.
- In corn, the strategy loses heavily at harvest, which is when transport volumes are most driven by the crop size itself rather than by demand surprises.

### 4.5 Iowa traffic diagnostic
FHWA rural-interstate volumes were given an **unrealistically short** 6-day publication lag (the real lag is months) to test whether the information exists at all. No transform is significant after correction; the best is week-over-week at h = 2, raw p = 0.04 for corn. Even with an impossible information advantage, the all-vehicle traffic counts carry no detectable price information.

## 5. Limitations

- **Sample size.** The Iowa anomaly needs two prior years, and the walk-forward needs three years of training, so the Iowa out-of-sample period is under five years. The only long-history variables (inspections, sales, costs, basis) show nothing out-of-sample over 10+ years.
- **Futures data.** Contract labels are unreliable, the USDA print is taken intraday, Yahoo rolls into delivery, and there is no open interest or term structure. These are the main threats to validity.
- **Publication timing.** The lags are conservative estimates, not archived release timestamps, and revisions to past values are not observable.
- **Seasonality.** This is handled with year-over-year seasonal anomalies, but crop size changes the seasonal shape from year to year. A proper control would use NASS crop-size and stocks expectations, which need an API key.
- **Market efficiency.** These reports are public, widely watched (the GTR is cited in market commentary) and released weekly. Any information in them is likely priced in within the release day, and a weekly sampling frequency cannot capture that.
- **Confounding.** Rail loadings and prices both respond to demand shocks, most clearly China in 2020–21. The in-sample relationship is consistent with both series reacting to a shared driver rather than transport predicting price.
- **Costs.** Costs are modest at one tick, and roll costs are captured as turnover. Execution at the weekly USDA print time is an approximation.

## 6. Conclusion

There is **no statistically defensible evidence** that weekly grain-transportation data, Iowa-specific or national, predicts CBOT corn or soybean returns well enough to trade after realistic costs. The in-sample rail effects are real features of the 2017–2022 data but do not generalize. This negative result comes from a methodology built to avoid the usual sources of false positives: point-in-time features, walk-forward embargoes, multiple-testing control, permutation nulls and roll-aware returns.

**If a follow-up is warranted:**
1. Re-test *soybeans vs. rail-loading anomalies* using CME contract-level data, with explicit settlement prices and a proper roll.
2. Model **surprises relative to expectations**, such as consensus export inspections or the gap between NASS crop progress and shipments, instead of raw volumes.
3. Test at a **daily/event frequency** around the Thursday GTR and Monday inspections releases.
4. Use the panel across crops and regions (Iowa vs. Illinois vs. Nebraska rail, PNW vs. Gulf) to gain cross-sectional power.
5. Ask Iowa DOT for continuous-count classification data (vehicle classes 8–13) to build the truck series this study could not obtain.
