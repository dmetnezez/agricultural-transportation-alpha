"""
Statistical tests and walk-forward forecasting models.

In-sample tests (descriptive, NOT evidence of tradability):
  * Pearson / Spearman correlations
  * univariate predictive regressions  r_{T->T+h} = a + b x_T + e
    with Newey-West (HAC) standard errors, lags = h + 2
  * multiple regressions with momentum, volatility and season controls
  * ADF stationarity, Granger-causality F-tests, rolling correlations
  * Benjamini-Hochberg false-discovery control across all tests run

Out-of-sample (the evidence that matters):
  * expanding-window walk-forward, minimum 156 weeks of training,
    refit every 4 weeks, with an h-week embargo so that no training target
    overlaps the decision date
  * OOS R^2 vs historical mean (Campbell-Thompson) and vs zero
  * Clark-West (2007) test for nested-model forecast improvement
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

TRANSPORT_BASES = ["ia_railcars", "us_railcars", "aar_grain_carloads", "barge_com_tons", "nola_unloads",
                   "insp_com_mt", "sales_com_exports", "sales_com_net_sales", "cost_truck",
                   "cost_shuttle_train", "cost_barge", "cost_gulf_vessel", "barge_rate_stl",
                   "basis_ia_com", "basis_gulf_com"]
IOWA_BASES = ["ia_railcars"]          # the only genuinely Iowa-specific transport measure
PRIMARY = [f"{b}__anomz" for b in TRANSPORT_BASES]   # pre-specified primary transform
CONTROLS = ["mom_4w", "mom_12w", "vol_12w", "harvest", "planting", "growing"]
HORIZONS = (1, 2, 4)


# --------------------------------------------------------------------------
# in-sample
# --------------------------------------------------------------------------
def hac_ols(y, X, lags):
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def univariate_table(df: pd.DataFrame, features, horizons=HORIZONS, start=None) -> pd.DataFrame:
    rows = []
    d = df if start is None else df.loc[start:]
    for h in horizons:
        y = d[f"fwd_{h}"]
        for f in features:
            if f not in d:
                continue
            m = pd.concat([y, d[f]], axis=1).dropna()
            if len(m) < 104:
                continue
            res = hac_ols(m.iloc[:, 0], m.iloc[:, 1], lags=h + 2)
            ci = res.conf_int().iloc[1]
            rho, p_s = stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])
            rows.append(dict(feature=f, horizon=h, n=len(m), start=m.index.min().date(),
                             beta=res.params.iloc[1], t_hac=res.tvalues.iloc[1], p_hac=res.pvalues.iloc[1],
                             ci_lo=ci.iloc[0], ci_hi=ci.iloc[1], r2=res.rsquared,
                             pearson=m.corr().iloc[0, 1], spearman=rho, p_spearman=p_s))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_bh"] = multipletests(t.p_hac, method="fdr_bh")[1]
        t["p_bonf"] = np.minimum(t.p_hac * len(t), 1)
    return t


def multiple_regression(df, features, horizon=1, controls=CONTROLS, start=None):
    d = df if start is None else df.loc[start:]
    cols = [c for c in features + controls if c in d]
    m = d[[f"fwd_{horizon}"] + cols].dropna()
    res = hac_ols(m.iloc[:, 0], m.iloc[:, 1:], lags=horizon + 2)
    return res, m


def adf_table(df, features):
    rows = []
    for f in features:
        s = df[f].dropna()
        if len(s) < 100:
            continue
        stat, p, *_ = adfuller(s, autolag="AIC")
        rows.append(dict(series=f, adf_stat=stat, p_value=p, n=len(s)))
    s = df["ret"].dropna()
    stat, p, *_ = adfuller(s, autolag="AIC")
    rows.append(dict(series="weekly_return", adf_stat=stat, p_value=p, n=len(s)))
    return pd.DataFrame(rows)


def granger_table(df, features, max_lag=4):
    """F-test that lags 1..L of x add to lags 1..L of returns in predicting r_t.
    Built by OLS so that roll-week NaNs are dropped row-wise rather than filled."""
    rows = []
    r = df["ret"]
    for f in features:
        x = df[f]
        for L in (1, 2, 4):
            if L > max_lag:
                continue
            Z = pd.DataFrame({"y": r})
            for k in range(1, L + 1):
                Z[f"r{k}"] = r.shift(k)
            for k in range(0, L):   # x at T is known at T (published before T)
                Z[f"x{k}"] = x.shift(k)
            Z["y"] = r.shift(-1)    # predict next week's return
            Z = Z.dropna()
            if len(Z) < 104:
                continue
            full = sm.OLS(Z.y, sm.add_constant(Z.drop(columns="y"))).fit()
            rest = sm.OLS(Z.y, sm.add_constant(Z[[c for c in Z if c.startswith("r")]])).fit()
            q = L
            F = ((rest.ssr - full.ssr) / q) / (full.ssr / full.df_resid)
            rows.append(dict(feature=f, lags=L, F=F, p_value=1 - stats.f.cdf(F, q, full.df_resid), n=len(Z)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_bh"] = multipletests(t.p_value, method="fdr_bh")[1]
    return t


def autocorr_returns(df, lags=8):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = df["ret"].dropna()
    ac = [r.autocorr(k) for k in range(1, lags + 1)]
    lb = acorr_ljungbox(r, lags=[4, 8], return_df=True)
    return pd.DataFrame({"lag": range(1, lags + 1), "autocorr": ac}), lb


def rolling_corr(df, feature, horizon=1, window=104):
    return df[f"fwd_{horizon}"].rolling(window, min_periods=int(window * .75)).corr(df[feature])


# --------------------------------------------------------------------------
# walk-forward out-of-sample
# --------------------------------------------------------------------------
def _ols_fit_predict(Xtr, ytr, Xte):
    b = np.linalg.lstsq(np.c_[np.ones(len(Xtr)), Xtr], ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ b


class WalkForward:
    """Expanding-window walk-forward with an embargo of `horizon` weeks."""

    def __init__(self, min_train=156, refit_every=4):
        self.min_train = min_train
        self.refit_every = refit_every

    def run(self, df, features, horizon, model="ols", start=None, end=None):
        y_col = f"fwd_{horizon}"
        d = df[[y_col] + list(features)]
        if start:
            d = d.loc[start:]
        if end:
            d = d.loc[:end]
        dates = d.index.values
        X = d[list(features)].to_numpy(dtype=float)
        y = d[y_col].to_numpy(dtype=float)
        x_ok = ~np.isnan(X).any(axis=1)
        complete = x_ok & ~np.isnan(y)
        preds = np.full(len(d), np.nan)
        info = np.full(len(d), None, dtype=object)
        embargo = np.timedelta64(7 * horizon, "D")
        fitted, last_fit = None, -10 ** 9
        for i in range(len(d)):
            if not x_ok[i]:
                continue
            # embargo: a training target must be fully realised by T -> week <= T - h weeks
            n_hist = np.searchsorted(dates, dates[i] - embargo, side="right")
            tr = complete[:n_hist]
            if tr.sum() < self.min_train:
                continue
            if fitted is None or i - last_fit >= self.refit_every:
                fitted = self._fit(X[:n_hist][tr], y[:n_hist][tr], model, list(features), None)
                last_fit = i
            preds[i] = fitted["predict"](X[i:i + 1])[0]
            info[i] = fitted.get("info")
        return pd.Series(preds, index=d.index), pd.Series(info, index=d.index, dtype=object)

    @staticmethod
    def _fit(X, y, model, features, train):
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - mu) / sd
        if model == "hist_mean":
            m = y.mean()
            return {"predict": lambda x: np.array([m])}
        if model == "ols":
            b = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], y, rcond=None)[0]
            return {"predict": lambda x: np.c_[np.ones(len(x)), (x - mu) / sd] @ b}
        cv = TimeSeriesSplit(n_splits=5)
        if model == "ridge":
            r = RidgeCV(alphas=np.logspace(-1, 4, 30), cv=cv).fit(Z, y)
            return {"predict": lambda x: r.predict((x - mu) / sd)}
        if model == "lasso":
            r = LassoCV(cv=cv, alphas=30, max_iter=5000).fit(Z, y)
            return {"predict": lambda x: r.predict((x - mu) / sd),
                    "info": ",".join(f for f, c in zip(features, r.coef_) if abs(c) > 0)}
        if model == "rf":
            r = RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=20,
                                      max_features=0.5, random_state=0, n_jobs=-1).fit(X, y)
            return {"predict": lambda x: r.predict(x)}
        if model == "best_uni":
            # pick the single feature with the highest in-sample |t| *inside the
            # training window* -> mimics a researcher data-mining in real time
            best, bt, bb = None, 0, None
            for j in range(Z.shape[1]):
                Xj = np.c_[np.ones(len(Z)), Z[:, j]]
                b, res, *_ = np.linalg.lstsq(Xj, y, rcond=None)
                e = y - Xj @ b
                se = np.sqrt(e.var(ddof=2) / (len(y) * Z[:, j].var() + 1e-12))
                t = abs(b[1] / se)
                if t > bt:
                    best, bt, bb = j, t, b
            j, b = best, bb
            return {"predict": lambda x: b[0] + b[1] * ((x[:, j] - mu[j]) / sd[j]),
                    "info": features[j]}
        raise ValueError(model)


def oos_stats(y, pred, bench):
    """OOS R^2 vs benchmark forecast, Clark-West test, hit rate."""
    m = pd.concat([y, pred, bench], axis=1, keys=["y", "p", "b"]).dropna()
    if len(m) < 30:
        return {}
    e_p, e_b = m.y - m.p, m.y - m.b
    r2 = 1 - (e_p ** 2).sum() / (e_b ** 2).sum()
    r2_zero = 1 - (e_p ** 2).sum() / (m.y ** 2).sum()
    f = e_b ** 2 - (e_p ** 2 - (m.b - m.p) ** 2)       # Clark-West adjusted
    cw = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    hit = (np.sign(m.p) == np.sign(m.y)).mean()
    # binomial test for hit rate vs 50%
    k = int((np.sign(m.p) == np.sign(m.y)).sum())
    p_hit = stats.binomtest(k, len(m), 0.5, alternative="greater").pvalue
    return dict(n_oos=len(m), oos_r2_vs_mean=r2, oos_r2_vs_zero=r2_zero,
                cw_t=cw.tvalues.iloc[0], cw_p=1 - stats.norm.cdf(cw.tvalues.iloc[0]),
                hit_rate=hit, p_hit=p_hit, corr_pred_actual=m.p.corr(m.y))
