"""Performance metrics for weekly strategy return series (log returns, net of costs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PPY = 52  # periods per year


def _trades(pos: pd.Series, pnl: pd.Series):
    """A trade = maximal run of consecutive weeks with the same non-zero position sign."""
    sgn = np.sign(pos.fillna(0))
    tid = (sgn != sgn.shift()).cumsum()
    out = []
    for _, g in pnl.groupby(tid):
        s = sgn.loc[g.index[0]]
        if s != 0:
            out.append(g.sum())
    return pd.Series(out, dtype=float)


def summarize(pnl: pd.Series, pos: pd.Series | None = None) -> dict:
    r = pnl.dropna()
    if len(r) == 0:
        return {}
    eq = r.cumsum()
    simple = np.expm1(r)
    ann_ret = r.mean() * PPY
    ann_vol = r.std() * np.sqrt(PPY)
    downside = r[r < 0].std() * np.sqrt(PPY) if (r < 0).sum() > 2 else np.nan
    dd = eq - eq.cummax()
    out = dict(
        weeks=len(r),
        total_return=np.expm1(r.sum()),
        ann_return=np.expm1(ann_ret),
        ann_vol=ann_vol,
        sharpe=ann_ret / ann_vol if ann_vol > 0 else np.nan,
        sortino=ann_ret / downside if downside and downside > 0 else np.nan,
        max_drawdown=np.expm1(dd.min()),
        sharpe_tstat=(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else np.nan,
    )
    if pos is not None:
        p = pos.reindex(r.index).fillna(0)
        tr = _trades(p, r)
        wins, losses = tr[tr > 0], tr[tr < 0]
        out.update(
            n_trades=len(tr),
            win_rate=(tr > 0).mean() if len(tr) else np.nan,
            weekly_hit_rate=(r[p != 0] > 0).mean() if (p != 0).any() else np.nan,
            profit_factor=wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.nan,
            avg_win=np.expm1(wins.mean()) if len(wins) else np.nan,
            avg_loss=np.expm1(losses.mean()) if len(losses) else np.nan,
            exposure=(p != 0).mean(),
            avg_gross_leverage=p.abs().mean(),
            turnover_ann=p.diff().abs().fillna(p.abs()).sum() / len(p) * PPY,
        )
    return out


def block_bootstrap_sharpe(r: pd.Series, n_boot=2000, block=8, seed=0):
    """Stationary-style moving-block bootstrap CI for the annualised Sharpe."""
    r = r.dropna().values
    rng = np.random.default_rng(seed)
    n = len(r)
    if n < 2 * block:
        return np.nan, np.nan
    sh = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block, n // block + 1)])[:n]
        x = r[idx]
        sh.append(x.mean() / x.std() * np.sqrt(PPY) if x.std() > 0 else 0)
    return np.percentile(sh, 2.5), np.percentile(sh, 97.5)
