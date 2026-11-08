"""
Weekly rules-based backtest driven by walk-forward (out-of-sample) forecasts.

Timeline per decision Friday T
  1. information set: transport data published before T, futures snapshot at T
  2. forecast  p_T  of the next-h-week return (model fitted only on targets
     realised by T, see statistical_models.WalkForward)
  3. standardise: z_T = p_T / std(p_{T-52..T-1})   (past OOS forecasts only)
  4. direction: +1 if z_T > k, -1 if z_T < -k, else 0  (k = signal threshold)
  5. size: direction * min(target_vol / vol_26w_T, max_leverage)
  6. entry at the Friday-T snapshot price, exit/rebalance at T+1 snapshot
  7. P&L_{T+1} = pos_T * r_{T->T+1} - cost * |pos_T - pos_{T-1}|

Contract roll: positions are held in the USDA reference contract, which rolls
2-3 weeks before first notice day (no delivery exposure). Whenever next week's
return is not measurable (the roll week, or a print removed by validation) the
strategy is FLAT that week, so no return is invented; exiting and re-entering
is charged as ordinary turnover. Roll weeks follow the contract calendar and
are known in advance; validation removals are rare (documented caveat).

Costs (per side, per contract of 5,000 bu):
  slippage 1 tick (0.25 c/bu) + commission/fees $2.50 (=0.05 c/bu)
  expressed as a fraction of the futures price; `cost_mult` scales it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK_CENTS = 0.25
COMMISSION_CENTS = 0.05


def cost_per_side(price_dollars: pd.Series, ticks=1.0, cost_mult=1.0) -> pd.Series:
    return cost_mult * (ticks * TICK_CENTS + COMMISSION_CENTS) / (price_dollars * 100)


def positions_from_forecast(pred: pd.Series, vol: pd.Series, k=0.5, target_vol=0.10,
                            max_lev=3.0, z_window=52, z_min=26) -> pd.Series:
    sd = pred.rolling(z_window, min_periods=z_min).std().shift(1)   # past forecasts only
    z = pred / sd
    direction = np.where(z > k, 1.0, np.where(z < -k, -1.0, 0.0))
    direction = pd.Series(direction, index=pred.index).where(z.notna(), 0.0)
    size = (target_vol / vol).clip(upper=max_lev)
    return (direction * size).fillna(0.0)


def run_backtest(df: pd.DataFrame, pos: pd.Series, cost_mult=1.0, ticks=1.0) -> pd.DataFrame:
    """df must contain fwd_1 (r_{T->T+1}) and price at T. Returns a frame indexed
    by decision date T with the P&L realised at T+1."""
    d = pd.DataFrame(index=pos.index)
    d["fwd_1"] = df["fwd_1"].reindex(pos.index)
    # flat whenever next week's return is excluded (roll / delivery window,
    # which is known in advance from the CBOT calendar)
    d["pos"] = pos.where(d.fwd_1.notna(), 0.0)
    d["price"] = df["price"].reindex(pos.index)
    c = cost_per_side(d.price.ffill().bfill(), ticks, cost_mult)  # price only scales tick costs
    trade = d.pos.diff().abs().fillna(d.pos.abs())
    if "roll" in df:
        # contract switch at this close: close the old contract AND open the new one
        r = df["roll"].reindex(pos.index).fillna(False).astype(bool)
        prev = d.pos.shift(1).fillna(0).abs()
        trade = trade.where(~r, prev + d.pos.abs())
    d["gross"] = (d.pos * d.fwd_1).fillna(0.0)
    d["cost"] = trade * c
    d["pnl"] = d.gross - d.cost
    d["roll_week"] = d.fwd_1.isna()
    return d


def benchmark_positions(df: pd.DataFrame, index, target_vol=0.10, max_lev=3.0) -> dict[str, pd.Series]:
    vol = df["vol_26w"].reindex(index)
    size = (target_vol / vol).clip(upper=max_lev).fillna(0)
    return {
        "cash": pd.Series(0.0, index=index),
        "buy_hold_1x": pd.Series(1.0, index=index),
        "long_voltarget": size,
        "momentum_12w": np.sign(df["mom_12w"].reindex(index)).fillna(0) * size,
        "momentum_4w": np.sign(df["mom_4w"].reindex(index)).fillna(0) * size,
    }
