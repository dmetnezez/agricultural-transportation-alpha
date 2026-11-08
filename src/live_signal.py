"""
Weekly live signal: the exact research model, applied to the latest data.

Run after the Thursday Grain Transportation Report is out (Friday is ideal).
The model is refitted on every week whose outcome is already known, makes one
forecast for the coming week, and turns it into a target position with the
same rules as the backtest (|z| > k, 10% vol target, 3x cap, flat in a week
whose return cannot be measured, e.g. a roll week on USDA prices).

Strategy configuration lives in LIVE_CONFIG. Corn is DISABLED by default
because the research found no out-of-sample evidence for it (Sharpe 0.03);
soybeans is enabled for PAPER trading only (OOS Sharpe 0.25, CI includes 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtesting as B
import statistical_models as S
from feature_engineering import build_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "live"

LIVE_CONFIG = {
    "zs": {"enabled": True, "feature": "ia_railcars__anomz", "k": 0.5},
    "zc": {"enabled": False, "feature": "ia_railcars__anomz", "k": 0.5},
    "target_vol": 0.10,
    "max_leverage": 3.0,
    "contract_bushels": 5000,
    "max_contracts": 5,          # hard cap per commodity regardless of sizing
}


def contracts_for(weight: float, equity: float, price_dollars: float, cfg=LIVE_CONFIG) -> int:
    """Signed number of contracts for a target notional weight (fraction of equity)."""
    if not np.isfinite(weight) or not np.isfinite(price_dollars) or price_dollars <= 0:
        return 0
    n = weight * equity / (price_dollars * cfg["contract_bushels"])
    n = int(np.sign(n) * np.floor(abs(n) + 0.5))       # round half away from zero
    return int(np.clip(n, -cfg["max_contracts"], cfg["max_contracts"]))


def compute_signals(equity: float, prices: str = "usda", cfg=LIVE_CONFIG) -> dict:
    data = build_dataset(save=False, source=prices, live=True)
    wf = S.WalkForward(min_train=156, refit_every=1)   # refit on all known outcomes
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "price_source": prices,
           "equity": equity, "signals": {}}
    for sym, df in data.items():
        c = cfg.get(sym, {})
        f = c.get("feature", "ia_railcars__anomz")
        pred, _ = wf.run(df, [f], 1, "ols")
        p = pred.dropna()
        if p.empty:
            out["signals"][sym] = {"error": "no forecast (feature missing for latest week)"}
            continue
        T = p.index[-1]
        sd = p.iloc[-53:-1].std()
        z = p.iloc[-1] / sd if sd and np.isfinite(sd) else np.nan
        vol = df.loc[T, "vol_26w"]
        k = c.get("k", 0.5)
        direction = 0 if not np.isfinite(z) or abs(z) <= k else int(np.sign(z))
        weight = direction * min(cfg["target_vol"] / vol, cfg["max_leverage"]) if np.isfinite(vol) else 0.0
        price = float(df.loc[T, "price"])
        n = contracts_for(weight, equity, price, cfg) if c.get("enabled") else 0
        latest_obs = df.loc[T, f]
        out["signals"][sym] = {
            "enabled": bool(c.get("enabled")),
            "decision_week": str(T.date()),
            "feature": f, "feature_value": None if pd.isna(latest_obs) else round(float(latest_obs), 3),
            "forecast_next_week_pct": round(float(p.iloc[-1]) * 100, 3),
            "z": None if not np.isfinite(z) else round(float(z), 2),
            "threshold_k": k,
            "direction": direction,
            "target_weight": round(float(weight), 3),
            "ref_price_usd_bu": round(price, 4),   # last known price; paper_trader re-sizes on the live quote
            "target_contracts": n,
            "run_date": str(pd.Timestamp.today().date()),
            "position_held_until": str((T + pd.Timedelta(days=7)).date()),
        }
    return out


def write_report(sig: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.today().strftime("%Y-%m-%d")
    (OUT / f"signal_{stamp}.json").write_text(json.dumps(sig, indent=2))
    (OUT / "latest.json").write_text(json.dumps(sig, indent=2))
    lines = [f"# Weekly signal {stamp}", "",
             f"Price source: {sig['price_source']}  |  equity: ${sig['equity']:,.0f}", "",
             "| | enabled | week | forecast | z | direction | contracts |", "|---|---|---|---|---|---|---|"]
    for s, v in sig["signals"].items():
        if "error" in v:
            lines.append(f"| {s} | - | - | {v['error']} | | | |")
            continue
        d = {1: "LONG", -1: "SHORT", 0: "FLAT"}[v["direction"]]
        lines.append(f"| {s.upper()} | {v['enabled']} | {v['decision_week']} | {v['forecast_next_week_pct']:+.2f}% "
                     f"| {v['z']} | {d} | {v['target_contracts']} |")
    lines += ["", "Research status: NOT validated for real money (see docs/research_report.md)."]
    p = OUT / "latest.md"
    p.write_text("\n".join(lines))
    return p
