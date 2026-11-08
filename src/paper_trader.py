"""
Paper trading through Interactive Brokers (IB Gateway or TWS, PAPER account).

Safety rails (cannot be switched off from the command line):
  * connects only to the paper port (7497 TWS / 4002 Gateway by default)
  * refuses to trade unless EVERY managed account id starts with "DU"
    (IB's prefix for paper accounts)
  * dry-run unless --submit is passed; every intended order is logged
  * hard cap of LIVE_CONFIG["max_contracts"] per commodity
  * only trades commodities enabled in LIVE_CONFIG

Setup (once):
  pip install ib_async
  IB paper account -> TWS or IB Gateway -> Settings > API: enable socket
  clients, port 7497 (TWS paper) or 4002 (Gateway paper), "Read-only API" OFF.
  CBOT futures market-data permissions are needed for live quotes; without
  them the trader falls back to the last price in the signal file.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

from live_signal import LIVE_CONFIG, OUT, contracts_for

PAPER_PORTS = {7497, 4002}
IB_SYMBOL = {"zc": "ZC", "zs": "ZS"}
CYCLE = {"zc": [3, 5, 7, 9, 12], "zs": [1, 3, 5, 7, 11]}


def contract_month_to_trade(sym: str, today: pd.Timestamp | None = None, roll_bdays: int = 5) -> str:
    """Same roll rule as the research: nearest cycle contract whose roll date
    (first notice day - 5 business days) is more than a week away. 'YYYYMM'."""
    today = (today or pd.Timestamp.today()).normalize()
    for y in (today.year, today.year + 1, today.year + 2):
        for m in CYCLE[sym]:
            fnd = pd.Timestamp(y, m, 1) - pd.offsets.BDay(1)
            if fnd - pd.offsets.BDay(roll_bdays) > today + pd.Timedelta(days=7):
                return f"{y}{m:02d}"
    raise RuntimeError("no contract found")


def plan_orders(signal: dict, positions: dict[str, dict[str, int]], equity: float,
                quotes: dict[str, float]) -> list[dict]:
    """Pure function (unit-tested): target vs current -> list of orders.
    positions: {sym: {"YYYYMM": qty}}; quotes: {sym: $/bu}."""
    orders = []
    for sym, s in signal["signals"].items():
        if "error" in s:
            continue
        month = contract_month_to_trade(sym)
        held = positions.get(sym, {})
        # 1) close anything in a month we no longer want (roll / stale contract)
        for m, q in held.items():
            if m != month and q != 0:
                orders.append(dict(sym=sym, month=m, action="SELL" if q > 0 else "BUY", qty=abs(q),
                                   reason="roll/close old contract"))
        # 2) move the active month to target
        price = quotes.get(sym) or s["ref_price_usd_bu"]
        target = contracts_for(s["target_weight"], equity, price) if s.get("enabled") else 0
        delta = target - held.get(month, 0)
        if delta != 0:
            orders.append(dict(sym=sym, month=month, action="BUY" if delta > 0 else "SELL", qty=abs(delta),
                               reason=f"target {target} (z={s['z']}, enabled={s.get('enabled')})"))
    return orders


def _log(orders, submitted: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "orders_log.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp_utc", "sym", "month", "action", "qty", "reason", "submitted"])
        for o in orders:
            w.writerow([pd.Timestamp.utcnow().isoformat(), o["sym"], o["month"], o["action"], o["qty"],
                        o["reason"], submitted])


def run(signal: dict, host="127.0.0.1", port=7497, client_id=17, submit=False):
    if port not in PAPER_PORTS:
        raise SystemExit(f"port {port} is not a paper-trading port {sorted(PAPER_PORTS)} - refusing")
    try:
        from ib_async import IB, Future, MarketOrder
    except ImportError:
        raise SystemExit("pip install ib_async")

    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=15)
    try:
        accts = ib.managedAccounts()
        if not accts or not all(a.startswith("DU") for a in accts):
            raise SystemExit(f"accounts {accts} are not all paper (DU...) accounts - refusing")
        summary = {v.tag: v.value for v in ib.accountSummary() if v.currency in ("USD", "")}
        equity = float(summary.get("NetLiquidation", "nan"))
        if not math.isfinite(equity):
            raise SystemExit("could not read NetLiquidation")

        positions: dict[str, dict[str, int]] = {}
        for p in ib.positions():
            c = p.contract
            for sym, root in IB_SYMBOL.items():
                if c.secType == "FUT" and c.symbol == root:
                    m = c.lastTradeDateOrContractMonth[:6]
                    positions.setdefault(sym, {})[m] = positions.get(sym, {}).get(m, 0) + int(p.position)

        quotes = {}
        for sym in signal["signals"]:
            fut = Future(IB_SYMBOL[sym], contract_month_to_trade(sym), "CBOT", currency="USD")
            ib.qualifyContracts(fut)
            t = ib.reqMktData(fut, snapshot=True)
            ib.sleep(3)
            px = t.marketPrice()
            if px and math.isfinite(px) and px > 0:
                q = px / 100.0                    # IB quotes CBOT grains in cents/bu
                ref = signal["signals"][sym].get("ref_price_usd_bu")
                if ref and abs(q / ref - 1) > 0.3 and abs(px / ref - 1) < 0.3:
                    q = px                        # quote was already in $/bu
                quotes[sym] = q
            ib.cancelMktData(fut)

        orders = plan_orders(signal, positions, equity, quotes)
        print(f"paper account(s) {accts}  equity ${equity:,.0f}  positions {positions}  quotes {quotes}")
        for o in orders:
            print(f"  {'SUBMIT' if submit else 'DRY-RUN'}: {o['action']} {o['qty']} {IB_SYMBOL[o['sym']]} {o['month']}  ({o['reason']})")
        if not orders:
            print("  no orders needed")
        if submit:
            for o in orders:
                fut = Future(IB_SYMBOL[o["sym"]], o["month"], "CBOT", currency="USD")
                ib.qualifyContracts(fut)
                ib.placeOrder(fut, MarketOrder(o["action"], o["qty"]))
            ib.sleep(2)
        _log(orders, submit)
        return orders
    finally:
        ib.disconnect()
