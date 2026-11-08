"""
Weekly live run (signals + optional IB paper trading).

    python live.py                          # refresh USDA data, print signal (no broker)
    python live.py --prices bloomberg       # size/vol from your Bloomberg export
    python live.py --paper                  # + show paper orders (dry run)
    python live.py --paper --submit         # + send orders to the IB PAPER account
    python live.py --equity 100000          # equity used when not connected to IB

Run it once a week, Friday, after the Thursday USDA Grain Transportation Report.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import data_collection as DC  # noqa: E402
import live_signal as L  # noqa: E402


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    if "--no-download" not in sys.argv:
        DC.collect_agtransport()
        DC.collect_futures()
    prices = arg("--prices", "usda")
    equity = float(arg("--equity", 100_000))
    sig = L.compute_signals(equity, prices)
    report = L.write_report(sig)
    print(report.read_text())
    if "--paper" in sys.argv:
        import paper_trader as P
        P.run(sig, port=int(arg("--port", 7497)), submit="--submit" in sys.argv)
    else:
        print(json.dumps(sig["signals"], indent=1))


if __name__ == "__main__":
    main()
