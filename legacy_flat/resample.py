"""CLI: python -m quantlab.data <command>

Commands:
  download --symbol BTCUSDT --timeframe 1m --start 2024-01-01 [--end ...] [--root data/]
  status   --symbol BTCUSDT --timeframe 1m [--root data/]
  sql      "SELECT symbol, count(*) FROM candles GROUP BY symbol" [--root data/]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time

from quantlab.core.types import Exchange, Instrument, Timeframe
from quantlab.data.adapters.binance import BinanceAdapter
from quantlab.data.downloader import Downloader
from quantlab.data.gaps import GapRegistry
from quantlab.data.store import CandleStore


def _parse_date(s: str) -> int:
    return int(
        dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp() * 1000
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="quantlab.data")
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download")
    dl.add_argument("--exchange", default="binance", choices=["binance"])
    dl.add_argument("--symbol", required=True)
    dl.add_argument("--timeframe", default="1m")
    dl.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    dl.add_argument("--end", default=None, help="YYYY-MM-DD (UTC), default: now")
    dl.add_argument("--root", default="data")
    dl.add_argument("--no-resume", action="store_true")

    st = sub.add_parser("status")
    st.add_argument("--exchange", default="binance", choices=["binance"])
    st.add_argument("--symbol", required=True)
    st.add_argument("--timeframe", default="1m")
    st.add_argument("--root", default="data")

    sq = sub.add_parser("sql")
    sq.add_argument("query")
    sq.add_argument("--root", default="data")

    args = p.parse_args(argv)
    store = CandleStore(args.root)

    if args.cmd == "download":
        tf = Timeframe.from_str(args.timeframe)
        end_ms = _parse_date(args.end) if args.end else int(time.time() * 1000)
        result = Downloader(BinanceAdapter(), store, GapRegistry(args.root)).backfill(
            args.symbol, tf, _parse_date(args.start), end_ms, resume=not args.no_resume
        )
        print(f"Done: {result.rows_written} rows written. {result.report.summary()}")
        return 0 if result.report.is_clean else 2

    if args.cmd == "status":
        tf = Timeframe.from_str(args.timeframe)
        inst = Instrument(Exchange(args.exchange), args.symbol)
        df = store.read(inst, tf)
        gaps = GapRegistry(args.root).load(inst, tf)
        if df.height == 0:
            print("No data stored.")
            return 1
        print(f"{inst} {tf.value}: {df.height} candles")
        print(f"  range: {df['ts_open'].min()} .. {df['ts_open'].max()}")
        print(f"  known gaps: {len(gaps)} ({sum(g.n_missing for g in gaps)} candles missing)")
        return 0

    if args.cmd == "sql":
        print(store.sql(args.query).df().to_string())
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
