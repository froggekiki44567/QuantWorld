"""Backfill orchestrator.

Pipeline per run: resolve resume point → stream pages from adapter →
persist each page immediately (crash-safe) → validate the full stored
range → record gaps.

Validation runs on what's IN THE STORE after the run, not on the in-flight
pages — the store is the ground truth a backtest will read, so that's what
must be certified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from quantlab.core.types import Candle, Instrument, Timeframe
from quantlab.data.adapters.base import ExchangeAdapter
from quantlab.data.gaps import GapRegistry
from quantlab.data.store import CandleStore
from quantlab.data.validation import ValidationReport, validate

log = logging.getLogger(__name__)


@dataclass(slots=True)
class BackfillResult:
    instrument: Instrument
    timeframe: Timeframe
    start_ms: int
    end_ms: int
    rows_written: int
    report: ValidationReport


class Downloader:
    def __init__(self, adapter: ExchangeAdapter, store: CandleStore, gaps: GapRegistry) -> None:
        self.adapter = adapter
        self.store = store
        self.gaps = gaps

    def backfill(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_ms: int,
        end_ms: int,
        resume: bool = True,
    ) -> BackfillResult:
        inst = Instrument(self.adapter.exchange, symbol)

        effective_start = start_ms
        if resume:
            last = self.store.max_ts(inst, timeframe)
            if last is not None and last + timeframe.ms > start_ms:
                effective_start = last + timeframe.ms
                log.info("%s %s: resuming from %s", inst, timeframe.value, effective_start)
        if effective_start >= end_ms:
            log.info("%s %s: nothing to fetch", inst, timeframe.value)

        rows = 0
        for page in self.adapter.fetch_klines(symbol, timeframe, effective_start, end_ms):
            rows += self.store.upsert(inst, timeframe, _to_df(page))
            log.info(
                "%s %s: +%d candles (through %s)",
                inst, timeframe.value, len(page), page[-1].ts_open,
            )

        stored = self.store.read(inst, timeframe, start_ms, end_ms)
        report = validate(stored, timeframe, expected_start_ms=start_ms, expected_end_ms=end_ms)
        if report.gaps:
            self.gaps.record(inst, timeframe, report.gaps)
            log.warning("%s %s: %s", inst, timeframe.value, report.summary())
        else:
            log.info("%s %s: clean (%s)", inst, timeframe.value, report.summary())

        return BackfillResult(inst, timeframe, start_ms, end_ms, rows, report)


def _to_df(candles: list[Candle]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_open": [c.ts_open for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
            "quote_volume": [c.quote_volume for c in candles],
            "n_trades": [c.n_trades for c in candles],
            "taker_buy_base": [c.taker_buy_base for c in candles],
        }
    )
