"""Adapter tests use httpx.MockTransport — no network, deterministic.

The fake exchange serves a configurable kline range and honors
startTime/endTime/limit exactly like Binance does, so pagination logic is
tested against realistic semantics.
"""

import json

import httpx

from quantlab.core.types import Exchange, Instrument, Timeframe
from quantlab.data.adapters.binance import BinanceAdapter
from quantlab.data.downloader import Downloader
from quantlab.data.gaps import GapRegistry
from quantlab.data.store import CandleStore

M = Timeframe.M1.ms


def kline(ts):
    return [ts, "100.0", "101.0", "99.0", "100.5", "10.0",
            ts + M - 1, "1000.0", 50, "5.0", "500.0", "0"]


class FakeBinance:
    """Serves klines for a fixed set of timestamps, Binance semantics."""

    def __init__(self, available_ts: list[int]):
        self.available = sorted(available_ts)
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        q = dict(request.url.params)
        start = int(q.get("startTime", 0))
        end = int(q.get("endTime", 2**62))
        limit = int(q.get("limit", 500))
        rows = [kline(t) for t in self.available if start <= t <= end][:limit]
        return httpx.Response(
            200, content=json.dumps(rows),
            headers={"x-mbx-used-weight-1m": "10"},
        )


def make_adapter(available_ts):
    fake = FakeBinance(available_ts)
    client = httpx.Client(
        base_url="https://api.binance.com", transport=httpx.MockTransport(fake.handler)
    )
    return BinanceAdapter(client=client), fake


class TestBinanceAdapter:
    def test_single_page(self):
        adapter, _ = make_adapter([0, M, 2 * M])
        pages = list(adapter.fetch_klines("BTCUSDT", Timeframe.M1, 0, 3 * M))
        assert len(pages) == 1
        assert [c.ts_open for c in pages[0]] == [0, M, 2 * M]
        c = pages[0][0]
        assert (c.open, c.high, c.low, c.close) == (100.0, 101.0, 99.0, 100.5)
        assert c.n_trades == 50 and c.taker_buy_base == 5.0

    def test_pagination(self):
        # 2500 candles -> 3 pages (1000/1000/500).
        ts = [i * M for i in range(2500)]
        adapter, fake = make_adapter(ts)
        pages = list(adapter.fetch_klines("BTCUSDT", Timeframe.M1, 0, 2500 * M))
        assert [len(p) for p in pages] == [1000, 1000, 500]
        got = [c.ts_open for p in pages for c in p]
        assert got == ts  # no duplicates, no holes, correct order
        assert fake.requests == 3

    def test_end_exclusive(self):
        adapter, _ = make_adapter([0, M, 2 * M])
        pages = list(adapter.fetch_klines("BTCUSDT", Timeframe.M1, 0, 2 * M))
        assert [c.ts_open for p in pages for c in p] == [0, M]

    def test_forming_candle_excluded(self):
        import time

        now = int(time.time() * 1000)
        current_open = now // M * M  # the forming candle
        adapter, _ = make_adapter([current_open - 2 * M, current_open - M, current_open])
        pages = list(
            adapter.fetch_klines("BTCUSDT", Timeframe.M1, current_open - 2 * M, now + M)
        )
        got = [c.ts_open for p in pages for c in p]
        assert current_open not in got  # never store an unclosed candle
        assert got == [current_open - 2 * M, current_open - M]

    def test_empty_range(self):
        adapter, fake = make_adapter([0, M])
        assert list(adapter.fetch_klines("BTCUSDT", Timeframe.M1, 5 * M, 5 * M)) == []
        assert fake.requests == 0

    def test_gap_in_exchange_data(self):
        # Exchange itself is missing minutes 2-3 (downtime).
        adapter, _ = make_adapter([0, M, 4 * M, 5 * M])
        pages = list(adapter.fetch_klines("BTCUSDT", Timeframe.M1, 0, 6 * M))
        got = [c.ts_open for p in pages for c in p]
        assert got == [0, M, 4 * M, 5 * M]  # returned as-is; validation flags it

    def test_retry_on_500(self):
        calls = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500)
            return httpx.Response(
                200, content=json.dumps([kline(0)]),
                headers={"x-mbx-used-weight-1m": "10"},
            )

        client = httpx.Client(
            base_url="https://api.binance.com", transport=httpx.MockTransport(flaky)
        )
        adapter = BinanceAdapter(client=client)
        pages = list(adapter.fetch_klines("BTCUSDT", Timeframe.M1, 0, M))
        assert [c.ts_open for p in pages for c in p] == [0]
        assert calls["n"] == 2


class TestDownloaderEndToEnd:
    def test_backfill_stores_and_validates(self, tmp_path):
        ts = [i * M for i in range(10)]
        adapter, _ = make_adapter(ts)
        store, gaps = CandleStore(tmp_path), GapRegistry(tmp_path)
        result = Downloader(adapter, store, gaps).backfill(
            "BTCUSDT", Timeframe.M1, 0, 10 * M
        )
        assert result.rows_written == 10
        assert result.report.is_clean
        inst = Instrument(Exchange.BINANCE, "BTCUSDT")
        assert store.read(inst, Timeframe.M1).height == 10
        assert gaps.load(inst, Timeframe.M1) == []

    def test_backfill_records_gaps(self, tmp_path):
        # Exchange missing minutes 3-4.
        adapter, _ = make_adapter([0, M, 2 * M, 5 * M, 6 * M])
        store, gaps = CandleStore(tmp_path), GapRegistry(tmp_path)
        result = Downloader(adapter, store, gaps).backfill(
            "BTCUSDT", Timeframe.M1, 0, 7 * M
        )
        assert not result.report.is_clean
        assert result.report.n_missing == 2
        inst = Instrument(Exchange.BINANCE, "BTCUSDT")
        stored_gaps = gaps.load(inst, Timeframe.M1)
        assert len(stored_gaps) == 1
        assert stored_gaps[0].n_missing == 2
        assert not gaps.is_clean(inst, Timeframe.M1, 0, 7 * M)

    def test_resume_skips_existing(self, tmp_path):
        ts = [i * M for i in range(10)]
        adapter, fake = make_adapter(ts)
        store, gaps = CandleStore(tmp_path), GapRegistry(tmp_path)
        dl = Downloader(adapter, store, gaps)
        dl.backfill("BTCUSDT", Timeframe.M1, 0, 10 * M)
        n_first = fake.requests

        # Second run over the same window: resume point = last stored candle,
        # start >= end -> adapter short-circuits, ZERO extra API requests.
        r2 = dl.backfill("BTCUSDT", Timeframe.M1, 0, 10 * M)
        assert r2.rows_written == 0
        assert fake.requests == n_first  # no wasted API calls on resume
        inst = Instrument(Exchange.BINANCE, "BTCUSDT")
        assert store.read(inst, Timeframe.M1).height == 10  # no duplication
