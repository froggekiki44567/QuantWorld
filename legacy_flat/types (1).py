"""Binance spot klines adapter.

Endpoint: GET /api/v3/klines  (public, no auth)
  - max 1000 candles per request
  - request weight 2 (as of 2025 API docs); IP limit 6000 weight/min
  - returns the currently-forming candle if endTime is not in the past —
    we always cap endTime at the last *closed* candle boundary.

Rate limiting: we track the X-MBX-USED-WEIGHT-1M response header and sleep
proactively when approaching the limit, instead of hammering until a 429.
On 429/418 we honor Retry-After. This is the difference between a polite
institutional downloader and a script that gets IP-banned mid-backfill.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import httpx

from quantlab.core.types import Candle, Exchange, Timeframe
from quantlab.data.adapters.base import ExchangeAdapter

log = logging.getLogger(__name__)

_BASE_URL = "https://api.binance.com"
_MAX_LIMIT = 1000
_WEIGHT_SOFT_CAP = 5000  # start throttling before the 6000/min hard limit
_MAX_RETRIES = 5


class BinanceAdapter(ExchangeAdapter):
    exchange = Exchange.BINANCE

    def __init__(self, client: httpx.Client | None = None) -> None:
        # Injectable client => trivially testable with httpx.MockTransport.
        self._client = client or httpx.Client(base_url=_BASE_URL, timeout=30.0)
        self._used_weight = 0

    # ------------------------------------------------------------------ API

    def fetch_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_ms: int,
        end_ms: int,
    ) -> Iterator[list[Candle]]:
        if start_ms >= end_ms:
            return
        # Never request the forming candle: cap at last closed boundary.
        now_ms = int(time.time() * 1000)
        last_closed_open = (now_ms // timeframe.ms - 1) * timeframe.ms
        end_ms = min(end_ms, last_closed_open + timeframe.ms)

        cursor = start_ms
        while cursor < end_ms:
            raw = self._request(
                "/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": timeframe.value,
                    "startTime": cursor,
                    "endTime": end_ms - 1,  # Binance endTime is inclusive
                    "limit": _MAX_LIMIT,
                },
            )
            if not raw:
                return
            page = [self._parse_kline(k) for k in raw]
            # Defensive: enforce contract even if the exchange misbehaves.
            page = [c for c in page if start_ms <= c.ts_open < end_ms]
            if not page:
                return
            yield page
            cursor = page[-1].ts_open + timeframe.ms
            if len(raw) < _MAX_LIMIT:
                return  # exchange has no more data in range

    def earliest_available(self, symbol: str, timeframe: Timeframe) -> int:
        raw = self._request(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": timeframe.value,
                "startTime": 0,
                "limit": 1,
            },
        )
        if not raw:
            raise ValueError(f"No data for {symbol} {timeframe.value}")
        return int(raw[0][0])

    # ------------------------------------------------------------- internals

    def _request(self, path: str, params: dict) -> list:
        for attempt in range(_MAX_RETRIES):
            self._throttle()
            resp = self._client.get(path, params=params)
            self._used_weight = int(
                resp.headers.get("x-mbx-used-weight-1m", self._used_weight)
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 418):
                wait = int(resp.headers.get("retry-after", 60))
                log.warning("Rate limited (%s), sleeping %ss", resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                backoff = 2**attempt
                log.warning("Server error %s, retry in %ss", resp.status_code, backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Failed after {_MAX_RETRIES} retries: {path} {params}")

    def _throttle(self) -> None:
        if self._used_weight >= _WEIGHT_SOFT_CAP:
            log.info("Weight %s >= soft cap, sleeping 10s", self._used_weight)
            time.sleep(10)
            self._used_weight = 0

    @staticmethod
    def _parse_kline(k: list) -> Candle:
        # Binance kline array layout:
        # [0] open time, [1] open, [2] high, [3] low, [4] close, [5] volume,
        # [6] close time, [7] quote volume, [8] n trades,
        # [9] taker buy base vol, [10] taker buy quote vol, [11] ignore
        return Candle(
            ts_open=int(k[0]),
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            quote_volume=float(k[7]),
            n_trades=int(k[8]),
            taker_buy_base=float(k[9]),
        )
