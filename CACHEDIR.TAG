"""Exchange adapter contract.

Each adapter is a thin, honest wrapper over one exchange's raw REST API.
Deliberately NOT ccxt: we want to see pagination boundaries, rate-limit
headers, and raw payload shape, because silent normalization in the data
layer is how bad data gets into backtests.

Adapters return raw Candle lists. Validation happens downstream (single
validation path for all exchanges), never inside the adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from quantlab.core.types import Candle, Exchange, Timeframe


class ExchangeAdapter(ABC):
    """Fetches historical klines. Stateless w.r.t. storage."""

    exchange: Exchange

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_ms: int,
        end_ms: int,
    ) -> Iterator[list[Candle]]:
        """Yield pages of candles with ts_open in [start_ms, end_ms).

        Yields pages (not one big list) so the downloader can persist
        incrementally and survive interruption — resumability is a
        requirement, not a nice-to-have, when backfilling years of 1m data.

        Guarantees the adapter must uphold:
        - candles within a page are sorted ascending by ts_open
        - pages are yielded in ascending time order
        - no candle outside [start_ms, end_ms)
        - the currently-forming (unclosed) candle is never returned
        """
        raise NotImplementedError

    @abstractmethod
    def earliest_available(self, symbol: str, timeframe: Timeframe) -> int:
        """First available ts_open for the symbol (listing date), ms UTC."""
        raise NotImplementedError
