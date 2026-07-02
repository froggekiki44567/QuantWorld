"""Exchange adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from quantlab.core.types import Candle, Exchange, Timeframe


class ExchangeAdapter(ABC):
    exchange: Exchange

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_ms: int,
        end_ms: int,
    ) -> Iterator[list[Candle]]:
        raise NotImplementedError

    @abstractmethod
    def earliest_available(self, symbol: str, timeframe: Timeframe) -> int:
        raise NotImplementedError
