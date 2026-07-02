"""Core domain types shared by data, strategy, and backtest modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Exchange(Enum):
    BINANCE = "binance"


class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def ms(self) -> int:
        return {
            Timeframe.M1: 60_000,
            Timeframe.M5: 5 * 60_000,
            Timeframe.M15: 15 * 60_000,
            Timeframe.M30: 30 * 60_000,
            Timeframe.H1: 60 * 60_000,
            Timeframe.H4: 4 * 60 * 60_000,
            Timeframe.D1: 24 * 60 * 60_000,
        }[self]

    @classmethod
    def from_str(cls, value: str) -> Timeframe:
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(tf.value for tf in cls)
            raise ValueError(f"Unsupported timeframe {value!r}; expected one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class Instrument:
    exchange: Exchange
    symbol: str

    def __str__(self) -> str:
        return f"{self.exchange.value}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class Candle:
    ts_open: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    n_trades: int = 0
    taker_buy_base: float = 0.0

    def is_valid_ohlc(self) -> bool:
        return (
            self.low > 0
            and self.volume >= 0
            and self.high >= self.low
            and self.high >= max(self.open, self.close)
            and self.low <= min(self.open, self.close)
        )
