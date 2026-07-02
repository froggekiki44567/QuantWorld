"""Point-in-time market data access.

THE anti-look-ahead mechanism. Strategies never touch raw DataFrames; they
receive a MarketView whose cursor is advanced only by the engine. Every
accessor slices [0, cursor]. Requesting anything past the cursor is not an
error the strategy can catch and work around — the data simply does not
exist from the strategy's point of view.

Internally numpy arrays for O(1) indexed access; the polars frame is
converted once at construction.
"""

from __future__ import annotations

import numpy as np
import polars as pl


class MarketView:
    __slots__ = ("_ts", "_open", "_high", "_low", "_close", "_volume", "_cursor")

    def __init__(self, df: pl.DataFrame) -> None:
        df = df.sort("ts_open")
        self._ts = df["ts_open"].to_numpy()
        self._open = df["open"].to_numpy()
        self._high = df["high"].to_numpy()
        self._low = df["low"].to_numpy()
        self._close = df["close"].to_numpy()
        self._volume = df["volume"].to_numpy()
        self._cursor = -1  # before first bar: nothing visible

    # ---- engine-only API (single underscore: convention-enforced) --------

    def _advance(self) -> bool:
        if self._cursor + 1 >= len(self._ts):
            return False
        self._cursor += 1
        return True

    # ---- strategy-facing API ---------------------------------------------

    @property
    def i(self) -> int:
        """Index of the current (latest visible) bar."""
        return self._cursor

    @property
    def ts(self) -> int:
        return int(self._ts[self._cursor])

    def close(self, lookback: int = 0) -> float:
        """close(0) = current bar close, close(1) = previous bar, ..."""
        return float(self._close[self._at(lookback)])

    def open(self, lookback: int = 0) -> float:
        return float(self._open[self._at(lookback)])

    def high(self, lookback: int = 0) -> float:
        return float(self._high[self._at(lookback)])

    def low(self, lookback: int = 0) -> float:
        return float(self._low[self._at(lookback)])

    def volume(self, lookback: int = 0) -> float:
        return float(self._volume[self._at(lookback)])

    def history(self, n: int, field: str = "close") -> np.ndarray:
        """Last n values of a field, ending at the current bar (inclusive).

        Returns a COPY: mutating history must not corrupt the source.
        Fewer than n bars available -> returns what exists (shorter array);
        strategies must check len() during warmup.
        """
        arr = getattr(self, f"_{field}")
        start = max(0, self._cursor + 1 - n)
        return arr[start : self._cursor + 1].copy()

    def _at(self, lookback: int) -> int:
        if lookback < 0:
            raise LookAheadError("Negative lookback is a future reference.")
        idx = self._cursor - lookback
        if idx < 0:
            raise IndexError(f"Only {self._cursor + 1} bars visible, asked lookback={lookback}")
        return idx

    def __len__(self) -> int:
        """Bars visible so far — NOT the total dataset length."""
        return self._cursor + 1


class LookAheadError(Exception):
    pass
