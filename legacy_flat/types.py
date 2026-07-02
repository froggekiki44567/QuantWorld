"""Gap registry.

Gaps are first-class data: a backtest that trades through an unrecorded
data hole produces silently wrong results (positions "held" through missing
periods, indicators computed over discontinuities). The registry lets any
consumer ask: is [start, end) fully covered for this instrument?

Stored as one parquet file per instrument+timeframe next to the candles.
Rewritten wholesale on update (gap lists are tiny).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quantlab.core.types import Instrument, Timeframe
from quantlab.data.validation import Gap

_SCHEMA = {"start_ms": pl.Int64, "end_ms": pl.Int64, "n_missing": pl.Int64}


class GapRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, inst: Instrument, tf: Timeframe) -> Path:
        return (
            self.root
            / "gaps"
            / f"exchange={inst.exchange.value}"
            / f"symbol={inst.symbol}"
            / f"timeframe={tf.value}.parquet"
        )

    def record(self, inst: Instrument, tf: Timeframe, gaps: list[Gap]) -> None:
        """Merge new gaps with existing ones (dedupe, coalesce adjacent)."""
        existing = self.load(inst, tf)
        combined = existing + gaps
        merged = _coalesce(combined, tf)
        path = self._path(inst, tf)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "start_ms": [g.start_ms for g in merged],
                "end_ms": [g.end_ms for g in merged],
                "n_missing": [g.n_missing for g in merged],
            },
            schema=_SCHEMA,
        ).write_parquet(path)

    def resolve(self, inst: Instrument, tf: Timeframe, filled: list[Gap]) -> None:
        """Remove gaps that a later backfill managed to fill."""
        remaining = [
            g
            for g in self.load(inst, tf)
            if not any(f.start_ms <= g.start_ms and f.end_ms >= g.end_ms for f in filled)
        ]
        path = self._path(inst, tf)
        if path.exists():
            path.unlink()
        if remaining:
            self.record(inst, tf, remaining)

    def load(self, inst: Instrument, tf: Timeframe) -> list[Gap]:
        path = self._path(inst, tf)
        if not path.exists():
            return []
        df = pl.read_parquet(path)
        return [
            Gap(int(r["start_ms"]), int(r["end_ms"]), int(r["n_missing"]))
            for r in df.iter_rows(named=True)
        ]

    def is_clean(self, inst: Instrument, tf: Timeframe, start_ms: int, end_ms: int) -> bool:
        """True if no known gap intersects [start_ms, end_ms)."""
        return not any(
            g.start_ms < end_ms and g.end_ms > start_ms for g in self.load(inst, tf)
        )


def _coalesce(gaps: list[Gap], tf: Timeframe) -> list[Gap]:
    """Sort, dedupe, and merge overlapping/adjacent gaps."""
    if not gaps:
        return []
    gaps = sorted(set(gaps), key=lambda g: g.start_ms)
    out: list[Gap] = [gaps[0]]
    for g in gaps[1:]:
        last = out[-1]
        if g.start_ms <= last.end_ms:  # overlap or adjacency
            end = max(last.end_ms, g.end_ms)
            out[-1] = Gap(last.start_ms, end, (end - last.start_ms) // tf.ms)
        else:
            out.append(g)
    return out
