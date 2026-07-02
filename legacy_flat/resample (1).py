"""Data validation.

Single validation path for all exchanges, run on ingest AND queryable later.

Philosophy: flag, never fix. Bad candles are recorded and quarantined, gaps
are registered — nothing is forward-filled or interpolated, because fabricated
data poisons every backtest downstream. A strategy must be able to ask
"was this period clean?" and get an honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from quantlab.core.types import Timeframe


@dataclass(frozen=True, slots=True)
class Gap:
    """Missing candles: expected grid points absent in [start_ms, end_ms)."""

    start_ms: int
    end_ms: int
    n_missing: int


@dataclass(slots=True)
class ValidationReport:
    n_rows: int = 0
    n_duplicates: int = 0
    n_non_monotonic: int = 0
    n_ohlc_violations: int = 0
    n_negative_volume: int = 0
    n_misaligned_ts: int = 0  # ts_open not on the timeframe grid
    gaps: list[Gap] = field(default_factory=list)

    @property
    def n_missing(self) -> int:
        return sum(g.n_missing for g in self.gaps)

    @property
    def is_clean(self) -> bool:
        return (
            self.n_duplicates == 0
            and self.n_non_monotonic == 0
            and self.n_ohlc_violations == 0
            and self.n_negative_volume == 0
            and self.n_misaligned_ts == 0
            and not self.gaps
        )

    def summary(self) -> str:
        return (
            f"rows={self.n_rows} dup={self.n_duplicates} "
            f"nonmono={self.n_non_monotonic} ohlc_bad={self.n_ohlc_violations} "
            f"neg_vol={self.n_negative_volume} misaligned={self.n_misaligned_ts} "
            f"gaps={len(self.gaps)} missing={self.n_missing}"
        )


def validate(
    df: pl.DataFrame,
    timeframe: Timeframe,
    expected_start_ms: int | None = None,
    expected_end_ms: int | None = None,
) -> ValidationReport:
    """Validate a candle DataFrame (CANDLE_SCHEMA columns).

    Gap detection runs against the expected grid between
    [expected_start_ms, expected_end_ms) if given, else between the min and
    max observed ts_open. Passing the expected range matters: without it,
    missing data at the very start/end of the requested window is invisible.
    """
    report = ValidationReport(n_rows=df.height)
    if df.height == 0:
        return report

    ts = df["ts_open"]

    report.n_duplicates = df.height - ts.n_unique()

    diffs = ts.diff().drop_nulls()
    report.n_non_monotonic = int((diffs <= 0).sum())

    ohlc_bad = (
        (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.col("low"))
        | (pl.col("low") <= 0)
    )
    report.n_ohlc_violations = df.filter(ohlc_bad).height
    report.n_negative_volume = df.filter(pl.col("volume") < 0).height
    report.n_misaligned_ts = df.filter(pl.col("ts_open") % timeframe.ms != 0).height

    report.gaps = _detect_gaps(
        ts.unique().sort(),
        timeframe,
        expected_start_ms if expected_start_ms is not None else int(ts.min()),
        expected_end_ms if expected_end_ms is not None else int(ts.max()) + timeframe.ms,
    )
    return report


def _detect_gaps(
    sorted_ts: pl.Series, tf: Timeframe, start_ms: int, end_ms: int
) -> list[Gap]:
    """Compare observed timestamps against the expected regular grid.

    Vectorized: O(n) over observed candles, independent of gap sizes —
    a year-long delisting gap costs the same as a single missing minute.
    """
    step = tf.ms
    # Align expected window to the grid.
    grid_start = ((start_ms + step - 1) // step) * step
    grid_end = (end_ms // step) * step
    if grid_start >= grid_end:
        return []

    obs = sorted_ts.filter((sorted_ts >= grid_start) & (sorted_ts < grid_end))
    gaps: list[Gap] = []

    prev = grid_start - step
    for t in obs.to_list():
        if t - prev > step:
            gaps.append(Gap(prev + step, t, (t - prev) // step - 1))
        prev = t
    if grid_end - prev > step:
        gaps.append(Gap(prev + step, grid_end, (grid_end - prev) // step - 1))
    return gaps
