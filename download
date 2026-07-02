"""Resample candles from a base timeframe to a higher one.

We store 1m and resample locally instead of downloading each timeframe,
because exchanges disagree on candle boundaries and this guarantees every
timeframe is derived from one consistent source. Also 5x less storage and
API traffic.

Rule: a resampled bucket is emitted only if COMPLETE (all constituent base
candles present). A 1h candle built from 57 of 60 minutes is fabricated
data — worse than a recorded gap, because it looks real.
"""

from __future__ import annotations

import polars as pl

from quantlab.core.types import Timeframe


def resample(df: pl.DataFrame, src: Timeframe, dst: Timeframe) -> pl.DataFrame:
    if dst.ms % src.ms != 0 or dst.ms <= src.ms:
        raise ValueError(f"Cannot resample {src.value} -> {dst.value}")
    factor = dst.ms // src.ms
    if df.height == 0:
        return df

    return (
        df.sort("ts_open")
        .with_columns((pl.col("ts_open") // dst.ms * dst.ms).alias("_bucket"))
        .group_by("_bucket", maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("quote_volume").sum(),
            pl.col("n_trades").sum(),
            pl.col("taker_buy_base").sum(),
            pl.len().alias("_n"),
        )
        .filter(pl.col("_n") == factor)  # completeness rule
        .drop("_n")
        .rename({"_bucket": "ts_open"})
        .select(
            "ts_open", "open", "high", "low", "close",
            "volume", "quote_volume", "n_trades", "taker_buy_base",
        )
    )
