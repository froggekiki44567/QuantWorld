"""Candle storage.

Layout: <root>/candles/exchange=X/symbol=Y/timeframe=Z/year=YYYY/month=MM.parquet

Decisions:
- Parquet is the source of truth: columnar, compressed (~10x vs CSV),
  portable, and DuckDB/polars read it natively with predicate pushdown on
  the hive partition columns.
- Monthly partitions: a 1m month is ~43k rows — big enough for compression
  to work, small enough that idempotent rewrite of one partition is cheap.
- Writes are idempotent upserts: read existing partition, merge, dedupe by
  ts_open keeping the newest write, sort, atomic replace (tmp + rename).
  Re-running a backfill is therefore always safe.
- DuckDB is a stateless query layer over the files — nothing to migrate,
  nothing to corrupt.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import duckdb
import polars as pl

from quantlab.core.types import Instrument, Timeframe

_SCHEMA = {
    "ts_open": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
    "n_trades": pl.Int64,
    "taker_buy_base": pl.Float64,
}


class CandleStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- paths

    def _partition_dir(self, inst: Instrument, tf: Timeframe) -> Path:
        return (
            self.root
            / "candles"
            / f"exchange={inst.exchange.value}"
            / f"symbol={inst.symbol}"
            / f"timeframe={tf.value}"
        )

    @staticmethod
    def _month_key(ts_ms: int) -> tuple[int, int]:
        d = dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc)
        return d.year, d.month

    def _month_path(self, inst: Instrument, tf: Timeframe, year: int, month: int) -> Path:
        return self._partition_dir(inst, tf) / f"year={year}" / f"month={month:02d}.parquet"

    # ------------------------------------------------------------- write

    def upsert(self, inst: Instrument, tf: Timeframe, df: pl.DataFrame) -> int:
        """Merge candles into monthly partitions. Returns rows written.

        Newest write wins on ts_open collision (allows re-ingesting
        corrected data). Atomic per partition.
        """
        if df.height == 0:
            return 0
        df = df.select(list(_SCHEMA)).cast(_SCHEMA)  # enforce schema strictly

        written = 0
        parts = df.with_columns(
            (pl.col("ts_open") // 1000)
            .map_elements(
                lambda s: dt.datetime.fromtimestamp(s, tz=dt.timezone.utc).year * 100
                + dt.datetime.fromtimestamp(s, tz=dt.timezone.utc).month,
                return_dtype=pl.Int64,
            )
            .alias("_ym")
        )
        for (ym,), month_df in parts.group_by("_ym"):
            year, month = divmod(int(ym), 100)
            path = self._month_path(inst, tf, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            new = month_df.drop("_ym")
            if path.exists():
                existing = pl.read_parquet(path)
                merged = pl.concat([existing, new]).unique(
                    subset="ts_open", keep="last", maintain_order=True
                )
            else:
                merged = new.unique(subset="ts_open", keep="last")
            merged = merged.sort("ts_open")
            self._atomic_write(merged, path)
            written += new.height
        return written

    @staticmethod
    def _atomic_write(df: pl.DataFrame, path: Path) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            df.write_parquet(tmp, compression="zstd")
            os.replace(tmp, path)  # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------- read

    def read(
        self,
        inst: Instrument,
        tf: Timeframe,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pl.DataFrame:
        pdir = self._partition_dir(inst, tf)
        files = sorted(pdir.rglob("*.parquet"))
        if not files:
            return pl.DataFrame(schema=_SCHEMA)
        lf = pl.scan_parquet([str(f) for f in files])
        if start_ms is not None:
            lf = lf.filter(pl.col("ts_open") >= start_ms)
        if end_ms is not None:
            lf = lf.filter(pl.col("ts_open") < end_ms)
        return lf.sort("ts_open").collect()

    def max_ts(self, inst: Instrument, tf: Timeframe) -> int | None:
        """Latest stored ts_open — the resume point for incremental backfill."""
        pdir = self._partition_dir(inst, tf)
        files = sorted(pdir.rglob("*.parquet"))
        if not files:
            return None
        # Only the lexicographically-last partition can hold the max.
        return int(pl.scan_parquet(str(files[-1])).select(pl.col("ts_open").max()).collect().item())

    # ------------------------------------------------------------- duckdb

    def sql(self, query: str) -> duckdb.DuckDBPyRelation:
        """Ad-hoc analytics over the whole store.

        Exposes a `candles` view with hive partition columns
        (exchange, symbol, timeframe) available as filters.
        """
        con = duckdb.connect()
        glob = str(self.root / "candles" / "**" / "*.parquet")
        con.execute(
            f"CREATE VIEW candles AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
        return con.sql(query)
