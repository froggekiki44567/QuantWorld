import polars as pl

from quantlab.core.types import Exchange, Instrument, Timeframe
from quantlab.data.gaps import GapRegistry
from quantlab.data.resample import resample
from quantlab.data.store import CandleStore
from quantlab.data.validation import Gap

M = Timeframe.M1.ms
INST = Instrument(Exchange.BINANCE, "BTCUSDT")

# 2024-01-31 23:58 UTC — two minutes before a month boundary.
JAN31_2358 = 1706745480000


def make_df(ts_list, close=None):
    n = len(ts_list)
    return pl.DataFrame(
        {
            "ts_open": ts_list,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": close or [100.5] * n,
            "volume": [10.0] * n,
            "quote_volume": [1000.0] * n,
            "n_trades": [50] * n,
            "taker_buy_base": [5.0] * n,
        }
    )


class TestStore:
    def test_roundtrip(self, tmp_path):
        store = CandleStore(tmp_path)
        df = make_df([JAN31_2358, JAN31_2358 + M])
        assert store.upsert(INST, Timeframe.M1, df) == 2
        out = store.read(INST, Timeframe.M1)
        assert out.height == 2
        assert out["ts_open"].to_list() == [JAN31_2358, JAN31_2358 + M]

    def test_idempotent_reupsert(self, tmp_path):
        store = CandleStore(tmp_path)
        df = make_df([JAN31_2358, JAN31_2358 + M])
        store.upsert(INST, Timeframe.M1, df)
        store.upsert(INST, Timeframe.M1, df)  # re-run same backfill
        assert store.read(INST, Timeframe.M1).height == 2

    def test_newest_write_wins(self, tmp_path):
        store = CandleStore(tmp_path)
        store.upsert(INST, Timeframe.M1, make_df([JAN31_2358], close=[100.5]))
        store.upsert(INST, Timeframe.M1, make_df([JAN31_2358], close=[200.0]))
        out = store.read(INST, Timeframe.M1)
        assert out.height == 1
        assert out["close"][0] == 200.0

    def test_month_boundary_split(self, tmp_path):
        store = CandleStore(tmp_path)
        # 23:58, 23:59 Jan 31 + 00:00, 00:01 Feb 1
        ts = [JAN31_2358 + i * M for i in range(4)]
        store.upsert(INST, Timeframe.M1, make_df(ts))
        files = list(tmp_path.rglob("*.parquet"))
        assert len(files) == 2  # two monthly partitions
        out = store.read(INST, Timeframe.M1)
        assert out["ts_open"].to_list() == ts  # read stitches them sorted

    def test_range_filter(self, tmp_path):
        store = CandleStore(tmp_path)
        ts = [JAN31_2358 + i * M for i in range(4)]
        store.upsert(INST, Timeframe.M1, make_df(ts))
        out = store.read(INST, Timeframe.M1, start_ms=ts[1], end_ms=ts[3])
        assert out["ts_open"].to_list() == [ts[1], ts[2]]

    def test_max_ts_resume_point(self, tmp_path):
        store = CandleStore(tmp_path)
        assert store.max_ts(INST, Timeframe.M1) is None
        ts = [JAN31_2358 + i * M for i in range(4)]  # spans two partitions
        store.upsert(INST, Timeframe.M1, make_df(ts))
        assert store.max_ts(INST, Timeframe.M1) == ts[-1]

    def test_isolation_between_instruments(self, tmp_path):
        store = CandleStore(tmp_path)
        other = Instrument(Exchange.BINANCE, "ETHUSDT")
        store.upsert(INST, Timeframe.M1, make_df([JAN31_2358]))
        assert store.read(other, Timeframe.M1).height == 0

    def test_duckdb_sql(self, tmp_path):
        store = CandleStore(tmp_path)
        store.upsert(INST, Timeframe.M1, make_df([JAN31_2358, JAN31_2358 + M]))
        n = store.sql("SELECT count(*) AS n FROM candles WHERE symbol='BTCUSDT'").df()
        assert int(n["n"][0]) == 2


class TestGapRegistry:
    def test_record_and_load(self, tmp_path):
        reg = GapRegistry(tmp_path)
        reg.record(INST, Timeframe.M1, [Gap(0, 2 * M, 2)])
        assert reg.load(INST, Timeframe.M1) == [Gap(0, 2 * M, 2)]

    def test_coalesce_adjacent(self, tmp_path):
        reg = GapRegistry(tmp_path)
        reg.record(INST, Timeframe.M1, [Gap(0, 2 * M, 2)])
        reg.record(INST, Timeframe.M1, [Gap(2 * M, 3 * M, 1)])
        assert reg.load(INST, Timeframe.M1) == [Gap(0, 3 * M, 3)]

    def test_is_clean(self, tmp_path):
        reg = GapRegistry(tmp_path)
        reg.record(INST, Timeframe.M1, [Gap(5 * M, 7 * M, 2)])
        assert reg.is_clean(INST, Timeframe.M1, 0, 5 * M)
        assert not reg.is_clean(INST, Timeframe.M1, 6 * M, 10 * M)

    def test_resolve_filled_gap(self, tmp_path):
        reg = GapRegistry(tmp_path)
        reg.record(INST, Timeframe.M1, [Gap(0, 2 * M, 2), Gap(5 * M, 6 * M, 1)])
        reg.resolve(INST, Timeframe.M1, [Gap(0, 2 * M, 2)])
        assert reg.load(INST, Timeframe.M1) == [Gap(5 * M, 6 * M, 1)]


class TestResample:
    def test_1m_to_5m_ohlcv(self):
        ts = list(range(0, 5 * M, M))
        df = pl.DataFrame(
            {
                "ts_open": ts,
                "open": [10.0, 11, 12, 13, 14],
                "high": [15.0, 11, 12, 20, 14],
                "low": [9.0, 8, 12, 13, 14],
                "close": [11.0, 12, 13, 14, 12.5],
                "volume": [1.0] * 5,
                "quote_volume": [10.0] * 5,
                "n_trades": [2] * 5,
                "taker_buy_base": [0.5] * 5,
            }
        )
        out = resample(df, Timeframe.M1, Timeframe.M5)
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["open"] == 10.0  # first
        assert row["high"] == 20.0  # max
        assert row["low"] == 8.0  # min
        assert row["close"] == 12.5  # last
        assert row["volume"] == 5.0  # sum
        assert row["n_trades"] == 10

    def test_incomplete_bucket_dropped(self):
        # 5m bucket with only 4 of 5 minutes -> must NOT be emitted.
        ts = [0, M, 2 * M, 3 * M]
        df = pl.DataFrame(
            {
                "ts_open": ts,
                "open": [10.0] * 4, "high": [11.0] * 4, "low": [9.0] * 4,
                "close": [10.0] * 4, "volume": [1.0] * 4,
                "quote_volume": [10.0] * 4, "n_trades": [1] * 4,
                "taker_buy_base": [0.5] * 4,
            }
        )
        assert resample(df, Timeframe.M1, Timeframe.M5).height == 0

    def test_invalid_direction_raises(self):
        import pytest

        with pytest.raises(ValueError):
            resample(pl.DataFrame(), Timeframe.H1, Timeframe.M5)
