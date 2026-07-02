import polars as pl
import pytest

from quantlab.core.types import Candle, Timeframe
from quantlab.data.validation import Gap, validate

M = Timeframe.M1.ms


def make_df(ts_list, **overrides):
    n = len(ts_list)
    base = {
        "ts_open": ts_list,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [10.0] * n,
        "quote_volume": [1000.0] * n,
        "n_trades": [50] * n,
        "taker_buy_base": [5.0] * n,
    }
    base.update(overrides)
    return pl.DataFrame(base)


class TestTimeframe:
    def test_ms_values(self):
        assert Timeframe.M1.ms == 60_000
        assert Timeframe.H1.ms == 3_600_000
        assert Timeframe.D1.ms == 86_400_000

    def test_from_str_roundtrip(self):
        for tf in Timeframe:
            assert Timeframe.from_str(tf.value) is tf

    def test_from_str_invalid(self):
        with pytest.raises(ValueError):
            Timeframe.from_str("7m")


class TestCandleInvariants:
    def test_valid(self):
        assert Candle(0, 100, 101, 99, 100.5, 10).is_valid_ohlc()

    def test_high_below_close(self):
        assert not Candle(0, 100, 100.2, 99, 100.5, 10).is_valid_ohlc()

    def test_low_above_open(self):
        assert not Candle(0, 100, 101, 100.5, 101, 10).is_valid_ohlc()

    def test_negative_price(self):
        assert not Candle(0, 100, 101, -1, 100, 10).is_valid_ohlc()

    def test_negative_volume(self):
        assert not Candle(0, 100, 101, 99, 100, -5).is_valid_ohlc()


class TestValidate:
    def test_clean_data(self):
        df = make_df([0, M, 2 * M, 3 * M])
        r = validate(df, Timeframe.M1)
        assert r.is_clean
        assert r.n_rows == 4

    def test_empty(self):
        r = validate(make_df([]), Timeframe.M1)
        assert r.is_clean

    def test_duplicates(self):
        r = validate(make_df([0, M, M, 2 * M]), Timeframe.M1)
        assert r.n_duplicates == 1
        assert not r.is_clean

    def test_non_monotonic(self):
        r = validate(make_df([0, 2 * M, M]), Timeframe.M1)
        assert r.n_non_monotonic == 1

    def test_ohlc_violation(self):
        df = make_df([0, M], high=[101.0, 99.5])  # 2nd: high < close
        r = validate(df, Timeframe.M1)
        assert r.n_ohlc_violations == 1

    def test_negative_volume(self):
        df = make_df([0, M], volume=[10.0, -1.0])
        r = validate(df, Timeframe.M1)
        assert r.n_negative_volume == 1

    def test_misaligned_timestamp(self):
        r = validate(make_df([0, M + 7]), Timeframe.M1)
        assert r.n_misaligned_ts == 1

    def test_single_internal_gap(self):
        r = validate(make_df([0, M, 4 * M, 5 * M]), Timeframe.M1)
        assert r.gaps == [Gap(2 * M, 4 * M, 2)]
        assert r.n_missing == 2

    def test_gap_at_start_and_end_with_expected_range(self):
        # Data covers minutes 2..3, but we expected 0..6.
        r = validate(
            make_df([2 * M, 3 * M]),
            Timeframe.M1,
            expected_start_ms=0,
            expected_end_ms=6 * M,
        )
        assert Gap(0, 2 * M, 2) in r.gaps
        assert Gap(4 * M, 6 * M, 2) in r.gaps
        assert r.n_missing == 4

    def test_no_false_gap_without_expected_range(self):
        # Without expected range, edges are defined by the data itself.
        r = validate(make_df([2 * M, 3 * M]), Timeframe.M1)
        assert r.gaps == []

    def test_multiple_gaps(self):
        r = validate(make_df([0, 2 * M, 5 * M]), Timeframe.M1)
        assert r.gaps == [Gap(M, 2 * M, 1), Gap(3 * M, 5 * M, 2)]
