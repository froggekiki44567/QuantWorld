import polars as pl
import pytest

from quantlab.backtest.costs import FeeModel, FixedBpsSlippage, VolumeImpactSlippage
from quantlab.backtest.marketview import MarketView
from quantlab.backtest.orders import Fill
from quantlab.backtest.portfolio import Portfolio
from quantlab.core.types import Exchange, Instrument, Timeframe

M = Timeframe.M1.ms
INST = Instrument(Exchange.BINANCE, "BTCUSDT")


def make_df(closes, opens=None, volumes=None):
    n = len(closes)
    opens = opens or closes
    return pl.DataFrame(
        {
            "ts_open": [i * M for i in range(n)],
            "open": [float(x) for x in opens],
            "high": [float(max(o, c)) + 1 for o, c in zip(opens, closes, strict=True)],
            "low": [float(min(o, c)) - 1 for o, c in zip(opens, closes, strict=True)],
            "close": [float(x) for x in closes],
            "volume": [float(v) for v in (volumes or [1000.0] * n)],
            "quote_volume": [0.0] * n,
            "n_trades": [1] * n,
            "taker_buy_base": [0.0] * n,
        }
    )


class TestMarketView:
    def test_nothing_visible_before_advance(self):
        v = MarketView(make_df([1, 2, 3]))
        assert len(v) == 0

    def test_cursor_limits_visibility(self):
        v = MarketView(make_df([10, 20, 30]))
        v._advance()
        assert len(v) == 1
        assert v.close() == 10.0
        # The future does not exist from the strategy's perspective:
        assert list(v.history(100)) == [10.0]

    def test_lookback(self):
        v = MarketView(make_df([10, 20, 30]))
        v._advance()
        v._advance()
        assert v.close(0) == 20.0
        assert v.close(1) == 10.0
        with pytest.raises(IndexError):
            v.close(2)

    def test_negative_lookback_rejected(self):
        from quantlab.backtest.marketview import LookAheadError

        v = MarketView(make_df([10, 20, 30]))
        v._advance()
        with pytest.raises(LookAheadError):
            v.close(-1)  # "give me the next bar" -> structural refusal

    def test_history_is_a_copy(self):
        v = MarketView(make_df([10, 20, 30]))
        v._advance()
        h = v.history(1)
        h[0] = 999.0
        assert v.close() == 10.0

    def test_advance_exhausts(self):
        v = MarketView(make_df([1, 2]))
        assert v._advance() and v._advance()
        assert not v._advance()

    def test_unsorted_input_is_sorted(self):
        df = make_df([10, 20, 30]).sort("ts_open", descending=True)
        v = MarketView(df)
        v._advance()
        assert v.close() == 10.0


class TestPortfolio:
    def test_long_open_and_close_profit(self):
        p = Portfolio(initial_cash=10_000)
        p.apply_fill(Fill(1, INST, 0, qty=1.0, price=100.0, fee=0.1))
        assert p.cash == pytest.approx(10_000 - 100 - 0.1)
        p.mark(INST, 110.0)
        assert p.equity == pytest.approx(10_000 - 100 - 0.1 + 110)
        assert p.unrealized_pnl == pytest.approx(10.0)

        p.apply_fill(Fill(2, INST, M, qty=-1.0, price=110.0, fee=0.11))
        assert p.realized_pnl == pytest.approx(10.0)
        assert p.position_qty(INST) == 0.0
        assert p.cash == pytest.approx(10_000 + 10 - 0.21)

    def test_short_pnl(self):
        p = Portfolio(initial_cash=10_000)
        p.apply_fill(Fill(1, INST, 0, qty=-2.0, price=100.0, fee=0.0))
        p.apply_fill(Fill(2, INST, M, qty=2.0, price=90.0, fee=0.0))
        assert p.realized_pnl == pytest.approx(20.0)  # short 2 @100, cover @90
        assert p.cash == pytest.approx(10_020.0)

    def test_average_cost_on_add(self):
        p = Portfolio(initial_cash=10_000)
        p.apply_fill(Fill(1, INST, 0, qty=1.0, price=100.0, fee=0.0))
        p.apply_fill(Fill(2, INST, M, qty=1.0, price=110.0, fee=0.0))
        assert p.positions[INST].avg_price == pytest.approx(105.0)

    def test_partial_close(self):
        p = Portfolio(initial_cash=10_000)
        p.apply_fill(Fill(1, INST, 0, qty=2.0, price=100.0, fee=0.0))
        p.apply_fill(Fill(2, INST, M, qty=-1.0, price=120.0, fee=0.0))
        assert p.realized_pnl == pytest.approx(20.0)
        assert p.position_qty(INST) == pytest.approx(1.0)
        assert p.positions[INST].avg_price == pytest.approx(100.0)  # unchanged

    def test_flip_long_to_short(self):
        p = Portfolio(initial_cash=10_000)
        p.apply_fill(Fill(1, INST, 0, qty=1.0, price=100.0, fee=0.0))
        p.apply_fill(Fill(2, INST, M, qty=-3.0, price=110.0, fee=0.0))
        assert p.realized_pnl == pytest.approx(10.0)  # long leg closed +10
        assert p.position_qty(INST) == pytest.approx(-2.0)
        assert p.positions[INST].avg_price == pytest.approx(110.0)  # new basis

    def test_fees_always_reduce_cash(self):
        p = Portfolio(initial_cash=10_000)
        p.apply_fill(Fill(1, INST, 0, qty=-1.0, price=100.0, fee=5.0))
        assert p.cash == pytest.approx(10_000 + 100 - 5)
        assert p.fees_paid == 5.0


class TestCosts:
    def test_taker_fee_on_notional(self):
        f = FeeModel(taker_bps=10)
        assert f.taker_fee(10_000) == pytest.approx(10.0)
        assert f.taker_fee(-10_000) == pytest.approx(10.0)  # abs

    def test_fixed_slippage_adversarial_both_sides(self):
        s = FixedBpsSlippage(bps=10)
        assert s.execution_price(100.0, qty=1, bar_volume=1e9) == pytest.approx(100.10)
        assert s.execution_price(100.0, qty=-1, bar_volume=1e9) == pytest.approx(99.90)

    def test_volume_impact_scales_with_participation(self):
        s = VolumeImpactSlippage(floor_bps=5, impact_bps_at_full=100)
        small = s.execution_price(100.0, qty=1, bar_volume=10_000)
        big = s.execution_price(100.0, qty=5_000, bar_volume=10_000)
        assert small < big
        # 50% participation -> 5 + 50 = 55 bps
        assert big == pytest.approx(100 * 1.0055)

    def test_impact_capped_at_full_volume(self):
        s = VolumeImpactSlippage(floor_bps=0, impact_bps_at_full=100)
        p = s.execution_price(100.0, qty=99_999, bar_volume=10)
        assert p == pytest.approx(101.0)  # capped, not extrapolated
