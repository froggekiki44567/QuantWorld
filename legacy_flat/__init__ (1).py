"""Engine integration tests on synthetic data.

The central test is timing: a signal on bar t must fill at bar t+1 OPEN.
We construct price paths where getting this wrong produces a measurably
different (better) result — the classic look-ahead signature.
"""

import polars as pl
import pytest

from quantlab.backtest.costs import FeeModel, FixedBpsSlippage
from quantlab.backtest.engine import BacktestEngine
from quantlab.backtest.metrics import max_drawdown, sharpe
from quantlab.backtest.orders import Intent
from quantlab.core.types import Exchange, Instrument, Timeframe
from quantlab.strategies.base import BaseStrategy, BuyAndHold, SmaCrossStrategy, StrategyContext

M = Timeframe.M1.ms
INST = Instrument(Exchange.BINANCE, "BTCUSDT")

NO_FEES = FeeModel(taker_bps=0, maker_bps=0)
NO_SLIP = FixedBpsSlippage(bps=0)


def make_df(opens, closes, volumes=None):
    n = len(closes)
    return pl.DataFrame(
        {
            "ts_open": [i * M for i in range(n)],
            "open": [float(x) for x in opens],
            "high": [float(max(o, c)) for o, c in zip(opens, closes, strict=True)],
            "low": [float(min(o, c)) for o, c in zip(opens, closes, strict=True)],
            "close": [float(x) for x in closes],
            "volume": [float(v) for v in (volumes or [1e9] * n)],
            "quote_volume": [0.0] * n,
            "n_trades": [1] * n,
            "taker_buy_base": [0.0] * n,
        }
    )


class EnterOnFirstBar(BaseStrategy):
    """Deterministic fixture: go 100% long on the very first bar."""

    def on_bar(self, view, ctx: StrategyContext):
        if ctx.position_weight == 0.0:
            return [Intent(ctx.instrument, 1.0, view.ts)]
        return []


class TestExecutionTiming:
    def test_fill_at_next_bar_open_not_signal_close(self):
        # Signal on bar 0 (close 100). Bar 1 GAPS UP to open at 150.
        # Honest engine buys at 150. Look-ahead engine would buy at 100.
        df = make_df(opens=[100, 150, 150], closes=[100, 150, 150])
        eng = BacktestEngine(EnterOnFirstBar(), INST, NO_FEES, NO_SLIP, initial_cash=15_000)
        res = eng.run(df)
        assert res.n_trades == 1
        assert res.fills[0].price == pytest.approx(150.0)  # gap paid, not dodged
        assert res.fills[0].ts == M  # filled on bar 1, not bar 0

    def test_no_fill_on_last_bar_signal(self):
        # Signal on the final bar has no next bar -> must never fill.
        df = make_df(opens=[100], closes=[100])
        res = BacktestEngine(EnterOnFirstBar(), INST, NO_FEES, NO_SLIP).run(df)
        assert res.n_trades == 0
        assert res.final_equity == pytest.approx(res.initial_cash)


class TestBuyAndHoldHandComputed:
    def test_exact_accounting_with_costs(self):
        # Bar0: signal at close 100. Bar1: fill at open 100 with 10bps slip
        # -> price 100.10, fee 10bps on notional. Price ends at 120.
        fees = FeeModel(taker_bps=10)
        slip = FixedBpsSlippage(bps=10)
        df = make_df(opens=[100, 100, 110], closes=[100, 110, 120])
        eng = BacktestEngine(BuyAndHold(), INST, fees, slip, initial_cash=10_000)
        res = eng.run(df)

        fill = res.fills[0]
        assert fill.price == pytest.approx(100.10)
        qty = fill.qty
        assert qty == pytest.approx(10_000 / 100)  # sized at bar0 close
        fee = qty * 100.10 * 0.0010
        assert fill.fee == pytest.approx(fee)
        expected_equity = 10_000 - qty * 100.10 - fee + qty * 120
        assert res.final_equity == pytest.approx(expected_equity)
        assert res.total_return == pytest.approx(expected_equity / 10_000 - 1)

    def test_costs_strictly_hurt(self):
        df = make_df(opens=[100, 100, 110], closes=[100, 110, 120])
        free = BacktestEngine(BuyAndHold(), INST, NO_FEES, NO_SLIP, initial_cash=10_000).run(df)
        costly = BacktestEngine(
            BuyAndHold(), INST, FeeModel(taker_bps=10),
            FixedBpsSlippage(bps=10), initial_cash=10_000,
        ).run(df)
        assert costly.final_equity < free.final_equity


class TestLiquidityCap:
    def test_partial_fill_and_rejection_recorded(self):
        # Volume 10/bar, cap 10% -> max 1 unit per bar. Order wants 100.
        df = make_df(opens=[100] * 3, closes=[100] * 3, volumes=[10] * 3)
        eng = BacktestEngine(
            EnterOnFirstBar(), INST, NO_FEES, NO_SLIP,
            initial_cash=10_000, max_participation=0.1,
        )
        res = eng.run(df)
        assert len(res.rejections) >= 1
        assert res.rejections[0].reason == "liquidity_cap"
        for f in res.fills:
            assert abs(f.qty) <= 1.0 + 1e-9  # never above the cap


class TestWarmup:
    def test_no_signals_during_warmup(self):
        closes = list(range(100, 140))
        df = make_df(opens=closes, closes=closes)
        strat = SmaCrossStrategy(fast=3, slow=10)
        res = BacktestEngine(strat, INST, NO_FEES, NO_SLIP).run(df)
        # earliest possible signal: bar index 9 (10th bar) -> fill on bar 10
        assert all(f.ts >= 10 * M for f in res.fills)
        assert res.n_trades >= 1  # uptrend -> it does eventually enter


class TestDeterminism:
    def test_identical_runs(self):
        closes = [100 + ((i * 7919) % 13) - 6 for i in range(200)]  # pseudo-random walk
        df = make_df(opens=closes, closes=closes)

        def run():
            return BacktestEngine(
                SmaCrossStrategy(fast=5, slow=20), INST,
                FeeModel(taker_bps=10), FixedBpsSlippage(bps=5), initial_cash=10_000,
            ).run(df)

        a, b = run(), run()
        assert a.final_equity == b.final_equity
        assert [f.price for f in a.fills] == [f.price for f in b.fills]
        assert a.equity_curve.equals(b.equity_curve)


class TestMetrics:
    def test_max_drawdown(self):
        eq = pl.Series([100.0, 120, 90, 95, 130, 110])
        assert max_drawdown(eq) == pytest.approx(90 / 120 - 1)  # -25%

    def test_sharpe_sign(self):
        up = pl.Series([100.0 * (1.001**i) for i in range(100)])
        assert sharpe(up, Timeframe.M1) > 0
        down = pl.Series([100.0 * (0.999**i) for i in range(100)])
        assert sharpe(down, Timeframe.M1) < 0

    def test_flat_equity_zero_sharpe(self):
        assert sharpe(pl.Series([100.0] * 50), Timeframe.M1) == 0.0
