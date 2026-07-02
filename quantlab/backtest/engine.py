"""Backtest engine.

Bar loop (the order of operations IS the anti-look-ahead guarantee):

  for each bar t:
    1. EXECUTE orders created on bar t-1, at bar t's OPEN (+slippage, fees)
    2. strategy sees data through bar t (cursor advanced), emits intents
    3. sizer converts intents -> orders using bar t CLOSE as decision price
       (these orders wait for step 1 of bar t+1)
    4. mark portfolio at bar t close, snapshot equity

A signal can never be filled on the bar that produced it. Decision price
(close of t) and execution price (open of t+1) differ — that difference is
real and every candle backtest that fills at signal-bar close hides it.

Sizing here is a deliberate pass-through stub (SimpleSizer): target weight
-> qty, plus a liquidity cap. The real risk engine replaces it in Phase 5
behind the same interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from quantlab.backtest.costs import FeeModel, SlippageModel
from quantlab.backtest.marketview import MarketView
from quantlab.backtest.orders import Fill, Intent, Order, Rejection
from quantlab.backtest.portfolio import Portfolio
from quantlab.core.types import Instrument
from quantlab.strategies.base import BaseStrategy, StrategyContext

log = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass(slots=True)
class SimpleSizer:
    """Intent -> Order. min_trade_notional suppresses dust rebalances that
    would otherwise bleed fees on every bar."""

    min_trade_notional: float = 10.0

    def size(
        self, intent: Intent, equity: float, price: float, current_qty: float
    ) -> Order | None:
        target_qty = equity * intent.target_weight / price
        delta = target_qty - current_qty
        if abs(delta * price) < self.min_trade_notional:
            return None
        return Order(instrument=intent.instrument, qty=delta, created_ts=intent.ts)


@dataclass(slots=True)
class FillSimulator:
    """Market orders fill at bar open, adversarial slippage, taker fees.

    max_participation caps fill size at a fraction of the bar's volume —
    the unfilled remainder is REJECTED and recorded, not silently filled.
    A backtest that ignores its own liquidity footprint reports capacity
    that does not exist.
    """

    fees: FeeModel
    slippage: SlippageModel
    max_participation: float = 0.1

    def execute(
        self, order: Order, ts: int, bar_open: float, bar_volume: float
    ) -> tuple[Fill | None, Rejection | None]:
        cap = bar_volume * self.max_participation
        fill_qty = max(-cap, min(cap, order.qty))
        rejection = None
        if abs(fill_qty) < abs(order.qty) - _EPS:
            rejection = Rejection(
                order.order_id, ts, "liquidity_cap", order.qty - fill_qty
            )
        if abs(fill_qty) < _EPS:
            return None, rejection
        price = self.slippage.execution_price(bar_open, fill_qty, bar_volume)
        fee = self.fees.taker_fee(fill_qty * price)
        return Fill(order.order_id, order.instrument, ts, fill_qty, price, fee), rejection


@dataclass(slots=True)
class BacktestResult:
    instrument: Instrument
    initial_cash: float
    final_equity: float
    equity_curve: pl.DataFrame
    fills: list[Fill]
    rejections: list[Rejection]
    fees_paid: float
    realized_pnl: float

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_cash - 1.0

    @property
    def n_trades(self) -> int:
        return len(self.fills)


@dataclass(slots=True)
class BacktestEngine:
    strategy: BaseStrategy
    instrument: Instrument
    fees: FeeModel
    slippage: SlippageModel
    initial_cash: float = 10_000.0
    max_participation: float = 0.1
    sizer: SimpleSizer = field(default_factory=SimpleSizer)

    def run(self, df: pl.DataFrame) -> BacktestResult:
        view = MarketView(df)
        portfolio = Portfolio(initial_cash=self.initial_cash)
        simulator = FillSimulator(self.fees, self.slippage, self.max_participation)

        pending: list[Order] = []
        fills: list[Fill] = []
        rejections: list[Rejection] = []
        bar_i = 0

        while view._advance():
            ts = view.ts

            # 1. Execute orders from the PREVIOUS bar at THIS bar's open.
            for order in pending:
                fill, rej = simulator.execute(order, ts, view.open(), view.volume())
                if fill:
                    portfolio.apply_fill(fill)
                    fills.append(fill)
                if rej:
                    rejections.append(rej)
            pending.clear()

            # 2-3. Strategy decides on completed bar t; orders wait for t+1.
            portfolio.mark(self.instrument, view.close())
            if bar_i + 1 >= self.strategy.warmup:
                equity = portfolio.equity
                pos_qty = portfolio.position_qty(self.instrument)
                weight = pos_qty * view.close() / equity if equity > _EPS else 0.0
                ctx = StrategyContext(self.instrument, position_weight=round(weight, 6))
                for intent in self.strategy.on_bar(view, ctx):
                    order = self.sizer.size(intent, equity, view.close(), pos_qty)
                    if order is not None:
                        pending.append(order)

            # 4. Mark & snapshot at close.
            portfolio.snapshot(ts)
            bar_i += 1

        curve = pl.DataFrame(
            {
                "ts": [p.ts for p in portfolio.equity_curve],
                "equity": [p.equity for p in portfolio.equity_curve],
                "cash": [p.cash for p in portfolio.equity_curve],
            },
            schema={"ts": pl.Int64, "equity": pl.Float64, "cash": pl.Float64},
        )
        return BacktestResult(
            instrument=self.instrument,
            initial_cash=self.initial_cash,
            final_equity=portfolio.equity,
            equity_curve=curve,
            fills=fills,
            rejections=rejections,
            fees_paid=portfolio.fees_paid,
            realized_pnl=portfolio.realized_pnl,
        )
