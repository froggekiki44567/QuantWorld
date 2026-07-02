"""Portfolio accounting.

Multi-instrument by design (dict-keyed positions) even though the Module 2
engine feeds one instrument — the accounting layer must not need a rewrite
when multi-asset arrives in Phase 3.

Accounting rules:
- Average-cost basis. Realized PnL is recognized on the closing portion of
  a fill; increasing a position re-averages the entry.
- Fees hit cash immediately, always.
- Position flips (long -> short in one fill) split into close + open.
- Equity = cash + sum(qty * mark_price). Marked at bar close by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantlab.backtest.orders import Fill
from quantlab.core.types import Instrument

_EPS = 1e-12


@dataclass(slots=True)
class Position:
    qty: float = 0.0
    avg_price: float = 0.0


@dataclass(slots=True)
class EquityPoint:
    ts: int
    equity: float
    cash: float


@dataclass(slots=True)
class Portfolio:
    initial_cash: float
    cash: float = field(init=False)
    positions: dict[Instrument, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    equity_curve: list[EquityPoint] = field(default_factory=list)
    _marks: dict[Instrument, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial_cash = float(self.initial_cash)
        self.cash = self.initial_cash

    # ------------------------------------------------------------- fills

    def apply_fill(self, fill: Fill) -> None:
        pos = self.positions.setdefault(fill.instrument, Position())
        qty, price = fill.qty, fill.price

        self.cash -= qty * price  # buy consumes cash, sell releases it
        self.cash -= fill.fee
        self.fees_paid += fill.fee

        if pos.qty * qty >= 0:  # same direction (or opening from flat)
            new_qty = pos.qty + qty
            if abs(new_qty) > _EPS:
                pos.avg_price = (pos.qty * pos.avg_price + qty * price) / new_qty
            pos.qty = new_qty
        else:  # reducing / closing / flipping
            closing = min(abs(qty), abs(pos.qty)) * (1 if qty > 0 else -1)
            # closing is opposite-signed to pos.qty. Works for both sides:
            # long closed by sell (closing<0): (avg-price)*closing = (price-avg)*|c|
            # short closed by buy (closing>0): (avg-price)*closing = (avg-price)*|c|
            self.realized_pnl += (pos.avg_price - price) * closing
            remainder = qty - closing
            pos.qty += closing
            if abs(pos.qty) < _EPS and abs(remainder) > _EPS:  # flip
                pos.qty = remainder
                pos.avg_price = price
            elif abs(pos.qty) < _EPS:
                pos.qty = 0.0
                pos.avg_price = 0.0

    # ------------------------------------------------------------- marking

    def mark(self, instrument: Instrument, price: float) -> None:
        self._marks[instrument] = price

    def snapshot(self, ts: int) -> EquityPoint:
        pt = EquityPoint(ts=ts, equity=self.equity, cash=self.cash)
        self.equity_curve.append(pt)
        return pt

    # ------------------------------------------------------------- views

    @property
    def equity(self) -> float:
        mtm = sum(
            p.qty * self._marks.get(inst, p.avg_price)
            for inst, p in self.positions.items()
        )
        return self.cash + mtm

    def position_qty(self, instrument: Instrument) -> float:
        p = self.positions.get(instrument)
        return p.qty if p else 0.0

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            p.qty * (self._marks.get(inst, p.avg_price) - p.avg_price)
            for inst, p in self.positions.items()
        )
