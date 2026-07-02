"""Order-flow domain objects.

Strategies emit Intents (desired exposure), never Orders. The sizer/risk
layer converts intents to orders. This separation is what lets the risk
engine (Phase 5) veto or resize without strategies knowing — and it means
strategy code contains zero position-sizing logic to overfit.

Signed quantities throughout: qty > 0 long/buy, qty < 0 short/sell.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

from quantlab.core.types import Instrument

_order_ids = itertools.count(1)


@dataclass(frozen=True, slots=True)
class Intent:
    """Desired portfolio state: target weight of equity in the instrument.

    target_weight = +1.0 -> fully long, -1.0 -> fully short, 0.0 -> flat.
    Emitted on bar t, actionable at bar t+1 at the earliest.
    """

    instrument: Instrument
    target_weight: float
    ts: int  # bar ts_open the intent was generated on (audit trail)


class OrderType(Enum):
    MARKET = "market"


@dataclass(slots=True)
class Order:
    instrument: Instrument
    qty: float  # signed
    order_type: OrderType = OrderType.MARKET
    created_ts: int = 0
    order_id: int = field(default_factory=lambda: next(_order_ids))


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: int
    instrument: Instrument
    ts: int  # bar ts_open on which the fill happened
    qty: float  # signed, may be < order qty (partial)
    price: float  # execution price incl. slippage
    fee: float  # quote currency, always >= 0


@dataclass(frozen=True, slots=True)
class Rejection:
    """A fill that didn't happen, and why. Rejections are data:
    a strategy whose orders are constantly capped by liquidity is a
    strategy whose backtest capacity is a fiction."""

    order_id: int
    ts: int
    reason: str
    unfilled_qty: float
