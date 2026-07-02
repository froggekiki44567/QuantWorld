"""Strategy contract (foundation for the Phase 3 plugin framework).

A strategy is a pure signal generator:
- sees the market only through MarketView (point-in-time enforced)
- may read its own position via the context (needed for stateful logic)
- returns Intents (target exposure), never orders, never sizes
- performs no I/O, holds no wall-clock, knows nothing about fees

This purity is what makes strategies unit-testable with 10 synthetic bars
and identically runnable in backtest, paper, and live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from quantlab.backtest.marketview import MarketView
from quantlab.backtest.orders import Intent
from quantlab.core.types import Instrument


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Read-only view of what the strategy is allowed to know beyond prices."""

    instrument: Instrument
    position_weight: float  # current exposure as fraction of equity


class BaseStrategy(ABC):
    """Subclass and implement on_bar. Registry/config wiring comes in Phase 3."""

    #: bars required before the strategy emits anything (engine enforces)
    warmup: int = 0

    @abstractmethod
    def on_bar(self, view: MarketView, ctx: StrategyContext) -> list[Intent]:
        raise NotImplementedError


class SmaCrossStrategy(BaseStrategy):
    """Reference strategy for engine verification — not an alpha claim.

    Long when fast SMA > slow SMA, flat otherwise. Exists because its
    behavior is hand-computable on synthetic data, which is exactly what
    engine tests need.
    """

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast, self.slow = fast, slow
        self.warmup = slow

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> list[Intent]:
        closes = view.history(self.slow, "close")
        fast_ma = closes[-self.fast :].mean()
        slow_ma = closes.mean()
        target = 1.0 if fast_ma > slow_ma else 0.0
        if target != ctx.position_weight:
            return [Intent(ctx.instrument, target, view.ts)]
        return []


class BuyAndHold(BaseStrategy):
    """The baseline every strategy must beat. Also an engine test fixture."""

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> list[Intent]:
        if ctx.position_weight == 0.0:
            return [Intent(ctx.instrument, 1.0, view.ts)]
        return []
