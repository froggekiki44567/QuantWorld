"""Transaction cost models.

Costs are where most retail backtests lie to their authors. Rules here:
- No zero-cost default. You must construct a model explicitly.
- Slippage is adversarial: always moves the price AGAINST the trader.
- Models are pure functions of observable inputs (order, bar) — no state,
  trivially testable, identical in backtest and paper mode.

Defaults reflect Binance spot VIP0: taker 10 bps, maker 10 bps (2026 fee
schedule with BNB discount off). Override per your actual tier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeeModel:
    taker_bps: float = 10.0
    maker_bps: float = 10.0

    def taker_fee(self, notional: float) -> float:
        return abs(notional) * self.taker_bps / 10_000


class SlippageModel(ABC):
    @abstractmethod
    def execution_price(
        self, ref_price: float, qty: float, bar_volume: float
    ) -> float:
        """Price actually paid. qty>0 pays MORE than ref, qty<0 receives LESS."""


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage(SlippageModel):
    """Constant spread-crossing cost. Floor model: appropriate for orders
    that are tiny relative to bar volume in liquid markets."""

    bps: float = 5.0

    def execution_price(self, ref_price: float, qty: float, bar_volume: float) -> float:
        sign = 1.0 if qty > 0 else -1.0
        return ref_price * (1 + sign * self.bps / 10_000)


@dataclass(frozen=True, slots=True)
class VolumeImpactSlippage(SlippageModel):
    """Fixed floor + linear impact in participation rate.

    impact_bps_at_full = cost in bps if the order were 100% of bar volume;
    scales linearly below that. Linear is a deliberately conservative
    simplification (square-root impact is gentler for small participations);
    with candle data, conservative beats precise.
    """

    floor_bps: float = 5.0
    impact_bps_at_full: float = 100.0

    def execution_price(self, ref_price: float, qty: float, bar_volume: float) -> float:
        sign = 1.0 if qty > 0 else -1.0
        participation = abs(qty) / bar_volume if bar_volume > 0 else 1.0
        bps = self.floor_bps + self.impact_bps_at_full * min(participation, 1.0)
        return ref_price * (1 + sign * bps / 10_000)
