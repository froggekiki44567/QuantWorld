"""Minimal performance metrics needed by Module 2 tests and sanity checks.

Full analytics (Sortino, Calmar, trade distributions, tearsheets) is
Phase 8. Only what's needed now, computed correctly:
- Sharpe is annualized from per-bar returns using the actual bar duration;
  crypto trades 24/7, so a year is 365 days, not 252 trading days.
- Max drawdown from the equity curve, peak-to-trough.
"""

from __future__ import annotations

import math

import polars as pl

from quantlab.core.types import Timeframe

_MS_PER_YEAR = 365 * 24 * 3_600_000


def bar_returns(equity: pl.Series) -> pl.Series:
    return (equity / equity.shift(1) - 1.0).drop_nulls()


def sharpe(equity: pl.Series, timeframe: Timeframe, risk_free_annual: float = 0.0) -> float:
    r = bar_returns(equity)
    if r.len() < 2:
        return 0.0
    periods_per_year = _MS_PER_YEAR / timeframe.ms
    rf_per_bar = risk_free_annual / periods_per_year
    excess = r - rf_per_bar
    std = excess.std()
    if std is None or std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity: pl.Series) -> float:
    """Max peak-to-trough decline as a NEGATIVE fraction (e.g. -0.23)."""
    peak = equity.cum_max()
    dd = equity / peak - 1.0
    return float(dd.min())
