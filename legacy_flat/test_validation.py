# quantlab

Quantitative research platform. **Module 1 — data layer** (Phase 1 of 8).

## What's implemented

- `quantlab/core/types.py` — Timeframe, Candle, Instrument (int-ms UTC timestamps everywhere)
- `quantlab/data/adapters/` — ExchangeAdapter contract + Binance REST adapter
  (pagination, weight-aware rate limiting, retry, forming-candle exclusion)
- `quantlab/data/validation.py` — OHLC invariants, duplicates, monotonicity,
  grid alignment, O(n) gap detection. Flag, never fix.
- `quantlab/data/store.py` — Parquet monthly partitions, idempotent atomic
  upsert, resume point, DuckDB SQL layer
- `quantlab/data/gaps.py` — persistent gap registry with coalescing,
  `is_clean(range)` query for backtests
- `quantlab/data/resample.py` — 1m -> higher TFs, incomplete buckets dropped
- `quantlab/data/downloader.py` — resumable backfill orchestrator
- `quantlab/data/__main__.py` — CLI

## Usage

```bash
pip install -e .
python -m quantlab.data download --symbol BTCUSDT --timeframe 1m --start 2024-01-01
python -m quantlab.data status   --symbol BTCUSDT --timeframe 1m
python -m quantlab.data sql "SELECT symbol, count(*) FROM candles GROUP BY symbol"
```

## Tests

```bash
pytest -q   # 44 tests: validation, store, gaps, resample, adapter (mocked), e2e
```

## Module 2 — backtest engine core (implemented)

- `backtest/marketview.py` — point-in-time data API; cursor-sliced, look-ahead structurally impossible
- `backtest/orders.py` — Intent / Order / Fill / Rejection (strategies emit intents, never orders)
- `backtest/costs.py` — FeeModel, FixedBpsSlippage, VolumeImpactSlippage (adversarial, no zero-cost default)
- `backtest/portfolio.py` — multi-instrument avg-cost accounting, realized/unrealized PnL, flips, equity curve
- `backtest/engine.py` — bar loop with strict next-bar-open execution, volume-participation fill caps, SimpleSizer stub (replaced by risk engine in Phase 5)
- `backtest/metrics.py` — Sharpe (365d annualization for 24/7 markets), max drawdown
- `strategies/base.py` — BaseStrategy contract + SmaCross and BuyAndHold reference fixtures

## Next: Module 3 — strategy plugin framework + walk-forward runner
