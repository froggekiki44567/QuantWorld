# QuantWorld / QuantLab

Quantitative research platform for crypto market data and candle backtesting.

## What is included

- `quantlab/core/types.py`: shared `Timeframe`, `Candle`, `Exchange`, and `Instrument` types.
- `quantlab/data/`: Binance ingestion, validation, parquet candle storage, gap tracking, resampling, and CLI commands.
- `quantlab/backtest/`: point-in-time market view, order/fill objects, fee and slippage models, portfolio accounting, engine, and metrics.
- `quantlab/strategies/`: simple strategy contract plus reference strategies.
- `tests/`: unit tests for data and backtest behavior.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run tests

```powershell
pytest -q
```

## Download candles

```powershell
python -m quantlab.data download --symbol BTCUSDT --timeframe 1m --start 2024-01-01
```

By default, data is written to `./data` as monthly parquet partitions.

## Check stored data

```powershell
python -m quantlab.data status --symbol BTCUSDT --timeframe 1m
```

## Query stored candles

```powershell
python -m quantlab.data sql "SELECT symbol, count(*) FROM candles GROUP BY symbol"
```
