# AI_trading

Paper-trading-first Binance USDT-M futures quant toolkit.

This project implements a rule-based composite strategy that combines EMA, MA,
BOLL, volume, open interest, long/short ratio, RSI, funding rate, ATR stops,
position sizing, staged take profit, and hard risk limits.

The default mode is research and paper trading. It does not place real orders.

## Strategy

The strategy avoids single-indicator prediction. It scores market state instead:

- Trend filter: `EMA20`, `EMA50`, `MA100`.
- Pullback and fake-breakout filter: `BOLL20, 2` plus close confirmation.
- Volume confirmation: current volume relative to `SMA20(volume)`.
- Derivatives confirmation: open interest change, long/short ratio, funding rate.
- Momentum filter: `RSI14`.
- Stop buffer: structural high/low plus ATR buffer.

Entry signals must pass the score threshold and hard vetoes:

- No long chase when RSI is overheated, price closes above upper BOLL, funding is hot, long side is crowded, or OI spikes.
- No short chase when RSI is oversold, price closes below lower BOLL, funding is very negative, short side is crowded, or OI spikes.
- Choppy EMA/BOLL/RSI conditions are treated as low-quality environments.

Risk is sized by maximum acceptable loss, not by leverage:

```text
risk_amount = equity * risk_per_trade
notional = risk_amount / stop_distance_percent
margin_required = notional / leverage
```

Default risk settings:

- Default leverage: `5x`.
- Max leverage: `10x`.
- Per-trade risk: `0.5%`.
- Max single-symbol margin: `10%`.
- Max total margin: `35%`.
- Max open positions: `3`.
- Daily loss limit: `2%`.
- Consecutive loss cooldown: `3` losses.

## Project Layout

```text
config/strategy.yaml        Strategy, risk, and execution defaults
src/ai_trading/indicators.py EMA/MA/BOLL/RSI/ATR/VOL calculations
src/ai_trading/strategy.py   Composite scoring and hard vetoes
src/ai_trading/risk.py       Position sizing, stop loss, take profit planning
src/ai_trading/backtest.py   Paper-style backtest engine
src/ai_trading/binance.py    Read-only Binance USDT-M market discovery
src/ai_trading/universe.py   Top20 + data quality filtering
src/ai_trading/api.py        FastAPI service skeleton
tests/                       Unit tests for strategy core
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run Demo CLI

```bash
ai-trading --demo
```

This runs the strategy on synthetic 15m-style data and prints the latest signal
plus a basic backtest summary.

## Run CSV Backtest

```bash
ai-trading --symbol BTCUSDT --candles-csv data/BTCUSDT-15m.csv --derivatives-csv data/BTCUSDT-derivatives-15m.csv --equity 10000
```

The CLI prints final equity, return, max drawdown, win rate, and the latest
closed trades.

## Run API

```bash
ai-trading-api
```

On Windows, from the project root you can also run:

```powershell
.\scripts\start_api.ps1
```

Endpoints:

- `GET /api/health`
- `GET /api/config`
- `GET /api/markets/top20`
- `GET /api/signals/demo`
- `POST /api/backtests/run`

OpenAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Data Format

CSV candle files should contain:

```text
timestamp,open,high,low,close,volume
```

Derivative CSV files should contain:

```text
timestamp,open_interest,long_short_ratio,funding_rate
```

Timestamps may be ISO strings or Unix seconds/milliseconds.

## Safety Notes

This code is not financial advice and cannot guarantee stable profit. Futures
trading can lose capital quickly, including at 5x-10x leverage when notional size
is high. Real trading integration should only be added after:

- Historical data tests pass.
- Paper trading runs without state errors.
- Order, position, and risk reconciliation are implemented.
- API key permissions are restricted.
- Emergency kill-switch and alerts are deployed.

The current Binance module is read-only by design.
