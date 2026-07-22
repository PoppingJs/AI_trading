# AI_trading

Paper-trading-first Binance USDT-M futures quant toolkit.

This project implements a rule-based composite strategy that combines EMA, MA,
BOLL, volume, open interest, long/short ratio, RSI, funding rate, ATR stops,
position sizing, staged take profit, and hard risk limits.

The default mode is research and paper trading. It does not place real orders.

## Strategy

Strategy changes must stay on one core path: universe selection, direction and
score, structural entry zone, price confirmation and vetoes, entry, structural
stop/targets, adding or rotation, then exit. Prefer correcting these shared
decisions over adding standalone parameters or alternate execution branches;
real-time paper trading and historical replay must continue to use the same
path.

The strategy avoids single-indicator prediction. It scores market state instead:

- Trend filter: `EMA20`, `EMA50`, `MA100`.
- Pullback and fake-breakout filter: `BOLL20, 2` plus close confirmation.
- Volume confirmation: current volume relative to `SMA20(volume)`.
- Derivatives confirmation: open interest change, long/short ratio, funding rate.
- Momentum filter: `RSI14`.
- Stop buffer: structural high/low plus ATR buffer.

Positive scoring evidence is deduplicated by family: EMA/BOLL location,
derivatives, daily/4h direction, and 15m trigger each contribute only their
strongest item. All reasons remain visible, while negative penalties and vetoes
remain independent and are never removed by deduplication.

Signals below 65 are `NO_TRADE`; scores from 65 to 79 remain under
observation, and scores of at least 80 establish a directional entry signal.
Hard vetoes remain visible on the signal and are enforced by automatic entry:

- No long chase when RSI is overheated, price closes above upper BOLL, funding is hot, long side is crowded, or OI spikes.
- No short chase when RSI is oversold, price closes below lower BOLL, funding is very negative, short side is crowded, or OI spikes.
- Choppy EMA/BOLL/RSI conditions are treated as low-quality environments.

Risk is sized by maximum acceptable loss, not by leverage:

```text
risk_amount = equity * risk_per_trade
notional = risk_amount / stop_distance_percent
margin_required = notional / leverage
```

Automatic trading capital allocation:

- Deployable margin: `95%` of current equity, split into five units.
- `A` entry: score `80-99`, at most one unit.
- `S` entry: score `100+`, at most two units; it uses one unit when only one remains.
- Scores below `80` do not open automatically.
- Default leverage: `5x`; maximum leverage: `10x`, further capped by stop distance.
- Maximum open positions: `5`.
- Daily/weekly loss limits, maximum drawdown and consecutive-loss cooldown are disabled by default during paper-strategy testing. Setting their limits above zero restores the corresponding hard entry gates.

The risk-per-trade sizing branch remains available only for legacy/manual
`RiskManager` callers. Production automatic trading and historical replay use
the fixed A/S capital-unit allocation above.

## Project Layout

```text
config/strategy.yaml        Strategy, risk, and execution defaults
src/ai_trading/indicators.py EMA/MA/BOLL/RSI/ATR/VOL calculations
src/ai_trading/strategy.py   Composite scoring and hard vetoes
src/ai_trading/risk.py       Position sizing, stop loss, take profit planning
src/ai_trading/historical.py Current-Top50 production-state-machine historical replay
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

## Run Demo Signal CLI

```bash
ai-trading --demo
```

This runs the strategy on synthetic 15m-style data and prints the latest signal.

## Inspect a CSV Signal

```bash
ai-trading --symbol BTCUSDT --candles-csv data/BTCUSDT-15m.csv --derivatives-csv data/BTCUSDT-derivatives-15m.csv
```

The CLI prints the latest signal. Historical replay is intentionally exposed
only through the dashboard so it always uses the live system defaults.

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
- `POST /api/backtests/jobs`
- `GET /api/backtests/jobs/{job_id}`
- `POST /api/backtests/jobs/{job_id}/cancel`

OpenAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

The paper trading dashboard is available at:

```text
http://127.0.0.1:8000/
```

The same top navigation opens the production-parity historical replay:

```text
http://127.0.0.1:8000/backtest
```

The replay fixes the current eligible Top50 at the first run, downloads Binance
K-lines/OI/long-short ratio/funding history, and drives the same
`PaperTradingEngine` used by real-time paper trading. Only the most recent
successful market dataset is cached; rerunning the same dates reuses market data
but executes the current strategy code again. Historical Top50 rankings are not
available, so reports explicitly disclose current-universe survivorship bias.

It starts with a local 1200 USDT paper account. You can:

- Start live Binance public market refresh.
- Enable strategy-driven paper auto trading.
- Manually open long or short futures positions.
- Close positions.
- Watch equity, available balance, used margin, realized PnL, unrealized PnL, fees, signals, and fills.

The dashboard is still paper trading only. It does not log in to Binance and
does not submit real orders.

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
