from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_trading.backtest import BacktestEngine
from ai_trading.binance import BinanceFuturesMarketData
from ai_trading.cli import _synthetic_market
from ai_trading.config import AppSettings, load_settings
from ai_trading.indicators import build_indicators
from ai_trading.models import SignalAction
from ai_trading.strategy import CompositeStrategy


class BacktestRequest(BaseModel):
    symbol: str = "DEMOUSDT"
    starting_equity: float = Field(default=10_000.0, gt=0)
    use_demo_data: bool = True


def create_app(settings_path: str | Path = "config/strategy.yaml") -> FastAPI:
    settings = load_settings(settings_path)
    app = FastAPI(
        title="AI Trading Strategy API",
        version="0.1.0",
        summary="Paper-trading-first Binance USDT-M futures strategy service.",
    )
    app.state.settings = settings

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "paper_trading": settings.execution.paper_trading,
            "symbols_mode": settings.symbols_mode,
            "timeframes": settings.timeframes,
        }

    @app.get("/api/config")
    def config() -> dict[str, object]:
        return settings.model_dump()

    @app.get("/api/markets/top20")
    async def top20_markets() -> dict[str, object]:
        client = BinanceFuturesMarketData()
        symbols = await client.top_usdt_perpetuals(limit=20)
        return {
            "rank_by": settings.symbol_rank_by,
            "symbols": [asdict(symbol) for symbol in symbols],
            "note": "Read-only market discovery; this endpoint never places orders.",
        }

    @app.get("/api/signals/demo")
    def demo_signal() -> dict[str, object]:
        candles, derivatives = _synthetic_market()
        indicators = _indicators(settings, candles, derivatives)
        signal = CompositeStrategy(settings.strategy).generate_signal("DEMOUSDT", candles, indicators)
        return _signal_payload(signal)

    @app.post("/api/backtests/run")
    def run_backtest(request: BacktestRequest) -> dict[str, object]:
        if not request.use_demo_data:
            raise HTTPException(
                status_code=400,
                detail="Only demo data is wired in this MVP. Add a data adapter before using live historical data.",
            )
        candles, derivatives = _synthetic_market()
        result = BacktestEngine(
            symbol=request.symbol,
            starting_equity=request.starting_equity,
            strategy_settings=settings.strategy,
            risk_settings=settings.risk,
            execution_settings=settings.execution,
        ).run(candles, derivatives)
        return {
            "symbol": request.symbol,
            "starting_equity": result.starting_equity,
            "ending_equity": result.ending_equity,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "trade_count": len(result.trades),
            "trades": [asdict(trade) for trade in result.trades],
            "notes": result.notes,
        }

    return app


def main() -> None:
    uvicorn.run("ai_trading.api:create_app", factory=True, host="127.0.0.1", port=8000, reload=False)


def _indicators(settings: AppSettings, candles, derivatives):
    return build_indicators(
        candles,
        derivatives,
        ema_fast=settings.strategy.ema_fast,
        ema_slow=settings.strategy.ema_slow,
        ma_trend=settings.strategy.ma_trend,
        bollinger_window=settings.strategy.bollinger_window,
        bollinger_stddev=settings.strategy.bollinger_stddev,
        rsi_window=settings.strategy.rsi_window,
        atr_window=settings.strategy.atr_window,
        volume_window=settings.strategy.volume_window,
    )


def _signal_payload(signal) -> dict[str, object]:
    return {
        "symbol": signal.symbol,
        "timestamp": signal.timestamp,
        "action": signal.action.value if isinstance(signal.action, SignalAction) else signal.action,
        "regime": signal.regime.value,
        "score": signal.score,
        "vetoes": signal.vetoes,
        "reasons": signal.reasons,
        "indicators": asdict(signal.indicators) if signal.indicators else None,
    }


app = create_app()
