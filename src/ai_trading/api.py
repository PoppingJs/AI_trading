from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import os
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ai_trading.binance import BinanceFuturesMarketData
from ai_trading.cli import _synthetic_market
from ai_trading.config import AppSettings, load_settings
from ai_trading.historical import HistoricalReplayManager
from ai_trading.indicators import build_indicators
from ai_trading.models import SignalAction
from ai_trading.paper import PAPER_DEFAULT_BALANCE, PaperTradingEngine
from ai_trading.strategy import CompositeStrategy
from ai_trading.web_pages import backtest_page


class BacktestJobRequest(BaseModel):
    start_date: str = Field(min_length=10, max_length=10)
    end_date: str = Field(min_length=10, max_length=10)


class PaperStartRequest(BaseModel):
    starting_balance: float | None = Field(default=None, gt=0)
    symbols: list[str] | None = None
    interval: str = "1h"
    auto_trade: bool = False
    poll_seconds: int = Field(default=20, ge=5, le=300)
    reset_account: bool = False


class PaperOrderRequest(BaseModel):
    symbol: str = "BTCUSDT"
    side: str = Field(pattern="^(LONG|SHORT)$")
    margin_usdt: float = Field(default=100.0, gt=0)
    leverage: int = Field(default=5, ge=1, le=10)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit_1: float | None = Field(default=None, gt=0)
    take_profit_2: float | None = Field(default=None, gt=0)


class PaperCloseRequest(BaseModel):
    symbol: str = "BTCUSDT"


def create_app(
    settings_path: str | Path = "config/strategy.yaml",
    state_path: str | Path | None = None,
    *,
    historical_cache_root: str | Path = "data/historical_cache",
    historical_market_data_factory: Callable[[], BinanceFuturesMarketData] | None = None,
) -> FastAPI:
    settings = load_settings(settings_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.paper_engine.start(auto_trade=False)
        try:
            yield
        finally:
            await app.state.paper_engine.close()
            app.state.historical_replays.close()

    app = FastAPI(
        title="AI Trading Strategy API",
        version="0.1.0",
        summary="Paper-trading-first Binance USDT-M futures strategy service.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    resolved_state_path = state_path if state_path is not None else os.getenv("AI_TRADING_PAPER_STATE", "data/paper_state.json")
    app.state.paper_engine = PaperTradingEngine(
        settings,
        starting_balance=PAPER_DEFAULT_BALANCE,
        state_path=resolved_state_path,
    )
    app.state.historical_replays = HistoricalReplayManager(
        settings,
        cache_root=historical_cache_root,
        market_data_factory=historical_market_data_factory or BinanceFuturesMarketData,
    )

    @app.get("/", response_class=HTMLResponse)
    def paper_dashboard() -> str:
        return PAPER_DASHBOARD_HTML

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest_dashboard() -> str:
        return backtest_page()

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            **app.state.paper_engine.health_status(),
            "paper_trading": settings.execution.paper_trading,
            "symbols_mode": settings.symbols_mode,
            "timeframes": settings.timeframes,
            "paper_dashboard": "http://127.0.0.1:8000/",
        }

    @app.get("/api/config")
    def config() -> dict[str, object]:
        return settings.model_dump()

    @app.get("/api/markets/top20")
    @app.get("/api/markets/top30")
    async def top30_markets() -> dict[str, object]:
        client = BinanceFuturesMarketData()
        try:
            symbols = await client.top_usdt_perpetuals(limit=30)
        finally:
            await client.aclose()
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

    @app.post("/api/backtests/jobs", dependencies=[Depends(_require_api_token)])
    def create_backtest_job(request: BacktestJobRequest) -> dict[str, object]:
        try:
            job = app.state.historical_replays.submit(request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return job.payload(include_result=False)

    @app.get("/api/backtests/jobs/{job_id}", dependencies=[Depends(_require_api_token)])
    def get_backtest_job(job_id: str, include_result: bool = True) -> dict[str, object]:
        job = app.state.historical_replays.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="backtest job not found")
        return job.payload(include_result=include_result)

    @app.post("/api/backtests/jobs/{job_id}/cancel", dependencies=[Depends(_require_api_token)])
    def cancel_backtest_job(job_id: str) -> dict[str, object]:
        job = app.state.historical_replays.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="backtest job not found")
        return job.payload(include_result=False)

    @app.get("/api/paper/status")
    async def paper_status() -> dict[str, object]:
        return await app.state.paper_engine.status_async()

    @app.post("/api/paper/start", dependencies=[Depends(_require_api_token)])
    async def paper_start(request: PaperStartRequest) -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        requested_symbols = [symbol.upper() for symbol in request.symbols or []]
        if not requested_symbols or _uses_default_symbol_pool(requested_symbols):
            engine.configure_symbols(["AUTO_TOP50"])
        else:
            engine.configure_symbols(requested_symbols)
        engine.configure_interval(request.interval)
        if request.reset_account:
            await engine.reset(starting_balance=request.starting_balance or PAPER_DEFAULT_BALANCE)
        elif request.starting_balance is not None and not engine.account.fills and not engine.account.positions:
            await engine.reset(starting_balance=request.starting_balance)
        await engine.start(auto_trade=request.auto_trade, poll_seconds=request.poll_seconds)
        return await engine.status_async()

    @app.post("/api/paper/stop", dependencies=[Depends(_require_api_token)])
    async def paper_stop() -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        await engine.stop()
        return await engine.status_async()

    @app.post("/api/paper/reset", dependencies=[Depends(_require_api_token)])
    async def paper_reset(starting_balance: float = PAPER_DEFAULT_BALANCE) -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        await engine.reset(starting_balance=starting_balance)
        return await engine.status_async()

    @app.post("/api/paper/refresh", dependencies=[Depends(_require_api_token)])
    async def paper_refresh() -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        try:
            await engine.refresh_once()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"币安行情刷新失败：{exc}") from exc
        return await engine.status_async()

    @app.post("/api/paper/order/open", dependencies=[Depends(_require_api_token)])
    async def paper_open_order(request: PaperOrderRequest) -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        try:
            await engine.open_position_with_risk(
                request.symbol,
                request.side,  # type: ignore[arg-type]
                margin_usdt=request.margin_usdt,
                leverage=request.leverage,
                stop_loss=request.stop_loss,
                take_profit_1=request.take_profit_1,
                take_profit_2=request.take_profit_2,
                reason="manual dashboard",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"币安价格获取失败：{exc}") from exc
        return await engine.status_async()

    @app.post("/api/paper/order/close", dependencies=[Depends(_require_api_token)])
    async def paper_close_order(request: PaperCloseRequest) -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        try:
            await engine.close_position(request.symbol)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"币安价格获取失败：{exc}") from exc
        return await engine.status_async()

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


def _uses_default_symbol_pool(symbols: list[str]) -> bool:
    cleaned = {symbol.replace("/", "").replace("-", "").upper() for symbol in symbols if symbol}
    return (
        not cleaned
        or cleaned == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        or cleaned in ({"AUTO_TOP50"}, {"AUTO_TOP30"})
    )


def _require_api_token(x_api_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("AI_TRADING_API_TOKEN", "").strip()
    if expected and x_api_token != expected:
        raise HTTPException(status_code=401, detail="missing or invalid API token")


PAPER_DASHBOARD_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI 量化交易平台</title>
  <style>
    :root { color-scheme: light; font-family: "Microsoft YaHei", Arial, sans-serif; }
    html, body { width: 100%; height: 100%; max-width: 100%; overflow: hidden; }
    body { margin: 0; background: #f6f7f9; color: #171717; }
    header { height: 60px; box-sizing: border-box; padding: 12px 24px; background: #111827; color: white; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    header h1 { margin: 0; font-size: 20px; }
    header p { margin: 4px 0 0; color: #cbd5e1; font-size: 13px; }
    .top-nav { height: 38px; box-sizing: border-box; padding: 0 18px; display: flex; align-items: stretch; gap: 4px; background: #fff; border-bottom: 1px solid #e5e7eb; overflow-x: auto; scrollbar-width: none; }
    .top-nav::-webkit-scrollbar { display: none; }
    .top-nav a { display: flex; align-items: center; padding: 0 13px; color: #64748b; text-decoration: none; font-size: 13px; font-weight: 700; white-space: nowrap; border-bottom: 2px solid transparent; }
    .top-nav a:hover { color: #1d4ed8; background: #f8fafc; }
    .top-nav a.active { color: #1d4ed8; border-bottom-color: #2563eb; }
    main { height: calc(100vh - 98px); padding: 6px 16px; width: 100%; box-sizing: border-box; margin: 0; overflow: hidden; display: flex; flex-direction: column; gap: 6px; }
    .grid { display: grid; gap: 10px; }
    .metrics { grid-template-columns: repeat(7, minmax(0, 1fr)); flex: 0 0 auto; }
    .layout { grid-template-columns: 460px minmax(0, 1fr); align-items: stretch; flex: 1 1 auto; min-height: 0; margin-top: 0 !important; }
    .layout > .card { min-height: 0; overflow: hidden; display: flex; flex-direction: column; }
    .layout > .grid { display: grid; grid-template-rows: 230px 405px minmax(210px, 1fr); min-height: 0; }
    .layout > .grid > .card { min-height: 0; overflow: hidden; }
    .layout > .grid > .card:nth-child(1) { display: flex; flex-direction: column; overflow: hidden; }
    .layout > .grid > .card:nth-child(1) .scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; scrollbar-gutter: stable; }
    .layout > .grid > .card:nth-child(1) thead th { position: sticky; top: 0; z-index: 1; }
    .layout > .grid > .card:nth-child(2) { display: flex; flex-direction: column; overflow: hidden; }
    .layout > .grid > .card:nth-child(2) .scroll { flex: 1 1 auto; min-height: 0; max-height: none; overflow-y: auto; }
    .layout > .grid > .card:nth-child(2) thead th { position: sticky; top: 0; z-index: 1; }
    .layout > .grid > .card:last-child { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
    .card { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .metric { min-width: 0; padding: 2px 10px; }
    .metric span { color: #6b7280; font-size: 12px; line-height: 1; }
    .metric strong { display: block; margin-top: 1px; font-size: 16px; line-height: 1.1; white-space: nowrap; }
    .positive { color: #047857; }
    .negative { color: #b91c1c; }
    label { display: block; font-size: 12px; color: #4b5563; margin: 10px 0 5px; }
    input, select { width: 100%; box-sizing: border-box; padding: 9px 10px; border: 1px solid #d1d5db; border-radius: 6px; background: white; }
    button { border: 0; border-radius: 6px; padding: 10px 12px; cursor: pointer; font-weight: 600; }
    button.primary { background: #2563eb; color: white; }
    button.long { background: #059669; color: white; }
    button.short { background: #dc2626; color: white; }
    button.neutral { background: #e5e7eb; color: #111827; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    table { width: max-content; max-width: 100%; border-collapse: collapse; font-size: 13px; table-layout: auto; }
    th, td { text-align: center; padding: 8px 14px; border-bottom: 1px solid #e5e7eb; vertical-align: middle; overflow-wrap: normal; }
    th:not(.reason-col), td:not(.reason-col) { white-space: nowrap; }
    th { color: #6b7280; font-weight: 600; background: #f9fafb; }
    .scroll { overflow-x: hidden; max-width: 100%; }
    .fills-scroll { flex: 1 1 auto; min-height: 0; overflow: hidden; padding-right: 0; }
    .fills-scroll thead th { position: sticky; top: 0; z-index: 1; }
    #positions, #signals, #fills { font-size: 11px; line-height: 1.05; }
    #positions th, #positions td, #signals th, #signals td, #fills th, #fills td { height: 16px; padding: 0 5px; line-height: 1.05; }
    #positions tbody tr, #signals tbody tr, #fills tbody tr { height: 16px; }
    #signals th, #signals td { height: 22px; padding: 2px 4px; line-height: 1.2; }
    #signals tbody tr { height: 22px; }
    #fills { width: 100%; max-width: 100%; table-layout: fixed; }
    #fills th, #fills td { min-width: 0; max-width: none; padding: 0 1px; overflow: hidden; text-overflow: clip; }
    #fills .time-col { font-size: 11px; }
    #fills .side-col { padding-left: 0; padding-right: 0; }
    #fills .reason-col { text-align: center; line-height: 1.2; overflow-wrap: anywhere; }
    #fills .empty-fill-row td { color: #111827; }
    #fills .empty-fill-row:not(:first-child) td { color: transparent; }
    .pager { display: flex; justify-content: center; align-items: center; gap: 6px; padding-top: 6px; flex: 0 0 auto; }
    .pager button { min-width: 34px; padding: 6px 9px; background: #e5e7eb; color: #111827; }
    .pager button.active { background: #2563eb; color: white; }
    .pager button:disabled { opacity: .45; cursor: not-allowed; }
    .pager .ellipsis { min-width: 18px; color: #6b7280; text-align: center; }
    .reason-col { white-space: normal; line-height: 1.45; }
    .time-col { min-width: 132px; }
    .num-col { text-align: center; min-width: 72px; }
    .symbol-col { min-width: 86px; }
    .side-col, .action-col { min-width: 56px; }
    .signals-table { width: 100%; table-layout: fixed; }
    .signals-table th, .signals-table td { vertical-align: middle; }
    .signals-table th:nth-child(-n+7), .signals-table td:nth-child(-n+7) { width: 78px; min-width: 0; max-width: 78px; text-align: center; }
    .signals-table .symbol-col, .signals-table .side-col, .signals-table .num-col { min-width: 0; }
    .signals-table .timing-col { white-space: normal; line-height: 1.15; padding-left: 2px; padding-right: 2px; }
    .signals-table .reason-col { width: auto; min-width: 0; max-width: none; text-align: left; }
    .signals-table .veto-col { width: 190px; text-align: center; white-space: normal; line-height: 1.2; }
    .signals-table th:nth-child(9), .signals-table td:nth-child(9) { width: 190px; min-width: 0; max-width: 190px; }
    .signals-table th.reason-col { text-align: center; }
    .signals-table th.veto-col { text-align: center; }
    .signals-table th:not(.reason-col), .signals-table td:not(.reason-col):not(.veto-col) { text-align: center; }
    .center-table th, .center-table td { text-align: center; vertical-align: middle; padding: 1px 6px; }
    .center-table .symbol-col { min-width: 78px; text-align: center; }
    .center-table .side-col { min-width: 46px; text-align: center; }
    .center-table .num-col { min-width: 64px; text-align: center; }
    .center-table .time-col { min-width: 120px; text-align: center; }
    .center-table th:last-child:not(.reason-col), .center-table td:last-child:not(.reason-col) { min-width: 62px; }
    .center-table .reason-col { min-width: 180px; max-width: 360px; text-align: left; white-space: normal; line-height: 1.2; }
    #positions { width: 100%; table-layout: auto; }
    #positions .reason-col { min-width: 560px; max-width: none; width: 100%; }
    #fills .symbol-col, #fills .side-col, #fills .num-col, #fills .time-col { min-width: 0; max-width: none; }
    #fills .reason-col { min-width: 0; max-width: none; text-align: center; white-space: normal; }
    #fills .entry-position-col { min-width: 0; max-width: none; text-align: left; white-space: normal; line-height: 1.2; overflow-wrap: anywhere; }
    #fills th.entry-position-col { text-align: center; }
    .center-table th.reason-col { text-align: center; }
    .status { font-size: 12px; color: #6b7280; }
    .pill { display: inline-block; padding: 3px 7px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; }
    .error { display: none; }
    .error-ticker {
      position: fixed;
      top: 98px;
      left: 0;
      right: 0;
      z-index: 50;
      display: none;
      height: 26px;
      overflow: hidden;
      pointer-events: none;
      background: rgba(254, 242, 242, 0.96);
      border-top: 1px solid #fecaca;
      border-bottom: 1px solid #fecaca;
      color: #b91c1c;
      font-size: 13px;
      line-height: 26px;
    }
    .error-ticker.show { display: block; }
    .error-ticker-text {
      display: inline-block;
      min-width: 100%;
      padding-left: 100%;
      white-space: nowrap;
      animation: error-marquee 15s linear infinite;
    }
    @keyframes error-marquee {
      from { transform: translateX(0); }
      to { transform: translateX(-100%); }
    }
    .left-main { flex: 0 0 auto; }
    .left-main label { margin: 6px 0 3px; }
    .left-main input, .left-main select { padding: 7px 10px; }
    .left-spacer { display: none; }
    .control-actions { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
    .control-actions button { height: 44px; padding: 0 8px; white-space: nowrap; }
    .daily-pnl {
      flex: 0 0 auto;
      padding-top: 0;
      margin-top: 15px;
      border-top: 0;
    }
    .daily-head, .daily-summary { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
    .daily-head h2 { font-size: 16px; margin: 0; }
    .daily-month { display: flex; justify-content: center; align-items: center; gap: 10px; margin: 6px 0 6px; color: #374151; font-weight: 700; }
    .month-btn { border: 0; background: transparent; color: #6b7280; padding: 2px 7px; font-size: 17px; line-height: 1; }
    .month-btn:disabled { color: #d1d5db; cursor: not-allowed; }
    .daily-week, .daily-calendar { display: grid; grid-template-columns: repeat(7, 1fr); gap: 5px; }
    .daily-week span { text-align: center; color: #6b7280; font-size: 11px; }
    .daily-day { min-height: 42px; border-radius: 7px; background: #f3f4f6; padding: 5px 4px; box-sizing: border-box; text-align: center; font-size: 12px; }
    .daily-day.empty { background: transparent; }
    .daily-day strong { display: block; font-size: 13px; line-height: 1.1; }
    .daily-day span { display: block; margin-top: 5px; font-size: 11px; color: #6b7280; }
    .daily-day.profit-day { background: #dcfce7; }
    .daily-day.loss-day { background: #fee2e2; }
    .daily-day.profit-day span { color: #059669; font-weight: 700; }
    .daily-day.loss-day span { color: #dc2626; font-weight: 700; }
    .chart-wrap {
      flex: 1 1 auto;
      min-height: 340px;
      margin-top: 15px;
      box-sizing: border-box;
      padding: 0;
      display: flex;
      flex-direction: column;
      background: transparent;
      border: 0;
      border-radius: 0;
      box-shadow: none;
    }
    .chart-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin: 0 0 8px; }
    .chart-title-row { display: flex; align-items: baseline; gap: 16px; min-width: 0; }
    .chart-head h2 { font-size: 16px; margin: 0; }
    .chart-head strong { font-size: 16px; font-weight: 800; white-space: nowrap; }
    .chart-head span { font-size: 12px; color: #6b7280; }
    #pnlChart { width: 100%; height: auto; min-height: 0; flex: 1 1 auto; display: block; background: transparent; border: 0; border-radius: 0; }
    @media (max-width: 980px) {
      .metrics, .layout { grid-template-columns: 1fr; }
      header { display: block; }
      .chart-wrap { position: static; width: 100%; margin-top: 14px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AI 量化交易平台</h1>
      <p>Binance USDT-M 实时行情，本地模拟账户，不会产生真实订单。</p>
    </div>
    <div class="status" id="updated">loading...</div>
  </header>
  <nav class="top-nav" aria-label="主要功能">
    <a class="active" href="/">实时交易</a>
    <a href="/backtest">历史回测</a>
  </nav>
  <div class="error-ticker" id="errorTicker" aria-live="polite">
    <span class="error-ticker-text" id="errorTickerText"></span>
  </div>
  <main>
    <section class="grid metrics" id="metrics"></section>
    <section class="grid layout" style="margin-top:14px;">
      <div class="card">
        <div class="left-main">
        <h2 style="font-size:16px;margin:0 0 10px;">控制台</h2>
        <label>模拟本金 USDT</label>
        <input id="startingBalance" type="number" value="1200" min="1" step="10" />
        <label>币种标的池</label>
        <input id="symbols" value="AUTO_TOP50" />
        <label>周期</label>
        <select id="interval"><option>15m</option><option selected>1h</option><option>4h</option><option>1d</option></select>
        <div class="control-actions">
          <button class="primary" onclick="startPaper(true)">启动策略</button>
          <button class="neutral" onclick="stopPaper()">关闭策略</button>
          <button class="neutral" onclick="resetPaper()">重置1200U</button>
        </div>
        <p class="error" id="error"></p>
        </div>
        <div class="daily-pnl" id="dailyPnl">
          <div class="daily-head">
            <h2>每日盈亏</h2>
            <span class="status">08:00 - 次日08:00</span>
          </div>
          <div class="daily-month">
            <button class="month-btn" id="dailyPrevMonth" onclick="shiftDailyMonth(-1)">‹</button>
            <span id="dailyMonth">--</span>
            <button class="month-btn" id="dailyNextMonth" onclick="shiftDailyMonth(1)">›</button>
          </div>
          <div class="daily-week"><span>日</span><span>一</span><span>二</span><span>三</span><span>四</span><span>五</span><span>六</span></div>
          <div class="daily-calendar" id="dailyCalendar"></div>
        </div>
        <div class="left-spacer"></div>
        <div class="chart-wrap">
          <div class="chart-head">
            <div class="chart-title-row">
              <h2>今日总收益</h2>
              <strong id="pnlHeaderValue">0.00U</strong>
            </div>
            <span>每15分钟采样，不含本金</span>
          </div>
          <canvas id="pnlChart" width="680" height="420"></canvas>
        </div>
      </div>
      <div class="grid">
        <div class="card">
          <h2 style="font-size:16px;margin:0 0 10px;">持仓</h2>
          <div class="scroll"><table id="positions"></table></div>
        </div>
        <div class="card">
          <h2 style="font-size:16px;margin:0 0 10px;">策略信号</h2>
          <div class="scroll"><table id="signals"></table></div>
        </div>
        <div class="card">
          <h2 style="font-size:16px;margin:0 0 10px;">成交记录</h2>
          <div class="scroll fills-scroll"><table id="fills"></table></div>
          <div class="pager" id="fillsPager"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    let pnlSamples = [];
    let latestTotalPnl = 0;
    let fillsPage = 1;
    let dailyMonthKey = null;
    let latestDailyPnl = null;
    async function api(path, options = {}) {
      const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
      const apiToken = localStorage.getItem('AI_TRADING_API_TOKEN');
      if (apiToken) headers['X-API-Token'] = apiToken;
      const response = await fetch(path, { ...options, headers });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || response.statusText);
      }
      return response.json();
    }
    function money(v) { return Number(v || 0).toFixed(2); }
    function signedMoney(v) {
      const n = Number(v || 0);
      if (!Number.isFinite(n)) return '0.00';
      if (Math.abs(n) < 0.005) return '0.00';
      return `${n > 0 ? '+' : ''}${n.toFixed(2)}`;
    }
    function wholeUsdt(v) {
      const n = Number(v || 0);
      if (!Number.isFinite(n)) return '0';
      return String(Math.round(n));
    }
    function priceText(v, significant = 4) {
      if (v === null || v === undefined || v === '') return '--';
      const n = Number(v);
      if (!Number.isFinite(n)) return String(v);
      if (n === 0) return '0';
      const abs = Math.abs(n);
      if (abs >= 1) {
        const integerDigits = Math.floor(abs).toString().length;
        const decimals = Math.max(0, significant - integerDigits);
        return n.toFixed(decimals);
      }
      const decimals = Math.min(10, Math.max(0, Math.ceil(-Math.log10(abs)) + significant - 1));
      return n.toFixed(decimals);
    }
    function pct(v) { return (Number(v || 0) * 100).toFixed(2) + '%'; }
    function pnlClass(v) { return Number(v || 0) >= 0 ? 'positive' : 'negative'; }
    function timeText(v) {
      if (!v) return '--';
      const date = new Date(v);
      if (Number.isNaN(date.getTime())) return String(v);
      const parts = new Intl.DateTimeFormat('zh-CN', {
        timeZone: 'Asia/Shanghai',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      }).formatToParts(date).reduce((acc, part) => {
        acc[part.type] = part.value;
        return acc;
      }, {});
      return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`;
    }
    const actionText = {
      ENTRY_LONG: '开多',
      ENTRY_SHORT: '开空',
      EXIT_LONG: '平多',
      EXIT_SHORT: '平空',
      WATCH: '观察',
      NO_TRADE: '不交易',
      OPEN: '开仓',
      CLOSE: '平仓',
      LONG: '多',
      SHORT: '空'
    };
    const regimeText = {
      TREND_LONG: '多头趋势',
      TREND_SHORT: '空头趋势',
      CHOP: '震荡',
      OVERCROWDED: '拥挤',
      INSUFFICIENT_DATA: '数据不足'
    };
    const smartMoneyPhaseText = {
      NEUTRAL: '中性',
      ACCUMULATION_REBUILD: '吸筹重建',
      SHORT_SQUEEZE_MARKUP: '逼空拉升',
      DISTRIBUTION_EXIT: '派发离场',
      TRAPPED_LONGS_MARKDOWN: '套多阴跌',
      CAPITULATION_ABSORB: '杀跌吸筹'
    };
    const trendStateText = {
      CHOP: '震荡',
      TREND_LONG: '多头趋势',
      TREND_SHORT: '空头趋势',
      ONE_WAY_UP: '单边上涨',
      ONE_WAY_DOWN: '单边下跌'
    };
    const riskStateText = {
      NORMAL: '正常',
      LONG_CROWD: '多头拥挤',
      SHORT_CROWD: '空头拥挤',
      OI_ABNORMAL: 'OI异常',
      FUNDING_HOT: '资金费率过热'
    };
    const reasonText = {
      'score below trading threshold': '评分低于交易阈值',
      'both sides failed hard filters': '多空双方都未通过硬性过滤',
      'not enough candles for MA trend filter': 'K线数量不足，无法计算 MA 趋势过滤',
      'latest candle lacks required indicators': '最新K线缺少必要指标',
      'EMA20 above EMA50 and EMA50 rising': 'EMA20 位于 EMA50 上方，且 EMA50 上行',
      'EMA20 below EMA50 and EMA50 falling': 'EMA20 位于 EMA50 下方，且 EMA50 下行',
      'EMA20 above EMA50 above EMA200': 'EMA20 > EMA50 > EMA200，三线多头排列',
      'EMA20 below EMA50 below EMA200': 'EMA20 < EMA50 < EMA200，三线空头排列',
      'price above MA100 trend filter': '价格位于 MA100 上方，符合多头趋势过滤',
      'price below MA100 trend filter': '价格位于 MA100 下方，符合空头趋势过滤',
      'close confirmed near EMA20/BOLL mid without chasing upper band': '收盘确认靠近 EMA20 或布林中轨，未追高上轨',
      'close confirmed failed retest near EMA20/BOLL mid': '收盘确认反抽 EMA20 或布林中轨失败',
      'BOLL mid confirmed for long continuation': '连续收盘站上 BOLL 中轨，多头延续确认',
      'BOLL mid confirmed for short continuation': '连续收盘跌破 BOLL 中轨，空头延续确认',
      'volume confirms move without extreme blow-off': '成交量确认走势，且未出现极端放量冲高',
      'volume confirms sell pressure without capitulation chase': '成交量确认卖压，且未追空恐慌下跌',
      'volume is acceptable but not strong': '成交量尚可，但强度不足',
      '4h open interest confirms new money entering': '4小时 OI 增加达到阈值，新资金入场确认',
      'open interest rising mildly with price': '价格配合 OI 温和上升',
      'open interest rising mildly with falling price': '价格下跌且 OI 温和上升',
      'open interest stable': 'OI 基本稳定',
      'long/short ratio is not overcrowded long': '多空比未出现多头过度拥挤',
      'long/short ratio is not overcrowded short': '多空比未出现空头过度拥挤',
      'top trader long/short ratio supports longs': '大户多空比支持做多',
      'top trader long/short ratio supports shorts': '大户多空比支持做空',
      'RSI in healthy long-trend range': 'RSI 处于健康多头区间',
      'RSI in healthy short-trend range': 'RSI 处于健康空头区间',
      'funding rate is not overheated for longs': '资金费率未对多头过热',
      'funding rate is not overheated for shorts': '资金费率未对空头过热',
      'funding rate is in long entry range': '资金费率处于多单入场区间',
      'funding rate is in short entry range': '资金费率处于空单入场区间',
      'RSI overheated for long entry': 'RSI 过热，禁止追多',
      'RSI oversold for short entry': 'RSI 超卖，禁止追空',
      'long side overcrowded': '多头过度拥挤',
      'short side overcrowded': '空头过度拥挤',
      'funding too hot for long entry': '资金费率过高，禁止追多',
      'funding too negative for short entry': '资金费率过低，禁止追空',
      'open interest spike risks liquidation sweep': 'OI 异常暴增，存在扫损/爆仓风险',
      'price closed above upper BOLL; no chase': '价格收在布林上轨外，禁止追高',
      'price closed below lower BOLL; no chase': '价格收在布林下轨外，禁止追空',
      'smart money accumulation: OI flushed into a 4h pocket, then rebuilt while price recovered': '主力吸筹：4H OI骤减到冰点后重新回升，确认形成OI洼地且价格修复',
      'smart money accumulation after OI flush; avoid chasing shorts': 'OI 爆减后疑似吸筹，避免追空',
      'short squeeze markup: price and OI rise while long/short ratio falls, shorts are being trapped': '逼空拉升：价格和 OI 同升，多空比下降，空头开始被套',
      'short crowd is vulnerable to a squeeze': '空头拥挤，存在被继续拉升爆空风险',
      'smart money distribution: repeated upper wicks with OI falling after markup': '主力派发：上涨后多次上插针，同时 OI 回落',
      'smart money distribution after upper wick sweeps': '上插针扫单后 OI 回落，疑似主力离场',
      'trapped longs markdown: price falls while OI and long/short ratio rise': '套多阴跌：价格下跌，但 OI 和多空比继续上升',
      'trapped longs are increasing while price falls': '价格下跌时多头继续拥挤，避免做多',
      'capitulation absorption: lower wick sweeps with OI flush and volume expansion': '杀跌吸筹：多次下插针，OI 爆减且成交量放大',
      'capitulation OI flush; avoid late shorts': '下跌尾段 OI 爆减，避免追空',
      'strict long blocked: EMA20/EMA50/EMA200 not bullish': '严格多单禁止：EMA20/EMA50/EMA200 未多头排列',
      'strict long blocked: BOLL mid not confirmed twice': '严格多单禁止：未连续站上 BOLL 中轨',
      'strict long blocked: RSI not in 52-72': '严格多单禁止：RSI 不在 52-72',
      'strict long blocked: volume below 1.5x average': '严格多单禁止：成交量低于均量 1.5 倍',
      'strict long blocked: 4h OI increase below 3%': '严格多单禁止：4小时 OI 增幅不足 3%',
      'strict long blocked: top long/short ratio below 1.1': '严格多单禁止：大户多空比低于 1.1',
      'strict long blocked: funding outside long range': '严格多单禁止：资金费率不在多单区间',
      'strict long blocked: EMA lines are too compressed': '严格多单禁止：EMA 三线粘合，趋势不明',
      'strict long blocked: RSI neutral zone': '严格多单禁止：RSI 位于中性区',
      'strict long blocked: 1h candle amplitude above 5%': '严格多单禁止：1小时振幅超过 5%',
      'strict short blocked: EMA20/EMA50/EMA200 not bearish': '严格空单禁止：EMA20/EMA50/EMA200 未空头排列',
      'strict short blocked: BOLL mid not confirmed twice': '严格空单禁止：未连续跌破 BOLL 中轨',
      'strict short blocked: RSI not in 28-48': '严格空单禁止：RSI 不在 28-48',
      'strict short blocked: volume below 1.5x average': '严格空单禁止：成交量低于均量 1.5 倍',
      'strict short blocked: 4h OI increase below 3%': '严格空单禁止：4小时 OI 增幅不足 3%',
      'strict short blocked: top long/short ratio above 0.9': '严格空单禁止：大户多空比高于 0.9',
      'strict short blocked: funding outside short range': '严格空单禁止：资金费率不在空单区间',
      'strict short blocked: EMA lines are too compressed': '严格空单禁止：EMA 三线粘合，趋势不明',
      'strict short blocked: RSI neutral zone': '严格空单禁止：RSI 位于中性区',
      'strict short blocked: 1h candle amplitude above 5%': '严格空单禁止：1小时振幅超过 5%',
      'market structure confirms long: breakout or retest held': '市场结构确认做多：突破或回踩压力位不破',
      'market structure confirms short: breakdown or retest failed': '市场结构确认做空：跌破或反抽支撑位失败',
      'market structure: resistance grind broke upward, shorts may be squeezed': '市场结构：前高压力位磨盘后向上突破，可能逼空',
      'market structure: support grind broke downward, longs may be liquidated': '市场结构：前低支撑位磨盘后向下跌破，可能爆多',
      'MA cluster breakout up': '均线密集区向上突破',
      'MA cluster retest held near MA20': '突破均线密集区后回踩MA20不破',
      'MA cluster dense; wait for breakout or MA20 retest': '均线密集缠绕，等待突破或回踩MA20确认',
      'MA cluster breakdown down': '均线密集区向下跌破',
      'MA cluster retest rejected near MA20': '跌破均线密集区后反抽MA20失败',
      'MA cluster dense; wait for breakdown or MA20 retest': '均线密集缠绕，等待跌破或反抽MA20确认',
      'washout confirmed: downside wick swept support, OI dropped, close reclaimed key level': '洗盘确认：下插针扫破支撑，OI下降，收盘收回关键位',
      'washout confirmed: upside wick swept resistance, OI dropped, close rejected key level': '洗盘确认：上插针扫过压力，OI下降，收盘跌回关键位',
      'downside sweep reclaimed support; stop-run filter favors long': '下插针扫损后收回支撑，偏向做多',
      'upside sweep rejected resistance; stop-run filter favors short': '上插针扫损后跌回压力，偏向做空',
      'upper wick sweep rejected; avoid chasing long': '上插针回落，等待回踩确认',
      'upper wick rejected; wait for pullback or reclaim before adding long': '上插针回落，等待回踩或重新收回',
      'lower wick sweep reclaimed; avoid chasing short': '下插针收回，禁止追空',
      'extreme volatility: skip new long entry': '极端波动，禁止新开多单',
      'extreme volatility: skip new short entry': '极端波动，禁止新开空单',
      '1h trigger opposes long entry': '1小时触发方向反对做多，禁止开多',
      '1h trigger opposes short entry': '1小时触发方向反对做空，禁止开空',
      '1d bearish bias; long position size reduced': '日线偏空，做多仓位减半',
      '1d bullish bias; short position size reduced': '日线偏多，做空仓位减半',
      '1d bullish bias supports long': '日线偏多，支持做多',
      '1d bearish bias supports short': '日线偏空，支持做空',
      '4h structure supports upside': '4小时结构支持上行',
      '4h structure supports downside': '4小时结构支持下行',
      'stage 1 high distribution: only a tiny short at a confirmed prior-high rejection': '第一阶段高位派发：仅在前高明确受阻时轻仓试空',
      'stage 2 descending distribution: projected lower-high area requires a real rejection': '第二阶段阶梯派发：下降顶点区域必须出现真实反抽拒绝',
      '4h direction remains bearish, but price is in the lower half; wait for a bounce': '4小时方向仍偏空，但价格位于箱体下半区，等待反抽',
      'stage 3 markdown acceleration: 1h/4h EMA/BOLL bounce rejection is eligible': '第三阶段加速主跌：允许在1H/4H EMA或BOLL反抽受阻时做空',
      '1h EMA20/EMA60 repeatedly breached; promote entry, stop, and target to 4h': '1H均线结构多次被刺穿，入场、止损和止盈统一升级到4H',
      'high distribution handoff complete: OI rebuilt while price failed and long/short ratio rose': '高位换手完成：OI已回补但价格未恢复，多空比继续上升',
      '4h OI sharp drop is an event, not a confirmed OI valley; wait for OI rebuilding and downside-wick reclaim': '4H OI骤减只是事件，尚未形成洼地；等待OI从冰点回升并出现下插针收回',
      '4h OI valley formed after retail capitulation; downside wick reclaimed support': '4H散户集中割肉后OI从冰点回升形成洼地，下插针已收回支撑',
      '4h OI valley formed; wait for a downside-wick support reclaim before long': '4H OI洼地已经形成，等待下插针收回支撑后再做多',
      '4h OI-valley absorption overrides the lagging lower-timeframe short bias': '4H OI洼地吸筹确认，覆盖滞后的低周期空头方向',
      '1h direction still lags the confirmed 4h reversal; retain A-size only': '1H方向仍滞后于已确认的4H反转，仅按A级仓位',
      '4h bottom absorbed repeated downside wicks; OI rebuilt with a stable long/short ratio and EMA55 was reclaimed': '4H底部多次下插针均被吸收，OI回升且多空比稳定，价格已收回EMA55',
      'absolute long/short ratio remains crowded; reversal quality capped at A': '多空比绝对值仍拥挤，反转机会最高按A级',
      '4h OI sharp drop is not an OI valley and does not confirm a short entry': '4H OI骤减不等于OI洼地，也不用于确认做空',
      '4h OI deleverage with price breakdown; avoid long entry': '4小时 OI 大幅去杠杆且价格破位，禁止做多',
      '4h OI deleveraged but 1h BOLL/EMA held; allow small long only': '4小时 OI 大幅去杠杆但1小时中轨/EMA守住，只允许小仓多',
      '4h OI deleveraged while long/short ratio rose; 1h support held, only tiny long allowed': '4小时 OI 大跌且多空比上升，1小时支撑守住，仅允许极小仓多',
      '4h OI rebounds after deleverage and price breaks out; strong long restored': '4小时 OI 去杠杆后重新回升且价格突破，恢复强多',
      '4h OI deleverage with price breakdown; short candidate improved': '4小时 OI 大幅去杠杆且价格破位，做空候选增强',
      '4h OI deleverage breakdown with failed bounce; short candidate improved': '4小时 OI 大幅去杠杆后价格破位，且反抽压力失败，做空候选增强',
      '4h OI deleverage breakdown; wait for resistance retest or upper-wick rejection before short': '4小时 OI 大幅去杠杆后价格破位，但不追空，等待阻力反抽或上插针失败',
      '4h OI deleveraged but 1h support held; avoid chasing short': '4小时 OI 大幅去杠杆但1小时支撑守住，避免追空',
      '1h breakout confirms long trigger': '1小时突破确认多头触发',
      '1h retest confirms long trigger': '1小时回踩确认多头触发',
      '1h fake_breakdown confirms long trigger': '1小时假跌破确认多头触发',
      '1h breakdown confirms short trigger': '1小时跌破确认空头触发',
      '1h retest confirms short trigger': '1小时反抽确认空头触发',
      '1h fake_breakout confirms short trigger': '1小时假突破确认空头触发',
      '1h BOLL/EMA pullback held with clean risk': '1小时 BOLL/EMA 回踩不破，风险干净，支持做多',
      '1h BOLL/EMA pullback rejected with clean risk': '1小时 BOLL/EMA 反抽失败，风险干净，支持做空',
      'high pullback with OI/funding/crowd risk; avoid long entry': '高位回踩叠加 OI/资金费率/拥挤风险，禁止开多',
      'low pullback with OI/funding/crowd risk; avoid short entry': '低位反抽叠加 OI/资金费率/拥挤风险，禁止开空',
      'high area without pullback confirmation; wait for 1h/4h pullback before long': '高位未完成回踩确认，等待1小时/4小时回调后再做多',
      'low area without bounce confirmation; wait for 1h/4h retest before short': '低位未完成反抽确认，等待1小时/4小时反抽后再做空',
      'one-way uptrend 15m BOLL/EMA9 pullback confirmed; allow tactical long': '单边上涨中15分钟BOLL中轨/EMA9回踩确认，允许战术做多',
      'one-way downtrend 15m BOLL/EMA9 bounce rejected; allow tactical short': '单边下跌中15分钟BOLL中轨/EMA9反抽失败，允许战术做空',
      'multi-timeframe context neutral': '多周期结构中性',
      'manual dashboard': '手动面板',
      'manual close': '手动平仓',
      'stop loss': '止损',
      'take profit': '止盈',
      'take profit 2': '止盈2',
      'trend invalidation exit': '趋势失效平仓',
      'floating profit trailing stop': '浮盈回撤止盈',
      'structure break stop': '结构破位止损',
      'ATR volatility stop': 'ATR波动止损',
      'take profit: target 2 reached': '止盈：达到第二止盈目标',
      'take profit: high distribution handoff complete': '止盈：高位换手完成',
      'take profit: 4h OI valley formed; downside trend exhausted': '止盈：4H OI洼地形成，下跌行情衰竭',
      'take profit: floating profit trailing stop': '止盈：浮盈回撤触发保护',
      'take profit: protected stop after profit lock': '止盈：盈利后保护止损触发',
      'stop loss: protected stop slipped below entry': '止损：保护止损成交后仍低于开仓价',
      'take profit: breakout protection stop': '止盈：突破保护止损触发',
      'stop loss: breakout protection stop': '止损：突破保护止损触发',
      'stop loss: signal structure failed': '止损：信号结构失效',
      'stop loss: signal direction or structure failed': '止损：信号方向或结构失效',
      'stop loss: ATR volatility hard stop': '止损：ATR波动硬止损',
      'stop loss: 15m entry structure stop': '止损：15分钟入场结构失效',
      'stop loss: 4h OI-valley absorption floor failed by close': '止损：4H实体收盘跌破OI洼地吸筹底部',
      'stop loss: 4h closed below EMA55 while OI increased; new shorts likely': '止损：4H收盘跌破EMA55且OI继续增加，疑似新增空头',
      'take profit: 1h/4h body closed below support or EMA/BOLL zone': '止盈：1小时/4小时实体跌破支撑或EMA/BOLL区域，保护利润',
      'stop loss: 1h/4h body closed below support or EMA/BOLL zone': '止损：1小时/4小时实体跌破支撑或EMA/BOLL区域',
      'take profit: 1h/4h body closed above resistance or EMA/BOLL zone': '止盈：1小时/4小时实体突破压力或EMA/BOLL区域，保护利润',
      'stop loss: 1h/4h body closed above resistance or EMA/BOLL zone': '止损：1小时/4小时实体突破压力或EMA/BOLL区域',
      'take profit: strong trend EMA50 structure invalidated': '止盈：强趋势EMA50结构失效，保护利润',
      'stop loss: strong trend EMA50 structure invalidated': '止损：强趋势EMA50结构失效',
      'take profit: floating profit drawdown protection': '止盈：浮盈回撤保护',
      'stop loss: floating profit drawdown protection': '止损：浮盈回撤后转亏离场',
      'take profit: near 4h resistance with profit protection': '止盈：靠近4小时压力位，保护利润',
      'take profit: near 4h support with profit protection': '止盈：靠近4小时支撑位，保护利润',
      'stop loss: near 4h resistance with profit protection': '止损：靠近4小时压力位后转弱',
      'stop loss: near 4h support with profit protection': '止损：靠近4小时支撑位后转弱',
      'take profit: 4h support plus short exhaustion confirmed': '止盈：4小时支撑位出现空头衰竭确认',
      'stop loss: 4h support plus short exhaustion confirmed': '止损：4小时支撑位反弹确认后离场',
      'take profit: short trend support protection stop': '止盈：空头趋势支撑位保护止损触发',
      'stop loss: short trend support protection stop': '止损：空头趋势支撑位保护止损触发',
      'take profit: profit drawdown after long crowd risk': '止盈：浮盈回撤叠加多头拥挤风险',
      'take profit: profit drawdown after short crowd risk': '止盈：浮盈回撤叠加空头拥挤风险',
      'take profit: profit drawdown after OI abnormal risk': '止盈：浮盈回撤叠加OI异常风险',
      'take profit: profit drawdown after funding overheated risk': '止盈：浮盈回撤叠加资金费率过热',
      'take profit: profit drawdown after OI drop risk': '止盈：浮盈回撤叠加OI下降风险',
      'take profit: profit drawdown after volume blow-off risk': '止盈：浮盈回撤叠加放量衰竭风险',
      'take profit: profit drawdown after RSI overheated risk': '止盈：浮盈回撤叠加RSI过热',
      'take profit: profit drawdown after RSI oversold risk': '止盈：浮盈回撤叠加RSI超卖',
      'take profit: long crowd risk': '止盈：多头拥挤风险，保护利润',
      'stop loss: long crowd risk': '止损：多头拥挤风险',
      'take profit: short crowd risk': '止盈：空头拥挤风险，保护利润',
      'stop loss: short crowd risk': '止损：空头拥挤风险',
      'take profit: OI abnormal risk': '止盈：OI异常风险，保护利润',
      'stop loss: OI abnormal risk': '止损：OI异常风险',
      'take profit: funding overheated risk': '止盈：资金费率过热风险，保护利润',
      'stop loss: funding overheated risk': '止损：资金费率过热风险',
      'crowded one-way exit': '单边行情散户拥挤，主动离场'
    };
    const riskExitReasonText = {
      LONG_CROWD: '风险：多头拥挤平仓',
      SHORT_CROWD: '风险：空头拥挤平仓',
      OI_ABNORMAL: '风险：OI异常平仓',
      FUNDING_HOT: '风险：资金费率过热平仓'
    };
    const vetoText = {
      'directional entry signal not established': '多/空方向未成立',
      'late trend stage blocks fresh entry': '趋势末段',
      'auto strategy disabled; new entries are paused': '策略已关闭',
      'symbol already has an open position': '已开仓',
      'high distribution handoff complete; avoid new long': '高位换手完成，禁止新开多单',
      '4h OI dropped while long/short ratio rose; retail longs are carrying the decline': '价格阴跌、OI下降且多空比上升，散户多头仍在扛单',
      '4h OI valley confirmed; downside trend exhaustion blocks new short': '4H OI洼地形成，下跌行情衰竭，禁止新开空单',
      '4h OI valley confirmed; low-area short chasing is blocked': '4H OI洼地形成，禁止在支撑或低位追空',
      'trend late stage and structure reward below minimum; wait for a new pullback': '趋势末端且结构收益空间不足，等待新的反抽机会',
      '15m tactical entry lacks a valid 15m structure stop or 1h/4h entry zone': '缺少有效止损或入场区',
      'BTC 4h extreme volatility; pause new altcoin entries': 'BTC极端波动',
      'market warm-up is still running': '行情预热中',
      'latest price is stale for more than 15 seconds': '实时价格超过15秒未更新',
      'OI/long-short ratio data is stale for more than 180 seconds': 'OI/多空比超过180秒未更新',
      'current funding rate data is stale for more than 15 minutes': '资金费率超过15分钟未更新',
      'extreme volatility: skip new long entry': '极端波动',
      'extreme volatility: skip new short entry': '极端波动',
      'long side overcrowded': '多头拥挤',
      'short side overcrowded': '空头拥挤',
      'funding too hot for long entry': '多单资金费率过热',
      'funding rate too hot for long entry': '多单资金费率过热',
      'funding too negative for short entry': '空单资金费率过低',
      'open interest spike risks liquidation sweep': 'OI激增，扫损风险',
      'RSI overheated for long entry': 'RSI过热',
      'RSI oversold for short entry': 'RSI超卖',
      'price closed above upper BOLL; no chase': '价格突破BOLL上轨',
      'price closed below lower BOLL; no chase': '价格跌破BOLL下轨',
      'upper wick sweep rejected; avoid chasing long': '上插针回落，等待回踩确认',
      'upper wick rejected; wait for pullback or reclaim before adding long': '上插针回落，等待回踩或重新收回',
      'lower wick sweep reclaimed; avoid chasing short': '下插针收回，禁止追空',
      'smart money accumulation after OI flush; avoid chasing shorts': 'OI爆减后疑似吸筹',
      'short crowd is vulnerable to a squeeze': '空头拥挤，逼空风险',
      'smart money distribution after upper wick sweeps': '上插针后疑似派发',
      'trapped longs are increasing while price falls': '价格下跌且多头拥挤',
      'capitulation OI flush; avoid late shorts': '下跌尾段OI爆减',
      'strict long blocked: EMA20/EMA50/EMA200 not bullish': 'EMA未多头排列',
      'strict long blocked: BOLL mid not confirmed twice': 'BOLL中轨未连续确认',
      'strict long blocked: RSI not in 52-72': 'RSI不在52-72',
      'strict long blocked: volume below 1.5x average': '成交量低于均量1.5倍',
      'strict long blocked: 4h OI increase below 3%': '4小时OI增幅不足3%',
      'strict long blocked: top long/short ratio below 1.1': '大户多空比低于1.1',
      'strict long blocked: funding outside long range': '资金费率不在多单区间',
      'strict long blocked: EMA lines are too compressed': 'EMA三线粘合',
      'strict long blocked: RSI neutral zone': 'RSI处于中性区',
      'strict long blocked: 1h candle amplitude above 5%': '1小时振幅超过5%',
      'strict short blocked: EMA20/EMA50/EMA200 not bearish': 'EMA未空头排列',
      'strict short blocked: BOLL mid not confirmed twice': 'BOLL中轨未连续确认',
      'strict short blocked: RSI not in 28-48': 'RSI不在28-48',
      'strict short blocked: volume below 1.5x average': '成交量低于均量1.5倍',
      'strict short blocked: 4h OI increase below 3%': '4小时OI增幅不足3%',
      'strict short blocked: top long/short ratio above 0.9': '大户多空比高于0.9',
      'strict short blocked: funding outside short range': '资金费率不在空单区间',
      'strict short blocked: EMA lines are too compressed': 'EMA三线粘合',
      'strict short blocked: RSI neutral zone': 'RSI处于中性区',
      'strict short blocked: 1h candle amplitude above 5%': '1小时振幅超过5%',
      '1h trigger opposes long entry': '1小时方向冲突',
      '1h trigger opposes short entry': '1小时方向冲突',
      '4h OI drained and volume is weak; EMA/BOLL bounce is not a clean long pullback': 'OI下降且量能不足',
      'high pullback with OI/funding/crowd risk; avoid long entry': '高位回踩风险',
      'low pullback with OI/funding/crowd risk; avoid short entry': '低位反抽风险',
      'high area without pullback confirmation; wait for 1h/4h pullback before long': '高位回踩未确认',
      'low area without bounce confirmation; wait for 1h/4h retest before short': '低位反抽未确认',
      'low area without 1h/4h resistance retest; wait for higher-timeframe bounce before short': '低位反抽未确认',
      '4h OI deleverage with price breakdown; avoid long entry': '4小时OI去杠杆破位',
      '4h OI deleverage breakdown; wait for resistance retest or upper-wick rejection before short': '4小时OI破位反抽未确认',
      '4h OI deleveraged but 1h support held; avoid chasing short': '1小时支撑仍有效',
      '4h OI drained; wait for 1h resistance retest or upper-wick rejection before short': 'OI下降且反抽未确认',
      'one-way uptrend RSI above 92; skip fresh long and protect existing profit': '单边上涨RSI超过92',
      'one-way uptrend RSI hot without 1h/15m pullback; wait before long': 'RSI过热且回踩未确认',
      'normal/chop trend RSI overheated; wait for 1h/4h pullback before long': 'RSI过热且回踩未确认',
      'one-way downtrend RSI below 8; skip fresh short and protect existing profit': '单边下跌RSI低于8',
      'one-way downtrend RSI cold without 1h/15m bounce rejection; wait before short': 'RSI超卖且反抽未确认',
      'normal/chop trend RSI oversold; wait for 1h/4h retest before short': 'RSI超卖且反抽未确认'
    };
    const mtfText = {
      BULL: '偏多',
      BEAR: '偏空',
      NEUTRAL: '中性',
      UNKNOWN: '未知',
      BREAKOUT_UP: '向上突破',
      BREAKDOWN_DOWN: '向下跌破',
      BOX_UPPER_HALF: '箱体上半区',
      BOX_LOWER_HALF: '箱体下半区',
      WAIT: '等待',
      BREAKOUT: '突破',
      BREAKDOWN: '跌破',
      RETEST: '回踩/反抽',
      FAKE_BREAKOUT: '假突破',
      FAKE_BREAKDOWN: '假跌破',
      HEALTHY_PULLBACK: '健康回踩',
      HIGH_PULLBACK: '高位回踩',
      LOW_PULLBACK: '低位反抽',
      NORMAL: '正常',
      DELEVERAGE_WAIT: 'OI去杠杆等待',
      DELEVERAGE_HOLD_LONG: 'OI去杠杆守住支撑',
      DELEVERAGE_CROWD_HOLD_LONG: 'OI去杠杆且多头拥挤但守住支撑',
      DELEVERAGE_BREAKDOWN: 'OI去杠杆后价格破位',
      DELEVERAGE_CROWD_BREAKDOWN: 'OI去杠杆且多头拥挤后破位',
      DELEVERAGE_CROWD_WAIT: 'OI去杠杆且多头拥挤等待',
      REBUILD_BREAKOUT_LONG: 'OI回升并突破',
      DENSE: '均线密集',
      SPREAD: '均线发散',
      RETEST_UP: '回踩均线不破',
      RETEST_DOWN: '反抽均线失败',
      LONG: '多头',
      SHORT: '空头',
      NONE: '无方向'
    };
    function tAction(value) { return actionText[value] || value || ''; }
    function tRegime(value) { return regimeText[value] || value || ''; }
    function tSmartMoneyPhase(value) { return smartMoneyPhaseText[value] || value || '中性'; }
    function tTrendState(value) { return trendStateText[value] || value || '震荡'; }
    function tRiskState(value) { return riskStateText[value] || value || '正常'; }
    function tReason(value) {
      if (!value) return '';
      const rawReason = String(value);
      if (
        rawReason === 'lower wick sweep reclaimed; avoid chasing short'
        || rawReason === 'lower wick reclaimed; wait for bounce or breakdown before adding short'
      ) return '下插针收回，等待反抽或重新跌回';
      if (reasonText[value]) return reasonText[value];
      const reason = rawReason;
      const maPrefixes = [
        ['MA cluster breakout up', '均线密集区向上突破'],
        ['MA cluster retest held near MA20', '突破均线密集区后回踩MA20不破'],
        ['MA cluster dense; wait for breakout or MA20 retest', '均线密集缠绕，等待突破或回踩MA20确认'],
        ['MA cluster breakdown down', '均线密集区向下跌破'],
        ['MA cluster retest rejected near MA20', '跌破均线密集区后反抽MA20失败'],
        ['MA cluster dense; wait for breakdown or MA20 retest', '均线密集缠绕，等待跌破或反抽MA20确认']
      ];
      for (const [prefix, text] of maPrefixes) {
        if (reason.startsWith(prefix)) return reason.replace(prefix, text).replace('price=', '，密集价=');
      }
      const dynamicPrefixes = [
        ['VWAP pullback held; average cost support favors long', 'VWAP回踩不破，平均成本支撑多头'],
        ['KC mid pullback held; volatility channel support favors long', 'KC中轨回踩不破，波动通道支撑多头'],
        ['QPS quote flow accelerates with price; traded value confirms long', '成交额速度放大且价格上行，确认多头'],
        ['QPS blow-off without price follow-through; long risk', '成交额放大但价格未跟进，多头风险'],
        ['price extended far above VWAP; chasing long risk', '价格远离VWAP，追多风险升高'],
        ['VWAP retest rejected; average cost resistance favors short', 'VWAP反抽不过，平均成本压制空头'],
        ['KC mid retest rejected; volatility channel resistance favors short', 'KC中轨反抽不过，波动通道压制空头'],
        ['QPS quote flow accelerates with price; traded value confirms short', '成交额速度放大且价格下行，确认空头'],
        ['QPS blow-off without price follow-through; short risk', '成交额放大但价格未跟进，空头风险'],
        ['price extended far below VWAP; chasing short risk', '价格远离VWAP，追空风险升高'],
        ['volume pattern confirms long: breakout volume, quiet retest, renewed buying', '放量突破、缩量回踩、再放量上行'],
        ['volume breakout above resistance; retest confirmation preferred', '放量突破压力，等待回踩确认更稳'],
        ['breakout retest held quietly; waiting renewed buying volume', '突破后缩量回踩不破，等待再放量'],
        ['volume pattern confirms short: breakdown volume, quiet retest, renewed selling', '放量跌破、缩量反抽、再放量下行'],
        ['volume breakdown below support; retest confirmation preferred', '放量跌破支撑，等待反抽确认更稳'],
        ['breakdown retest rejected quietly; waiting renewed selling volume', '跌破后缩量反抽不过，等待再放量']
      ];
      for (const [prefix, text] of dynamicPrefixes) {
        if (reason.startsWith(prefix)) return text;
      }
      if (reason.startsWith('entry reward/risk ')) {
        const preferred = reason.match(/entry reward\\/risk ([0-9.]+)R below preferred ([0-9.]+)R/);
        if (preferred) return `实际盈亏比${preferred[1]}R，低于偏好${preferred[2]}R，降级开仓`;
        const values = reason.match(/entry reward\\/risk ([0-9.]+)R below minimum ([0-9.]+)R/);
        return values ? `实际盈亏比${values[1]}R，低于最低${values[2]}R` : '实际盈亏比低于最低要求';
      }
      if (reason.startsWith('final score ')) {
        return '评分低于85';
      }
      if (reason === 'directional entry signal not established') {
        return '多空方向信号尚未成立';
      }
      if (reason.startsWith('等待 ') || reason.startsWith('已进入建议区') || reason === '暂无有效建议入场区') {
        return reason;
      }
      if (reason.startsWith('current entry position is not excellent:')) {
        return '当前入场位置不优秀';
      }
      if (reason === 'late trend stage blocks fresh entry') {
        return '趋势已进入末段，禁止新开仓';
      }
      if (reason === 'auto strategy disabled; new entries are paused') {
        return '策略总开关已关闭，暂停新开仓';
      }
      if (reason === 'symbol already has an open position') {
        return '该币种已有持仓，禁止重复开仓';
      }
      if (reason.startsWith('available entry margin ')) {
        const values = reason.match(/available entry margin ([0-9.]+) USDT below minimum ([0-9.]+) USDT/);
        return values ? `可用开仓保证金${values[1]}U，低于最低${values[2]}U` : '可用开仓保证金不足';
      }
      if (reason.startsWith('position capacity full:')) {
        const value = (reason.match(/position capacity full: (\\d+)/) || [])[1] || '';
        return `持仓已满${value ? `（上限${value}个）` : ''}，等待空余仓位或满足调仓条件`;
      }
      if (reason === '15m tactical entry lacks a valid 15m structure stop or 1h/4h entry zone') {
        return '15分钟战术入场缺少有效止损结构，且未处于1H/4H入场区';
      }
      if (reason === 'BTC 4h extreme volatility; pause new altcoin entries') {
        return 'BTC 4小时极端波动，暂停山寨币新开仓';
      }
      if (reason === 'market warm-up is still running') {
        return '多周期行情正在预热，完成前禁止新开仓';
      }
      if (reason === 'latest price is stale for more than 15 seconds') {
        return '实时价格超过15秒未更新';
      }
      if (reason === 'OI/long-short ratio data is stale for more than 180 seconds') {
        return 'OI/多空比超过180秒未更新';
      }
      if (reason === 'current funding rate data is stale for more than 15 minutes') {
        return '资金费率超过15分钟未更新';
      }
      if (reason === 'required multi-timeframe K-line context is missing or discontinuous') {
        return '交易周期缺失或不连续';
      }
      const timeframeGap = reason.match(/^(15m|1h|4h) K-line context is missing or discontinuous$/);
      if (timeframeGap) {
        return `${timeframeGap[1]}周期缺失或不连续`;
      }
      if (reason.startsWith('auto entry execution failed:')) {
        return `自动开仓执行失败：${reason.replace('auto entry execution failed:', '').trim()}`;
      }
      if (String(value).startsWith('risk exit:')) {
        const key = String(value).replace('risk exit:', '').trim();
        return riskExitReasonText[key] || `风险：${key}平仓`;
      }
      if (String(value).startsWith('take profit:')) return String(value).replace('take profit:', '止盈：');
      if (String(value).startsWith('stop loss:')) return String(value).replace('stop loss:', '止损：');
      if (String(value).startsWith('rotation exit:')) {
        const text = String(value);
        const symbol = (text.match(/symbol=([^\\s]+)/) || [])[1] || '';
        const score = (text.match(/score=(\\d+)/) || [])[1] || '';
        const type = text.includes('efficiency rotation') ? '效率调仓' : text.includes('trend invalidated') ? '趋势失效调仓' : '调仓';
        const display = symbol ? displaySymbol(symbol) : '更强标的';
        return `${type}：5仓已满，换入高评分强趋势标的 ${display}${score ? `，评分=${score}` : ''}`;
      }
      if (String(value).startsWith('pyramid add:')) return String(value).replace('pyramid add:', '强趋势盈利回踩加仓：').replace('score=', '评分=').replace('state=', '状态=');
      if (String(value).startsWith('auto strategy score=')) return String(value).replace('auto strategy score=', '自动策略开仓，评分=');
      if (String(value).startsWith('MTF:')) return tMtfSummary(value);
      return value;
    }
    function tVeto(value) {
      const reason = String(value || '').trim();
      if (!reason) return '';
      if (
        reason === 'lower wick sweep reclaimed; avoid chasing short'
        || reason === 'lower wick reclaimed; wait for bounce or breakdown before adding short'
      ) return '下插针收回，等待反抽或重新跌回';
      if (vetoText[reason]) return vetoText[reason];
      if (reason === 'directional entry signal not established') return '多/空方向未成立';
      if (reason.startsWith('final score ')) return '评分低于85';
      if (reason.startsWith('等待 ') || reason.startsWith('已进入建议区') || reason === '暂无有效建议入场区') {
        return reason;
      }
      if (reason.startsWith('current entry position is not excellent:')) {
        return '当前入场位置不优秀';
      }
      const timeframeGap = reason.match(/^(15m|1h|4h) K-line context is missing or discontinuous$/);
      if (timeframeGap) return `${timeframeGap[1]}周期缺失或不连续`;
      if (reason.startsWith('entry reward/risk ')) {
        const preferred = (reason.match(/below preferred ([0-9.]+)R/) || [])[1] || '';
        if (preferred) return `盈亏比偏低，降级开仓（偏好${preferred}R）`;
        const minimum = (reason.match(/below minimum ([0-9.]+)R/) || [])[1] || '';
        return minimum ? `盈亏比低于${minimum}R` : '盈亏比不足';
      }
      if (reason.startsWith('available entry margin ')) return '可用保证金不足';
      if (reason.startsWith('position capacity full:')) return '持仓已满';
      if (reason === 'entry lacks a valid structure stop') return '缺少可验证的结构止损';
      const reentryWait = reason.match(/^waiting for new (15m|1h|4h|1d) closed candle after (?:full exit|stop loss)$/);
      if (reentryWait) return `完整平仓后等待新的${reentryWait[1]}收盘K线`;
      if (reason.startsWith('auto entry execution failed:')) return '自动开仓执行失败';
      if (
        reason.startsWith('MA cluster dense; wait for breakout or MA20 retest')
        || reason.startsWith('MA cluster dense; wait for breakdown or MA20 retest')
      ) {
        return '均线密集，方向未确认';
      }
      if (/[\u3400-\u9fff]/.test(reason)) return reason;
      return '其他风控条件未满足';
    }
    function tMtfSummary(value) {
      const text = String(value || '');
      const d1 = (text.match(/1d=([^;]+)/) || [])[1] || 'UNKNOWN';
      const h4 = (text.match(/4h=([^;]+)/) || [])[1] || 'UNKNOWN';
      const h1 = (text.match(/1h=([^/;]+)/) || [])[1] || 'UNKNOWN';
      const h1Dir = (text.match(new RegExp('1h=[^/;]+/([^;]+)')) || [])[1] || 'NONE';
      const pullback = (text.match(/pullback=([^/;]+)/) || [])[1] || 'UNKNOWN';
      const pullbackDir = (text.match(new RegExp('pullback=[^/;]+/([^;]+)')) || [])[1] || 'NONE';
      const oi4h = (text.match(/oi4h=([^;]+)/) || [])[1] || 'UNKNOWN';
      const ma4h = (text.match(/ma4h=([^@;]+)/) || [])[1] || 'UNKNOWN';
      const ma4hPrice = (text.match(/ma4h=[^@;]+@([^;]+)/) || [])[1] || '--';
      const ma1h = (text.match(/ma1h=([^@;]+)/) || [])[1] || 'UNKNOWN';
      const ma1hPrice = (text.match(/ma1h=[^@;]+@([^;]+)/) || [])[1] || '--';
      return `多周期：日线=${mtfText[d1] || d1}；4小时=${mtfText[h4] || h4}；1小时=${mtfText[h1] || h1}/${mtfText[h1Dir] || h1Dir}；回踩=${mtfText[pullback] || pullback}/${mtfText[pullbackDir] || pullbackDir}；4H OI=${mtfText[oi4h] || oi4h}；4H均线=${mtfText[ma4h] || ma4h}@${ma4hPrice}；1H均线=${mtfText[ma1h] || ma1h}@${ma1hPrice}`;
    }
    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }
    function wrapReason(value, limit = 50) {
      const text = String(value || '');
      const lines = [];
      for (const paragraph of text.split(/\\r?\\n/)) {
        if (!paragraph) {
          lines.push('');
          continue;
        }
        if (paragraph.length <= limit) {
          lines.push(paragraph);
          continue;
        }
        const parts = paragraph.split(/([；;])/);
        let line = '';
        for (let i = 0; i < parts.length; i += 1) {
          const part = parts[i] || '';
          if (!part) continue;
          const next = line + part;
          if (next.length > limit && line) {
            lines.push(line);
            line = part.replace(/^[；;]\\s*/, '');
          } else {
            line = next;
          }
        }
        if (line) lines.push(line);
      }
      return lines.flatMap(item => {
        if (item.length <= limit) return [item];
        const chunks = [];
        for (let i = 0; i < item.length; i += limit) chunks.push(item.slice(i, i + limit));
        return chunks;
      }).map(escapeHtml).join('<br>');
    }
    function flowTableText(value) {
      return escapeHtml(String(value || '').replace(/\\s*\\r?\\n\\s*/g, ' ').trim());
    }
    function flowReasonText(value) {
      return String(value || '')
        .split(/\\r?\\n/)
        .map(line => escapeHtml(line.trim()))
        .join('<br>');
    }
    function tReasons(values) { return (values || []).map(tReason).join('；'); }
    function tVetoes(values) {
      return [...new Set((values || []).map(tVeto).filter(Boolean))].join('；');
    }
    function reasonList(values) {
      return (values || []).map(value => String(value || '').trim()).filter(Boolean);
    }
    function reasonTextList(values) {
      return reasonList(values).map(tReason);
    }
    function reasonHas(reasons, patterns) {
      return reasonList(reasons).some(reason => patterns.some(pattern => reason.includes(pattern)));
    }
    function firstReason(reasons, patterns) {
      return reasonList(reasons).find(reason => patterns.some(pattern => reason.includes(pattern))) || '';
    }
    function approxPriceFromReason(reason) {
      const text = String(reason || '');
      const match = text.match(/(?:price=|密集价=|@)([0-9]+(?:\\.[0-9]+)?)/);
      if (!match) return '';
      const value = Number(match[1]);
      if (!Number.isFinite(value)) return '';
      if (Math.abs(value) >= 100) return value.toFixed(1);
      if (Math.abs(value) >= 10) return value.toFixed(2);
      if (Math.abs(value) >= 1) return value.toFixed(3);
      return value.toPrecision(4);
    }
    function mtfPart(reasons, key) {
      const mtf = firstReason(reasons, ['MTF:']);
      const match = mtf.match(new RegExp(`${key}=([^;]+)`));
      return match ? match[1] : '';
    }
    function mtfPrice(reasons, key) {
      const mtf = firstReason(reasons, ['MTF:']);
      const match = mtf.match(new RegExp(`${key}=[^;@]*@([0-9]+(?:\\.[0-9]+)?)`));
      if (!match) return '';
      return approxPriceFromReason(`@${match[1]}`);
    }
    function numberValue(value) {
      const num = Number(value);
      return Number.isFinite(num) ? num : null;
    }
    function priceLabel(value) {
      const num = numberValue(value);
      if (num === null) return '';
      if (Math.abs(num) >= 100) return num.toFixed(1);
      if (Math.abs(num) >= 10) return num.toFixed(2);
      if (Math.abs(num) >= 1) return num.toFixed(3);
      return num.toPrecision(4);
    }
    function pctLabel(value) {
      const num = numberValue(value);
      return num === null ? '' : `${(num * 100).toFixed(1)}%`;
    }
    function zoneLabel(low, high, fallback) {
      const lo = numberValue(low);
      const hi = numberValue(high);
      const fb = numberValue(fallback);
      if (lo !== null && hi !== null) {
        const a = Math.min(lo, hi);
        const b = Math.max(lo, hi);
        const aText = priceLabel(a);
        const bText = priceLabel(b);
        return aText && bText && aText !== bText ? `${aText}-${bText}` : (aText || bText);
      }
      return fb !== null ? priceLabel(fb) : '';
    }
    function objectValue(value) {
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    }
    function clusterObject(signal) {
      const h1 = objectValue(signal.h1_ma_cluster);
      const h4 = objectValue(signal.h4_ma_cluster);
      const h1State = String(h1.state || 'UNKNOWN');
      const h4State = String(h4.state || 'UNKNOWN');
      const activeStates = ['DENSE', 'BREAKOUT_UP', 'BREAKDOWN_DOWN', 'RETEST_UP', 'RETEST_DOWN'];
      if (activeStates.includes(h1State)) return h1;
      if (activeStates.includes(h4State)) return h4;
      if (h1State && h1State !== 'UNKNOWN') return h1;
      return h4;
    }
    function clusterState(signal) {
      return String(clusterObject(signal).state || 'UNKNOWN');
    }
    function clusterZone(signal, fallbackPrice) {
      const cluster = clusterObject(signal);
      return zoneLabel(cluster.lower, cluster.upper, cluster.price) || fallbackPrice || '';
    }
    function structureZone(signal, direction, fallbackPrice) {
      const h4 = objectValue(signal.h4_structure);
      const h1s = objectValue(signal.h1_structure);
      const h1 = objectValue(signal.h1_trigger);
      if (direction === '多头') {
        return zoneLabel(h1s.support_zone_low, h1s.support_zone_high, h1s.support)
          || zoneLabel(h1.support_zone_low, h1.support_zone_high, null)
          || zoneLabel(h4.support_zone_low, h4.support_zone_high, h4.support)
          || fallbackPrice || '';
      }
      if (direction === '空头') {
        return zoneLabel(h1s.resistance_zone_low, h1s.resistance_zone_high, h1s.resistance)
          || zoneLabel(h1.resistance_zone_low, h1.resistance_zone_high, null)
          || zoneLabel(h4.resistance_zone_low, h4.resistance_zone_high, h4.resistance)
          || fallbackPrice || '';
      }
      return fallbackPrice || '';
    }
    function structureZoneByFrame(signal, direction, frame, fallbackPrice = '') {
      const h4 = objectValue(signal.h4_structure);
      const h1s = objectValue(signal.h1_structure);
      const h1 = objectValue(signal.h1_trigger);
      const source = frame === '4h' ? h4 : h1s;
      if (direction === '多头') {
        return zoneLabel(source.support_zone_low, source.support_zone_high, source.support)
          || (frame === '1h' ? zoneLabel(h1.support_zone_low, h1.support_zone_high, h1.support) : '')
          || fallbackPrice || '';
      }
      if (direction === '空头') {
        return zoneLabel(source.resistance_zone_low, source.resistance_zone_high, source.resistance)
          || (frame === '1h' ? zoneLabel(h1.resistance_zone_low, h1.resistance_zone_high, h1.resistance) : '')
          || fallbackPrice || '';
      }
      return fallbackPrice || '';
    }
    function atZone(text, zone) {
      return zone ? `${text}≈${zone}` : text;
    }
    function entrySideLevels(signal, direction) {
      const levels = objectValue(signal.entry_levels);
      if (direction === '多头') return objectValue(levels.long);
      if (direction === '空头') return objectValue(levels.short);
      return {};
    }
    function entryLevelZone(level) {
      const item = objectValue(level);
      return zoneLabel(item.low, item.high, item.price);
    }
    function addEntryCandidate(items, label, level) {
      const zone = typeof level === 'string' ? level : entryLevelZone(level);
      if (!zone) return;
      items.push({ label, zone });
    }
    function mergedEntryText(items, limit = 3) {
      const groups = [];
      for (const item of items) {
        const found = groups.find(group => group.zone === item.zone);
        if (found) {
          if (!found.labels.includes(item.label)) found.labels.push(item.label);
        } else {
          groups.push({ zone: item.zone, labels: [item.label] });
        }
      }
      return groups.slice(0, limit).map(group => `${group.labels.join('/')}≈${group.zone}`);
    }
    function conciseReason(reasons, signal = {}, options = {}) {
      const rawReasons = reasonList(reasons);
      const action = String(signal.action || '');
      const trend = String(signal.trend_state || signal.regime || '');
      let direction = '震荡';
      if (action.includes('SHORT') || trend.includes('SHORT') || trend.includes('DOWN')) direction = '空头';
      if (action.includes('LONG') || trend.includes('LONG') || trend.includes('UP')) direction = '多头';

      const has1hHeld = reasonHas(rawReasons, ['1h BOLL/EMA pullback held', '1小时 BOLL/EMA 回踩不破']);
      const has1hRejected = reasonHas(rawReasons, ['1h BOLL/EMA pullback rejected', '1小时 BOLL/EMA 反抽失败']);
      const has15mConfirm = reasonHas(rawReasons, ['15m pullback', '15分钟回踩', '15m BOLL']);
      const hasVwapLong = reasonHas(rawReasons, ['VWAP pullback held']);
      const hasVwapShort = reasonHas(rawReasons, ['VWAP retest rejected']);
      const hasVwapLongRisk = reasonHas(rawReasons, ['far above VWAP']);
      const hasVwapShortRisk = reasonHas(rawReasons, ['far below VWAP']);
      const hasVolumeLongConfirm = reasonHas(rawReasons, ['volume pattern confirms long']);
      const hasVolumeShortConfirm = reasonHas(rawReasons, ['volume pattern confirms short']);
      const hasVolumeLongBreakout = reasonHas(rawReasons, ['volume breakout above resistance', 'breakout retest held quietly']);
      const hasVolumeShortBreakdown = reasonHas(rawReasons, ['volume breakdown below support', 'breakdown retest rejected quietly']);
      const maRetestUp = firstReason(rawReasons, ['MA cluster retest held near MA20', '均线密集区后回踩MA20不破']);
      const maRetestDown = firstReason(rawReasons, ['MA cluster retest rejected near MA20', '均线密集区反抽MA20失败']);
      const maBreakUp = firstReason(rawReasons, ['MA cluster breakout up', '突破均线密集区']);
      const maBreakDown = firstReason(rawReasons, ['MA cluster breakdown down', '跌破均线密集区']);
      const maReason = maRetestUp || maRetestDown || maBreakUp || maBreakDown;
      const maPrice = approxPriceFromReason(maReason);
      const maZone = clusterZone(signal, maPrice || mtfPrice(rawReasons, 'ma1h') || mtfPrice(rawReasons, 'ma4h'));
      const supportResistanceZone = structureZone(signal, direction, maPrice || mtfPrice(rawReasons, 'ma1h') || mtfPrice(rawReasons, 'ma4h'));
      const h1StructureZone = structureZoneByFrame(signal, direction, '1h', supportResistanceZone);
      const h4StructureZone = structureZoneByFrame(signal, direction, '4h', supportResistanceZone);
      const isOneWayUp = trend.includes('ONE_WAY_UP');
      const isOneWayDown = trend.includes('ONE_WAY_DOWN');
      const hasDownsideSweep = reasonHas(rawReasons, ['washout confirmed: downside wick swept support', 'downside sweep reclaimed support', 'lower wick sweeps', 'capitulation absorption']);
      const hasUpsideSweep = reasonHas(rawReasons, ['washout confirmed: upside wick swept resistance', 'upside sweep rejected resistance', 'upper wick sweeps', 'repeated upper wicks', 'smart money distribution']);
      const h4Oi = objectValue(signal.h4_oi);
      const h4OiState = String(h4Oi.state || mtfPart(rawReasons, 'oi4h') || '');
      const h4OiValley = objectValue(signal.h4_oi_valley);
      const hasOiValley = String(h4OiValley.state || '') === 'CONFIRMED';
      const h1Trigger = objectValue(signal.h1_trigger);
      const h1State = String(h1Trigger.state || '');
      const h4Structure = objectValue(signal.h4_structure);
      const h4State = String(h4Structure.state || '');
      const riskState = String(signal.risk_state || '');
      const rsiValue = numberValue(signal.rsi14);
      const volumeRatio = numberValue(signal.volume_ratio);
      const oiDropFromHigh = numberValue(h4Oi.drop_from_high_pct);
      const oiRebound = numberValue(h4Oi.rebound_pct);

      const activeClusterStates = ['DENSE', 'BREAKOUT_UP', 'BREAKDOWN_DOWN', 'RETEST_UP', 'RETEST_DOWN'];
      let maStructure = '';
      if (maReason || activeClusterStates.includes(clusterState(signal))) maStructure = `上一个均线密集${maZone ? `≈${maZone}` : ''}`;

      let structureLine = '方向与结构：震荡，等待边界确认。';
      if (direction === '多头') {
        structureLine = `方向与结构：多头方向成立：日线不空，4H结构偏多，1H站稳EMA20/EMA50或BOLL中轨，价格回踩支撑位${supportResistanceZone ? `≈${supportResistanceZone}` : ''}不破并重新收回${maStructure ? `，${maStructure}` : ''}。`;
        if (hasVolumeLongConfirm || hasVolumeLongBreakout) structureLine = structureLine.replace('。', '，放量突破后回踩不破。');
      } else if (direction === '空头') {
        structureLine = `方向与结构：空头方向成立：日线不多，4H结构偏空，1H跌破EMA20/EMA50或BOLL中轨，价格反抽压力位${supportResistanceZone ? `≈${supportResistanceZone}` : ''}不破并重新回落${maStructure ? `，${maStructure}` : ''}。`;
        if (hasVolumeShortConfirm || hasVolumeShortBreakdown) structureLine = structureLine.replace('。', '，放量跌破后反抽不过。');
      }

      let entryBits = [];
      const backendEntryText = String(signal.suggested_entry_text || '').trim();
      const levels = entrySideLevels(signal, direction);
      const candidates = [];
      if (direction === '多头') {
        if (String(signal.entry_timeframe_override || '') === '4h') {
          addEntryCandidate(candidates, '4H支撑回踩不破', levels.h4_support);
          addEntryCandidate(candidates, '4H BOLL中轨趋势回踩', levels.h4_boll_mid);
          addEntryCandidate(candidates, '4H EMA20或EMA60趋势回踩', levels.h4_ema20_ema60);
        }
        if (isOneWayUp && has15mConfirm) addEntryCandidate(candidates, '强单边15m EMA20/EMA60回踩收回', levels.m15_ema20_ema60);
        if (hasDownsideSweep) addEntryCandidate(candidates, '下插针扫损后重新收回支撑', levels.sweep_reclaim_support);
        if (hasVwapLong) addEntryCandidate(candidates, 'VWAP/成交密集区回踩不破', levels.vwap_pullback);
        if (hasVolumeLongConfirm || hasVolumeLongBreakout) addEntryCandidate(candidates, '前压力突破后回踩确认', levels.breakout_retest);
        if (has1hHeld) addEntryCandidate(candidates, '1H支撑回踩不破', levels.h1_support);
        if (has1hHeld) addEntryCandidate(candidates, '1H BOLL中轨回踩不破', levels.h1_boll_mid);
        if (maBreakUp) addEntryCandidate(candidates, '1H/4H K线上穿均线密集', levels.ma_cluster_breakout);
        if (maRetestUp) addEntryCandidate(candidates, '突破均线密集后回踩MA20不破', levels.ma20_retest);
        if (!candidates.length) {
          if (isOneWayUp) {
            addEntryCandidate(candidates, '强单边15m EMA20/EMA60回踩收回', levels.m15_ema20_ema60);
            addEntryCandidate(candidates, '1H/4H EMA20或EMA60回踩不破', levels.h1_ema20_ema60);
            addEntryCandidate(candidates, '4H EMA20或EMA60趋势回踩', levels.h4_ema20_ema60);
            addEntryCandidate(candidates, '4H BOLL中轨趋势回踩', levels.h4_boll_mid);
          } else {
            addEntryCandidate(candidates, '1H支撑回踩不破', levels.h1_support);
            addEntryCandidate(candidates, '1H BOLL中轨回踩不破', levels.h1_boll_mid);
            addEntryCandidate(candidates, '前压力突破后回踩确认', levels.breakout_retest);
          }
          addEntryCandidate(candidates, '均线密集区突破或回踩MA20不破', levels.ma_cluster_breakout || levels.ma20_retest);
        }
      } else if (direction === '空头') {
        const distributionStage = String(signal.distribution_short_stage || '');
        if (distributionStage === 'HIGH_DISTRIBUTION_RANGE') {
          addEntryCandidate(candidates, '第一阶段：高位派发箱体上沿小仓试空', levels.distribution_range_high);
        }
        if (distributionStage === 'DESCENDING_DISTRIBUTION') {
          addEntryCandidate(candidates, '第二阶段：下降顶点趋势线反抽做空', levels.descending_high_trendline);
        }
        if (hasUpsideSweep) addEntryCandidate(candidates, '上插针扫空后重新跌回压力', levels.sweep_reject_resistance);
        if (hasVwapShort) addEntryCandidate(candidates, 'VWAP/成交密集区反抽不过', levels.vwap_retest);
        if (hasVolumeShortConfirm || hasVolumeShortBreakdown) addEntryCandidate(candidates, '前支撑跌破后反抽确认', levels.breakdown_retest);
        if (has1hRejected) addEntryCandidate(candidates, '1H压力反抽不过', levels.h1_resistance);
        if (has1hRejected) addEntryCandidate(candidates, '1H BOLL中轨反抽失败', levels.h1_boll_mid);
        if (maBreakDown) addEntryCandidate(candidates, '1H/4H K线下穿均线密集', levels.ma_cluster_breakdown);
        if (maRetestDown) addEntryCandidate(candidates, '跌破均线密集后反抽MA20失败', levels.ma20_retest);
        if (!candidates.length) {
          if (isOneWayDown) {
            addEntryCandidate(candidates, '1H/4H EMA20或EMA60反抽不过', levels.h1_ema20_ema60);
            addEntryCandidate(candidates, '4H EMA20或EMA60趋势反抽', levels.h4_ema20_ema60);
          } else {
            addEntryCandidate(candidates, '1H压力反抽不过', levels.h1_resistance);
            addEntryCandidate(candidates, '1H BOLL中轨反抽失败', levels.h1_boll_mid);
            addEntryCandidate(candidates, '前支撑跌破后反抽确认', levels.breakdown_retest);
          }
          addEntryCandidate(candidates, '均线密集区跌破或反抽MA20失败', levels.ma_cluster_breakdown || levels.ma20_retest);
        }
      }
      entryBits = mergedEntryText(candidates);
      if (backendEntryText) entryBits = [backendEntryText];
      if (!entryBits.length) entryBits.push('边界确认处：暂无有效区间');
      const entryLabel = options.entryLabel || '入场位置';
      const entryLine = `${entryLabel}：${entryBits.slice(0, 3).join('；')}`;

      const riskBits = [];
      if (oiDropFromHigh !== null && oiDropFromHigh <= -0.18) {
        riskBits.push(`OI骤减${pctLabel(oiDropFromHigh)}`);
      } else if (reasonHas(rawReasons, ['open interest rising mildly', 'OI 温和上升'])) {
        riskBits.push('OI温和上升');
      } else if (reasonHas(rawReasons, ['open interest stable', 'OI 基本稳定', '4h OI=NORMAL'])) {
        riskBits.push('OI基本稳定');
      }
      if (volumeRatio !== null && volumeRatio >= 1.2) {
        const shortStuck = direction === '空头' && ['WAIT', 'FAKE_BREAKOUT'].includes(h1State) && ['BOX_UPPER_HALF', 'BREAKOUT_UP'].includes(h4State);
        const longStuck = direction === '多头' && ['WAIT', 'FAKE_BREAKDOWN'].includes(h1State) && ['BOX_LOWER_HALF', 'BREAKDOWN_DOWN'].includes(h4State);
        if (shortStuck) riskBits.push('成交量放大后价格1H横盘涨不动');
        else if (longStuck) riskBits.push('成交量放大后价格1H横盘跌不动');
        else riskBits.push('成交量放大');
      }
      if (hasVolumeLongConfirm) riskBits.push('突破放量，回踩缩量，再放量上行');
      if (hasVolumeShortConfirm) riskBits.push('跌破放量，反抽缩量，再放量下行');
      if (hasVwapLong || hasVwapShort) riskBits.push('VWAP成本位确认');
      if (hasVwapLongRisk) riskBits.push('价格远离VWAP，追多风险');
      if (hasVwapShortRisk) riskBits.push('价格远离VWAP，追空风险');
      if (rsiValue !== null) {
        if (rsiValue >= 75 || rsiValue <= 25) riskBits.push(`RSI=${rsiValue.toFixed(0)}`);
        else if (reasonHas(rawReasons, ['RSI in healthy', 'RSI 处于健康'])) riskBits.push('RSI健康');
      } else if (reasonHas(rawReasons, ['RSI in healthy', 'RSI 处于健康'])) {
        riskBits.push('RSI健康');
      }
      if (direction === '多头' && hasDownsideSweep) riskBits.push('多次下插针已清杠杆');
      if (direction === '空头' && hasUpsideSweep) riskBits.push('多次上插针已清杠杆');
      if (oiRebound !== null && oiRebound >= 0.003) riskBits.push('OI下降后回稳');
      else if (hasOiValley) riskBits.push('4H OI洼地已形成');
      if (riskState === 'LONG_CROWD') riskBits.push('多头情绪拥挤');
      else if (riskState === 'SHORT_CROWD') riskBits.push('空头情绪拥挤');
      else if (riskState === 'OI_ABNORMAL') riskBits.push('OI异常');
      else if (riskState === 'FUNDING_HOT') riskBits.push('资金费率过热');
      else if (reasonHas(rawReasons, ['long/short ratio is not overcrowded', '多空比未出现'])) riskBits.push('情绪未拥挤');
      const structureHeld = has1hHeld || has1hRejected || ['RETEST', 'BREAKOUT', 'BREAKDOWN', 'FAKE_BREAKOUT', 'FAKE_BREAKDOWN'].includes(h1State);
      if (structureHeld) riskBits.push('结构未失效');
      if (!riskBits.length) riskBits.push('风险正常');
      const riskLine = `指标与风险：${riskBits.join('；')}。`;

      return `${structureLine}\n${entryLine}\n${riskLine}`;
    }
    function signalReasonText(signal) {
      return conciseReason(signal.reasons || [], signal, { entryLabel: '建议入场位置' });
    }
    function signalEntryPosition(signal) {
      if (signal.entry_timing) {
        const timingText = { GOOD: '优秀', WAIT: '等待', BLOCK: '禁止' };
        timingText.GOOD = '优秀';
        timingText.WAIT = '等待';
        timingText.BLOCK = '禁止';
        const timingClass = signal.entry_timing === 'GOOD' ? 'pos' : signal.entry_timing === 'BLOCK' ? 'neg' : 'muted';
        return `<span class="${timingClass}">${timingText[signal.entry_timing] || signal.entry_timing}</span>`;
      }
      const action = String(signal.action || '');
      return action === 'ENTRY_LONG' || action === 'ENTRY_SHORT' ? '等待' : '禁止';
    }
    function tEntryPositionReason(value) {
      const text = String(value || '');
      const map = {
        'entry position wait: latest price unavailable': '等待最新价格',
        'entry position good: live price is inside a scored long entry zone': '实时价格已进入评分认可的做多区间',
        'entry position good: live price is inside a scored short entry zone': '实时价格已进入评分认可的做空区间',
        'entry position wait: live price has not reached a scored long entry zone': '实时价格尚未进入评分认可的做多区间',
        'entry position wait: live price has not reached a scored short entry zone': '实时价格尚未进入评分认可的做空区间',
        'entry position wait: live price is at or below 1h BOLL lower; wait for a higher-timeframe bounce': '实时价格位于或跌破1H BOLL下轨，等待1H/4H反抽后再做空',
        'entry position blocked: suggested entry zone unavailable': '评分未生成有效建议入场区'
      };
      return map[text] || text;
    }
    function entryReasonText(position) {
      const rawReason = String(position.reason || '');
      const rawEntryReason = String(position.entry_reason || '');
      const entryReasons = reasonList(position.entry_reasons || []);
      if (rawEntryReason === '手动' || rawEntryReason === '鎵嬪姩' || (!entryReasons.length && rawReason.toLowerCase().includes('manual'))) return '手动';
      const score = position.entry_score || (rawReason.match(/score=(\\d+)/) || [])[1] || '';
      const prefix = score ? `自动，评分：${score}` : '自动';
      if (entryReasons.length) {
        const sideAction = String(position.side || '').includes('SHORT') ? 'ENTRY_SHORT' : 'ENTRY_LONG';
        return `${prefix}；\n${conciseReason(entryReasons, { action: sideAction, ...(position.entry_context || {}) })}`;
      }
      return prefix;
    }
    function apiSymbol(value) {
      return String(value || '').trim().toUpperCase().replace('/', '').replace('-', '');
    }
    function displaySymbol(value) {
      const raw = apiSymbol(value);
      if (raw.endsWith('USDT') && raw.length > 4) return `${raw.slice(0, -4)}/USDT`;
      return raw;
    }
    function symbols() { return document.getElementById('symbols').value.split(',').map(apiSymbol).filter(Boolean); }
    async function startPaper(autoTrade) {
      await call(() => api('/api/paper/start', { method: 'POST', body: JSON.stringify({
        starting_balance: Number(document.getElementById('startingBalance').value),
        symbols: symbols(),
        interval: document.getElementById('interval').value,
        auto_trade: autoTrade,
        reset_account: false,
        poll_seconds: 5
      }) }));
    }
    async function stopPaper() { await call(() => api('/api/paper/stop', { method: 'POST' })); }
    async function refreshPaper() { await call(() => api('/api/paper/refresh', { method: 'POST' })); }
    async function resetPaper() {
      await call(() => api('/api/paper/reset?starting_balance=' + encodeURIComponent(document.getElementById('startingBalance').value), { method: 'POST' }));
    }
    async function closePosition(symbol) {
      await call(() => api('/api/paper/order/close', { method: 'POST', body: JSON.stringify({ symbol: apiSymbol(symbol) }) }));
    }
    function showError(message) {
      const text = String(message || '').trim();
      const ticker = document.getElementById('errorTicker');
      const tickerText = document.getElementById('errorTickerText');
      if (!ticker || !tickerText) return;
      if (!text) {
        hideError();
        return;
      }
      if (tickerText.textContent !== text) {
        tickerText.textContent = text;
        tickerText.style.animation = 'none';
        void tickerText.offsetWidth;
        tickerText.style.animation = '';
      }
      ticker.classList.add('show');
    }
    function hideError() {
      const ticker = document.getElementById('errorTicker');
      const tickerText = document.getElementById('errorTickerText');
      if (!ticker || !tickerText) return;
      ticker.classList.remove('show');
      tickerText.textContent = '';
    }
    async function call(fn) {
      hideError();
      try { render(await fn()); } catch (err) { showError(err.message); }
    }
    async function load() {
      try { render(await api('/api/paper/status')); } catch (err) { showError(err.message); }
    }
    function captureTableScrollAnchor(tableElement) {
      const scrollElement = tableElement?.closest('.scroll');
      if (!tableElement || !scrollElement) return null;
      if (scrollElement.scrollTop <= 1) return { scrollTop: 0, key: '', offset: 0 };
      const viewportTop = Math.max(
        scrollElement.getBoundingClientRect().top,
        tableElement.tHead?.getBoundingClientRect().bottom || 0
      );
      const row = [...tableElement.querySelectorAll('tbody tr')].find(
        item => item.getBoundingClientRect().bottom > viewportTop
      );
      return {
        scrollTop: scrollElement.scrollTop,
        key: row?.cells[0]?.textContent?.trim() || '',
        offset: row ? row.getBoundingClientRect().top - viewportTop : 0
      };
    }
    function restoreTableScrollAnchor(tableElement, anchor) {
      const scrollElement = tableElement?.closest('.scroll');
      if (!tableElement || !scrollElement || !anchor) return;
      if (!anchor.key) {
        scrollElement.scrollTop = anchor.scrollTop;
        return;
      }
      const row = [...tableElement.querySelectorAll('tbody tr')].find(
        item => item.cells[0]?.textContent?.trim() === anchor.key
      );
      if (!row) {
        scrollElement.scrollTop = anchor.scrollTop;
        return;
      }
      const viewportTop = Math.max(
        scrollElement.getBoundingClientRect().top,
        tableElement.tHead?.getBoundingClientRect().bottom || 0
      );
      scrollElement.scrollTop += row.getBoundingClientRect().top - viewportTop - anchor.offset;
    }
    function render(data) {
      const marketAge = data.market_updated_at ? (Date.now() - new Date(data.market_updated_at).getTime()) / 1000 : Infinity;
      const marketState = marketAge > 60 ? '行情延迟' : '行情正常';
      const entryBlockText = {
        SERVICE_STOPPED: '后台服务停止',
        AUTO_TRADE_DISABLED: '策略关闭',
        DAILY_LOSS_LIMIT: '日亏损锁定',
        WEEKLY_LOSS_LIMIT: '周亏损锁定',
        MAX_DRAWDOWN: '最大回撤锁定',
        LOSS_COOLDOWN: '连败冷静期',
        CONSECUTIVE_LOSSES: '连续亏损锁定'
      };
      const newEntriesAllowed = data.new_entries_allowed ?? data.auto_trade;
      const entryBlockCodes = Array.isArray(data.new_entry_block_codes) ? data.new_entry_block_codes : [];
      const entryState = newEntriesAllowed
        ? '允许'
        : `禁止${entryBlockCodes.length ? `（${entryBlockCodes.map(code => entryBlockText[code] || code).join('、')}）` : ''}`;
      document.getElementById('updated').textContent = `${data.running ? '行情与持仓管理运行中' : '后台服务已停止'} | 新开仓 ${entryState} | ${marketState} | ${timeText(data.market_updated_at || data.updated_at)}`;
      const metrics = [
        ['资金', money(data.equity) + ' U'],
        ['可用', money(data.available_balance) + ' U'],
        ['占用保证金', money(data.used_margin) + ' U'],
        ['已实现', money(data.realized_pnl) + ' U'],
        ['未实现', money(data.unrealized_pnl) + ' U'],
        ['手续费', '-' + money(data.fees_paid) + ' U'],
        ['总收益', money(data.total_pnl) + ' U / ' + pct(data.total_pnl_pct)]
      ];
      document.getElementById('metrics').innerHTML = metrics.map(([k,v]) => `<div class="card metric"><span>${k}</span><strong class="${k.includes('收益') || k.includes('实现') || k.includes('手续费') ? pnlClass(String(v).split(' ')[0]) : ''}">${v}</strong></div>`).join('');
      document.getElementById('positions').className = 'center-table';
      document.getElementById('positions').innerHTML = table(['币种','方向','杠杆','入场','现价','数量','保证金','浮盈亏','收益率','止损','止盈','入场原因','操作'], data.positions.map(p => [
        displaySymbol(p.symbol),
        tAction(p.side),
        `${p.leverage || 0}x`,
        priceText(p.entry_price),
        priceText(p.mark_price),
        wholeUsdt(p.notional),
        money(p.margin_usdt),
        `<span class="${pnlClass(p.unrealized_pnl)}">${money(p.unrealized_pnl)}</span>`,
        `<span class="${pnlClass(p.unrealized_pnl_pct_on_margin)}">${pct(p.unrealized_pnl_pct_on_margin)}</span>`,
        priceText(p.stop_price),
        priceText(p.take_profit_2),
        wrapReason(entryReasonText(p), 100),
        `<button class="neutral" onclick="closePosition('${displaySymbol(p.symbol)}')">平仓</button>`
      ]));
      const signalRows = Object.entries(data.latest_signals || {})
        .map(([symbol, s]) => [displaySymbol(symbol), `<span class="pill">${tAction(s.action)}</span>`, tTrendState(s.trend_state || s.regime), tRiskState(s.risk_state), tSmartMoneyPhase(s.smart_money_phase), s.score, signalEntryPosition(s), flowReasonText(signalReasonText(s)), tVetoes(s.vetoes)]);
      const signalsTable = document.getElementById('signals');
      const signalsScrollAnchor = captureTableScrollAnchor(signalsTable);
      signalsTable.className = 'signals-table';
      signalsTable.innerHTML = table(
        ['币种','动作','状态','风险','主力周期','分数','入场位置','原因','否决'],
        signalRows,
        { 6: 'timing-col' },
      );
      restoreTableScrollAnchor(signalsTable, signalsScrollAnchor);
      balanceSignalAndFillPanels();
      const fillHeaders = ['币种','方向','杠杆','开仓均价','平仓均价','数量','止损','止盈','收益率','实现盈亏','手续费','开仓时间','平仓时间','入场位置','出场原因'];
      const allFills = [...(data.fills || [])].filter(f => f.action === 'CLOSE' || f.closed_at).reverse();
      const fillRows = allFills.map(f => [
        displaySymbol(f.symbol),
        tAction(f.side),
        `${f.leverage || 0}x`,
        priceText(f.entry_price || f.price),
        priceText(f.price),
        wholeUsdt(Number(f.price || 0) * Number(f.quantity || 0)),
        priceText(f.stop_price),
        priceText(f.take_profit_2),
        `<span class="${pnlClass(f.return_pct)}">${pct(f.return_pct)}</span>`,
        `<span class="${pnlClass(f.realized_pnl)}">${money(f.realized_pnl)}</span>`,
        money(f.fee),
        timeText(f.opened_at),
        timeText(f.closed_at),
        flowTableText(f.entry_position || '--'),
        wrapReason(tReason(f.reason), 50)
      ]);
      const fillPages = paginateFillRows(fillHeaders, fillRows);
      const totalFillPages = Math.max(fillPages.length, 1);
      fillsPage = Math.min(Math.max(fillsPage, 1), totalFillPages);
      document.getElementById('fills').innerHTML = fillsTable(fillHeaders, fillPages[fillsPage - 1] || []);
      renderFillsPager(totalFillPages);
      renderDailyPnl(data.daily_pnl);
      if (data.last_error) showError(data.last_error);
      else hideError();
      updatePnlHistory(data);
      drawPnlChart();
    }
    function table(headers, rows, classOverrides = {}) {
      const classes = headers.map((header, index) => classOverrides[index] || tableClass(header));
      if (!rows.length) return `<thead><tr>${headers.map((h, i) => `<th class="${classes[i]}">${h}</th>`).join('')}</tr></thead><tbody><tr><td colspan="${headers.length}">暂无数据</td></tr></tbody>`;
      return `<thead><tr>${headers.map((h, i) => `<th class="${classes[i]}">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((v, i) => `<td class="${classes[i]}">${v}</td>`).join('')}</tr>`).join('')}</tbody>`;
    }
    function balanceSignalAndFillPanels() {
      const signalCard = document.getElementById('signals')?.closest('.card');
      const dataGrid = signalCard?.parentElement;
      if (!dataGrid) return;
      dataGrid.style.gridTemplateRows = '230px 405px minmax(210px, 1fr)';
    }
    function fillsTable(headers, rows) {
      const classes = headers.map(tableClass);
      const widths = headers.map(fillColumnWidth);
      const colgroup = `<colgroup>${widths.map(width => `<col style="width:${width}">`).join('')}</colgroup>`;
      const head = `${colgroup}<thead><tr>${headers.map((h, i) => `<th class="${classes[i]}">${h}</th>`).join('')}</tr></thead>`;
      if (rows.length) {
        return `${head}<tbody>${rows.map(row => `<tr>${row.map((v, i) => `<td class="${classes[i]}">${v}</td>`).join('')}</tr>`).join('')}</tbody>`;
      }
      return `${head}<tbody><tr class="empty-fill-row"><td colspan="${headers.length}">暂无数据</td></tr></tbody>`;
    }
    function paginateFillRows(headers, rows) {
      const tableElement = document.getElementById('fills');
      const scrollElement = tableElement.closest('.fills-scroll');
      const pager = document.getElementById('fillsPager');
      tableElement.className = 'center-table';
      if (!rows.length || !scrollElement || !pager) return [rows];

      pager.style.display = 'none';
      pager.innerHTML = '';
      tableElement.innerHTML = fillsTable(headers, rows);
      let pages = measuredFillPages(tableElement, scrollElement, rows);

      if (pages.length > 1) {
        pager.style.display = 'flex';
        pager.innerHTML = '<button style="visibility:hidden">上一页</button>';
        pages = measuredFillPages(tableElement, scrollElement, rows);
      }
      return pages;
    }
    function measuredFillPages(tableElement, scrollElement, rows) {
      const headHeight = Math.ceil(tableElement.tHead?.getBoundingClientRect().height || 0);
      const availableHeight = Math.max(scrollElement.clientHeight - headHeight - 1, 16);
      const rowElements = [...tableElement.tBodies[0].rows];
      const pages = [];
      let page = [];
      let usedHeight = 0;

      rowElements.forEach((rowElement, index) => {
        const rowHeight = Math.max(Math.ceil(rowElement.getBoundingClientRect().height), 16);
        if (page.length && usedHeight + rowHeight > availableHeight) {
          pages.push(page);
          page = [];
          usedHeight = 0;
        }
        page.push(rows[index]);
        usedHeight += rowHeight;
      });
      if (page.length) pages.push(page);
      return pages.length ? pages : [[]];
    }
    function fillColumnWidth(header) {
      if (header === '币种') return '5%';
      if (header === '方向') return '2.4%';
      if (header === '杠杆') return '2.6%';
      if (['开仓均价', '平仓均价', '止损', '止盈'].includes(header)) return '4.4%';
      if (header === '数量') return '3.8%';
      if (['收益率', '实现盈亏'].includes(header)) return '4.2%';
      if (header === '手续费') return '3.2%';
      if (header.includes('时间')) return '8.5%';
      if (header === '入场位置') return '18%';
      if (header === '出场原因') return '22%';
      if (header === '原因') return '22%';
      return '5%';
    }
    function tableClass(header) {
      if (header === '入场位置') return 'entry-position-col';
      if (header === '原因' || header === '入场原因' || header === '出场原因') return 'reason-col';
      if (header === '否决') return 'veto-col';
      if (header.includes('时间')) return 'time-col';
      if (['价格', '入场', '现价', '成交价', '开仓均价', '平仓均价', '数量', '保证金', '净盈亏', '浮盈亏', '收益率', '止损', '止损价', '止盈', '收益率', '实现盈亏', '手续费', '分数'].includes(header)) return 'num-col';
      if (header === '币种') return 'symbol-col';
      if (['方向', '动作', '状态', '风险', '操作', '杠杆'].includes(header)) return 'side-col';
      return '';
    }
    function renderFillsPager(totalPages) {
      const pager = document.getElementById('fillsPager');
      if (!pager) return;
      if (totalPages <= 1) {
        pager.style.display = 'none';
        pager.innerHTML = '';
        return;
      }
      pager.style.display = 'flex';
      const items = [];
      const addPage = page => items.push(`<button class="${page === fillsPage ? 'active' : ''}" onclick="setFillsPage(${page})">${page}</button>`);
      const addEllipsis = () => items.push('<span class="ellipsis">...</span>');
      if (totalPages <= 7) {
        for (let i = 1; i <= totalPages; i += 1) addPage(i);
      } else {
        addPage(1);
        const start = Math.max(2, fillsPage - 1);
        const end = Math.min(totalPages - 1, fillsPage + 1);
        if (start > 2) addEllipsis();
        for (let i = start; i <= end; i += 1) addPage(i);
        if (end < totalPages - 1) addEllipsis();
        addPage(totalPages);
      }
      pager.innerHTML = `
        <button onclick="setFillsPage(${fillsPage - 1})" ${fillsPage <= 1 ? 'disabled' : ''}>上一页</button>
        ${items.join('')}
        <button onclick="setFillsPage(${fillsPage + 1})" ${fillsPage >= totalPages ? 'disabled' : ''}>下一页</button>
      `;
    }
    function setFillsPage(page) {
      fillsPage = page;
      load();
    }
    function shiftDailyMonth(delta) {
      if (!latestDailyPnl) return;
      const months = dailyAvailableMonths(latestDailyPnl);
      const index = months.indexOf(dailyMonthKey);
      const next = months[Math.min(Math.max(index + delta, 0), months.length - 1)];
      if (!next || next === dailyMonthKey) return;
      dailyMonthKey = next;
      renderDailyPnl(latestDailyPnl);
    }
    function dailyAvailableMonths(daily) {
      const months = new Set((daily?.days || []).map(day => String(day.date || '').slice(0, 7)).filter(Boolean));
      months.add(String(daily?.today || tradingDayKey(new Date())).slice(0, 7));
      return [...months].sort();
    }
    function renderDailyPnl(daily) {
      const calendar = document.getElementById('dailyCalendar');
      const monthLabel = document.getElementById('dailyMonth');
      const prevButton = document.getElementById('dailyPrevMonth');
      const nextButton = document.getElementById('dailyNextMonth');
      if (!calendar || !monthLabel) return;
      latestDailyPnl = daily;
      const today = daily?.today || tradingDayKey(new Date());
      const months = dailyAvailableMonths(daily);
      if (!dailyMonthKey || !months.includes(dailyMonthKey)) dailyMonthKey = today.slice(0, 7);
      const [year, month] = dailyMonthKey.split('-').map(Number);
      monthLabel.textContent = dailyMonthKey;
      if (prevButton) prevButton.disabled = months.indexOf(dailyMonthKey) <= 0;
      if (nextButton) nextButton.disabled = months.indexOf(dailyMonthKey) >= months.length - 1;
      const daysByDate = new Map((daily?.days || []).map(day => [day.date, day]));
      const first = new Date(year, month - 1, 1);
      const daysInMonth = new Date(year, month, 0).getDate();
      const cells = [];
      for (let i = 0; i < first.getDay(); i += 1) {
        cells.push('<div class="daily-day empty"></div>');
      }
      for (let day = 1; day <= daysInMonth; day += 1) {
        const key = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
        const item = daysByDate.get(key);
        const pnl = Number(item?.net_pnl || 0);
        const cls = pnl > 0 ? 'profit-day' : pnl < 0 ? 'loss-day' : '';
        cells.push(`<div class="daily-day ${cls}"><strong>${day}</strong><span>${signedMoney(pnl)}</span></div>`);
      }
      calendar.innerHTML = cells.join('');
    }
    function tradingDayStart(date) {
      const d = new Date(date);
      const start = new Date(d.getFullYear(), d.getMonth(), d.getDate(), 8, 0, 0, 0);
      if (d < start) start.setDate(start.getDate() - 1);
      return start;
    }
    function tradingDayKey(date) {
      const start = tradingDayStart(date);
      return `${start.getFullYear()}-${String(start.getMonth() + 1).padStart(2, '0')}-${String(start.getDate()).padStart(2, '0')}`;
    }
    function updatePnlHistory(data) {
      const todayBaseline = Number(data.daily_pnl?.today_baseline || 0);
      const value = Number(data.daily_pnl?.today_pnl ?? (Number(data.total_pnl || 0) - todayBaseline));
      latestTotalPnl = value;
      const headline = document.getElementById('pnlHeaderValue');
      if (headline) {
        headline.textContent = `${signedMoney(latestTotalPnl)}U`;
        headline.className = latestTotalPnl >= 0 ? 'positive' : 'negative';
      }
      pnlSamples = (data.pnl_history || [])
        .map(point => ({
          date: new Date(point.timestamp),
          value: Number(point.total_pnl || 0) - todayBaseline
        }))
        .filter(point => !Number.isNaN(point.date.getTime()));
    }
    function quarterHourBucketStart(date) {
      const d = new Date(date);
      d.setMinutes(Math.floor(d.getMinutes() / 15) * 15, 0, 0);
      return d;
    }
    function drawPnlChart() {
      const canvas = document.getElementById('pnlChart');
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(Math.floor(rect.width * dpr), 320);
      const height = Math.max(Math.floor(rect.height * dpr), 180);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, width, height);
      const pad = { left: 52 * dpr, right: 6 * dpr, top: 10 * dpr, bottom: 20 * dpr };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const plotBottom = pad.top + plotH;
      const timeTickTop = plotBottom + 3 * dpr;
      const timeLabelY = plotBottom + 8 * dpr;
      ctx.fillStyle = '#fbfcfe';
      ctx.fillRect(0, 0, width, height);

      const now = new Date();
      const dayStart = tradingDayStart(now);
      const dayEnd = new Date(dayStart.getTime() + 24 * 60 * 60 * 1000);
      const chartPoints = pnlSamples
        .filter(point => point.date >= dayStart && point.date <= dayEnd)
        .sort((a, b) => a.date - b.date);
      let plotPoints = chartPoints;
      if (plotPoints.length > 1 && Math.abs(plotPoints[0].value) < 0.005) {
        plotPoints = plotPoints.slice(1);
      }
      const values = plotPoints.length ? plotPoints.map(point => point.value) : [0];
      let min = Math.min(...values, 0);
      let max = Math.max(...values, 0);
      if (Math.abs(max - min) < 0.01) {
        max = 1;
        min = -1;
      }
      const yFor = value => pad.top + (max - value) / (max - min) * plotH;
      const xForDate = date => {
        const ratio = (date.getTime() - dayStart.getTime()) / (dayEnd.getTime() - dayStart.getTime());
        return pad.left + Math.max(0, Math.min(1, ratio)) * plotW;
      };

      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1 * dpr;
      for (let i = 0; i <= 4; i += 1) {
        const y = pad.top + (plotH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
      }
      const zeroY = yFor(0);
      ctx.strokeStyle = '#94a3b8';
      ctx.setLineDash([5 * dpr, 4 * dpr]);
      ctx.beginPath();
      ctx.moveTo(pad.left, zeroY);
      ctx.lineTo(width - pad.right, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);

      if (plotPoints.length) {
        ctx.strokeStyle = values[values.length - 1] >= 0 ? '#059669' : '#dc2626';
        ctx.lineWidth = 2.2 * dpr;
        ctx.beginPath();
        plotPoints.forEach((point, index) => {
          const x = xForDate(point.date);
          const y = yFor(point.value);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        const last = plotPoints[plotPoints.length - 1];
        ctx.fillStyle = last.value >= 0 ? '#059669' : '#dc2626';
        ctx.beginPath();
        ctx.arc(xForDate(last.date), yFor(last.value), 4 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = '#475569';
      ctx.font = `${11 * dpr}px "Microsoft YaHei", Arial`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      [max, (max + min) / 2, min].forEach(value => {
        ctx.fillText(`${value.toFixed(2)}U`, pad.left - 8 * dpr, yFor(value));
      });

      ctx.fillStyle = '#64748b';
      ctx.font = `${8.5 * dpr}px "Microsoft YaHei", Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let i = 0; i < 24; i += 1) {
        const tickDate = new Date(dayStart.getTime() + i * 60 * 60 * 1000);
        const x = xForDate(tickDate);
        const label = i === 16 ? '24' : String((8 + i) % 24);
        ctx.beginPath();
        ctx.moveTo(x, timeTickTop);
        ctx.lineTo(x, timeTickTop + 3 * dpr);
        ctx.strokeStyle = '#cbd5e1';
        ctx.stroke();
        ctx.fillText(label, x, timeLabelY);
      }

    }
    load();
    setInterval(load, 5000);
    window.addEventListener('resize', drawPnlChart);
  </script>
</body>
</html>
"""


app = create_app()
