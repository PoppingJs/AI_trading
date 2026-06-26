from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import os
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ai_trading.backtest import BacktestEngine
from ai_trading.binance import BinanceFuturesMarketData
from ai_trading.cli import _synthetic_market
from ai_trading.config import AppSettings, load_settings
from ai_trading.indicators import build_indicators
from ai_trading.models import SignalAction
from ai_trading.paper import PAPER_DEFAULT_BALANCE, PaperTradingEngine
from ai_trading.strategy import CompositeStrategy


class BacktestRequest(BaseModel):
    symbol: str = "DEMOUSDT"
    starting_equity: float = Field(default=10_000.0, gt=0)
    use_demo_data: bool = True


class PaperStartRequest(BaseModel):
    starting_balance: float | None = Field(default=None, gt=0)
    symbols: list[str] | None = None
    interval: str = "15m"
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


def create_app(settings_path: str | Path = "config/strategy.yaml", state_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.paper_engine.close()

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

    @app.get("/", response_class=HTMLResponse)
    def paper_dashboard() -> str:
        return PAPER_DASHBOARD_HTML

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
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

    @app.post("/api/backtests/run", dependencies=[Depends(_require_api_token)])
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

    @app.get("/api/paper/status")
    async def paper_status() -> dict[str, object]:
        return await app.state.paper_engine.status_async()

    @app.post("/api/paper/start", dependencies=[Depends(_require_api_token)])
    async def paper_start(request: PaperStartRequest) -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        requested_symbols = [symbol.upper() for symbol in request.symbols or []]
        if not requested_symbols or _uses_default_symbol_pool(requested_symbols):
            engine.symbols = ["AUTO_TOP30"]
        else:
            engine.symbols = requested_symbols
        engine.interval = request.interval
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
            raise HTTPException(status_code=502, detail=f"Binance market refresh failed: {exc}") from exc
        return await engine.status_async()

    @app.post("/api/paper/order/open", dependencies=[Depends(_require_api_token)])
    async def paper_open_order(request: PaperOrderRequest) -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        try:
            await engine.open_position(
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
            raise HTTPException(status_code=502, detail=f"Binance price fetch failed: {exc}") from exc
        return await engine.status_async()

    @app.post("/api/paper/order/close", dependencies=[Depends(_require_api_token)])
    async def paper_close_order(request: PaperCloseRequest) -> dict[str, object]:
        engine: PaperTradingEngine = app.state.paper_engine
        try:
            await engine.close_position(request.symbol)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Binance price fetch failed: {exc}") from exc
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
    return not cleaned or cleaned == {"BTCUSDT", "ETHUSDT", "SOLUSDT"} or cleaned == {"AUTO_TOP30"}


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
  <title>AI 閲忓寲浜ゆ槗骞冲彴</title>
  <style>
    :root { color-scheme: light; font-family: "Microsoft YaHei", Arial, sans-serif; }
    html, body { width: 100%; height: 100%; max-width: 100%; overflow: hidden; }
    body { margin: 0; background: #f6f7f9; color: #171717; }
    header { height: 60px; box-sizing: border-box; padding: 12px 24px; background: #111827; color: white; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    header h1 { margin: 0; font-size: 20px; }
    header p { margin: 4px 0 0; color: #cbd5e1; font-size: 13px; }
    main { height: calc(100vh - 60px); padding: 10px 16px; width: 100%; box-sizing: border-box; margin: 0; overflow: hidden; display: flex; flex-direction: column; gap: 10px; }
    .grid { display: grid; gap: 10px; }
    .metrics { grid-template-columns: repeat(7, minmax(0, 1fr)); flex: 0 0 auto; }
    .layout { grid-template-columns: 460px minmax(0, 1fr); align-items: stretch; flex: 1 1 auto; min-height: 0; margin-top: 0 !important; }
    .layout > .card { min-height: 0; overflow: hidden; display: flex; flex-direction: column; }
    .layout > .grid { display: grid; grid-template-rows: 230px minmax(70px, 1fr) 210px; min-height: 0; }
    .layout > .grid > .card { min-height: 0; overflow: hidden; }
    .layout > .grid > .card:nth-child(1) { display: flex; flex-direction: column; overflow: hidden; }
    .layout > .grid > .card:nth-child(1) .scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; scrollbar-gutter: stable; }
    .layout > .grid > .card:nth-child(1) thead th { position: sticky; top: 0; z-index: 1; }
    .layout > .grid > .card:nth-child(2) { display: flex; flex-direction: column; overflow: hidden; }
    .layout > .grid > .card:nth-child(2) .scroll { flex: 1 1 auto; min-height: 0; max-height: none; overflow-y: auto; }
    .layout > .grid > .card:nth-child(2) thead th { position: sticky; top: 0; z-index: 1; }
    .layout > .grid > .card:last-child { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
    .card { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .metric { min-width: 0; padding: 9px 10px; }
    .metric span { color: #6b7280; font-size: 12px; }
    .metric strong { display: block; margin-top: 4px; font-size: 18px; white-space: nowrap; }
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
    #fills { width: max-content; max-width: 100%; table-layout: auto; }
    #fills th, #fills td { padding: 0 1px; overflow: hidden; text-overflow: clip; }
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
    #fills .reason-col { min-width: 50em; max-width: 50em; }
    .center-table th.reason-col { text-align: center; }
    .status { font-size: 12px; color: #6b7280; }
    .pill { display: inline-block; padding: 3px 7px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; }
    .error { display: none; }
    .error-ticker {
      position: fixed;
      top: 60px;
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
    #pnlChart { width: 100%; height: 315px; display: block; background: transparent; border: 0; border-radius: 0; }
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
      <h1>AI 閲忓寲浜ゆ槗骞冲彴</h1>
      <p>Binance USDT-M 瀹炴椂琛屾儏锛屾湰鍦版ā鎷熻处鎴凤紝涓嶄細浜х敓鐪熷疄璁㈠崟銆?/p>
    </div>
    <div class="status" id="updated">loading...</div>
  </header>
  <div class="error-ticker" id="errorTicker" aria-live="polite">
    <span class="error-ticker-text" id="errorTickerText"></span>
  </div>
  <main>
    <section class="grid metrics" id="metrics"></section>
    <section class="grid layout" style="margin-top:14px;">
      <div class="card">
        <div class="left-main">
        <h2 style="font-size:16px;margin:0 0 10px;">鎺у埗鍙?/h2>
        <label>妯℃嫙鏈噾 USDT</label>
        <input id="startingBalance" type="number" value="1200" min="1" step="10" />
        <label>甯佺鏍囩殑姹?/label>
        <input id="symbols" value="AUTO_TOP30" />
        <label>鍛ㄦ湡</label>
        <select id="interval"><option>15m</option><option selected>1h</option><option>4h</option><option>1d</option></select>
        <div class="control-actions">
          <button class="primary" onclick="startPaper(true)">鍚姩绛栫暐</button>
          <button class="neutral" onclick="stopPaper()">鍋滄</button>
          <button class="neutral" onclick="resetPaper()">閲嶇疆1200U</button>
        </div>
        <p class="error" id="error"></p>
        </div>
        <div class="daily-pnl" id="dailyPnl">
          <div class="daily-head">
            <h2>姣忔棩鐩堜簭</h2>
            <span class="status">08:00 - 娆℃棩08:00</span>
          </div>
          <div class="daily-month">
            <button class="month-btn" id="dailyPrevMonth" onclick="shiftDailyMonth(-1)">鈥?/button>
            <span id="dailyMonth">--</span>
            <button class="month-btn" id="dailyNextMonth" onclick="shiftDailyMonth(1)">鈥?/button>
          </div>
          <div class="daily-week"><span>鏃?/span><span>涓€</span><span>浜?/span><span>涓?/span><span>鍥?/span><span>浜?/span><span>鍏?/span></div>
          <div class="daily-calendar" id="dailyCalendar"></div>
        </div>
        <div class="left-spacer"></div>
        <div class="chart-wrap">
          <div class="chart-head">
            <div class="chart-title-row">
              <h2>浠婃棩鎬绘敹鐩?/h2>
              <strong id="pnlHeaderValue">0.00U</strong>
            </div>
            <span>姣?5鍒嗛挓閲囨牱锛屼笉鍚湰閲?/span>
          </div>
          <canvas id="pnlChart" width="680" height="420"></canvas>
        </div>
      </div>
      <div class="grid">
        <div class="card">
          <h2 style="font-size:16px;margin:0 0 10px;">鎸佷粨</h2>
          <div class="scroll"><table id="positions"></table></div>
        </div>
        <div class="card">
          <h2 style="font-size:16px;margin:0 0 10px;">绛栫暐淇″彿</h2>
          <div class="scroll"><table id="signals"></table></div>
        </div>
        <div class="card">
          <h2 style="font-size:16px;margin:0 0 10px;">鎴愪氦璁板綍</h2>
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
    const fillsPageSize = 6;
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
      ENTRY_LONG: '寮€澶?,
      ENTRY_SHORT: '寮€绌?,
      EXIT_LONG: '骞冲',
      EXIT_SHORT: '骞崇┖',
      WATCH: '瑙傚療',
      NO_TRADE: '涓嶄氦鏄?,
      OPEN: '寮€浠?,
      CLOSE: '骞充粨',
      LONG: '澶?,
      SHORT: '绌?
    };
    const regimeText = {
      TREND_LONG: '澶氬ご瓒嬪娍',
      TREND_SHORT: '绌哄ご瓒嬪娍',
      CHOP: '闇囪崱',
      OVERCROWDED: '鎷ユ尋',
      INSUFFICIENT_DATA: '鏁版嵁涓嶈冻'
    };
    const smartMoneyPhaseText = {
      NEUTRAL: '涓€?,
      ACCUMULATION_REBUILD: '鍚哥閲嶅缓',
      SHORT_SQUEEZE_MARKUP: '閫肩┖鎷夊崌',
      DISTRIBUTION_EXIT: '娲惧彂绂诲満',
      TRAPPED_LONGS_MARKDOWN: '濂楀闃磋穼',
      CAPITULATION_ABSORB: '鏉€璺屽惛绛?
    };
    const trendStateText = {
      CHOP: '闇囪崱',
      TREND_LONG: '澶氬ご瓒嬪娍',
      TREND_SHORT: '绌哄ご瓒嬪娍',
      ONE_WAY_UP: '鍗曡竟涓婃定',
      ONE_WAY_DOWN: '鍗曡竟涓嬭穼'
    };
    const riskStateText = {
      NORMAL: '姝ｅ父',
      LONG_CROWD: '澶氬ご鎷ユ尋',
      SHORT_CROWD: '绌哄ご鎷ユ尋',
      OI_ABNORMAL: 'OI寮傚父',
      FUNDING_HOT: '璧勯噾璐圭巼杩囩儹'
    };
    const reasonText = {
      'score below trading threshold': '璇勫垎浣庝簬浜ゆ槗闃堝€?,
      'both sides failed hard filters': '澶氱┖鍙屾柟閮芥湭閫氳繃纭€ц繃婊?,
      'not enough candles for MA trend filter': 'K绾挎暟閲忎笉瓒筹紝鏃犳硶璁＄畻 MA 瓒嬪娍杩囨护',
      'latest candle lacks required indicators': '鏈€鏂癒绾跨己灏戝繀瑕佹寚鏍?,
      'EMA20 above EMA50 and EMA50 rising': 'EMA20 浣嶄簬 EMA50 涓婃柟锛屼笖 EMA50 涓婅',
      'EMA20 below EMA50 and EMA50 falling': 'EMA20 浣嶄簬 EMA50 涓嬫柟锛屼笖 EMA50 涓嬭',
      'EMA20 above EMA50 above EMA200': 'EMA20 > EMA50 > EMA200锛屼笁绾垮澶存帓鍒?,
      'EMA20 below EMA50 below EMA200': 'EMA20 < EMA50 < EMA200锛屼笁绾跨┖澶存帓鍒?,
      'price above MA100 trend filter': '浠锋牸浣嶄簬 MA100 涓婃柟锛岀鍚堝澶磋秼鍔胯繃婊?,
      'price below MA100 trend filter': '浠锋牸浣嶄簬 MA100 涓嬫柟锛岀鍚堢┖澶磋秼鍔胯繃婊?,
      'close confirmed near EMA20/BOLL mid without chasing upper band': '鏀剁洏纭闈犺繎 EMA20 鎴栧竷鏋椾腑杞紝鏈拷楂樹笂杞?,
      'close confirmed failed retest near EMA20/BOLL mid': '鏀剁洏纭鍙嶆娊 EMA20 鎴栧竷鏋椾腑杞ㄥけ璐?,
      'BOLL mid confirmed for long continuation': '杩炵画鏀剁洏绔欎笂 BOLL 涓建锛屽澶村欢缁‘璁?,
      'BOLL mid confirmed for short continuation': '杩炵画鏀剁洏璺岀牬 BOLL 涓建锛岀┖澶村欢缁‘璁?,
      'volume confirms move without extreme blow-off': '鎴愪氦閲忕‘璁よ蛋鍔匡紝涓旀湭鍑虹幇鏋佺鏀鹃噺鍐查珮',
      'volume confirms sell pressure without capitulation chase': '鎴愪氦閲忕‘璁ゅ崠鍘嬶紝涓旀湭杩界┖鎭愭厡涓嬭穼',
      'volume is acceptable but not strong': '鎴愪氦閲忓皻鍙紝浣嗗己搴︿笉瓒?,
      '4h open interest confirms new money entering': '4灏忔椂 OI 澧炲姞杈惧埌闃堝€硷紝鏂拌祫閲戝叆鍦虹‘璁?,
      'open interest rising mildly with price': '浠锋牸閰嶅悎 OI 娓╁拰涓婂崌',
      'open interest rising mildly with falling price': '浠锋牸涓嬭穼涓?OI 娓╁拰涓婂崌',
      'open interest stable': 'OI 鍩烘湰绋冲畾',
      'long/short ratio is not overcrowded long': '澶氱┖姣旀湭鍑虹幇澶氬ご杩囧害鎷ユ尋',
      'long/short ratio is not overcrowded short': '澶氱┖姣旀湭鍑虹幇绌哄ご杩囧害鎷ユ尋',
      'top trader long/short ratio supports longs': '澶ф埛澶氱┖姣旀敮鎸佸仛澶?,
      'top trader long/short ratio supports shorts': '澶ф埛澶氱┖姣旀敮鎸佸仛绌?,
      'RSI in healthy long-trend range': 'RSI 澶勪簬鍋ュ悍澶氬ご鍖洪棿',
      'RSI in healthy short-trend range': 'RSI 澶勪簬鍋ュ悍绌哄ご鍖洪棿',
      'funding rate is not overheated for longs': '璧勯噾璐圭巼鏈澶氬ご杩囩儹',
      'funding rate is not overheated for shorts': '璧勯噾璐圭巼鏈绌哄ご杩囩儹',
      'funding rate is in long entry range': '璧勯噾璐圭巼澶勪簬澶氬崟鍏ュ満鍖洪棿',
      'funding rate is in short entry range': '璧勯噾璐圭巼澶勪簬绌哄崟鍏ュ満鍖洪棿',
      'RSI overheated for long entry': 'RSI 杩囩儹锛岀姝㈣拷澶?,
      'RSI oversold for short entry': 'RSI 瓒呭崠锛岀姝㈣拷绌?,
      'long side overcrowded': '澶氬ご杩囧害鎷ユ尋',
      'short side overcrowded': '绌哄ご杩囧害鎷ユ尋',
      'funding too hot for long entry': '璧勯噾璐圭巼杩囬珮锛岀姝㈣拷澶?,
      'funding too negative for short entry': '璧勯噾璐圭巼杩囦綆锛岀姝㈣拷绌?,
      'open interest spike risks liquidation sweep': 'OI 寮傚父鏆村锛屽瓨鍦ㄦ壂鎹?鐖嗕粨椋庨櫓',
      'price closed above upper BOLL; no chase': '浠锋牸鏀跺湪甯冩灄涓婅建澶栵紝绂佹杩介珮',
      'price closed below lower BOLL; no chase': '浠锋牸鏀跺湪甯冩灄涓嬭建澶栵紝绂佹杩界┖',
      'smart money accumulation: OI flushed into a 4h pocket, then rebuilt while price recovered': '涓诲姏鍚哥锛?灏忔椂 OI 鐖嗗噺褰㈡垚娲煎湴锛岄殢鍚?OI 鍥炲崌涓斾环鏍间慨澶?,
      'smart money accumulation after OI flush; avoid chasing shorts': 'OI 鐖嗗噺鍚庣枒浼煎惛绛癸紝閬垮厤杩界┖',
      'short squeeze markup: price and OI rise while long/short ratio falls, shorts are being trapped': '閫肩┖鎷夊崌锛氫环鏍煎拰 OI 鍚屽崌锛屽绌烘瘮涓嬮檷锛岀┖澶村紑濮嬭濂?,
      'short crowd is vulnerable to a squeeze': '绌哄ご鎷ユ尋锛屽瓨鍦ㄨ缁х画鎷夊崌鐖嗙┖椋庨櫓',
      'smart money distribution: repeated upper wicks with OI falling after markup': '涓诲姏娲惧彂锛氫笂娑ㄥ悗澶氭涓婃彃閽堬紝鍚屾椂 OI 鍥炶惤',
      'smart money distribution after upper wick sweeps': '涓婃彃閽堟壂鍗曞悗 OI 鍥炶惤锛岀枒浼间富鍔涚鍦?,
      'trapped longs markdown: price falls while OI and long/short ratio rise': '濂楀闃磋穼锛氫环鏍间笅璺岋紝浣?OI 鍜屽绌烘瘮缁х画涓婂崌',
      'trapped longs are increasing while price falls': '浠锋牸涓嬭穼鏃跺澶寸户缁嫢鎸わ紝閬垮厤鍋氬',
      'capitulation absorption: lower wick sweeps with OI flush and volume expansion': '鏉€璺屽惛绛癸細澶氭涓嬫彃閽堬紝OI 鐖嗗噺涓旀垚浜ら噺鏀惧ぇ',
      'capitulation OI flush; avoid late shorts': '涓嬭穼灏炬 OI 鐖嗗噺锛岄伩鍏嶈拷绌?,
      'strict long blocked: EMA20/EMA50/EMA200 not bullish': '涓ユ牸澶氬崟绂佹锛欵MA20/EMA50/EMA200 鏈澶存帓鍒?,
      'strict long blocked: BOLL mid not confirmed twice': '涓ユ牸澶氬崟绂佹锛氭湭杩炵画绔欎笂 BOLL 涓建',
      'strict long blocked: RSI not in 52-72': '涓ユ牸澶氬崟绂佹锛歊SI 涓嶅湪 52-72',
      'strict long blocked: volume below 1.5x average': '涓ユ牸澶氬崟绂佹锛氭垚浜ら噺浣庝簬鍧囬噺 1.5 鍊?,
      'strict long blocked: 4h OI increase below 3%': '涓ユ牸澶氬崟绂佹锛?灏忔椂 OI 澧炲箙涓嶈冻 3%',
      'strict long blocked: top long/short ratio below 1.1': '涓ユ牸澶氬崟绂佹锛氬ぇ鎴峰绌烘瘮浣庝簬 1.1',
      'strict long blocked: funding outside long range': '涓ユ牸澶氬崟绂佹锛氳祫閲戣垂鐜囦笉鍦ㄥ鍗曞尯闂?,
      'strict long blocked: EMA lines are too compressed': '涓ユ牸澶氬崟绂佹锛欵MA 涓夌嚎绮樺悎锛岃秼鍔夸笉鏄?,
      'strict long blocked: RSI neutral zone': '涓ユ牸澶氬崟绂佹锛歊SI 浣嶄簬涓€у尯',
      'strict long blocked: 1h candle amplitude above 5%': '涓ユ牸澶氬崟绂佹锛?灏忔椂鎸箙瓒呰繃 5%',
      'strict short blocked: EMA20/EMA50/EMA200 not bearish': '涓ユ牸绌哄崟绂佹锛欵MA20/EMA50/EMA200 鏈┖澶存帓鍒?,
      'strict short blocked: BOLL mid not confirmed twice': '涓ユ牸绌哄崟绂佹锛氭湭杩炵画璺岀牬 BOLL 涓建',
      'strict short blocked: RSI not in 28-48': '涓ユ牸绌哄崟绂佹锛歊SI 涓嶅湪 28-48',
      'strict short blocked: volume below 1.5x average': '涓ユ牸绌哄崟绂佹锛氭垚浜ら噺浣庝簬鍧囬噺 1.5 鍊?,
      'strict short blocked: 4h OI increase below 3%': '涓ユ牸绌哄崟绂佹锛?灏忔椂 OI 澧炲箙涓嶈冻 3%',
      'strict short blocked: top long/short ratio above 0.9': '涓ユ牸绌哄崟绂佹锛氬ぇ鎴峰绌烘瘮楂樹簬 0.9',
      'strict short blocked: funding outside short range': '涓ユ牸绌哄崟绂佹锛氳祫閲戣垂鐜囦笉鍦ㄧ┖鍗曞尯闂?,
      'strict short blocked: EMA lines are too compressed': '涓ユ牸绌哄崟绂佹锛欵MA 涓夌嚎绮樺悎锛岃秼鍔夸笉鏄?,
      'strict short blocked: RSI neutral zone': '涓ユ牸绌哄崟绂佹锛歊SI 浣嶄簬涓€у尯',
      'strict short blocked: 1h candle amplitude above 5%': '涓ユ牸绌哄崟绂佹锛?灏忔椂鎸箙瓒呰繃 5%',
      'market structure confirms long: breakout or retest held': '甯傚満缁撴瀯纭鍋氬锛氱獊鐮存垨鍥炶俯鍘嬪姏浣嶄笉鐮?,
      'market structure confirms short: breakdown or retest failed': '甯傚満缁撴瀯纭鍋氱┖锛氳穼鐮存垨鍙嶆娊鏀拺浣嶅け璐?,
      'market structure: resistance grind broke upward, shorts may be squeezed': '甯傚満缁撴瀯锛氬墠楂樺帇鍔涗綅纾ㄧ洏鍚庡悜涓婄獊鐮达紝鍙兘閫肩┖',
      'market structure: support grind broke downward, longs may be liquidated': '甯傚満缁撴瀯锛氬墠浣庢敮鎾戜綅纾ㄧ洏鍚庡悜涓嬭穼鐮达紝鍙兘鐖嗗',
      'MA cluster breakout up': '鍧囩嚎瀵嗛泦鍖哄悜涓婄獊鐮?,
      'MA cluster retest held near MA20': '绐佺牬鍧囩嚎瀵嗛泦鍖哄悗鍥炶俯MA20涓嶇牬',
      'MA cluster dense; wait for breakout or MA20 retest': '鍧囩嚎瀵嗛泦缂犵粫锛岀瓑寰呯獊鐮存垨鍥炶俯MA20纭',
      'MA cluster breakdown down': '鍧囩嚎瀵嗛泦鍖哄悜涓嬭穼鐮?,
      'MA cluster retest rejected near MA20': '璺岀牬鍧囩嚎瀵嗛泦鍖哄悗鍙嶆娊MA20澶辫触',
      'MA cluster dense; wait for breakdown or MA20 retest': '鍧囩嚎瀵嗛泦缂犵粫锛岀瓑寰呰穼鐮存垨鍙嶆娊MA20纭',
      'washout confirmed: downside wick swept support, OI dropped, close reclaimed key level': '娲楃洏纭锛氫笅鎻掗拡鎵牬鏀拺锛孫I涓嬮檷锛屾敹鐩樻敹鍥炲叧閿綅',
      'washout confirmed: upside wick swept resistance, OI dropped, close rejected key level': '娲楃洏纭锛氫笂鎻掗拡鎵繃鍘嬪姏锛孫I涓嬮檷锛屾敹鐩樿穼鍥炲叧閿綅',
      'downside sweep reclaimed support; stop-run filter favors long': '涓嬫彃閽堟壂鎹熷悗鏀跺洖鏀拺锛屽亸鍚戝仛澶?,
      'upside sweep rejected resistance; stop-run filter favors short': '涓婃彃閽堟壂鎹熷悗璺屽洖鍘嬪姏锛屽亸鍚戝仛绌?,
      'upper wick sweep rejected; avoid chasing long': '涓婃彃閽堝洖钀斤紝绂佹杩藉',
      'lower wick sweep reclaimed; avoid chasing short': '涓嬫彃閽堟敹鍥烇紝绂佹杩界┖',
      'extreme volatility: skip new long entry': '鏋佺娉㈠姩锛岀姝㈡柊寮€澶氬崟',
      'extreme volatility: skip new short entry': '鏋佺娉㈠姩锛岀姝㈡柊寮€绌哄崟',
      '1h trigger opposes long entry': '1灏忔椂瑙﹀彂鏂瑰悜鍙嶅鍋氬锛岀姝㈠紑澶?,
      '1h trigger opposes short entry': '1灏忔椂瑙﹀彂鏂瑰悜鍙嶅鍋氱┖锛岀姝㈠紑绌?,
      '1d bearish bias; long position size reduced': '鏃ョ嚎鍋忕┖锛屽仛澶氫粨浣嶅噺鍗?,
      '1d bullish bias; short position size reduced': '鏃ョ嚎鍋忓锛屽仛绌轰粨浣嶅噺鍗?,
      '1d bullish bias supports long': '鏃ョ嚎鍋忓锛屾敮鎸佸仛澶?,
      '1d bearish bias supports short': '鏃ョ嚎鍋忕┖锛屾敮鎸佸仛绌?,
      '4h structure supports upside': '4灏忔椂缁撴瀯鏀寔涓婅',
      '4h structure supports downside': '4灏忔椂缁撴瀯鏀寔涓嬭',
      '4h OI deleverage with price breakdown; avoid long entry': '4灏忔椂 OI 澶у箙鍘绘潬鏉嗕笖浠锋牸鐮翠綅锛岀姝㈠仛澶?,
      '4h OI deleveraged but 1h BOLL/EMA held; allow small long only': '4灏忔椂 OI 澶у箙鍘绘潬鏉嗕絾1灏忔椂涓建/EMA瀹堜綇锛屽彧鍏佽灏忎粨澶?,
      '4h OI deleveraged while long/short ratio rose; 1h support held, only tiny long allowed': '4灏忔椂 OI 澶ц穼涓斿绌烘瘮涓婂崌锛?灏忔椂鏀拺瀹堜綇锛屼粎鍏佽鏋佸皬浠撳',
      '4h OI rebounds after deleverage and price breaks out; strong long restored': '4灏忔椂 OI 鍘绘潬鏉嗗悗閲嶆柊鍥炲崌涓斾环鏍肩獊鐮达紝鎭㈠寮哄',
      '4h OI deleverage with price breakdown; short candidate improved': '4灏忔椂 OI 澶у箙鍘绘潬鏉嗕笖浠锋牸鐮翠綅锛屽仛绌哄€欓€夊寮?,
      '4h OI deleverage breakdown with failed bounce; short candidate improved': '4灏忔椂 OI 澶у箙鍘绘潬鏉嗗悗浠锋牸鐮翠綅锛屼笖鍙嶆娊鍘嬪姏澶辫触锛屽仛绌哄€欓€夊寮?,
      '4h OI deleverage breakdown; wait for resistance retest or upper-wick rejection before short': '4灏忔椂 OI 澶у箙鍘绘潬鏉嗗悗浠锋牸鐮翠綅锛屼絾涓嶈拷绌猴紝绛夊緟闃诲姏鍙嶆娊鎴栦笂鎻掗拡澶辫触',
      '4h OI deleveraged but 1h support held; avoid chasing short': '4灏忔椂 OI 澶у箙鍘绘潬鏉嗕絾1灏忔椂鏀拺瀹堜綇锛岄伩鍏嶈拷绌?,
      '1h breakout confirms long trigger': '1灏忔椂绐佺牬纭澶氬ご瑙﹀彂',
      '1h retest confirms long trigger': '1灏忔椂鍥炶俯纭澶氬ご瑙﹀彂',
      '1h fake_breakdown confirms long trigger': '1灏忔椂鍋囪穼鐮寸‘璁ゅ澶磋Е鍙?,
      '1h breakdown confirms short trigger': '1灏忔椂璺岀牬纭绌哄ご瑙﹀彂',
      '1h retest confirms short trigger': '1灏忔椂鍙嶆娊纭绌哄ご瑙﹀彂',
      '1h fake_breakout confirms short trigger': '1灏忔椂鍋囩獊鐮寸‘璁ょ┖澶磋Е鍙?,
      '1h BOLL/EMA pullback held with clean risk': '1灏忔椂 BOLL/EMA 鍥炶俯涓嶇牬锛岄闄╁共鍑€锛屾敮鎸佸仛澶?,
      '1h BOLL/EMA pullback rejected with clean risk': '1灏忔椂 BOLL/EMA 鍙嶆娊澶辫触锛岄闄╁共鍑€锛屾敮鎸佸仛绌?,
      'high pullback with OI/funding/crowd risk; avoid long entry': '楂樹綅鍥炶俯鍙犲姞 OI/璧勯噾璐圭巼/鎷ユ尋椋庨櫓锛岀姝㈠紑澶?,
      'low pullback with OI/funding/crowd risk; avoid short entry': '浣庝綅鍙嶆娊鍙犲姞 OI/璧勯噾璐圭巼/鎷ユ尋椋庨櫓锛岀姝㈠紑绌?,
      'high area without pullback confirmation; wait for 1h/4h pullback before long': '楂樹綅鏈畬鎴愬洖韪╃‘璁わ紝绛夊緟1灏忔椂/4灏忔椂鍥炶皟鍚庡啀鍋氬',
      'low area without bounce confirmation; wait for 1h/4h retest before short': '浣庝綅鏈畬鎴愬弽鎶界‘璁わ紝绛夊緟1灏忔椂/4灏忔椂鍙嶆娊鍚庡啀鍋氱┖',
      'one-way uptrend 15m BOLL/EMA9 pullback confirmed; allow tactical long': '鍗曡竟涓婃定涓?5鍒嗛挓BOLL涓建/EMA9鍥炶俯纭锛屽厑璁告垬鏈仛澶?,
      'one-way downtrend 15m BOLL/EMA9 bounce rejected; allow tactical short': '鍗曡竟涓嬭穼涓?5鍒嗛挓BOLL涓建/EMA9鍙嶆娊澶辫触锛屽厑璁告垬鏈仛绌?,
      'multi-timeframe context neutral': '澶氬懆鏈熺粨鏋勪腑鎬?,
      'manual dashboard': '鎵嬪姩闈㈡澘',
      'manual close': '鎵嬪姩骞充粨',
      'stop loss': '姝㈡崯',
      'take profit': '姝㈢泩',
      'take profit 2': '姝㈢泩2',
      'trend invalidation exit': '瓒嬪娍澶辨晥骞充粨',
      'floating profit trailing stop': '娴泩鍥炴挙姝㈢泩',
      'structure break stop': '缁撴瀯鐮翠綅姝㈡崯',
      'ATR volatility stop': 'ATR娉㈠姩姝㈡崯',
      'take profit: target 2 reached': '姝㈢泩锛氳揪鍒扮浜屾鐩堢洰鏍?,
      'take profit: floating profit trailing stop': '姝㈢泩锛氭诞鐩堝洖鎾よЕ鍙戜繚鎶?,
      'take profit: protected stop after profit lock': '姝㈢泩锛氱泩鍒╁悗淇濇姢姝㈡崯瑙﹀彂',
      'stop loss: protected stop slipped below entry': '姝㈡崯锛氫繚鎶ゆ鎹熸垚浜ゅ悗浠嶄綆浜庡紑浠撲环',
      'take profit: breakout protection stop': '姝㈢泩锛氱獊鐮翠繚鎶ゆ鎹熻Е鍙?,
      'stop loss: breakout protection stop': '姝㈡崯锛氱獊鐮翠繚鎶ゆ鎹熻Е鍙?,
      'stop loss: signal structure failed': '姝㈡崯锛氫俊鍙风粨鏋勫け鏁?,
      'stop loss: signal direction or structure failed': '姝㈡崯锛氫俊鍙锋柟鍚戞垨缁撴瀯澶辨晥',
      'stop loss: ATR volatility hard stop': '姝㈡崯锛欰TR娉㈠姩纭鎹?,
      'stop loss: 15m entry structure stop': '姝㈡崯锛?5鍒嗛挓鍏ュ満缁撴瀯澶辨晥',
      'take profit: 1h/4h body closed below support or EMA/BOLL zone': '姝㈢泩锛?灏忔椂/4灏忔椂瀹炰綋璺岀牬鏀拺鎴朎MA/BOLL鍖哄煙锛屼繚鎶ゅ埄娑?,
      'stop loss: 1h/4h body closed below support or EMA/BOLL zone': '姝㈡崯锛?灏忔椂/4灏忔椂瀹炰綋璺岀牬鏀拺鎴朎MA/BOLL鍖哄煙',
      'take profit: 1h/4h body closed above resistance or EMA/BOLL zone': '姝㈢泩锛?灏忔椂/4灏忔椂瀹炰綋绐佺牬鍘嬪姏鎴朎MA/BOLL鍖哄煙锛屼繚鎶ゅ埄娑?,
      'stop loss: 1h/4h body closed above resistance or EMA/BOLL zone': '姝㈡崯锛?灏忔椂/4灏忔椂瀹炰綋绐佺牬鍘嬪姏鎴朎MA/BOLL鍖哄煙',
      'take profit: strong trend EMA50 structure invalidated': '姝㈢泩锛氬己瓒嬪娍EMA50缁撴瀯澶辨晥锛屼繚鎶ゅ埄娑?,
      'stop loss: strong trend EMA50 structure invalidated': '姝㈡崯锛氬己瓒嬪娍EMA50缁撴瀯澶辨晥',
      'take profit: floating profit drawdown protection': '姝㈢泩锛氭诞鐩堝洖鎾や繚鎶?,
      'stop loss: floating profit drawdown protection': '姝㈡崯锛氭诞鐩堝洖鎾ゅ悗杞簭绂诲満',
      'take profit: near 4h resistance with profit protection': '姝㈢泩锛氶潬杩?灏忔椂鍘嬪姏浣嶏紝淇濇姢鍒╂鼎',
      'take profit: near 4h support with profit protection': '姝㈢泩锛氶潬杩?灏忔椂鏀拺浣嶏紝淇濇姢鍒╂鼎',
      'stop loss: near 4h resistance with profit protection': '姝㈡崯锛氶潬杩?灏忔椂鍘嬪姏浣嶅悗杞急',
      'stop loss: near 4h support with profit protection': '姝㈡崯锛氶潬杩?灏忔椂鏀拺浣嶅悗杞急',
      'take profit: 4h support plus short exhaustion confirmed': '姝㈢泩锛?灏忔椂鏀拺浣嶅嚭鐜扮┖澶磋“绔‘璁?,
      'stop loss: 4h support plus short exhaustion confirmed': '姝㈡崯锛?灏忔椂鏀拺浣嶅弽寮圭‘璁ゅ悗绂诲満',
      'take profit: short trend support protection stop': '姝㈢泩锛氱┖澶磋秼鍔挎敮鎾戜綅淇濇姢姝㈡崯瑙﹀彂',
      'stop loss: short trend support protection stop': '姝㈡崯锛氱┖澶磋秼鍔挎敮鎾戜綅淇濇姢姝㈡崯瑙﹀彂',
      'take profit: profit drawdown after long crowd risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔犲澶存嫢鎸ら闄?,
      'take profit: profit drawdown after short crowd risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔犵┖澶存嫢鎸ら闄?,
      'take profit: profit drawdown after OI abnormal risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔燨I寮傚父椋庨櫓',
      'take profit: profit drawdown after funding overheated risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔犺祫閲戣垂鐜囪繃鐑?,
      'take profit: profit drawdown after OI drop risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔燨I涓嬮檷椋庨櫓',
      'take profit: profit drawdown after volume blow-off risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔犳斁閲忚“绔闄?,
      'take profit: profit drawdown after RSI overheated risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔燫SI杩囩儹',
      'take profit: profit drawdown after RSI oversold risk': '姝㈢泩锛氭诞鐩堝洖鎾ゅ彔鍔燫SI瓒呭崠',
      'take profit: long crowd risk': '姝㈢泩锛氬澶存嫢鎸ら闄╋紝淇濇姢鍒╂鼎',
      'stop loss: long crowd risk': '姝㈡崯锛氬澶存嫢鎸ら闄?,
      'take profit: short crowd risk': '姝㈢泩锛氱┖澶存嫢鎸ら闄╋紝淇濇姢鍒╂鼎',
      'stop loss: short crowd risk': '姝㈡崯锛氱┖澶存嫢鎸ら闄?,
      'take profit: OI abnormal risk': '姝㈢泩锛歄I寮傚父椋庨櫓锛屼繚鎶ゅ埄娑?,
      'stop loss: OI abnormal risk': '姝㈡崯锛歄I寮傚父椋庨櫓',
      'take profit: funding overheated risk': '姝㈢泩锛氳祫閲戣垂鐜囪繃鐑闄╋紝淇濇姢鍒╂鼎',
      'stop loss: funding overheated risk': '姝㈡崯锛氳祫閲戣垂鐜囪繃鐑闄?,
      'crowded one-way exit': '鍗曡竟琛屾儏鏁ｆ埛鎷ユ尋锛屼富鍔ㄧ鍦?
    };
    const riskExitReasonText = {
      LONG_CROWD: '椋庨櫓锛氬澶存嫢鎸ゅ钩浠?,
      SHORT_CROWD: '椋庨櫓锛氱┖澶存嫢鎸ゅ钩浠?,
      OI_ABNORMAL: '椋庨櫓锛歄I寮傚父骞充粨',
      FUNDING_HOT: '椋庨櫓锛氳祫閲戣垂鐜囪繃鐑钩浠?
    };
    const mtfText = {
      BULL: '鍋忓',
      BEAR: '鍋忕┖',
      NEUTRAL: '涓€?,
      UNKNOWN: '鏈煡',
      BREAKOUT_UP: '鍚戜笂绐佺牬',
      BREAKDOWN_DOWN: '鍚戜笅璺岀牬',
      BOX_UPPER_HALF: '绠变綋涓婂崐鍖?,
      BOX_LOWER_HALF: '绠变綋涓嬪崐鍖?,
      WAIT: '绛夊緟',
      BREAKOUT: '绐佺牬',
      BREAKDOWN: '璺岀牬',
      RETEST: '鍥炶俯/鍙嶆娊',
      FAKE_BREAKOUT: '鍋囩獊鐮?,
      FAKE_BREAKDOWN: '鍋囪穼鐮?,
      HEALTHY_PULLBACK: '鍋ュ悍鍥炶俯',
      HIGH_PULLBACK: '楂樹綅鍥炶俯',
      LOW_PULLBACK: '浣庝綅鍙嶆娊',
      NORMAL: '姝ｅ父',
      DELEVERAGE_WAIT: 'OI鍘绘潬鏉嗙瓑寰?,
      DELEVERAGE_HOLD_LONG: 'OI鍘绘潬鏉嗗畧浣忔敮鎾?,
      DELEVERAGE_CROWD_HOLD_LONG: 'OI鍘绘潬鏉嗕笖澶氬ご鎷ユ尋浣嗗畧浣忔敮鎾?,
      DELEVERAGE_BREAKDOWN: 'OI鍘绘潬鏉嗗悗浠锋牸鐮翠綅',
      DELEVERAGE_CROWD_BREAKDOWN: 'OI鍘绘潬鏉嗕笖澶氬ご鎷ユ尋鍚庣牬浣?,
      DELEVERAGE_CROWD_WAIT: 'OI鍘绘潬鏉嗕笖澶氬ご鎷ユ尋绛夊緟',
      REBUILD_BREAKOUT_LONG: 'OI鍥炲崌骞剁獊鐮?,
      DENSE: '鍧囩嚎瀵嗛泦',
      SPREAD: '鍧囩嚎鍙戞暎',
      RETEST_UP: '鍥炶俯鍧囩嚎涓嶇牬',
      RETEST_DOWN: '鍙嶆娊鍧囩嚎澶辫触',
      LONG: '澶氬ご',
      SHORT: '绌哄ご',
      NONE: '鏃犳柟鍚?
    };
    function tAction(value) { return actionText[value] || value || ''; }
    function tRegime(value) { return regimeText[value] || value || ''; }
    function tSmartMoneyPhase(value) { return smartMoneyPhaseText[value] || value || '涓€?; }
    function tTrendState(value) { return trendStateText[value] || value || '闇囪崱'; }
    function tRiskState(value) { return riskStateText[value] || value || '姝ｅ父'; }
    function tReason(value) {
      if (!value) return '';
      if (reasonText[value]) return reasonText[value];
      const reason = String(value);
      const maPrefixes = [
        ['MA cluster breakout up', '鍧囩嚎瀵嗛泦鍖哄悜涓婄獊鐮?],
        ['MA cluster retest held near MA20', '绐佺牬鍧囩嚎瀵嗛泦鍖哄悗鍥炶俯MA20涓嶇牬'],
        ['MA cluster dense; wait for breakout or MA20 retest', '鍧囩嚎瀵嗛泦缂犵粫锛岀瓑寰呯獊鐮存垨鍥炶俯MA20纭'],
        ['MA cluster breakdown down', '鍧囩嚎瀵嗛泦鍖哄悜涓嬭穼鐮?],
        ['MA cluster retest rejected near MA20', '璺岀牬鍧囩嚎瀵嗛泦鍖哄悗鍙嶆娊MA20澶辫触'],
        ['MA cluster dense; wait for breakdown or MA20 retest', '鍧囩嚎瀵嗛泦缂犵粫锛岀瓑寰呰穼鐮存垨鍙嶆娊MA20纭']
      ];
      for (const [prefix, text] of maPrefixes) {
        if (reason.startsWith(prefix)) return reason.replace(prefix, text).replace('price=', '锛屽瘑闆嗕环=');
      }
      const dynamicPrefixes = [
        ['VWAP pullback held; average cost support favors long', 'VWAP鍥炶俯涓嶇牬锛屽钩鍧囨垚鏈敮鎾戝澶?],
        ['KC mid pullback held; volatility channel support favors long', 'KC涓建鍥炶俯涓嶇牬锛屾尝鍔ㄩ€氶亾鏀拺澶氬ご'],
        ['QPS quote flow accelerates with price; traded value confirms long', '鎴愪氦棰濋€熷害鏀惧ぇ涓斾环鏍间笂琛岋紝纭澶氬ご'],
        ['QPS blow-off without price follow-through; long risk', '鎴愪氦棰濇斁澶т絾浠锋牸鏈窡杩涳紝澶氬ご椋庨櫓'],
        ['price extended far above VWAP; chasing long risk', '浠锋牸杩滅VWAP锛岃拷澶氶闄╁崌楂?],
        ['VWAP retest rejected; average cost resistance favors short', 'VWAP鍙嶆娊涓嶈繃锛屽钩鍧囨垚鏈帇鍒剁┖澶?],
        ['KC mid retest rejected; volatility channel resistance favors short', 'KC涓建鍙嶆娊涓嶈繃锛屾尝鍔ㄩ€氶亾鍘嬪埗绌哄ご'],
        ['QPS quote flow accelerates with price; traded value confirms short', '鎴愪氦棰濋€熷害鏀惧ぇ涓斾环鏍间笅琛岋紝纭绌哄ご'],
        ['QPS blow-off without price follow-through; short risk', '鎴愪氦棰濇斁澶т絾浠锋牸鏈窡杩涳紝绌哄ご椋庨櫓'],
        ['price extended far below VWAP; chasing short risk', '浠锋牸杩滅VWAP锛岃拷绌洪闄╁崌楂?],
        ['volume pattern confirms long: breakout volume, quiet retest, renewed buying', '鏀鹃噺绐佺牬銆佺缉閲忓洖韪┿€佸啀鏀鹃噺涓婅'],
        ['volume breakout above resistance; retest confirmation preferred', '鏀鹃噺绐佺牬鍘嬪姏锛岀瓑寰呭洖韪╃‘璁ゆ洿绋?],
        ['breakout retest held quietly; waiting renewed buying volume', '绐佺牬鍚庣缉閲忓洖韪╀笉鐮达紝绛夊緟鍐嶆斁閲?],
        ['volume pattern confirms short: breakdown volume, quiet retest, renewed selling', '鏀鹃噺璺岀牬銆佺缉閲忓弽鎶姐€佸啀鏀鹃噺涓嬭'],
        ['volume breakdown below support; retest confirmation preferred', '鏀鹃噺璺岀牬鏀拺锛岀瓑寰呭弽鎶界‘璁ゆ洿绋?],
        ['breakdown retest rejected quietly; waiting renewed selling volume', '璺岀牬鍚庣缉閲忓弽鎶戒笉杩囷紝绛夊緟鍐嶆斁閲?]
      ];
      for (const [prefix, text] of dynamicPrefixes) {
        if (reason.startsWith(prefix)) return text;
      }
      if (String(value).startsWith('risk exit:')) {
        const key = String(value).replace('risk exit:', '').trim();
        return riskExitReasonText[key] || `椋庨櫓锛?{key}骞充粨`;
      }
      if (String(value).startsWith('take profit:')) return String(value).replace('take profit:', '姝㈢泩锛?);
      if (String(value).startsWith('stop loss:')) return String(value).replace('stop loss:', '姝㈡崯锛?);
      if (String(value).startsWith('rotation exit:')) {
        const text = String(value);
        const symbol = (text.match(/symbol=([^\\s]+)/) || [])[1] || '';
        const score = (text.match(/score=(\\d+)/) || [])[1] || '';
        const type = text.includes('efficiency rotation') ? '鏁堢巼璋冧粨' : text.includes('trend invalidated') ? '瓒嬪娍澶辨晥璋冧粨' : '璋冧粨';
        const display = symbol ? displaySymbol(symbol) : '鏇村己鏍囩殑';
        return `${type}锛?浠撳凡婊★紝鎹㈠叆楂樿瘎鍒嗗己瓒嬪娍鏍囩殑 ${display}${score ? `锛岃瘎鍒?${score}` : ''}`;
      }
      if (String(value).startsWith('pyramid add:')) return String(value).replace('pyramid add:', '寮鸿秼鍔跨泩鍒╁洖韪╁姞浠擄細').replace('score=', '璇勫垎=').replace('state=', '鐘舵€?');
      if (String(value).startsWith('auto strategy score=')) return String(value).replace('auto strategy score=', '鑷姩绛栫暐寮€浠擄紝璇勫垎=');
      if (String(value).startsWith('MTF:')) return tMtfSummary(value);
      return value;
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
      return `澶氬懆鏈燂細鏃ョ嚎=${mtfText[d1] || d1}锛?灏忔椂=${mtfText[h4] || h4}锛?灏忔椂=${mtfText[h1] || h1}/${mtfText[h1Dir] || h1Dir}锛涘洖韪?${mtfText[pullback] || pullback}/${mtfText[pullbackDir] || pullbackDir}锛?H OI=${mtfText[oi4h] || oi4h}锛?H鍧囩嚎=${mtfText[ma4h] || ma4h}@${ma4hPrice}锛?H鍧囩嚎=${mtfText[ma1h] || ma1h}@${ma1hPrice}`;
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
        const parts = paragraph.split(/([锛?])/);
        let line = '';
        for (let i = 0; i < parts.length; i += 1) {
          const part = parts[i] || '';
          if (!part) continue;
          const next = line + part;
          if (next.length > limit && line) {
            lines.push(line);
            line = part.replace(/^[锛?]\\s*/, '');
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
    function tReasons(values) { return (values || []).map(tReason).join('锛?); }
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
      const match = text.match(/(?:price=|瀵嗛泦浠?|@)([0-9]+(?:\\.[0-9]+)?)/);
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
      if (direction === '澶氬ご') {
        return zoneLabel(h1s.support_zone_low, h1s.support_zone_high, h1s.support)
          || zoneLabel(h1.support_zone_low, h1.support_zone_high, null)
          || zoneLabel(h4.support_zone_low, h4.support_zone_high, h4.support)
          || fallbackPrice || '';
      }
      if (direction === '绌哄ご') {
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
      if (direction === '澶氬ご') {
        return zoneLabel(source.support_zone_low, source.support_zone_high, source.support)
          || (frame === '1h' ? zoneLabel(h1.support_zone_low, h1.support_zone_high, h1.support) : '')
          || fallbackPrice || '';
      }
      if (direction === '绌哄ご') {
        return zoneLabel(source.resistance_zone_low, source.resistance_zone_high, source.resistance)
          || (frame === '1h' ? zoneLabel(h1.resistance_zone_low, h1.resistance_zone_high, h1.resistance) : '')
          || fallbackPrice || '';
      }
      return fallbackPrice || '';
    }
    function atZone(text, zone) {
      return zone ? `${text}鈮?{zone}` : text;
    }
    function entrySideLevels(signal, direction) {
      const levels = objectValue(signal.entry_levels);
      if (direction === '澶氬ご') return objectValue(levels.long);
      if (direction === '绌哄ご') return objectValue(levels.short);
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
      return groups.slice(0, limit).map(group => `${group.labels.join('/')}鈮?{group.zone}`);
    }
    function conciseReason(reasons, signal = {}, options = {}) {
      const rawReasons = reasonList(reasons);
      const action = String(signal.action || '');
      const trend = String(signal.trend_state || signal.regime || '');
      let direction = '闇囪崱';
      if (action.includes('SHORT') || trend.includes('SHORT') || trend.includes('DOWN')) direction = '绌哄ご';
      if (action.includes('LONG') || trend.includes('LONG') || trend.includes('UP')) direction = '澶氬ご';

      const has1hHeld = reasonHas(rawReasons, ['1h BOLL/EMA pullback held', '1灏忔椂 BOLL/EMA 鍥炶俯涓嶇牬']);
      const has1hRejected = reasonHas(rawReasons, ['1h BOLL/EMA pullback rejected', '1灏忔椂 BOLL/EMA 鍙嶆娊澶辫触']);
      const has15mConfirm = reasonHas(rawReasons, ['15m pullback', '15鍒嗛挓鍥炶俯', '15m BOLL']);
      const hasVwapLong = reasonHas(rawReasons, ['VWAP pullback held']);
      const hasVwapShort = reasonHas(rawReasons, ['VWAP retest rejected']);
      const hasVwapLongRisk = reasonHas(rawReasons, ['far above VWAP']);
      const hasVwapShortRisk = reasonHas(rawReasons, ['far below VWAP']);
      const hasVolumeLongConfirm = reasonHas(rawReasons, ['volume pattern confirms long']);
      const hasVolumeShortConfirm = reasonHas(rawReasons, ['volume pattern confirms short']);
      const hasVolumeLongBreakout = reasonHas(rawReasons, ['volume breakout above resistance', 'breakout retest held quietly']);
      const hasVolumeShortBreakdown = reasonHas(rawReasons, ['volume breakdown below support', 'breakdown retest rejected quietly']);
      const maRetestUp = firstReason(rawReasons, ['MA cluster retest held near MA20', '鍧囩嚎瀵嗛泦鍖哄悗鍥炶俯MA20涓嶇牬']);
      const maRetestDown = firstReason(rawReasons, ['MA cluster retest rejected near MA20', '鍧囩嚎瀵嗛泦鍖哄弽鎶組A20澶辫触']);
      const maBreakUp = firstReason(rawReasons, ['MA cluster breakout up', '绐佺牬鍧囩嚎瀵嗛泦鍖?]);
      const maBreakDown = firstReason(rawReasons, ['MA cluster breakdown down', '璺岀牬鍧囩嚎瀵嗛泦鍖?]);
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
      const hasOiValley = h4OiState.includes('DELEVERAGE') || h4OiState.includes('REBUILD') || reasonHas(rawReasons, ['OI deleveraged', 'OI rebounds after deleverage', 'OI娲煎湴', 'OI 鍘绘潬鏉?]);
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
      if (maReason || activeClusterStates.includes(clusterState(signal))) maStructure = `涓婁竴涓潎绾垮瘑闆?{maZone ? `鈮?{maZone}` : ''}`;

      let structureLine = '鏂瑰悜涓庣粨鏋勶細闇囪崱锛岀瓑寰呰竟鐣岀‘璁ゃ€?;
      if (direction === '澶氬ご') {
        structureLine = `鏂瑰悜涓庣粨鏋勶細澶氬ご鏂瑰悜鎴愮珛锛氭棩绾夸笉绌猴紝4H缁撴瀯鍋忓锛?H绔欑ǔEMA20/EMA50鎴朆OLL涓建锛屼环鏍煎洖韪╂敮鎾戜綅${supportResistanceZone ? `鈮?{supportResistanceZone}` : ''}涓嶇牬骞堕噸鏂版敹鍥?{maStructure ? `锛?{maStructure}` : ''}銆俙;
        if (hasVolumeLongConfirm || hasVolumeLongBreakout) structureLine = structureLine.replace('銆?, '锛屾斁閲忕獊鐮村悗鍥炶俯涓嶇牬銆?);
      } else if (direction === '绌哄ご') {
        structureLine = `鏂瑰悜涓庣粨鏋勶細绌哄ご鏂瑰悜鎴愮珛锛氭棩绾夸笉澶氾紝4H缁撴瀯鍋忕┖锛?H璺岀牬EMA20/EMA50鎴朆OLL涓建锛屼环鏍煎弽鎶藉帇鍔涗綅${supportResistanceZone ? `鈮?{supportResistanceZone}` : ''}涓嶇牬骞堕噸鏂板洖钀?{maStructure ? `锛?{maStructure}` : ''}銆俙;
        if (hasVolumeShortConfirm || hasVolumeShortBreakdown) structureLine = structureLine.replace('銆?, '锛屾斁閲忚穼鐮村悗鍙嶆娊涓嶈繃銆?);
      }

      let entryBits = [];
      const levels = entrySideLevels(signal, direction);
      const candidates = [];
      if (direction === '澶氬ご') {
        if (isOneWayUp && has15mConfirm) addEntryCandidate(candidates, '寮哄崟杈?5m EMA20/EMA60鍥炶俯鏀跺洖', levels.m15_ema20_ema60);
        if (hasDownsideSweep) addEntryCandidate(candidates, '涓嬫彃閽堟壂鎹熷悗閲嶆柊鏀跺洖鏀拺', levels.sweep_reclaim_support);
        if (hasOiValley) addEntryCandidate(candidates, 'OI娲煎湴姝㈣穼鏀鹃噺鍥炵ǔ', levels.oi_valley_recovery);
        if (hasVwapLong) addEntryCandidate(candidates, 'VWAP/鎴愪氦瀵嗛泦鍖哄洖韪╀笉鐮?, levels.vwap_pullback);
        if (hasVolumeLongConfirm || hasVolumeLongBreakout) addEntryCandidate(candidates, '鍓嶅帇鍔涚獊鐮村悗鍥炶俯纭', levels.breakout_retest);
        if (has1hHeld) addEntryCandidate(candidates, '1H鏀拺鍥炶俯涓嶇牬', levels.h1_support);
        if (has1hHeld) addEntryCandidate(candidates, '1H BOLL涓建鍥炶俯涓嶇牬', levels.h1_boll_mid);
        if (maBreakUp) addEntryCandidate(candidates, '1H/4H K绾夸笂绌垮潎绾垮瘑闆?, levels.ma_cluster_breakout);
        if (maRetestUp) addEntryCandidate(candidates, '绐佺牬鍧囩嚎瀵嗛泦鍚庡洖韪㎝A20涓嶇牬', levels.ma20_retest);
        if (!candidates.length) {
          if (isOneWayUp) {
            addEntryCandidate(candidates, '寮哄崟杈?5m EMA20/EMA60鍥炶俯鏀跺洖', levels.m15_ema20_ema60);
            addEntryCandidate(candidates, '1H/4H EMA20鎴朎MA60鍥炶俯涓嶇牬', levels.h1_ema20_ema60);
            addEntryCandidate(candidates, '4H EMA20鎴朎MA60瓒嬪娍鍥炶俯', levels.h4_ema20_ema60);
          } else {
            addEntryCandidate(candidates, '1H鏀拺鍥炶俯涓嶇牬', levels.h1_support);
            addEntryCandidate(candidates, '1H BOLL涓建鍥炶俯涓嶇牬', levels.h1_boll_mid);
            addEntryCandidate(candidates, '鍓嶅帇鍔涚獊鐮村悗鍥炶俯纭', levels.breakout_retest);
          }
          addEntryCandidate(candidates, '鍧囩嚎瀵嗛泦鍖虹獊鐮存垨鍥炶俯MA20涓嶇牬', levels.ma_cluster_breakout || levels.ma20_retest);
        }
      } else if (direction === '绌哄ご') {
        if (hasUpsideSweep) addEntryCandidate(candidates, '涓婃彃閽堟壂绌哄悗閲嶆柊璺屽洖鍘嬪姏', levels.sweep_reject_resistance);
        if (hasOiValley) addEntryCandidate(candidates, '楂樹綅OI涓嬮檷妯洏娑ㄤ笉鍔?, levels.oi_distribution);
        if (hasVwapShort) addEntryCandidate(candidates, 'VWAP/鎴愪氦瀵嗛泦鍖哄弽鎶戒笉杩?, levels.vwap_retest);
        if (hasVolumeShortConfirm || hasVolumeShortBreakdown) addEntryCandidate(candidates, '鍓嶆敮鎾戣穼鐮村悗鍙嶆娊纭', levels.breakdown_retest);
        if (has1hRejected) addEntryCandidate(candidates, '1H鍘嬪姏鍙嶆娊涓嶈繃', levels.h1_resistance);
        if (has1hRejected) addEntryCandidate(candidates, '1H BOLL涓建鍙嶆娊澶辫触', levels.h1_boll_mid);
        if (maBreakDown) addEntryCandidate(candidates, '1H/4H K绾夸笅绌垮潎绾垮瘑闆?, levels.ma_cluster_breakdown);
        if (maRetestDown) addEntryCandidate(candidates, '璺岀牬鍧囩嚎瀵嗛泦鍚庡弽鎶組A20澶辫触', levels.ma20_retest);
        if (!candidates.length) {
          if (isOneWayDown) {
            addEntryCandidate(candidates, '1H/4H EMA20鎴朎MA60鍙嶆娊涓嶈繃', levels.h1_ema20_ema60);
            addEntryCandidate(candidates, '4H EMA20鎴朎MA60瓒嬪娍鍙嶆娊', levels.h4_ema20_ema60);
          } else {
            addEntryCandidate(candidates, '1H鍘嬪姏鍙嶆娊涓嶈繃', levels.h1_resistance);
            addEntryCandidate(candidates, '1H BOLL涓建鍙嶆娊澶辫触', levels.h1_boll_mid);
            addEntryCandidate(candidates, '鍓嶆敮鎾戣穼鐮村悗鍙嶆娊纭', levels.breakdown_retest);
          }
          addEntryCandidate(candidates, '鍧囩嚎瀵嗛泦鍖鸿穼鐮存垨鍙嶆娊MA20澶辫触', levels.ma_cluster_breakdown || levels.ma20_retest);
        }
      }
      entryBits = mergedEntryText(candidates);
      if (!entryBits.length) entryBits.push('杈圭晫纭澶勶細鏆傛棤鏈夋晥鍖洪棿');
      const entryLabel = options.entryLabel || '鍏ュ満浣嶇疆';
      const entryLine = `${entryLabel}锛?{entryBits.slice(0, 3).join('锛?)}`;

      const riskBits = [];
      if (oiDropFromHigh !== null && oiDropFromHigh <= -0.18) {
        riskBits.push(`OI楠ゅ噺${pctLabel(oiDropFromHigh)}`);
      } else if (reasonHas(rawReasons, ['open interest rising mildly', 'OI 娓╁拰涓婂崌'])) {
        riskBits.push('OI娓╁拰涓婂崌');
      } else if (reasonHas(rawReasons, ['open interest stable', 'OI 鍩烘湰绋冲畾', '4h OI=NORMAL'])) {
        riskBits.push('OI鍩烘湰绋冲畾');
      }
      if (volumeRatio !== null && volumeRatio >= 1.2) {
        const shortStuck = direction === '绌哄ご' && ['WAIT', 'FAKE_BREAKOUT'].includes(h1State) && ['BOX_UPPER_HALF', 'BREAKOUT_UP'].includes(h4State);
        const longStuck = direction === '澶氬ご' && ['WAIT', 'FAKE_BREAKDOWN'].includes(h1State) && ['BOX_LOWER_HALF', 'BREAKDOWN_DOWN'].includes(h4State);
        if (shortStuck) riskBits.push('鎴愪氦閲忔斁澶у悗浠锋牸1H妯洏娑ㄤ笉鍔?);
        else if (longStuck) riskBits.push('鎴愪氦閲忔斁澶у悗浠锋牸1H妯洏璺屼笉鍔?);
        else riskBits.push('鎴愪氦閲忔斁澶?);
      }
      if (hasVolumeLongConfirm) riskBits.push('绐佺牬鏀鹃噺锛屽洖韪╃缉閲忥紝鍐嶆斁閲忎笂琛?);
      if (hasVolumeShortConfirm) riskBits.push('璺岀牬鏀鹃噺锛屽弽鎶界缉閲忥紝鍐嶆斁閲忎笅琛?);
      if (hasVwapLong || hasVwapShort) riskBits.push('VWAP鎴愭湰浣嶇‘璁?);
      if (hasVwapLongRisk) riskBits.push('浠锋牸杩滅VWAP锛岃拷澶氶闄?);
      if (hasVwapShortRisk) riskBits.push('浠锋牸杩滅VWAP锛岃拷绌洪闄?);
      if (rsiValue !== null) {
        if (rsiValue >= 75 || rsiValue <= 25) riskBits.push(`RSI=${rsiValue.toFixed(0)}`);
        else if (reasonHas(rawReasons, ['RSI in healthy', 'RSI 澶勪簬鍋ュ悍'])) riskBits.push('RSI鍋ュ悍');
      } else if (reasonHas(rawReasons, ['RSI in healthy', 'RSI 澶勪簬鍋ュ悍'])) {
        riskBits.push('RSI鍋ュ悍');
      }
      if (direction === '澶氬ご' && hasDownsideSweep) riskBits.push('澶氭涓嬫彃閽堝凡娓呮潬鏉?);
      if (direction === '绌哄ご' && hasUpsideSweep) riskBits.push('澶氭涓婃彃閽堝凡娓呮潬鏉?);
      if (oiRebound !== null && oiRebound >= 0.003) riskBits.push('OI涓嬮檷鍚庡洖绋?);
      else if (hasOiValley && !(oiDropFromHigh !== null && oiDropFromHigh <= -0.18)) riskBits.push('OI涓嬮檷鍚庡洖绋?);
      if (riskState === 'LONG_CROWD') riskBits.push('澶氬ご鎯呯华鎷ユ尋');
      else if (riskState === 'SHORT_CROWD') riskBits.push('绌哄ご鎯呯华鎷ユ尋');
      else if (riskState === 'OI_ABNORMAL') riskBits.push('OI寮傚父');
      else if (riskState === 'FUNDING_HOT') riskBits.push('璧勯噾璐圭巼杩囩儹');
      else if (reasonHas(rawReasons, ['long/short ratio is not overcrowded', '澶氱┖姣旀湭鍑虹幇'])) riskBits.push('鎯呯华鏈嫢鎸?);
      const structureHeld = has1hHeld || has1hRejected || ['RETEST', 'BREAKOUT', 'BREAKDOWN', 'FAKE_BREAKOUT', 'FAKE_BREAKDOWN'].includes(h1State);
      if (structureHeld) riskBits.push('缁撴瀯鏈け鏁?);
      if (!riskBits.length) riskBits.push('椋庨櫓姝ｅ父');
      const riskLine = `鎸囨爣涓庨闄╋細${riskBits.join('锛?)}銆俙;

      return `${structureLine}\n${entryLine}\n${riskLine}`;
    }
    function signalReasonText(signal) {
      return conciseReason(signal.reasons || [], signal, { entryLabel: '寤鸿鍏ュ満浣嶇疆' });
    }
    function signalEntryTiming(signal) {
      if (signal.entry_timing) {
        const timingText = { GOOD: '浼樼', WAIT: '绛夊緟', BLOCK: '绂佹' };
        const timingClass = signal.entry_timing === 'GOOD' ? 'pos' : signal.entry_timing === 'BLOCK' ? 'neg' : 'muted';
        return `<span class="${timingClass}">${timingText[signal.entry_timing] || signal.entry_timing}</span>`;
      }
      const vetoes = reasonList(signal.vetoes || []);
      if (vetoes.length) return '绂佹';
      const score = Number(signal.score || 0);
      const action = String(signal.action || '');
      const reasons = reasonList(signal.reasons || []);
      const h1 = objectValue(signal.h1_trigger);
      const pullback = objectValue(signal.h1_pullback);
      const m15 = objectValue(signal.m15_precision);
      const h1Direction = String(h1.direction || '');
      const h1State = String(h1.state || '');
      const pullbackDirection = String(pullback.direction || '');
      const pullbackState = String(pullback.state || '');
      const m15Pullback = String(m15.pullback || '');
      const wantsLong = action === 'ENTRY_LONG' || action === 'WATCH' && reasonHas(reasons, ['EMA20 above EMA50', 'supports long', 'long']);
      const wantsShort = action === 'ENTRY_SHORT' || action === 'WATCH' && reasonHas(reasons, ['EMA20 below EMA50', 'supports short', 'short']);
      const longPositionReady =
        h1Direction === 'LONG' && ['BREAKOUT', 'RETEST', 'FAKE_BREAKDOWN'].includes(h1State) ||
        pullbackDirection === 'LONG' && ['HEALTHY_PULLBACK', 'HIGH_PULLBACK'].includes(pullbackState) ||
        m15Pullback === 'M15_LONG_PULLBACK' ||
        reasonHas(reasons, [
          'close confirmed near EMA20/BOLL mid',
          '1h BOLL/EMA pullback held',
          'VWAP pullback held',
          'volume pattern confirms long',
          'breakout retest held quietly',
          'K绾跨獊鐮村潎绾垮瘑闆?,
        ]);
      const shortPositionReady =
        h1Direction === 'SHORT' && ['BREAKDOWN', 'RETEST', 'FAKE_BREAKOUT'].includes(h1State) ||
        pullbackDirection === 'SHORT' && ['HEALTHY_PULLBACK', 'LOW_PULLBACK'].includes(pullbackState) ||
        reasonHas(reasons, [
          'close confirmed failed retest',
          '1h BOLL/EMA pullback rejected',
          'VWAP retest rejected',
          'volume pattern confirms short',
          'breakdown retest rejected quietly',
          'K绾胯穼鐮村潎绾垮瘑闆?,
        ]);
      const positionReady = wantsShort ? shortPositionReady : wantsLong ? longPositionReady : longPositionReady || shortPositionReady;
      if (score >= 82 && positionReady) return '浼樼';
      if (score >= 82) return wantsShort ? '绛夊弽鎶? : wantsLong ? '绛夊洖韪? : '绛夊緟浣嶇疆';
      if (positionReady) return '瑙傚療';
      return '绛夊緟';
    }
    function tEntryTimingReason(value) {
      const text = String(value || '');
      const map = {
        'entry timing not required': '鏃犻渶鍏ュ満鏃舵満鍒ゆ柇',
        'legacy signal without entry levels': '鏃т俊鍙风己灏戝叆鍦哄尯闂达紝鍏煎閫氳繃',
        'entry timing wait: latest price unavailable': '绛夊緟鏈€鏂颁环鏍?,
        'entry timing good: strong one-way 15m pullback/rejection confirmed': '寮哄崟杈癸紝15鍒嗛挓鍥炶俯/鍙嶆娊纭',
        'entry timing good: price is inside 1h/4h support, EMA/BOLL, VWAP, or retest entry zone': '浠锋牸宸插埌1H/4H鏀拺銆丒MA/BOLL銆乂WAP鎴栧洖韪╃‘璁ゅ尯',
        'entry timing good: price is inside 1h/4h resistance, EMA/BOLL, VWAP, or retest entry zone': '浠锋牸宸插埌1H/4H鍘嬪姏銆丒MA/BOLL銆乂WAP鎴栧弽鎶界‘璁ゅ尯',
        'entry timing good: 1h pullback held support/EMA/BOLL': '1H鍥炶俯鏀拺/EMA/BOLL涓嶇牬',
        'entry timing good: 1h bounce rejected resistance/EMA/BOLL': '1H鍙嶆娊鍘嬪姏/EMA/BOLL澶辫触',
        'entry timing good: 1h retest or fake breakdown reclaimed support': '1H鍥炶俯鎴栧亣璺岀牬鍚庢敹鍥炴敮鎾?,
        'entry timing good: 1h retest or fake breakout rejected resistance': '1H鍙嶆娊鎴栧亣绐佺牬鍚庤穼鍥炲帇鍔?,
        'entry timing wait: breakout needs pullback confirmation before fresh long': '绐佺牬鍚庣瓑寰呭洖韪╃‘璁ゅ啀鍋氬',
        'entry timing wait: breakdown needs resistance retest before fresh short': '璺岀牬鍚庣瓑寰呭弽鎶界‘璁ゅ啀鍋氱┖',
        'entry timing blocked: crowding/OI/funding risk not clean': '鎯呯华/OI/璧勯噾璐圭巼椋庨櫓涓嶅共鍑€',
        'entry timing blocked: late trend stage needs a new pullback': '瓒嬪娍鏈锛岀瓑寰呮柊鐨勫洖韪?鍙嶆娊纭',
        'entry timing blocked: reward/risk too low': '鐩爣绌洪棿涓嶈冻锛岀泩浜忔瘮涓嶅',
        'entry timing wait: long needs 1h/4h support, EMA/BOLL, VWAP, or breakout retest': '鍋氬绛夊緟1H/4H鏀拺銆丒MA/BOLL銆乂WAP鎴栫獊鐮村洖韪╁尯',
        'entry timing wait: short needs 1h/4h resistance, EMA/BOLL, VWAP, or breakdown retest': '鍋氱┖绛夊緟1H/4H鍘嬪姏銆丒MA/BOLL銆乂WAP鎴栬穼鐮村弽鎶藉尯'
      };
      return map[text] || text;
    }
    function entryReasonText(position) {
      const rawReason = String(position.reason || '');
      const rawEntryReason = String(position.entry_reason || '');
      const entryReasons = reasonList(position.entry_reasons || []);
      if (rawEntryReason === '鎵嬪姩' || rawEntryReason === '閹靛濮? || (!entryReasons.length && rawReason.toLowerCase().includes('manual'))) return '鎵嬪姩';
      const score = position.entry_score || (rawReason.match(/score=(\\d+)/) || [])[1] || '';
      const prefix = score ? `鑷姩锛岃瘎鍒嗭細${score}` : '鑷姩';
      if (entryReasons.length) {
        const sideAction = String(position.side || '').includes('SHORT') ? 'ENTRY_SHORT' : 'ENTRY_LONG';
        return `${prefix}锛沑n${conciseReason(entryReasons, { action: sideAction, ...(position.entry_context || {}) })}`;
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
        poll_seconds: 20
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
    function render(data) {
      const marketAge = data.market_updated_at ? (Date.now() - new Date(data.market_updated_at).getTime()) / 1000 : Infinity;
      const marketState = marketAge > 60 ? '琛屾儏寤惰繜' : '琛屾儏姝ｅ父';
      document.getElementById('updated').textContent = `${data.running ? '杩愯涓? : '宸插仠姝?} | 鑷姩绛栫暐 ${data.auto_trade ? '寮€' : '鍏?} | ${marketState} | ${timeText(data.market_updated_at || data.updated_at)}`;
      const metrics = [
        ['璧勯噾', money(data.equity) + ' U'],
        ['鍙敤', money(data.available_balance) + ' U'],
        ['鍗犵敤淇濊瘉閲?, money(data.used_margin) + ' U'],
        ['宸插疄鐜?, money(data.realized_pnl) + ' U'],
        ['鏈疄鐜?, money(data.unrealized_pnl) + ' U'],
        ['鎵嬬画璐?, '-' + money(data.fees_paid) + ' U'],
        ['鎬绘敹鐩?, money(data.total_pnl) + ' U / ' + pct(data.total_pnl_pct)]
      ];
      document.getElementById('metrics').innerHTML = metrics.map(([k,v]) => `<div class="card metric"><span>${k}</span><strong class="${k.includes('鏀剁泭') || k.includes('瀹炵幇') || k.includes('鎵嬬画璐?) ? pnlClass(String(v).split(' ')[0]) : ''}">${v}</strong></div>`).join('');
      document.getElementById('positions').className = 'center-table';
      document.getElementById('positions').innerHTML = table(['甯佺','鏂瑰悜','鏉犳潌','鍏ュ満','鐜颁环','鏁伴噺','淇濊瘉閲?,'娴泩浜?,'鏀剁泭鐜?,'姝㈡崯','姝㈢泩','鍏ュ満鍘熷洜','鎿嶄綔'], data.positions.map(p => [
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
        `<button class="neutral" onclick="closePosition('${displaySymbol(p.symbol)}')">骞充粨</button>`
      ]));
      const signalRows = Object.entries(data.latest_signals || {})
        .filter(([, s]) => {
          const reasons = s.reasons || [];
          return !(s.action === 'NO_TRADE' && reasons.includes('score below trading threshold'));
        })
        .map(([symbol, s]) => [displaySymbol(symbol), `<span class="pill">${tAction(s.action)}</span>`, tTrendState(s.trend_state || s.regime), tRiskState(s.risk_state), tSmartMoneyPhase(s.smart_money_phase), s.score, signalEntryTiming(s), wrapReason(signalReasonText(s), 100), tReasons(s.vetoes)]);
      document.getElementById('signals').className = 'signals-table';
      document.getElementById('signals').innerHTML = table(['甯佺','鍔ㄤ綔','鐘舵€?,'椋庨櫓','涓诲姏鍛ㄦ湡','鍒嗘暟','鍏ュ満鏃舵満','鍘熷洜','鍚﹀喅'], signalRows);
      const allFills = [...(data.fills || [])].filter(f => f.action === 'CLOSE' || f.closed_at).reverse();
      const totalFillPages = Math.max(Math.ceil(allFills.length / fillsPageSize), 1);
      fillsPage = Math.min(Math.max(fillsPage, 1), totalFillPages);
      const pageFills = allFills.slice((fillsPage - 1) * fillsPageSize, fillsPage * fillsPageSize);
      const fills = pageFills.map(f => [
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
        wrapReason(tReason(f.reason), 50)
      ]);
      document.getElementById('fills').className = 'center-table';
      document.getElementById('fills').innerHTML = fillsTable(['甯佺','鏂瑰悜','鏉犳潌','寮€浠撳潎浠?,'骞充粨鍧囦环','鏁伴噺','姝㈡崯','姝㈢泩','鏀剁泭鐜?,'瀹炵幇鐩堜簭','鎵嬬画璐?,'寮€浠撴椂闂?,'骞充粨鏃堕棿','鍑哄満鍘熷洜'], fills);
      renderFillsPager(totalFillPages);
      renderDailyPnl(data.daily_pnl);
      if (data.last_error) showError(data.last_error);
      else hideError();
      updatePnlHistory(data);
      drawPnlChart();
    }
    function table(headers, rows) {
      const classes = headers.map(tableClass);
      if (!rows.length) return `<thead><tr>${headers.map((h, i) => `<th class="${classes[i]}">${h}</th>`).join('')}</tr></thead><tbody><tr><td colspan="${headers.length}">鏆傛棤鏁版嵁</td></tr></tbody>`;
      return `<thead><tr>${headers.map((h, i) => `<th class="${classes[i]}">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((v, i) => `<td class="${classes[i]}">${v}</td>`).join('')}</tr>`).join('')}</tbody>`;
    }
    function fillsTable(headers, rows) {
      const classes = headers.map(tableClass);
      const widths = headers.map(fillColumnWidth);
      const colgroup = `<colgroup>${widths.map(width => `<col style="width:${width}">`).join('')}</colgroup>`;
      const head = `${colgroup}<thead><tr>${headers.map((h, i) => `<th class="${classes[i]}">${h}</th>`).join('')}</tr></thead>`;
      if (rows.length) {
        return `${head}<tbody>${rows.map(row => `<tr>${row.map((v, i) => `<td class="${classes[i]}">${v}</td>`).join('')}</tr>`).join('')}</tbody>`;
      }
      return `${head}<tbody>${Array.from({ length: fillsPageSize }).map((_, idx) => `<tr class="empty-fill-row"><td colspan="${headers.length}">${idx === 0 ? '鏆傛棤鏁版嵁' : '&nbsp;'}</td></tr>`).join('')}</tbody>`;
    }
    function fillColumnWidth(header) {
      if (header === '甯佺') return '7em';
      if (header === '鏂瑰悜') return '4em';
      if (header === '鏉犳潌') return '4em';
      if (['寮€浠撳潎浠?, '骞充粨鍧囦环', '鏁伴噺', '姝㈡崯', '姝㈢泩', '鏀剁泭鐜?, '瀹炵幇鐩堜簭'].includes(header)) return '6em';
      if (header === '鎵嬬画璐?) return '5em';
      if (header.includes('鏃堕棿')) return '8.2%';
      if (header === '鍑哄満鍘熷洜') return '50em';
      if (header === '鍘熷洜') return '30em';
      return '6em';
    }
    function tableClass(header) {
      if (header === '鍘熷洜' || header === '鍏ュ満鍘熷洜' || header === '鍑哄満鍘熷洜') return 'reason-col';
      if (header === '鍏ュ満鏃舵満') return 'timing-col';
      if (header === '鍚﹀喅') return 'veto-col';
      if (header.includes('鏃堕棿')) return 'time-col';
      if (['浠锋牸', '鍏ュ満', '鐜颁环', '鎴愪氦浠?, '寮€浠撳潎浠?, '骞充粨鍧囦环', '鏁伴噺', '淇濊瘉閲?, '鍑€鐩堜簭', '娴泩浜?, '鏀剁泭鐜?, '姝㈡崯', '姝㈡崯浠?, '姝㈢泩', '鏀剁泭鐜?, '瀹炵幇鐩堜簭', '鎵嬬画璐?, '鍒嗘暟'].includes(header)) return 'num-col';
      if (header === '甯佺') return 'symbol-col';
      if (['鏂瑰悜', '鍔ㄤ綔', '鐘舵€?, '椋庨櫓', '鎿嶄綔', '鏉犳潌'].includes(header)) return 'side-col';
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
        <button onclick="setFillsPage(${fillsPage - 1})" ${fillsPage <= 1 ? 'disabled' : ''}>涓婁竴椤?/button>
        ${items.join('')}
        <button onclick="setFillsPage(${fillsPage + 1})" ${fillsPage >= totalPages ? 'disabled' : ''}>涓嬩竴椤?/button>
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
      const pad = { left: 52 * dpr, right: 6 * dpr, top: 10 * dpr, bottom: 34 * dpr };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const plotBottom = pad.top + plotH;
      const timeTickTop = plotBottom + 3 * dpr;
      const timeLabelY = plotBottom + 12 * dpr;
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
    setInterval(load, 3000);
    window.addEventListener('resize', drawPnlChart);
  </script>
</body>
</html>
"""


app = create_app()
