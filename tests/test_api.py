from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from ai_trading.api import create_app
from ai_trading.binance import FuturesSymbol
from ai_trading.models import Candle


def test_api_health_and_removed_legacy_surfaces(tmp_path: Path) -> None:
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["paper_trading"] is True
    assert client.post("/api/backtests/run", json={}).status_code == 404
    assert client.get("/api/review/summary").status_code == 404
    assert client.get("/review").status_code == 404


def test_demo_signal_exposes_explicit_direction_decision(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))

    payload = client.get("/api/signals/demo").json()

    assert payload["direction"] in {"LONG", "SHORT", "NONE"}
    assert "direction_decision" in payload
    assert "long_score" in payload
    assert "short_score" in payload


def test_api_token_protects_mutating_endpoints(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_TRADING_API_TOKEN", "secret")
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))
    assert client.post("/api/paper/stop").status_code == 401
    assert client.post("/api/paper/stop", headers={"X-API-Token": "secret"}).status_code == 200


def test_new_backtest_page_keeps_only_realtime_and_historical_navigation(tmp_path: Path) -> None:
    with TestClient(create_app(state_path=tmp_path / "paper_state.json")) as client:
        realtime = client.get("/")
        backtest = client.get("/backtest")
    assert realtime.status_code == backtest.status_code == 200
    assert 'href="/backtest"' in realtime.text
    assert "当前评分${values[1]}，低于本通道要求${values[2]}" in realtime.text
    assert "评分低于75" not in realtime.text
    assert "policy_blocks" in realtime.text
    assert "auto_entry_blocks" in realtime.text
    assert "研究模拟通道" in realtime.text
    assert "\u6781\u7aef\u6ce2\u52a8\u89c2\u5bdf\uff08\u6a21\u62df\u76d8\u4e0d\u963b\u65ad\uff09" in realtime.text
    assert "\u8bc4\u5206\u4f4e\u4e8e82" not in realtime.text
    assert 'href="/review"' not in realtime.text
    assert "历史回测" in backtest.text
    assert "交易复盘" not in backtest.text
    assert "开始回测时间" in backtest.text
    assert "分析总结" in backtest.text
    assert "等待启动" not in backtest.text
    assert "行情缓存" not in backtest.text
    assert "<span>完成交易</span>" not in backtest.text
    assert "后台固定使用实时交易同源配置" not in backtest.text
    assert "按亏损贡献汇总" not in backtest.text
    assert "grid-template-columns:460px minmax(0,1fr)" in backtest.text
    for label in ("资金", "可用", "占用保证金", "已实现", "未实现", "手续费", "胜率", "最大回撤", "总收益"):
        assert label in backtest.text
    assert "overflow-y:scroll" in backtest.text
    assert "symbol_summaries" in backtest.text
    assert 'class="day-tick"' in backtest.text
    assert "成交记录与失败归因" not in backtest.text
    assert "<h2>成交记录</h2>" in backtest.text
    assert "已完成成交" not in backtest.text
    assert "策略信号" not in backtest.text
    assert "<th>主力周期</th>" not in backtest.text
    assert "<th>失败根因</th>" not in backtest.text
    assert "<th>证据</th>" not in backtest.text
    assert "<th>操作</th>" not in backtest.text
    assert "closePosition(" not in backtest.text
    assert "结束时间仍持仓则按市值计入总权益" not in backtest.text
    assert "固定当前Top50 · 实际可用" not in backtest.text
    assert "跳过${result.skipped_symbols.length}个数据不完整币种" not in backtest.text
    assert "<tr><th>币种</th><th>方向</th><th>杠杆</th><th>入场</th><th>现价</th><th>数量</th><th>保证金</th><th>浮盈亏</th><th>收益率</th><th>止损</th><th>止盈</th><th>入场原因</th></tr>" in backtest.text
    assert "<tr><th>币种</th><th>方向</th><th>杠杆</th><th>开仓均价</th><th>平仓均价</th><th>数量</th><th>止损</th><th>止盈</th><th>收益率</th><th>实现盈亏</th><th>手续费</th><th>开仓时间</th><th>平仓时间</th><th>入场位置</th><th>出场原因</th></tr>" in backtest.text
    assert 'id="analysisSection"' in backtest.text
    assert 'id="analysisSection" hidden' not in backtest.text
    assert 'id="failureSummary"></div>' in backtest.text
    assert 'id="equityChart"></div>' in backtest.text
    assert "replayState" not in backtest.text
    assert "job.snapshot" in backtest.text
    assert "entryReasonText" in backtest.text
    assert "exitReasonText" in backtest.text
    assert ".filter(row=>row.action==='CLOSE')" not in backtest.text
    assert "回测中 ${job.progress||0}%" in backtest.text
    assert "本金" not in backtest.text
    assert "标的池" not in backtest.text
    assert "<label>周期</label>" not in backtest.text


def test_realtime_dashboard_reserves_height_for_pnl_time_axis(tmp_path: Path) -> None:
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))
    page = client.get("/").text
    assert ".metric { min-width: 0; padding: 2px 10px; }" in page
    assert "main { height: calc(100vh - 98px); padding: 6px 16px;" in page
    assert "#pnlChart { width: 100%; height: auto; min-height: 0; flex: 1 1 auto;" in page
    assert "const timeLabelY = plotBottom + 8 * dpr;" in page
    assert "data.new_entries_allowed ?? data.auto_trade" in page
    assert "MAX_DRAWDOWN: '最大回撤锁定'" in page


def test_realtime_dashboard_reads_split_market_context_states(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))

    page = client.get("/").text

    assert "marketDirectionText(s)" in page
    assert "marketRiskText(s)" in page
    assert "context.direction_state" in page
    assert "context.crowding_state" in page
    assert "context.liquidity_state" in page
    assert "context.system_risk_state" in page


def test_trade_exit_reasons_are_rendered_with_chinese_only_fallbacks(tmp_path: Path) -> None:
    with TestClient(create_app(state_path=tmp_path / "paper_state.json")) as client:
        realtime = client.get("/").text
        backtest = client.get("/backtest").text

    assert "wrapReason(exitReasonText(f.reason), 50)" in realtime
    assert "wrapReason(tReason(f.reason), 50)" not in realtime
    for english, chinese in (
        (
            "direction unvalidated and higher-timeframe structure failed",
            "方向尚未验证且高周期结构失效",
        ),
        (
            "structure close confirmed beyond the primary setup",
            "实体收盘确认突破主交易结构",
        ),
        ("target 1 reached", "第一止盈目标达成"),
        (
            "4h body closed below support or EMA/BOLL zone",
            "4小时实体跌破支撑或EMA/BOLL区域",
        ),
        (
            "4h body closed above resistance or EMA/BOLL zone",
            "4小时实体突破压力或EMA/BOLL区域",
        ),
        (
            "no 0.5r progress, price progress ineffective and structure did not advance",
            "未达到0.5R、价格推进无效且结构未继续发展",
        ),
    ):
        assert english in realtime
        assert chinese in realtime
        assert english.lower() in backtest.lower()
        assert chinese in backtest
    assert "return '止损：交易条件失效';" in realtime
    assert "return '止盈：达到目标或保护条件';" in realtime
    assert "return '策略退出：触发既定离场条件';" in realtime


def test_historical_job_replays_current_engine_and_reuses_latest_market_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("ai_trading.historical.UNIVERSE_SIZE", 1)
    monkeypatch.setattr("ai_trading.historical.MIN_USABLE_SYMBOLS", 1)
    factory = _FakeHistoricalFactory()
    app = create_app(
        state_path=tmp_path / "paper_state.json",
        historical_cache_root=tmp_path / "historical_cache",
        historical_market_data_factory=factory,
    )
    with TestClient(app) as client:
        first = _run_job(client)
        assert first["status"] == "COMPLETED", first
        assert first["result"]["summary"]["starting_equity"] == 1200.0
        assert "failure_summary" in first["result"]["analysis"]
        assert "lifecycles" not in first["result"]["analysis"]
        assert "failure_causes" not in first["result"]["analysis"]
        assert first["result"]["universe"] == ["TESTUSDT"]
        assert first["result"]["notes"][0].startswith("使用实时模拟交易")
        assert first["snapshot"] is not None
        assert "latest_signals" not in first["snapshot"]["account"]
        assert "positions" in first["snapshot"]["account"]
        assert "fills" in first["snapshot"]["account"]

        second = _run_job(client)
        assert second["status"] == "COMPLETED"
        assert second["cache_hit"] is True
        assert factory.top_calls == 1


def _run_job(client: TestClient) -> dict:
    submitted = client.post(
        "/api/backtests/jobs",
        json={"start_date": "2026-07-10", "end_date": "2026-07-10"},
    )
    assert submitted.status_code == 200, submitted.text
    payload = submitted.json()
    job_id = payload["id"]
    deadline = time.monotonic() + 20
    while payload["status"] in {"QUEUED", "RUNNING"} and time.monotonic() < deadline:
        time.sleep(0.02)
        payload = client.get(f"/api/backtests/jobs/{job_id}").json()
    return payload


class _FakeHistoricalFactory:
    def __init__(self) -> None:
        self.top_calls = 0

    def __call__(self):
        return _FakeHistoricalMarket(self)


class _FakeHistoricalMarket:
    def __init__(self, owner: _FakeHistoricalFactory) -> None:
        self.owner = owner

    async def top_usdt_perpetuals(self, limit: int = 20):
        self.owner.top_calls += 1
        return [
            FuturesSymbol(
                symbol="TESTUSDT",
                quote_asset="USDT",
                contract_type="PERPETUAL",
                status="TRADING",
                quote_volume=100_000_000.0,
                trade_count=1_000,
            )
        ]

    async def klines(
        self,
        symbol: str,
        interval: str = "15m",
        *,
        limit: int = 500,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ):
        seconds = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[interval]
        start = datetime.fromtimestamp((start_time_ms or 0) / 1000, tz=UTC)
        end = datetime.fromtimestamp((end_time_ms or 0) / 1000, tz=UTC)
        rows = []
        timestamp = start
        index = 0
        while timestamp < end and len(rows) < limit:
            base = 100.0 + math.sin(index / 9) * 1.5 + index * 0.002
            rows.append(
                Candle(
                    timestamp=timestamp,
                    open=base,
                    high=base * 1.003,
                    low=base * 0.997,
                    close=base * 1.0005,
                    volume=2_000_000.0,
                )
            )
            timestamp += timedelta(seconds=seconds)
            index += 1
        return rows

    async def open_interest_history(
        self,
        symbol,
        period="15m",
        *,
        limit=500,
        start_time_ms=None,
        end_time_ms=None,
    ):
        return self._series(period, limit, start_time_ms, end_time_ms, 10_000_000.0)

    async def global_long_short_ratio(
        self,
        symbol,
        period="15m",
        *,
        limit=500,
        start_time_ms=None,
        end_time_ms=None,
    ):
        return self._series(period, limit, start_time_ms, end_time_ms, 1.1)

    async def funding_rates(
        self,
        symbol,
        *,
        limit=100,
        start_time_ms=None,
        end_time_ms=None,
    ):
        return self._series("8h", limit, start_time_ms, end_time_ms, 0.0001)

    @staticmethod
    def _series(period, limit, start_time_ms, end_time_ms, value):
        seconds = {"15m": 900, "1h": 3600, "4h": 14400, "8h": 28800}[period]
        timestamp = datetime.fromtimestamp((start_time_ms or 0) / 1000, tz=UTC)
        end = datetime.fromtimestamp((end_time_ms or 0) / 1000, tz=UTC)
        result = {}
        while timestamp < end and len(result) < limit:
            result[timestamp] = value
            timestamp += timedelta(seconds=seconds)
        return result

    async def aclose(self):
        return None
