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
    assert 'href="/review"' not in realtime.text
    assert "历史回测" in backtest.text
    assert "交易复盘" not in backtest.text
    assert "开始回测时间" in backtest.text
    assert "失败分析总结" in backtest.text
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
        assert first["result"]["universe"] == ["TESTUSDT"]
        assert first["result"]["notes"][0].startswith("使用实时模拟交易")

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
