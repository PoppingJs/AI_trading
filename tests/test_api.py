from __future__ import annotations

from pathlib import Path
import time

from fastapi.testclient import TestClient

from ai_trading.api import create_app


def test_api_health_and_demo_backtest(tmp_path: Path) -> None:
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["paper_trading"] is True

    backtest = client.post("/api/backtests/run", json={"symbol": "DEMOUSDT", "starting_equity": 10_000})
    assert backtest.status_code == 200
    payload = backtest.json()
    assert payload["symbol"] == "DEMOUSDT"
    assert payload["ending_equity"] > 0


def test_api_token_protects_mutating_endpoints(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_TRADING_API_TOKEN", "secret")
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))

    rejected = client.post("/api/paper/stop")
    assert rejected.status_code == 401

    accepted = client.post("/api/paper/stop", headers={"X-API-Token": "secret"})
    assert accepted.status_code == 200


def test_dashboard_uses_concise_chinese_veto_copy(tmp_path: Path) -> None:
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))

    page = client.get("/").text

    assert "'directional entry signal not established': '多/空方向未成立'" in page
    assert "'symbol already has an open position': '已开仓'" in page
    assert "'symbol excluded from automatic universe':" not in page
    assert "'entry reward/risk target unavailable':" not in page
    assert "'extreme volatility: skip new long entry': '极端波动'" in page
    assert "'low area without 1h/4h resistance retest; wait for higher-timeframe bounce before short': '低位反抽未确认'" in page
    assert "return '其他风控条件未满足';" in page
    assert "tVetoes(s.vetoes)" in page
    assert "return '评分低于82';" in page
    assert "return '当前入场位置不优秀';" in page
    assert "return `${timeframeGap[1]}周期缺失或不连续`;" in page
    assert "signalEntryPosition(s)" in page
    assert "signalEntryTiming(s)" not in page
    assert "reasons.includes('score below trading threshold')" not in page
    assert "['币种','动作','状态','风险','主力周期','分数','入场位置','原因','否决']" in page


def test_backtest_and_review_pages_keep_realtime_navigation(tmp_path: Path) -> None:
    with TestClient(create_app(state_path=tmp_path / "paper_state.json")) as client:
        realtime = client.get("/")
        backtest = client.get("/backtest")
        review = client.get("/review")

    assert realtime.status_code == backtest.status_code == review.status_code == 200
    assert 'href="/backtest"' in realtime.text
    assert 'href="/review"' in realtime.text
    assert "历史回测" in backtest.text
    assert "交易复盘" in review.text


def test_realtime_dashboard_reserves_height_for_pnl_time_axis(tmp_path: Path) -> None:
    client = TestClient(create_app(state_path=tmp_path / "paper_state.json"))

    page = client.get("/").text

    assert ".metric { min-width: 0; padding: 2px 10px; }" in page
    assert "main { height: calc(100vh - 98px); padding: 6px 16px;" in page
    assert "#pnlChart { width: 100%; height: auto; min-height: 0; flex: 1 1 auto;" in page
    assert "const timeLabelY = plotBottom + 8 * dpr;" in page


def test_async_demo_backtest_job_and_review_summary(tmp_path: Path) -> None:
    with TestClient(create_app(state_path=tmp_path / "paper_state.json")) as client:
        submitted = client.post(
            "/api/backtests/jobs",
            json={
                "data_source": "demo",
                "symbol": "DEMOUSDT",
                "starting_equity": 10_000,
                "mode": "production",
                "base_timeframe": "15m",
            },
        )
        assert submitted.status_code == 200
        job_id = submitted.json()["id"]

        payload = submitted.json()
        deadline = time.monotonic() + 5
        while payload["status"] in {"QUEUED", "RUNNING"} and time.monotonic() < deadline:
            time.sleep(0.02)
            payload = client.get(f"/api/backtests/jobs/{job_id}").json()

        assert payload["status"] == "COMPLETED"
        assert payload["result"]["mode"] == "production"
        assert "analysis" in payload["result"]

        summary = client.get("/api/review/summary")
        assert summary.status_code == 200
        assert summary.json()["metrics"]["completed"] == 0
