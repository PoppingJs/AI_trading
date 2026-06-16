from __future__ import annotations

from pathlib import Path

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
