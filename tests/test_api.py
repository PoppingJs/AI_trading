from __future__ import annotations

from fastapi.testclient import TestClient

from ai_trading.api import create_app


def test_api_health_and_demo_backtest() -> None:
    client = TestClient(create_app())

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["paper_trading"] is True

    backtest = client.post("/api/backtests/run", json={"symbol": "DEMOUSDT", "starting_equity": 10_000})
    assert backtest.status_code == 200
    payload = backtest.json()
    assert payload["symbol"] == "DEMOUSDT"
    assert payload["ending_equity"] > 0
