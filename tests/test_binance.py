from __future__ import annotations

import asyncio

import httpx

from ai_trading.binance import _get_json_with_retry, _websocket_proxy


def test_binance_retry_handles_rate_limit_response() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as client:
            return await _get_json_with_retry(client, "/fapi/v1/exchangeInfo")

    assert asyncio.run(run()) == {"ok": True}
    assert attempts == 2


def test_websocket_proxy_uses_shared_all_proxy(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_WS_PROXY", raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")

    assert _websocket_proxy() == "socks5h://127.0.0.1:1080"


def test_websocket_proxy_prefers_dedicated_setting(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setenv("BINANCE_WS_PROXY", "socks5://127.0.0.1:2080")

    assert _websocket_proxy() == "socks5h://127.0.0.1:2080"
