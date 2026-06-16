from __future__ import annotations

import asyncio

import httpx

from ai_trading.binance import _get_json_with_retry


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
