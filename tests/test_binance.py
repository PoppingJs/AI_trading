from __future__ import annotations

import asyncio

import httpx

from ai_trading.binance import (
    BinanceFuturesMarketData,
    _get_json_with_retry,
    _websocket_proxy,
)


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


def test_top_trader_series_degrades_to_empty_without_api_key(
    monkeypatch,
) -> None:
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    market = BinanceFuturesMarketData(api_key=None)

    async def run() -> tuple[dict, dict]:
        return (
            await market.top_trader_account_ratio("BTCUSDT"),
            await market.top_trader_position_ratio("BTCUSDT"),
        )

    assert asyncio.run(run()) == ({}, {})


def test_taker_buy_sell_volume_parses_participant_flow_fields() -> None:
    class FakeMarket(BinanceFuturesMarketData):
        async def _get_json(self, path: str, **kwargs):  # type: ignore[override]
            assert path == "/futures/data/takerlongshortRatio"
            return [
                {
                    "timestamp": 1_700_000_000_000,
                    "buySellRatio": "1.12",
                    "buyVol": "112",
                    "sellVol": "100",
                }
            ]

    result = asyncio.run(
        FakeMarket().taker_buy_sell_volume("BTCUSDT")
    )
    ratio, buy_volume, sell_volume = next(iter(result.values()))

    assert ratio == 1.12
    assert buy_volume == 112.0
    assert sell_volume == 100.0
