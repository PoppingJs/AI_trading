from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from ai_trading.models import Candle, DerivativesSnapshot


FAPI_BASE_URL = "https://fapi.binance.com"


@dataclass(frozen=True)
class FuturesSymbol:
    symbol: str
    quote_asset: str
    contract_type: str
    status: str
    quote_volume: float
    trade_count: int


class BinanceFuturesMarketData:
    """Read-only helpers for Binance USDT-M market discovery.

    This class intentionally avoids authenticated trading endpoints. Live order
    placement should be added only after paper trading and risk tests pass.
    """

    def __init__(self, base_url: str = FAPI_BASE_URL, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def top_usdt_perpetuals(self, limit: int = 20) -> list[FuturesSymbol]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            exchange_info, tickers = await _fetch_exchange_and_tickers(client)

        valid_symbols = {
            item["symbol"]: item
            for item in exchange_info["symbols"]
            if item.get("quoteAsset") == "USDT"
            and item.get("contractType") == "PERPETUAL"
            and item.get("status") == "TRADING"
        }
        out: list[FuturesSymbol] = []
        for ticker in tickers:
            symbol = ticker.get("symbol")
            if symbol not in valid_symbols:
                continue
            out.append(
                FuturesSymbol(
                    symbol=symbol,
                    quote_asset="USDT",
                    contract_type="PERPETUAL",
                    status="TRADING",
                    quote_volume=float(ticker.get("quoteVolume", 0.0)),
                    trade_count=int(ticker.get("count", 0)),
                )
            )
        out.sort(key=lambda item: (item.quote_volume, item.trade_count), reverse=True)
        return out[:limit]

    async def klines(
        self,
        symbol: str,
        interval: str = "15m",
        *,
        limit: int = 500,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Candle]:
        params: dict[str, object] = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            rows = await _get_json_with_retry(client, "/fapi/v1/klines", params=params)
        return [
            Candle(
                timestamp=_from_ms(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
        ]

    async def open_interest_history(self, symbol: str, period: str = "15m", *, limit: int = 500) -> dict[datetime, float]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            rows = await _get_json_with_retry(
                client,
                "/futures/data/openInterestHist",
                params={"symbol": symbol.upper(), "period": period, "limit": limit},
            )
        return {_from_ms(row["timestamp"]): float(row["sumOpenInterest"]) for row in rows}

    async def global_long_short_ratio(self, symbol: str, period: str = "15m", *, limit: int = 500) -> dict[datetime, float]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            rows = await _get_json_with_retry(
                client,
                "/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol.upper(), "period": period, "limit": limit},
            )
        return {_from_ms(row["timestamp"]): float(row["longShortRatio"]) for row in rows}

    async def funding_rates(self, symbol: str, *, limit: int = 100) -> dict[datetime, float]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            rows = await _get_json_with_retry(client, "/fapi/v1/fundingRate", params={"symbol": symbol.upper(), "limit": limit})
        return {_from_ms(row["fundingTime"]): float(row["fundingRate"]) for row in rows}

    async def historical_bundle(self, symbol: str, interval: str = "15m", *, limit: int = 500) -> tuple[list[Candle], list[DerivativesSnapshot]]:
        candles = await self.klines(symbol, interval, limit=limit)
        oi_result, ratio_result, funding_result = await asyncio.gather(
            self.open_interest_history(symbol, interval, limit=min(limit, 500)),
            self.global_long_short_ratio(symbol, interval, limit=min(limit, 500)),
            self.funding_rates(symbol, limit=100),
            return_exceptions=True,
        )
        oi_by_time = oi_result if isinstance(oi_result, dict) else {}
        ratio_by_time = ratio_result if isinstance(ratio_result, dict) else {}
        funding_by_time = funding_result if isinstance(funding_result, dict) else {}
        funding_times = sorted(funding_by_time)

        derivatives: list[DerivativesSnapshot] = []
        for candle in candles:
            nearest_funding = _nearest_before(funding_times, candle.timestamp)
            derivatives.append(
                DerivativesSnapshot(
                    timestamp=candle.timestamp,
                    open_interest=oi_by_time.get(candle.timestamp),
                    long_short_ratio=ratio_by_time.get(candle.timestamp),
                    funding_rate=funding_by_time.get(nearest_funding) if nearest_funding else None,
                )
            )
        return candles, derivatives


async def _fetch_exchange_and_tickers(client: httpx.AsyncClient) -> tuple[dict, list[dict]]:
    exchange_info, tickers = await asyncio.gather(
        _get_json_with_retry(client, "/fapi/v1/exchangeInfo"),
        _get_json_with_retry(client, "/fapi/v1/ticker/24hr"),
    )
    return exchange_info, tickers


async def _get_json_with_retry(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, object] | None = None,
    attempts: int = 3,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.PoolTimeout) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(0.25 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _from_ms(value: int | str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _nearest_before(values: list[datetime], timestamp: datetime) -> datetime | None:
    nearest: datetime | None = None
    for value in values:
        if value > timestamp:
            break
        nearest = value
    return nearest
