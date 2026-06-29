from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from typing import Any, AsyncIterator, Iterable

import httpx

from ai_trading.models import Candle, DerivativesSnapshot


FAPI_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
FSTREAM_BASE_URL = os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com")


class BinanceMarketDataError(RuntimeError):
    """Raised when Binance public market data cannot be fetched cleanly."""


@dataclass(frozen=True)
class FuturesSymbol:
    symbol: str
    quote_asset: str
    contract_type: str
    status: str
    quote_volume: float
    trade_count: int
    last_price: float | None = None
    price_change_percent: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    open_price: float | None = None


class BinanceFuturesMarketData:
    """Read-only helpers for Binance USDT-M market discovery.

    This class intentionally avoids authenticated trading endpoints. Live order
    placement should be added only after paper trading and risk tests pass.
    """

    def __init__(
        self,
        base_url: str = FAPI_BASE_URL,
        timeout: float = 10.0,
        request_concurrency: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_semaphore = asyncio.Semaphore(max(int(request_concurrency), 1))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> Any:
        client = await self._get_client()
        async with self._request_semaphore:
            return await _get_json_with_retry(client, path, params=params)

    async def top_usdt_perpetuals(self, limit: int = 20) -> list[FuturesSymbol]:
        exchange_info, tickers = await asyncio.gather(
            self._get_json("/fapi/v1/exchangeInfo"),
            self._get_json("/fapi/v1/ticker/24hr"),
        )

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
                    last_price=_optional_float(ticker.get("lastPrice")),
                    price_change_percent=_optional_float(ticker.get("priceChangePercent")),
                    high_price=_optional_float(ticker.get("highPrice")),
                    low_price=_optional_float(ticker.get("lowPrice")),
                    open_price=_optional_float(ticker.get("openPrice")),
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
        rows = await self._get_json("/fapi/v1/klines", params=params)
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
        rows = await self._get_json(
            "/futures/data/openInterestHist",
            params={"symbol": symbol.upper(), "period": period, "limit": limit},
        )
        return {_from_ms(row["timestamp"]): float(row["sumOpenInterest"]) for row in rows}

    async def global_long_short_ratio(self, symbol: str, period: str = "15m", *, limit: int = 500) -> dict[datetime, float]:
        rows = await self._get_json(
            "/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol.upper(), "period": period, "limit": limit},
        )
        return {_from_ms(row["timestamp"]): float(row["longShortRatio"]) for row in rows}

    async def funding_rates(self, symbol: str, *, limit: int = 100) -> dict[datetime, float]:
        rows = await self._get_json(
            "/fapi/v1/fundingRate",
            params={"symbol": symbol.upper(), "limit": limit},
        )
        return {_from_ms(row["fundingTime"]): float(row["fundingRate"]) for row in rows}

    async def current_funding_rates(self, symbols: Iterable[str] | None = None) -> dict[str, float]:
        """Return current predicted funding rates from Binance premium index."""
        rows = await self._get_json("/fapi/v1/premiumIndex")
        rows = rows if isinstance(rows, list) else [rows]
        wanted = {symbol.upper() for symbol in symbols} if symbols is not None else None
        rates: dict[str, float] = {}
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if not symbol or (wanted is not None and symbol not in wanted):
                continue
            rate = _optional_float(row.get("lastFundingRate"))
            if rate is not None:
                rates[symbol] = rate
        return rates

    async def mark_prices(self, symbols: Iterable[str] | None = None) -> dict[str, float]:
        """Return current USDT-M mark prices in one REST request.

        This is intentionally a fallback path. Normal real-time updates should
        come from ``stream_mark_prices``.
        """
        rows = await self._get_json("/fapi/v1/premiumIndex")
        rows = rows if isinstance(rows, list) else [rows]
        wanted = {symbol.upper() for symbol in symbols} if symbols is not None else None
        prices: dict[str, float] = {}
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if not symbol or (wanted is not None and symbol not in wanted):
                continue
            price = _optional_float(row.get("markPrice"))
            if price is not None and price > 0:
                prices[symbol] = price
        return prices

    async def ticker_prices(self, symbols: Iterable[str] | None = None) -> dict[str, float]:
        """Backward-compatible alias that now consistently returns mark prices."""
        return await self.mark_prices(symbols)

    async def derivatives_bundle(
        self,
        symbol: str,
        interval: str,
        candle_times: Iterable[datetime],
        *,
        include_funding: bool = True,
    ) -> list[DerivativesSnapshot]:
        """Refresh derivatives data without downloading K-lines again."""
        timestamps = sorted(set(candle_times))
        if not timestamps:
            return []
        limit = min(max(len(timestamps), 30), 500)
        funding_call = (
            self.current_funding_rates([symbol])
            if include_funding
            else _empty_mapping()
        )
        oi_result, ratio_result, funding_result = await asyncio.gather(
            self.open_interest_history(symbol, interval, limit=limit),
            self.global_long_short_ratio(symbol, interval, limit=limit),
            funding_call,
            return_exceptions=True,
        )
        if (
            not isinstance(oi_result, dict)
            or not isinstance(ratio_result, dict)
            or (include_funding and not isinstance(funding_result, dict))
        ):
            raise BinanceMarketDataError(f"{symbol.upper()} derivatives data is incomplete")
        oi_by_time = oi_result if isinstance(oi_result, dict) else {}
        ratio_by_time = ratio_result if isinstance(ratio_result, dict) else {}
        current_funding = (
            funding_result.get(symbol.upper())
            if isinstance(funding_result, dict)
            else None
        )
        return [
            DerivativesSnapshot(
                timestamp=timestamp,
                open_interest=oi_by_time.get(timestamp),
                long_short_ratio=ratio_by_time.get(timestamp),
                funding_rate=current_funding,
            )
            for timestamp in timestamps
        ]

    async def stream_mark_prices(self, symbols: Iterable[str]) -> AsyncIterator[dict[str, float]]:
        """Yield Binance mark-price updates for the requested symbols.

        ``websockets`` is imported lazily so REST-only deployments keep
        working even when the optional stream dependency is unavailable.
        """
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - deployment fallback
            raise BinanceMarketDataError("websockets dependency is unavailable") from exc

        streams = "/".join(
            f"{symbol.lower()}@markPrice@1s"
            for symbol in sorted({symbol.upper() for symbol in symbols if symbol})
        )
        if not streams:
            return
        url = f"{FSTREAM_BASE_URL.rstrip('/')}/stream?streams={streams}"
        async with websockets.connect(
            url,
            proxy=_websocket_proxy(),
            open_timeout=self.timeout,
            close_timeout=5,
            ping_interval=20,
            ping_timeout=20,
            max_queue=32,
        ) as websocket:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                data = message.get("data", message)
                rows = data if isinstance(data, list) else [data]
                prices: dict[str, float] = {}
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    symbol = str(row.get("s") or "").upper()
                    price = _optional_float(row.get("p"))
                    if symbol and price is not None and price > 0:
                        prices[symbol] = price
                if prices:
                    yield prices
    async def historical_bundle(self, symbol: str, interval: str = "15m", *, limit: int = 500) -> tuple[list[Candle], list[DerivativesSnapshot]]:
        candles = await self.klines(symbol, interval, limit=limit)
        oi_result, ratio_result, funding_result = await asyncio.gather(
            self.open_interest_history(symbol, interval, limit=min(limit, 500)),
            self.global_long_short_ratio(symbol, interval, limit=min(limit, 500)),
            self.current_funding_rates([symbol]),
            return_exceptions=True,
        )
        if (
            not isinstance(oi_result, dict)
            or not isinstance(ratio_result, dict)
            or not isinstance(funding_result, dict)
        ):
            raise BinanceMarketDataError(f"{symbol.upper()} historical bundle is incomplete")
        oi_by_time = oi_result
        ratio_by_time = ratio_result
        current_funding = funding_result.get(symbol.upper())

        derivatives: list[DerivativesSnapshot] = []
        for candle in candles:
            derivatives.append(
                DerivativesSnapshot(
                    timestamp=candle.timestamp,
                    open_interest=oi_by_time.get(candle.timestamp),
                    long_short_ratio=ratio_by_time.get(candle.timestamp),
                    funding_rate=current_funding,
                )
            )
        return candles, derivatives


def _websocket_proxy() -> str | bool:
    """Use the explicit proxy and resolve Binance DNS through the SSH tunnel."""
    proxy = os.getenv("BINANCE_WS_PROXY") or os.getenv("ALL_PROXY")
    if not proxy:
        return True
    if proxy.startswith("socks5://"):
        return f"socks5h://{proxy.removeprefix('socks5://')}"
    if proxy.startswith("socks4://"):
        return f"socks4a://{proxy.removeprefix('socks4://')}"
    return proxy


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
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in {418, 429} and attempt < attempts - 1:
                retry_after = _retry_after_seconds(exc.response)
                await asyncio.sleep(retry_after or 0.75 * (attempt + 1))
                continue
            if status_code == 451:
                raise BinanceMarketDataError(
                    "Binance 合约行情接口返回 451：当前网络或地区被 Binance 限制访问 fapi.binance.com。"
                    "这通常发生在本地中国网络出口，建议在海外 VPS 上运行策略，或更换可访问 Binance 合约 API 的服务器网络。"
                ) from exc
            raise BinanceMarketDataError(f"Binance 合约行情接口返回 HTTP {status_code}：{path}") from exc
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.PoolTimeout) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(0.25 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _from_ms(value: int | str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nearest_before(values: list[datetime], timestamp: datetime) -> datetime | None:
    nearest: datetime | None = None
    for value in values:
        if value > timestamp:
            break
        nearest = value
    return nearest


async def _empty_mapping() -> dict[datetime, float]:
    return {}
