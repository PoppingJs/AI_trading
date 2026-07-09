from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any
from typing import Literal

from ai_trading.binance import BinanceFuturesMarketData
from ai_trading.config import AppSettings
from ai_trading.indicators import build_indicators, ema, sma
from ai_trading.models import (
    SETUP_DISTRIBUTION_STAGE1_SHORT,
    SETUP_DISTRIBUTION_STAGE2_SHORT,
    SETUP_DISTRIBUTION_STAGE3_SHORT,
    SETUP_H1_PULLBACK_LONG,
    SETUP_H1_PULLBACK_SHORT,
    SETUP_H1_STRUCTURE_LONG,
    SETUP_H1_STRUCTURE_SHORT,
    SETUP_H4_PULLBACK_LONG,
    SETUP_H4_PULLBACK_SHORT,
    SETUP_M15_SQUEEZE_TACTICAL_LONG,
    SETUP_OI_VALLEY_REVERSAL_LONG,
    Candle,
    DerivativesSnapshot,
    IndicatorSnapshot,
    Position,
    PositionSide,
    SignalAction,
    Trade,
)
from ai_trading.strategy import CompositeStrategy


PaperSide = Literal["LONG", "SHORT"]
AUTO_UNIVERSE_SYMBOL = "AUTO_TOP50"
LEGACY_AUTO_UNIVERSE_SYMBOL = "AUTO_TOP30"
AUTO_UNIVERSE_EXCLUDED_SYMBOLS = {"BTCUSDT", "BTCUSDC", "ETHUSDT", "SOLUSDT", "XAUUSDT"}
AUTO_UNIVERSE_SCAN_LIMIT = 80
AUTO_MAIN_POOL_LIMIT = 50
AUTO_MAIN_POOL_MIN_SCORE = 65
AUTO_POOL_REBALANCE_TIMEFRAME = "15m"
CANDIDATE_MIN_QUOTE_VOLUME = 50_000_000
BTC_EXTREME_4H_AMPLITUDE = 0.08
ROTATION_MIN_SCORE_GAP = 20
ROTATION_MIN_ATR_PCT = 0.008
ROTATION_MIN_VOLUME_RATIO = 1.2
PYRAMID_MIN_SCORE = 85
PYRAMID_MARGIN_FRACTION = 0.35
PYRAMID_MAX_ADDS = 1
PAPER_DEFAULT_BALANCE = 1200.0
INITIAL_ENTRY_MARGIN_CAP = 0.95
TOTAL_OPEN_RISK_CAP = 0.04
PYRAMID_TOTAL_MARGIN_CAP = 0.95
ENTRY_MARGIN_DEPLOYMENT_CAP = 0.95
ENTRY_MARGIN_BASE_SLOTS = 5
ENTRY_ADVANTAGE_ZONE_FRACTION = 0.60
ENTRY_QUALITY_S = "S"
ENTRY_QUALITY_A = "A"
ENTRY_QUALITY_B = "B"
ENTRY_QUALITY_STOP_MAX_10X = 0.065
ENTRY_QUALITY_STOP_MAX_7X = 0.095
ENTRY_QUALITY_STOP_MAX_5X = 0.13
RSI_NORMAL_LONG_MAX = 75.0
RSI_NORMAL_SHORT_MIN = 25.0
RSI_STRONG_LONG_PULLBACK = 75.0
RSI_STRONG_LONG_SEVERE = 90.0
RSI_STRONG_LONG_HARD = 92.0
RSI_STRONG_SHORT_PULLBACK = 25.0
RSI_STRONG_SHORT_SEVERE = 10.0
RSI_STRONG_SHORT_HARD = 8.0
ENTRY_TIMING_GOOD = "GOOD"
ENTRY_TIMING_WAIT = "WAIT"
ENTRY_TIMING_BLOCK = "BLOCK"
TREND_STAGE_EARLY = "EARLY"
TREND_STAGE_MID = "MID"
TREND_STAGE_LATE = "LATE"
TREND_STAGE_NEUTRAL = "NEUTRAL"
DISTRIBUTION_STAGE_RANGE = "HIGH_DISTRIBUTION_RANGE"
DISTRIBUTION_STAGE_DESCENDING = "DESCENDING_DISTRIBUTION"
DISTRIBUTION_STAGE_MARKDOWN = "MARKDOWN_ACCELERATION"
AUTO_ENTRY_MIN_SCORE = 82
CORE_STRUCTURE_SCORE_CAP = AUTO_ENTRY_MIN_SCORE - 1
MIN_ENTRY_REWARD_R = 1.2
LARGE_TIMEFRAME_WICK_REJECTION_PCT = 0.02
PROFIT_LOCK_SLIPPAGE_PCT = 0.0008
MIN_PRECISION_STOP_PCT = 0.012
MIN_PRECISION_STOP_ATR_MULTIPLE = 0.65
MARKET_PRICE_STALE_SECONDS = 15.0
MARKET_PRICE_FALLBACK_SECONDS = 10.0
WEBSOCKET_STALE_SECONDS = 15.0
DERIVATIVES_STALE_SECONDS = 180.0
LIVE_RISK_REFRESH_SECONDS = 3.0
LIVE_ENTRY_REFRESH_SECONDS = 5.0
POSITION_DERIVATIVES_REFRESH_SECONDS = 60.0
BACKGROUND_DERIVATIVES_REFRESH_SECONDS = 180.0
FUNDING_REFRESH_SECONDS = 600.0
UNIVERSE_REFRESH_SECONDS = 900.0
# Trade/account mutations persist immediately at their call sites. This slower
# checkpoint only preserves non-critical live snapshots such as marks/signals.
STATE_SAVE_SECONDS = 300.0
FULL_DATA_CHECK_SECONDS = 120.0
MARKET_REQUEST_CONCURRENCY = 3
WARMUP_BATCH_SIZE = 5
TIMEFRAME_CLOSE_GRACE_SECONDS = 5.0
SUPPORTED_TIMEFRAMES = ("15m", "1h", "4h", "1d")
POSITION_SAFETY_REFRESH_SECONDS = 1.0


class PaperStateError(RuntimeError):
    """Raised when persisted paper-account state cannot be loaded safely."""


class ExitPlanAssertionError(ValueError):
    """Raised when generated stop-loss/take-profit prices violate execution invariants."""


class _StateFileLock:
    def __init__(self, state_path: Path | None) -> None:
        self.path = (
            state_path.with_name(f"{state_path.name}.lock")
            if state_path is not None
            else None
        )
        self._handle: Any | None = None

    def acquire(self) -> None:
        if self.path is None or self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle: Any | None = None
        try:
            handle = self.path.open("a+b")
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if handle is not None:
                handle.close()
            raise PaperStateError(
                f"状态文件正在被另一个交易进程使用：{self.path}"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _candidate_prefilter_allowed(item: object) -> bool:
    quote_volume = float(getattr(item, "quote_volume", 0.0) or 0.0)
    return quote_volume >= CANDIDATE_MIN_QUOTE_VOLUME


@dataclass(frozen=True)
class PaperFill:
    timestamp: datetime
    symbol: str
    side: PositionSide
    action: str
    price: float
    entry_price: float
    quantity: float
    realized_pnl: float
    fee: float
    reason: str
    leverage: int
    margin_usdt: float
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    opened_at: datetime
    entry_position: str = ""
    closed_at: datetime | None = None
    return_pct: float = 0.0


@dataclass
class PaperAccount:
    starting_balance: float = PAPER_DEFAULT_BALANCE
    wallet_balance: float = PAPER_DEFAULT_BALANCE
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[PaperFill] = field(default_factory=list)
    daily_pnl_baselines: dict[str, float] = field(default_factory=dict)
    pnl_history: dict[str, float] = field(default_factory=dict)
    latest_signals: dict[str, dict[str, object]] = field(default_factory=dict)
    reentry_barriers: dict[str, dict[str, str]] = field(default_factory=dict)


class PaperTradingEngine:
    """Local Binance-market paper account.

    It reads public Binance futures data, but all orders are simulated locally.
    No authenticated Binance endpoint is used here.
    """

    def __init__(
        self,
        settings: AppSettings,
        *,
        starting_balance: float = PAPER_DEFAULT_BALANCE,
        symbols: list[str] | None = None,
        interval: str = "1h",
        market_data: BinanceFuturesMarketData | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.state_path = Path(state_path) if state_path else None
        self.account = _load_paper_account(self.state_path) or PaperAccount(starting_balance=starting_balance, wallet_balance=starting_balance)
        requested_symbols = [
            symbol.upper()
            for symbol in (symbols or [AUTO_UNIVERSE_SYMBOL])
        ]
        self._auto_universe = requested_symbols in (
            [AUTO_UNIVERSE_SYMBOL],
            [LEGACY_AUTO_UNIVERSE_SYMBOL],
        )
        self.symbols = (
            [AUTO_UNIVERSE_SYMBOL]
            if self._auto_universe
            else requested_symbols
        )
        self._universe_symbols: list[str] = []
        self._candidate_symbols: list[str] = []
        self._post_close_pool_review: set[str] = set()
        self.interval = interval
        self.market_data = market_data or BinanceFuturesMarketData()
        self.strategy = CompositeStrategy(settings.strategy)
        self.latest_prices: dict[str, float] = {
            symbol: mark_price
            for symbol, position in self.account.positions.items()
            if (mark_price := _stored_mark_price(position)) is not None
        }
        self.latest_signals: dict[str, dict[str, object]] = {symbol: dict(signal) for symbol, signal in self.account.latest_signals.items()}
        self.latest_indicators: dict[str, list[IndicatorSnapshot]] = {}
        self.latest_timeframe_contexts: dict[str, dict[str, object]] = {}
        self.latest_timeframe_indicators: dict[str, dict[str, list[IndicatorSnapshot]]] = {}
        self.running = False
        self.auto_trade = False
        self.last_error: str | None = None
        self.last_market_update_at: datetime | None = None
        self._task: asyncio.Task | None = None
        self._price_stream_task: asyncio.Task | None = None
        self._price_fallback_task: asyncio.Task | None = None
        self._position_safety_task: asyncio.Task | None = None
        self._poll_seconds = 20
        self._lock = asyncio.Lock()
        self._scan_lock = asyncio.Lock()
        self._funding_lock = asyncio.Lock()
        self._price_update_event = asyncio.Event()
        self._state_file_lock = _StateFileLock(self.state_path)
        self._timeframe_candles: dict[str, dict[str, list[Candle]]] = {}
        self._timeframe_derivatives: dict[str, dict[str, list[DerivativesSnapshot]]] = {}
        self._live_m15_candles: dict[str, Candle] = {}
        self._price_updated_at: dict[str, datetime] = {}
        self._signal_updated_at: dict[str, datetime] = {}
        self._derivatives_updated_at: dict[str, datetime] = {}
        self._oi_ratio_updated_at: dict[str, datetime] = {}
        self._funding_updated_at: dict[str, datetime] = {}
        self._derivatives_source_at: dict[str, datetime] = {}
        self._current_funding_rates: dict[str, float] = {}
        self._last_ws_update_at: datetime | None = None
        self._last_price_stream_error: str | None = None
        self._last_rest_price_refresh_at: datetime | None = None
        self._last_universe_refresh_at: datetime | None = None
        self._last_funding_refresh_at: datetime | None = None
        self._last_full_data_check_at: datetime | None = None
        self._last_state_save_at: datetime | None = None
        self._last_btc_extreme_check_at: datetime | None = None
        self._btc_extreme_cached = False
        self._last_closed_candle_slot: dict[str, datetime] = {}
        self._next_candle_retry_at: datetime | None = None
        self._warmup_complete = False
        self._stream_symbols: tuple[str, ...] = ()
        self._last_position_management_at: datetime | None = None

    async def start(self, *, auto_trade: bool = False, poll_seconds: int = 20) -> None:
        async with self._lock:
            self.auto_trade = auto_trade
            self._poll_seconds = max(poll_seconds, 5)
            if self.running:
                return
            self._state_file_lock.acquire()
            self.running = True
            self._position_safety_task = asyncio.create_task(
                self._position_safety_loop()
            )
            self._price_fallback_task = asyncio.create_task(
                self._price_fallback_loop()
            )
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Disable new automated entries while keeping market and position management alive."""
        async with self._lock:
            self.auto_trade = False

    async def shutdown(self) -> None:
        """Stop the background worker during application shutdown."""
        async with self._lock:
            self.running = False
            self.auto_trade = False
            task = self._task
            price_stream_task = self._price_stream_task
            price_fallback_task = self._price_fallback_task
            position_safety_task = self._position_safety_task
            self._task = None
            self._price_stream_task = None
            self._price_fallback_task = None
            self._position_safety_task = None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if price_stream_task:
            price_stream_task.cancel()
            try:
                await price_stream_task
            except asyncio.CancelledError:
                pass
        if price_fallback_task:
            price_fallback_task.cancel()
            try:
                await price_fallback_task
            except asyncio.CancelledError:
                pass
        if position_safety_task:
            position_safety_task.cancel()
            try:
                await position_safety_task
            except asyncio.CancelledError:
                pass
        self._state_file_lock.release()

    async def reset(self, starting_balance: float = PAPER_DEFAULT_BALANCE) -> None:
        await self.stop()
        async with self._lock:
            self.account = PaperAccount(starting_balance=starting_balance, wallet_balance=starting_balance)
            self.latest_prices.clear()
            self.latest_signals.clear()
            self.latest_indicators.clear()
            self.latest_timeframe_contexts.clear()
            self.latest_timeframe_indicators.clear()
            self._timeframe_candles.clear()
            self._timeframe_derivatives.clear()
            self._live_m15_candles.clear()
            self._price_updated_at.clear()
            self._signal_updated_at.clear()
            self._derivatives_updated_at.clear()
            self._oi_ratio_updated_at.clear()
            self._funding_updated_at.clear()
            self._derivatives_source_at.clear()
            self._current_funding_rates.clear()
            self._universe_symbols.clear()
            self._candidate_symbols.clear()
            self._post_close_pool_review.clear()
            if self._auto_universe:
                self.symbols = [AUTO_UNIVERSE_SYMBOL]
            self.last_error = None
            self.last_market_update_at = None
            self._warmup_complete = False
            self.auto_trade = False
            self._save_state_unlocked()

    def configure_symbols(self, symbols: list[str]) -> None:
        requested = [symbol.upper() for symbol in symbols if symbol]
        use_auto_universe = (
            not requested
            or requested == [AUTO_UNIVERSE_SYMBOL]
            or requested == [LEGACY_AUTO_UNIVERSE_SYMBOL]
        )
        if (
            use_auto_universe
            and self._auto_universe
            and self.symbols != [AUTO_UNIVERSE_SYMBOL]
        ):
            return
        next_symbols = [AUTO_UNIVERSE_SYMBOL] if use_auto_universe else requested
        if self._auto_universe == use_auto_universe and self.symbols == next_symbols:
            return
        self._auto_universe = use_auto_universe
        self.symbols = next_symbols
        self._universe_symbols.clear()
        self._candidate_symbols.clear()
        self._warmup_complete = False
        if not use_auto_universe:
            self._prune_market_cache()

    def configure_interval(self, interval: str) -> None:
        if interval not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"unsupported paper interval: {interval}")
        if self.interval == interval:
            return
        self.interval = interval
        self._warmup_complete = False

    async def close(self) -> None:
        await self.shutdown()
        close = getattr(self.market_data, "aclose", None)
        if close is not None:
            await close()

    async def refresh_once(self) -> None:
        if self._scan_lock.locked():
            return
        async with self._scan_lock:
            await self.refresh_universe_if_needed(force=not self._warmup_complete)
            errors: list[str] = []
            refreshed = 0
            symbols = self._managed_symbols()
            await self._refresh_current_funding_cache(
                symbols,
                datetime.now(UTC),
            )
            semaphore = asyncio.Semaphore(MARKET_REQUEST_CONCURRENCY)

            async def refresh_with_limit(symbol: str) -> None:
                nonlocal refreshed
                async with semaphore:
                    try:
                        if await self._refresh_symbol(symbol):
                            refreshed += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{symbol}: {exc}")

            for start in range(0, len(symbols), WARMUP_BATCH_SIZE):
                batch = symbols[start : start + WARMUP_BATCH_SIZE]
                await asyncio.gather(*(refresh_with_limit(symbol) for symbol in batch))
                await asyncio.sleep(0)
            self._rebalance_auto_signal_pools(
                fresh_symbols=self._symbols_with_latest_closed_timeframe(
                    AUTO_POOL_REBALANCE_TIMEFRAME,
                ),
            )
            self._warmup_complete = True
            if refreshed:
                now = datetime.now(UTC)
                self.last_market_update_at = now
                self.last_error = _format_partial_market_errors(errors) if errors else None
                self._save_state_unlocked()
            elif errors:
                self.last_error = f"行情刷新失败，等待网络恢复：{'; '.join(errors[:3])}"

    async def refresh_open_position_prices(self) -> int:
        """Refresh persisted positions without starting the strategy loop."""
        symbols = list(self.account.positions)
        if not symbols:
            return 0

        mark_prices = getattr(self.market_data, "mark_prices", None)
        ticker_prices = getattr(self.market_data, "ticker_prices", None)
        try:
            if mark_prices is not None:
                prices = await mark_prices(symbols)
            elif ticker_prices is not None:
                prices = await ticker_prices(symbols)
            else:
                semaphore = asyncio.Semaphore(MARKET_REQUEST_CONCURRENCY)

                async def fetch(symbol: str) -> tuple[str, float] | None:
                    async with semaphore:
                        candles = await self.market_data.klines(
                            symbol,
                            self.interval,
                            limit=2,
                        )
                        if not candles:
                            return None
                        return symbol, float(candles[-1].close)

                results = await asyncio.gather(
                    *(fetch(symbol) for symbol in symbols)
                )
                prices = dict(
                    result for result in results if result is not None
                )
        except Exception:  # noqa: BLE001 - keep persisted marks when network is unavailable
            return 0
        if not prices:
            return 0

        async with self._lock:
            for symbol, price in prices.items():
                self._remember_mark_price(symbol, price)
            self.last_market_update_at = datetime.now(UTC)
            self._save_state_unlocked()
        return len(prices)

    async def refresh_universe_if_needed(self, *, force: bool = False) -> None:
        if not self._auto_universe:
            return
        now = datetime.now(UTC)
        if (
            not force
            and self._last_universe_refresh_at is not None
            and (now - self._last_universe_refresh_at).total_seconds() < UNIVERSE_REFRESH_SECONDS
        ):
            return
        eligible = await self._eligible_auto_universe_symbols()
        universe = eligible[:AUTO_UNIVERSE_SCAN_LIMIT]
        current_main = [
            symbol
            for symbol in self.symbols
            if symbol in universe
            and symbol not in {
                AUTO_UNIVERSE_SYMBOL,
                LEGACY_AUTO_UNIVERSE_SYMBOL,
            }
        ]
        self._universe_symbols = universe
        self.symbols = (
            current_main
            if current_main
            else universe[:AUTO_MAIN_POOL_LIMIT]
        )
        main_set = set(self.symbols)
        self._candidate_symbols = [
            symbol for symbol in universe if symbol not in main_set
        ]
        self._last_universe_refresh_at = now
        self._prune_market_cache()
        await self._restart_price_stream_if_needed()

    async def _eligible_auto_universe_symbols(self) -> list[str]:
        top_symbols = await self.market_data.top_usdt_perpetuals(
            limit=AUTO_UNIVERSE_SCAN_LIMIT
        )
        eligible = [item for item in top_symbols if item.symbol.upper() not in AUTO_UNIVERSE_EXCLUDED_SYMBOLS]
        return [
            item.symbol
            for item in eligible
            if _candidate_prefilter_allowed(item)
        ]

    def _rebalance_auto_signal_pools(
        self,
        *,
        fresh_symbols: set[str] | None = None,
    ) -> None:
        if not self._auto_universe or not self._universe_symbols:
            return
        fresh = (
            set(self._universe_symbols)
            if fresh_symbols is None
            else set(fresh_symbols)
        )
        position_symbols = list(self.account.positions)
        position_set = set(position_symbols)
        current_main = [
            symbol
            for symbol in self.symbols
            if (
                symbol not in position_set
                and symbol
                not in {
                    AUTO_UNIVERSE_SYMBOL,
                    LEGACY_AUTO_UNIVERSE_SYMBOL,
                }
            )
        ]
        waiting_post_close = (
            self._post_close_pool_review - fresh
        )
        locked_main = [
            symbol
            for symbol in current_main
            if (
                symbol not in fresh
                or symbol in waiting_post_close
            )
        ]
        self._post_close_pool_review.difference_update(fresh)
        turnover_rank = {
            symbol: index
            for index, symbol in enumerate(self._universe_symbols)
        }
        eligible: list[str] = []
        for symbol in self._universe_symbols:
            if symbol not in fresh or symbol in position_set:
                continue
            signal = self.latest_signals.get(symbol)
            if not isinstance(signal, dict):
                continue
            score = int(signal.get("score") or 0)
            if score >= AUTO_MAIN_POOL_MIN_SCORE:
                eligible.append(symbol)
        eligible.sort(
            key=lambda symbol: (
                int(
                    self.latest_signals.get(symbol, {}).get("score") or 0
                ),
                -turnover_rank[symbol],
            ),
            reverse=True,
        )
        pinned = list(
            dict.fromkeys(
                [
                    *position_symbols,
                    *locked_main,
                ]
            )
        )
        available_slots = max(
            AUTO_MAIN_POOL_LIMIT - len(pinned),
            0,
        )
        selected = [
            symbol
            for symbol in eligible
            if symbol not in set(pinned)
        ][:available_slots]
        self.symbols = [
            *pinned[:AUTO_MAIN_POOL_LIMIT],
            *selected,
        ]
        main_set = set(self.symbols)
        self._candidate_symbols = [
            symbol
            for symbol in self._universe_symbols
            if symbol not in main_set
        ]

    def _managed_symbols(self) -> list[str]:
        if self._auto_universe:
            symbols = list(
                self._universe_symbols
                or [
                    symbol
                    for symbol in self.symbols
                    if symbol not in {
                        AUTO_UNIVERSE_SYMBOL,
                        LEGACY_AUTO_UNIVERSE_SYMBOL,
                    }
                ]
                or self.latest_signals
            )
        else:
            symbols = list(self.symbols)
        for symbol in self.account.positions:
            if symbol not in symbols:
                symbols.append(symbol)
        for symbol in self._post_close_pool_review:
            if symbol not in symbols:
                symbols.append(symbol)
        return symbols

    def _symbols_with_latest_closed_timeframe(
        self,
        timeframe: str,
        now: datetime | None = None,
    ) -> set[str]:
        current_time = now or datetime.now(UTC)
        expected_open = _latest_closed_slot(
            timeframe,
            current_time,
        ) - timedelta(seconds=_timeframe_seconds(timeframe))
        return {
            symbol
            for symbol in self._managed_symbols()
            if (
                symbol in self.latest_signals
                and self._timeframe_candles.get(symbol, {}).get(timeframe)
                and self._timeframe_candles[symbol][timeframe][-1].timestamp
                >= expected_open
            )
        }

    def _rebalance_auto_signal_pools_if_due(
        self,
        due_timeframes: tuple[str, ...],
        now: datetime,
    ) -> None:
        if AUTO_POOL_REBALANCE_TIMEFRAME not in due_timeframes:
            return
        self._rebalance_auto_signal_pools(
            fresh_symbols=self._symbols_with_latest_closed_timeframe(
                AUTO_POOL_REBALANCE_TIMEFRAME,
                now,
            ),
        )

    def _auto_main_symbols(self) -> set[str]:
        if not self._auto_universe:
            return set(self.latest_signals)
        if self._universe_symbols:
            return set(self.symbols)
        return set(self.latest_signals)

    def _prune_market_cache(self) -> None:
        managed = set(self._managed_symbols())
        mappings = (
            self.latest_prices,
            self.latest_signals,
            self.latest_indicators,
            self.latest_timeframe_contexts,
            self.latest_timeframe_indicators,
            self._timeframe_candles,
            self._timeframe_derivatives,
            self._live_m15_candles,
            self._price_updated_at,
            self._signal_updated_at,
            self._derivatives_updated_at,
            self._oi_ratio_updated_at,
            self._funding_updated_at,
            self._derivatives_source_at,
            self._current_funding_rates,
            self.account.latest_signals,
        )
        for mapping in mappings:
            for symbol in list(mapping):
                if symbol not in managed:
                    mapping.pop(symbol, None)

    async def open_position(
        self,
        symbol: str,
        side: PaperSide,
        *,
        margin_usdt: float = 100.0,
        leverage: int | None = None,
        stop_loss: float | None = None,
        take_profit_1: float | None = None,
        take_profit_2: float | None = None,
        reason: str = "manual",
        entry_reasons: list[str] | tuple[str, ...] | None = None,
        entry_context: dict[str, object] | None = None,
    ) -> Position:
        symbol = symbol.upper()
        leverage = min(leverage or self.settings.risk.leverage_default, self.settings.risk.leverage_max)
        if margin_usdt <= 0:
            raise ValueError("margin_usdt must be positive")
        if symbol in self.account.positions:
            raise ValueError(f"{symbol} already has an open paper position")
        price = await self._price(symbol)
        async with self._lock:
            available = self._available_balance_unlocked()
            if margin_usdt > available:
                raise ValueError(f"insufficient paper balance: available {available:.2f} USDT")
            notional = margin_usdt * leverage
            quantity = notional / price
            fee = notional * self.settings.execution.taker_fee_rate
            self.account.wallet_balance -= fee
            self.account.fees_paid += fee
            side_enum = PositionSide(side)
            stop_loss = stop_loss or _default_stop(side_enum, price, leverage)
            take_profit_1 = take_profit_1 or _default_take_profit(side_enum, price, stop_loss, 1)
            take_profit_2 = take_profit_2 or _default_take_profit(side_enum, price, stop_loss, 2)
            _assert_valid_exit_plan(
                side_enum,
                price,
                stop_loss,
                take_profit_1,
                take_profit_2,
            )
            normalized_entry_context = _normalize_entry_context(entry_context)
            entry_position = _entry_position_text(side_enum, price, normalized_entry_context, reason)
            position = Position(
                symbol=symbol,
                side=side_enum,
                entry_price=price,
                quantity=quantity,
                opened_at=datetime.now(UTC),
                stop_price=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                metadata={
                    "margin_usdt": margin_usdt,
                    "leverage": leverage,
                    "reason": reason,
                    "entry_score": _score_from_reason(reason),
                    "entry_reason": _entry_reason_text(reason, entry_reasons),
                    "entry_reasons": list(entry_reasons or ()),
                    "entry_context": normalized_entry_context,
                    "entry_position": entry_position,
                    "adds": 0,
                    "initial_stop_distance": abs(price - stop_loss),
                    "last_mark_price": price,
                },
            )
            self.account.positions[symbol] = position
            self.account.fills.append(
                PaperFill(
                    timestamp=position.opened_at,
                    symbol=symbol,
                    side=side_enum,
                    action="OPEN",
                    price=price,
                    entry_price=price,
                    quantity=quantity,
                    realized_pnl=0.0,
                    fee=fee,
                    reason=reason,
                    leverage=leverage,
                    margin_usdt=margin_usdt,
                    stop_price=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    opened_at=position.opened_at,
                    entry_position=entry_position,
                )
            )
            self._save_state_unlocked()
            return position

    async def close_position(self, symbol: str, *, reason: str = "manual close") -> Trade:
        symbol = symbol.upper()
        price = await self._price(symbol)
        async with self._lock:
            position = self.account.positions.get(symbol)
            if position is None:
                raise ValueError(f"{symbol} has no open paper position")
            return self._close_position_unlocked(position, price, reason)

    async def status_async(self) -> dict[str, object]:
        async with self._lock:
            return self._status_unlocked()

    def status(self) -> dict[str, object]:
        return self._status_unlocked()

    def health_status(self) -> dict[str, object]:
        now = datetime.now(UTC)
        stale_positions = [
            symbol
            for symbol in self.account.positions
            if (updated_at := self._price_updated_at.get(symbol)) is None
            or (now - updated_at).total_seconds() > MARKET_PRICE_STALE_SECONDS
        ]
        scheduler_alive = self._task is not None and not self._task.done()
        safety_alive = (
            self._position_safety_task is not None
            and not self._position_safety_task.done()
        )
        return {
            "status": (
                "ok"
                if self.running and scheduler_alive and safety_alive
                and not stale_positions
                else "degraded"
            ),
            "running": self.running,
            "auto_trade": self.auto_trade,
            "warmup_complete": self._warmup_complete,
            "scheduler_alive": scheduler_alive,
            "position_safety_alive": safety_alive,
            "price_stream_alive": (
                self._price_stream_task is not None
                and not self._price_stream_task.done()
            ),
            "price_fallback_alive": (
                self._price_fallback_task is not None
                and not self._price_fallback_task.done()
            ),
            "price_stream_error": self._last_price_stream_error,
            "stale_position_symbols": stale_positions,
            "position_management_at": (
                self._last_position_management_at.isoformat()
                if self._last_position_management_at
                else None
            ),
            "market_updated_at": (
                self.last_market_update_at.isoformat()
                if self.last_market_update_at
                else None
            ),
            "last_error": self.last_error,
        }

    def _status_unlocked(self) -> dict[str, object]:
        positions = []
        unrealized = 0.0
        used_margin = 0.0
        latest_prices = dict(self.latest_prices)
        latest_signals: dict[str, dict[str, object]] = {}
        main_symbols = set(self.symbols)
        ordered_signal_symbols = (
            list(
                dict.fromkeys(
                    [
                        *self.account.positions,
                        *self.symbols,
                    ]
                )
            )
            if self._auto_universe and self._universe_symbols
            else list(
                dict.fromkeys(
                    [
                        *self.account.positions,
                        *self.symbols,
                        *self.latest_signals,
                    ]
                )
            )
        )
        for symbol in ordered_signal_symbols:
            signal = self.latest_signals.get(symbol)
            if not isinstance(signal, dict):
                continue
            has_position = symbol in self.account.positions
            payload = _auto_entry_status_signal(
                symbol,
                signal,
                auto_trade=self.auto_trade,
                has_position=has_position,
            )
            if self.running and not has_position:
                for reason in self._data_freshness_blocks(symbol, payload):
                    _record_auto_entry_block(payload, reason)
            payload["pool"] = (
                "POSITION"
                if has_position
                else "MAIN"
                if symbol in main_symbols
                else "OUTSIDE"
            )
            latest_signals[symbol] = payload
        latest_timeframe_contexts = {
            symbol: dict(self.latest_timeframe_contexts[symbol])
            for symbol in ordered_signal_symbols
            if symbol in self.latest_timeframe_contexts
        }
        positions_snapshot = list(self.account.positions.values())
        fills_snapshot = list(self.account.fills[-100:])
        for position in positions_snapshot:
            price = latest_prices.get(position.symbol) or _stored_mark_price(position) or position.entry_price
            remaining_quantity = (
                position.quantity * max(position.remaining_fraction, 0.0)
            )
            pnl = _pnl(
                position.side,
                position.entry_price,
                price,
                remaining_quantity,
            )
            margin = float(position.metadata.get("margin_usdt", 0.0))
            entry_reasons = tuple(str(reason) for reason in (position.metadata.get("entry_reasons") or ()))
            entry_context = position.metadata.get("entry_context")
            if not isinstance(entry_context, dict):
                entry_context = {}
            signal_snapshot = latest_signals.get(position.symbol, {})
            if not entry_reasons:
                entry_reasons = tuple(str(reason) for reason in (signal_snapshot.get("reasons") or ()))
            if not entry_context:
                entry_context = _entry_context_from_signal(signal_snapshot)
            entry_score = position.metadata.get("entry_score") or _score_from_reason(str(position.metadata.get("reason", ""))) or signal_snapshot.get("score")
            unrealized += pnl
            used_margin += margin
            positions.append(
                {
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "entry_price": position.entry_price,
                    "mark_price": price,
                    "quantity": remaining_quantity,
                    "notional": price * remaining_quantity,
                    "margin_usdt": margin,
                    "leverage": position.metadata.get("leverage", self.settings.risk.leverage_default),
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_pct_on_margin": pnl / margin if margin else 0.0,
                    "stop_price": position.stop_price,
                    "take_profit_1": position.take_profit_1,
                    "take_profit_2": position.take_profit_2,
                    "opened_at": position.opened_at.isoformat(),
                    "reason": position.metadata.get("reason", ""),
                    "entry_score": entry_score,
                    "entry_reasons": entry_reasons,
                    "entry_context": entry_context,
                    "entry_position": position.metadata.get("entry_position")
                    or _entry_position_text(position.side, position.entry_price, entry_context, str(position.metadata.get("reason", ""))),
                    "entry_reason": _entry_reason_text(str(position.metadata.get("reason", "")), entry_reasons, entry_score),
                }
            )
        equity = self.account.wallet_balance + unrealized
        total_pnl = equity - self.account.starting_balance
        gross_realized_pnl = (
            self.account.wallet_balance
            - self.account.starting_balance
            + self.account.fees_paid
        )
        now = datetime.now(UTC)
        pnl_history = _pnl_history_payload(self.account.pnl_history, total_pnl, now=now)
        return {
            "running": self.running,
            "auto_trade": self.auto_trade,
            "symbols": self.symbols,
            "candidate_symbols": self._candidate_symbols,
            "universe_symbols": self._universe_symbols,
            "main_pool_count": len(self.symbols),
            "candidate_pool_count": len(self._candidate_symbols),
            "interval": self.interval,
            "starting_balance": self.account.starting_balance,
            "wallet_balance": self.account.wallet_balance,
            "equity": equity,
            "available_balance": max(equity - used_margin, 0.0),
            "used_margin": used_margin,
            "realized_pnl": gross_realized_pnl,
            "unrealized_pnl": unrealized,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.account.starting_balance,
            "fees_paid": self.account.fees_paid,
            "latest_prices": latest_prices,
            "latest_signals": latest_signals,
            "latest_timeframe_contexts": latest_timeframe_contexts,
            "positions": positions,
            "fills": [_fill_payload(fill) for fill in fills_snapshot],
            "daily_pnl": _daily_pnl_payload(self.account.daily_pnl_baselines, total_pnl, now=now),
            "pnl_history": pnl_history,
            "last_error": self.last_error,
            "market_updated_at": self.last_market_update_at.isoformat() if self.last_market_update_at else None,
            "updated_at": now.isoformat(),
        }

    async def _run_loop(self) -> None:
        loop = asyncio.get_running_loop()
        last_live_entry_refresh = 0.0
        try:
            await self._restart_price_stream_if_needed()
            await self._warm_restored_positions()
            await self.refresh_once()
            now = datetime.now(UTC)
            self._last_closed_candle_slot = {
                timeframe: _latest_closed_slot(timeframe, now)
                for timeframe in SUPPORTED_TIMEFRAMES
            }
        except Exception as exc:  # noqa: BLE001 - keep restored positions manageable
            self._warmup_complete = True
            self.last_error = str(exc)

        while self.running:
            monotonic_now = loop.time()
            now = datetime.now(UTC)
            try:
                if not self._warmup_complete:
                    await self._warm_configured_symbols()
                due_timeframes = self._due_closed_timeframes(now)
                if due_timeframes:
                    await self._refresh_closed_timeframes(due_timeframes)
                await self._refresh_due_derivatives(now)
                self._rebalance_auto_signal_pools_if_due(
                    due_timeframes,
                    now,
                )
                await self._refresh_universe_and_new_symbols(now)
                await self._validate_cached_market_data(now)
                if monotonic_now - last_live_entry_refresh >= LIVE_ENTRY_REFRESH_SECONDS:
                    self._refresh_live_entry_timing()
                    if self.auto_trade:
                        await self._auto_trade_once()
                    last_live_entry_refresh = monotonic_now
                if (
                    self._last_state_save_at is None
                    or (now - self._last_state_save_at).total_seconds() >= STATE_SAVE_SECONDS
                ):
                    self._record_account_snapshots(now)
                    self._last_state_save_at = now
            except Exception as exc:  # noqa: BLE001 - user-facing paper loop should keep running
                self.last_error = str(exc)
            await asyncio.sleep(1)

    async def _position_safety_loop(self) -> None:
        """Manage restored and live positions independently from network scans."""
        while self.running:
            try:
                self._manage_open_positions(fresh_only=True)
                self._last_position_management_at = datetime.now(UTC)
            except Exception as exc:  # noqa: BLE001 - safety loop must stay alive
                self.last_error = f"持仓安全管理异常：{exc}"
            self._price_update_event.clear()
            try:
                await asyncio.wait_for(
                    self._price_update_event.wait(),
                    timeout=POSITION_SAFETY_REFRESH_SECONDS,
                )
            except TimeoutError:
                pass

    async def _price_fallback_loop(self) -> None:
        """Keep live prices fresh even while full market scans are blocked."""
        force = True
        while self.running:
            try:
                await self._refresh_rest_price_fallback(
                    datetime.now(UTC),
                    force=force,
                )
                force = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry without blocking scans
                self.last_error = f"实时价格 REST 兜底刷新失败：{exc}"
            await asyncio.sleep(1)

    async def _warm_restored_positions(self) -> None:
        symbols = list(self.account.positions)
        if not symbols or self._scan_lock.locked():
            return
        await self._refresh_current_funding_cache(
            symbols,
            datetime.now(UTC),
        )
        async with self._scan_lock:
            semaphore = asyncio.Semaphore(MARKET_REQUEST_CONCURRENCY)

            async def warm(symbol: str) -> None:
                async with semaphore:
                    await self._refresh_symbol(symbol)

            await asyncio.gather(
                *(warm(symbol) for symbol in symbols),
                return_exceptions=True,
            )

    async def _warm_configured_symbols(self) -> None:
        if self._scan_lock.locked():
            return
        await self.refresh_once()
        await self._restart_price_stream_if_needed()

    async def _restart_price_stream_if_needed(self) -> None:
        stream = getattr(self.market_data, "stream_mark_prices", None)
        if stream is None or not self.running:
            return
        symbols = tuple(sorted(self._managed_symbols()))
        if self._price_stream_task is not None and not self._price_stream_task.done() and symbols == self._stream_symbols:
            return
        old_task = self._price_stream_task
        self._price_stream_task = None
        if old_task is not None:
            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass
        self._stream_symbols = symbols
        if symbols:
            self._price_stream_task = asyncio.create_task(self._price_stream_loop(symbols))

    async def _price_stream_loop(self, symbols: tuple[str, ...]) -> None:
        delay = 1.0
        while self.running and symbols == self._stream_symbols:
            try:
                async for prices in self.market_data.stream_mark_prices(symbols):
                    now = datetime.now(UTC)
                    for symbol, price in prices.items():
                        self._remember_mark_price(symbol, float(price))
                    self._last_ws_update_at = now
                    self._last_price_stream_error = None
                    self.last_market_update_at = now
                    delay = 1.0
                    if not self.running or symbols != self._stream_symbols:
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - REST fallback keeps held positions safe
                self._last_price_stream_error = str(exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    async def _refresh_rest_price_fallback(
        self,
        now: datetime,
        *,
        force: bool = False,
    ) -> None:
        if (
            not force
            and self._last_rest_price_refresh_at is not None
            and (now - self._last_rest_price_refresh_at).total_seconds() < LIVE_ENTRY_REFRESH_SECONDS
        ):
            return
        symbols = self._managed_symbols()
        stale_symbols = [
            symbol
            for symbol in symbols
            if force
            or (updated_at := self._price_updated_at.get(symbol)) is None
            or (now - updated_at).total_seconds() > MARKET_PRICE_FALLBACK_SECONDS
        ]
        if not stale_symbols:
            return
        self._last_rest_price_refresh_at = now
        mark_prices = getattr(self.market_data, "mark_prices", None)
        if mark_prices is not None:
            prices = await mark_prices(stale_symbols)
            for symbol, price in prices.items():
                self._remember_mark_price(symbol, float(price))
            if prices:
                self.last_market_update_at = now
            return
        ticker_prices = getattr(self.market_data, "ticker_prices", None)
        if ticker_prices is not None:
            prices = await ticker_prices(stale_symbols)
            for symbol, price in prices.items():
                self._remember_mark_price(symbol, float(price))
            if prices:
                self.last_market_update_at = now
            return
        await self.refresh_open_position_prices()

    def _refresh_live_entry_timing(self) -> None:
        live_symbols = (
            self._auto_main_symbols()
            | set(self.account.positions)
        )
        for symbol, signal in self.latest_signals.items():
            if self._auto_universe and symbol not in live_symbols:
                continue
            score = int(signal.get("score") or 0)
            if symbol not in self.account.positions and score < AUTO_ENTRY_MIN_SCORE:
                continue
            price = self.latest_prices.get(symbol)
            if price is None:
                continue
            _clear_transient_auto_entry_blocks(signal)
            signal["price"] = price
            self._apply_live_m15_overlay(symbol, signal)
            _update_entry_position_fields(signal)
            self.account.latest_signals[symbol] = dict(signal)

    def _apply_live_m15_overlay(self, symbol: str, signal: dict[str, object]) -> None:
        live_candle = self._live_m15_candles.get(symbol)
        confirmed = self._timeframe_candles.get(symbol, {}).get("15m", [])
        if live_candle is None or len(confirmed) < 60:
            return
        candles = _merge_candles(confirmed, [live_candle], max_length=max(240, self.settings.strategy.ma_trend + 40))
        indicators = _build_price_only_indicators(candles, self.settings)
        precision = _fifteen_minute_precision(candles, indicators, self.settings)
        signal["m15_precision"] = precision
        signal["live_rsi14"] = indicators[-1].rsi14 if indicators else None
        levels = signal.get("entry_levels")
        if not isinstance(levels, dict):
            return
        copied_levels = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in levels.items()
        }
        long_levels = copied_levels.get("long")
        if isinstance(long_levels, dict):
            long_levels["m15_ema20_ema60"] = _zone_from_object(precision.get("long_pullback_zone"))
        signal["entry_levels"] = copied_levels

    def _due_closed_timeframes(self, now: datetime) -> tuple[str, ...]:
        if self._next_candle_retry_at is not None and now < self._next_candle_retry_at:
            return ()
        due: list[str] = []
        for timeframe in SUPPORTED_TIMEFRAMES:
            slot = _latest_closed_slot(timeframe, now)
            previous = self._last_closed_candle_slot.get(timeframe)
            if previous is None:
                self._last_closed_candle_slot[timeframe] = slot
            elif slot > previous:
                due.append(timeframe)
        return tuple(due)

    async def _refresh_closed_timeframes(self, timeframes: tuple[str, ...]) -> None:
        if self._scan_lock.locked():
            return
        async with self._scan_lock:
            symbols = self._priority_symbols()
            semaphore = asyncio.Semaphore(MARKET_REQUEST_CONCURRENCY)
            errors: list[str] = []

            async def refresh_symbol(symbol: str) -> None:
                async with semaphore:
                    try:
                        changed = False
                        for timeframe in timeframes:
                            changed = await self._refresh_timeframe_incremental(symbol, timeframe) or changed
                        if changed:
                            if self.interval in timeframes or "4h" in timeframes:
                                self._derivatives_updated_at.pop(symbol, None)
                            self._publish_symbol_from_cache(symbol)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{symbol}: {exc}")

            for start in range(0, len(symbols), WARMUP_BATCH_SIZE):
                await asyncio.gather(*(refresh_symbol(symbol) for symbol in symbols[start : start + WARMUP_BATCH_SIZE]))
            if errors:
                self.last_error = _format_partial_market_errors(errors)
                self._next_candle_retry_at = datetime.now(UTC) + timedelta(seconds=5)
            else:
                self._next_candle_retry_at = None
                for timeframe in timeframes:
                    self._last_closed_candle_slot[timeframe] = _latest_closed_slot(timeframe, datetime.now(UTC))

    async def _refresh_timeframe_incremental(self, symbol: str, timeframe: str) -> bool:
        rows = await self.market_data.klines(symbol, timeframe, limit=3)
        closed = _closed_candles(rows, timeframe)
        if not closed:
            raise ValueError(f"{symbol} {timeframe} closed candle is not available yet")
        expected_open = _latest_closed_slot(timeframe, datetime.now(UTC)) - timedelta(
            seconds=_timeframe_seconds(timeframe)
        )
        if closed[-1].timestamp < expected_open:
            raise ValueError(f"{symbol} {timeframe} latest closed candle is delayed")
        existing = self._timeframe_candles.setdefault(symbol, {}).get(timeframe, [])
        merged = _merge_candles(existing, closed, max_length=max(240, self.settings.strategy.ma_trend + 40))
        if existing and merged[-1].timestamp == existing[-1].timestamp and merged[-1] == existing[-1]:
            return False
        self._timeframe_candles[symbol][timeframe] = merged
        return True

    async def _refresh_due_derivatives(self, now: datetime) -> None:
        refresh = getattr(self.market_data, "derivatives_bundle", None)
        if refresh is None or self._scan_lock.locked():
            return
        position_symbols = set(self.account.positions)
        ranked_candidates = [
            symbol
            for symbol, signal in sorted(
                self.latest_signals.items(),
                key=lambda item: int(item[1].get("score") or 0),
                reverse=True,
            )
            if int(signal.get("score") or 0) >= AUTO_ENTRY_MIN_SCORE
            and symbol not in position_symbols
        ]
        fast_symbols = position_symbols | set(ranked_candidates[:10])
        fast_due_symbols = [
            symbol
            for symbol in self._managed_symbols()
            if symbol in fast_symbols
            and (
                self._oi_ratio_updated_at.get(symbol) is None
                or (now - self._oi_ratio_updated_at[symbol]).total_seconds()
                >= POSITION_DERIVATIVES_REFRESH_SECONDS
            )
        ]
        background_due_symbols = [
            symbol
            for symbol in self._managed_symbols()
            if symbol not in fast_symbols
            and (
                self._oi_ratio_updated_at.get(symbol) is None
                or (now - self._oi_ratio_updated_at[symbol]).total_seconds()
                >= BACKGROUND_DERIVATIVES_REFRESH_SECONDS
            )
        ]
        due_symbols = fast_due_symbols + background_due_symbols[:WARMUP_BATCH_SIZE]
        if not due_symbols:
            return
        funding_due_symbols = [
            symbol
            for symbol in due_symbols
            if self._funding_updated_at.get(symbol) is None
            or (now - self._funding_updated_at[symbol]).total_seconds()
            >= FUNDING_REFRESH_SECONDS
        ]
        successful_refreshes = 0
        async with self._scan_lock:
            semaphore = asyncio.Semaphore(MARKET_REQUEST_CONCURRENCY)
            funding_rates: dict[str, float] = {}
            funding_error: Exception | None = None
            current_funding_rates = getattr(
                self.market_data,
                "current_funding_rates",
                None,
            )
            if funding_due_symbols and current_funding_rates is not None:
                try:
                    funding_rates = await current_funding_rates(
                        funding_due_symbols
                    )
                    self._current_funding_rates.update(funding_rates)
                except Exception as exc:  # noqa: BLE001 - OI/ratio refresh can still succeed
                    funding_error = exc

            async def refresh_symbol(symbol: str) -> bool:
                async with semaphore:
                    changed = False
                    source_timestamps: list[datetime] = []
                    for timeframe in {self.interval, "1h", "4h"}:
                        candles = self._timeframe_candles.get(symbol, {}).get(timeframe, [])
                        if not candles:
                            continue
                        snapshots = await refresh(
                            symbol,
                            timeframe,
                            (candle.timestamp for candle in candles),
                            include_funding=False,
                        )
                        existing = self._timeframe_derivatives.setdefault(symbol, {}).get(timeframe, [])
                        merged = _merge_derivatives(
                            existing,
                            snapshots,
                            preserve_funding=True,
                        )
                        if symbol in funding_rates:
                            merged = _with_current_funding(
                                merged,
                                funding_rates[symbol],
                            )
                        self._timeframe_derivatives[symbol][timeframe] = merged
                        source_timestamps.extend(
                            snapshot.timestamp
                            for snapshot in snapshots
                            if snapshot.open_interest is not None
                            and snapshot.long_short_ratio is not None
                        )
                        changed = changed or bool(snapshots)
                    if changed:
                        self._oi_ratio_updated_at[symbol] = now
                        self._derivatives_updated_at[symbol] = now
                        if source_timestamps:
                            self._derivatives_source_at[symbol] = max(
                                source_timestamps
                            )
                        if symbol in funding_rates:
                            self._funding_updated_at[symbol] = now
                        self._publish_symbol_from_cache(symbol)
                    return changed

            for start in range(0, len(due_symbols), WARMUP_BATCH_SIZE):
                results = await asyncio.gather(
                    *(refresh_symbol(symbol) for symbol in due_symbols[start : start + WARMUP_BATCH_SIZE]),
                    return_exceptions=True,
                )
                successful_refreshes += sum(result is True for result in results)
                errors = [result for result in results if isinstance(result, Exception)]
                if errors:
                    self.last_error = f"衍生品数据刷新失败：{errors[0]}"
            if funding_error is not None:
                self.last_error = f"当前资金费率刷新失败，已保留上次有效值：{funding_error}"
        if funding_rates and successful_refreshes:
            self._last_funding_refresh_at = now

    async def _refresh_current_funding_cache(
        self,
        symbols: list[str],
        now: datetime,
    ) -> None:
        due_symbols = [
            symbol
            for symbol in symbols
            if self._funding_updated_at.get(symbol) is None
            or (now - self._funding_updated_at[symbol]).total_seconds()
            >= FUNDING_REFRESH_SECONDS
        ]
        current_funding_rates = getattr(
            self.market_data,
            "current_funding_rates",
            None,
        )
        if not due_symbols or current_funding_rates is None:
            return
        async with self._funding_lock:
            still_due = [
                symbol
                for symbol in due_symbols
                if self._funding_updated_at.get(symbol) is None
                or (now - self._funding_updated_at[symbol]).total_seconds()
                >= FUNDING_REFRESH_SECONDS
            ]
            if not still_due:
                return
            try:
                rates = await current_funding_rates(still_due)
            except Exception as exc:  # noqa: BLE001 - stale funding blocks entries
                self.last_error = (
                    f"当前资金费率刷新失败，已保留上次有效值：{exc}"
                )
                return
            self._current_funding_rates.update(rates)
            for symbol in still_due:
                if symbol in rates:
                    self._funding_updated_at[symbol] = now
            if rates:
                self._last_funding_refresh_at = now

    async def _refresh_universe_and_new_symbols(self, now: datetime) -> None:
        if not self._auto_universe:
            return
        if (
            self._last_universe_refresh_at is not None
            and (now - self._last_universe_refresh_at).total_seconds() < UNIVERSE_REFRESH_SECONDS
        ):
            return
        if self._scan_lock.locked():
            return
        async with self._scan_lock:
            previous_universe = list(
                self._universe_symbols
                or dict.fromkeys(
                    [
                        *self.symbols,
                        *self._candidate_symbols,
                    ]
                )
            )
            previous_main = list(self.symbols)
            eligible = await self._eligible_auto_universe_symbols()
            desired_universe = eligible[:AUTO_UNIVERSE_SCAN_LIMIT]
            previous_set = set(previous_universe)
            symbols_to_warm = [
                symbol
                for symbol in desired_universe
                if (
                    symbol not in previous_set
                    or symbol not in self.latest_signals
                    or not self._symbol_cache_valid(symbol)
                )
            ]
            semaphore = asyncio.Semaphore(MARKET_REQUEST_CONCURRENCY)
            warmed: set[str] = set()

            async def warm(symbol: str) -> None:
                async with semaphore:
                    if await self._refresh_symbol(symbol):
                        warmed.add(symbol)

            for start in range(0, len(symbols_to_warm), WARMUP_BATCH_SIZE):
                await asyncio.gather(
                    *(warm(symbol) for symbol in symbols_to_warm[start : start + WARMUP_BATCH_SIZE]),
                    return_exceptions=True,
                )
            ready = [
                symbol
                for symbol in desired_universe
                if (
                    symbol in warmed
                    or (
                        symbol in previous_set
                        and symbol in self.latest_signals
                        and self._symbol_cache_valid(symbol)
                    )
                )
            ]
            target_size = len(desired_universe)
            eligible_set = set(eligible)
            for symbol in previous_universe:
                if len(ready) >= target_size:
                    break
                if (
                    symbol in eligible_set
                    and symbol not in ready
                    and symbol in self.latest_signals
                ):
                    ready.append(symbol)
            self._universe_symbols = ready
            ready_set = set(ready)
            protected_main = (
                set(self.account.positions)
                | self._post_close_pool_review
            )
            self.symbols = [
                symbol
                for symbol in previous_main
                if (
                    symbol in ready_set
                    or symbol in protected_main
                )
            ]
            main_set = set(self.symbols)
            self._candidate_symbols = [
                symbol
                for symbol in ready
                if symbol not in main_set
            ]
            self._last_universe_refresh_at = now
            self._prune_market_cache()
            await self._restart_price_stream_if_needed()

    async def _validate_cached_market_data(self, now: datetime) -> None:
        if (
            self._last_full_data_check_at is not None
            and (now - self._last_full_data_check_at).total_seconds() < FULL_DATA_CHECK_SECONDS
        ):
            return
        self._last_full_data_check_at = now
        invalid = [symbol for symbol in self._managed_symbols() if not self._symbol_cache_valid(symbol)]
        if not invalid or self._scan_lock.locked():
            return
        async with self._scan_lock:
            semaphore = asyncio.Semaphore(MARKET_REQUEST_CONCURRENCY)

            async def repair(symbol: str) -> None:
                async with semaphore:
                    await self._refresh_symbol(symbol)

            await asyncio.gather(*(repair(symbol) for symbol in invalid[:WARMUP_BATCH_SIZE]), return_exceptions=True)
        if len(invalid) > WARMUP_BATCH_SIZE:
            self._last_full_data_check_at = now - timedelta(
                seconds=FULL_DATA_CHECK_SECONDS - TIMEFRAME_CLOSE_GRACE_SECONDS
            )

    def _symbol_cache_valid(self, symbol: str) -> bool:
        cache = self._timeframe_candles.get(symbol, {})
        return all(_candles_contiguous(cache.get(timeframe, []), timeframe) for timeframe in SUPPORTED_TIMEFRAMES)

    def _priority_symbols(self) -> list[str]:
        positions = list(self.account.positions)
        candidates = [
            symbol
            for symbol, signal in sorted(
                self.latest_signals.items(),
                key=lambda item: int(item[1].get("score") or 0),
                reverse=True,
            )
            if int(signal.get("score") or 0) >= AUTO_ENTRY_MIN_SCORE and symbol not in positions
        ]
        remainder = [
            symbol for symbol in self._managed_symbols()
            if symbol not in positions and symbol not in candidates
        ]
        return positions + candidates + remainder

    def _record_account_snapshots(self, now: datetime | None = None) -> None:
        total_pnl = self._current_total_pnl()
        _pnl_history_payload(self.account.pnl_history, total_pnl, now=now)
        _daily_pnl_payload(self.account.daily_pnl_baselines, total_pnl, now=now)
        self._save_state_unlocked()

    def _record_pnl_history_sample(self, now: datetime | None = None) -> None:
        _pnl_history_payload(self.account.pnl_history, self._current_total_pnl(), now=now)

    def _current_total_pnl(self) -> float:
        unrealized = 0.0
        for position in self.account.positions.values():
            price = self.latest_prices.get(position.symbol) or _stored_mark_price(position) or position.entry_price
            unrealized += _pnl(position.side, position.entry_price, price, position.quantity)
        return self.account.wallet_balance + unrealized - self.account.starting_balance

    async def _refresh_symbol(self, symbol: str) -> bool:
        history_limit = max(240, self.settings.strategy.ma_trend + 40)
        timeframe_candles: dict[str, list[Candle]] = {}
        timeframe_derivatives: dict[str, list[DerivativesSnapshot]] = {}
        for timeframe in SUPPORTED_TIMEFRAMES:
            candles = await self.market_data.klines(
                symbol,
                timeframe,
                limit=history_limit,
            )
            timeframe_candles[timeframe] = _closed_candles(candles, timeframe)
        if not timeframe_candles.get(self.interval):
            return False
        refresh_derivatives = getattr(self.market_data, "derivatives_bundle", None)
        if refresh_derivatives is not None:
            for timeframe in {self.interval, "1h", "4h"}:
                closed = timeframe_candles.get(timeframe, [])
                if not closed:
                    continue
                snapshots = await refresh_derivatives(
                    symbol,
                    timeframe,
                    (candle.timestamp for candle in closed),
                    include_funding=False,
                )
                existing = self._timeframe_derivatives.get(symbol, {}).get(
                    timeframe,
                    [],
                )
                timeframe_derivatives[timeframe] = _merge_derivatives(
                    existing,
                    snapshots,
                    preserve_funding=True,
                )

        funding_rate = self._current_funding_rates.get(symbol.upper())
        if funding_rate is not None:
            for timeframe, snapshots in timeframe_derivatives.items():
                timeframe_derivatives[timeframe] = _with_current_funding(
                    snapshots,
                    funding_rate,
                )

        self._timeframe_candles[symbol] = timeframe_candles
        self._timeframe_derivatives[symbol] = timeframe_derivatives
        now = datetime.now(UTC)
        if all(
            _derivatives_complete(timeframe_derivatives.get(timeframe, []))
            for timeframe in {self.interval, "1h", "4h"}
        ):
            self._oi_ratio_updated_at[symbol] = now
            self._derivatives_updated_at[symbol] = now
            source_timestamps = [
                snapshot.timestamp
                for snapshots in timeframe_derivatives.values()
                for snapshot in snapshots
                if snapshot.open_interest is not None
                and snapshot.long_short_ratio is not None
            ]
            if source_timestamps:
                self._derivatives_source_at[symbol] = max(source_timestamps)
        if symbol not in self.latest_prices:
            self._remember_mark_price(
                symbol,
                float(timeframe_candles[self.interval][-1].close),
                fresh=False,
            )
        return self._publish_symbol_from_cache(symbol)

    def _publish_symbol_from_cache(self, symbol: str) -> bool:
        timeframe_candles = self._timeframe_candles.get(symbol, {})
        signal_timeframe = _entry_signal_timeframe(self.interval)
        base_candles = timeframe_candles.get(signal_timeframe, [])
        if not base_candles:
            return False
        timeframe_indicators: dict[str, list[IndicatorSnapshot]] = {}
        for timeframe in SUPPORTED_TIMEFRAMES:
            candles = timeframe_candles.get(timeframe, [])
            if not candles:
                continue
            derivatives = self._timeframe_derivatives.get(symbol, {}).get(timeframe, [])
            timeframe_indicators[timeframe] = (
                _build_indicators(candles, derivatives, self.settings)
                if derivatives
                else _build_price_only_indicators(candles, self.settings)
            )
        indicators = timeframe_indicators.get(signal_timeframe, [])
        if not indicators:
            return False
        mtf_context = _build_multi_timeframe_context(timeframe_candles, timeframe_indicators, self.settings)
        self.latest_indicators[symbol] = indicators
        self.latest_timeframe_indicators[symbol] = timeframe_indicators
        self.latest_timeframe_contexts[symbol] = mtf_context
        strategy = CompositeStrategy(
            replace(
                self.settings.strategy,
                smart_money_window=_bars_for_4h(signal_timeframe),
            )
        )
        signal = strategy.generate_signal(symbol, base_candles, indicators)
        cycle = strategy.smart_money_cycle(base_candles, indicators)
        signal_side = signal.direction.value if signal.direction else None
        candidate_action = (
            SignalAction.ENTRY_LONG.value
            if signal.direction == PositionSide.LONG
            else SignalAction.ENTRY_SHORT.value
            if signal.direction == PositionSide.SHORT
            else None
        )
        trend_state = strategy.trend_state(signal_side, indicators)
        risk_state = strategy.risk_state(indicators)
        current = indicators[-1] if indicators else None
        live_price = self.latest_prices.get(symbol) or (current.close if current else base_candles[-1].close)
        payload = {
            "timestamp": signal.timestamp.isoformat(),
            "signal_timeframe": signal_timeframe,
            "action": signal.action.value,
            "candidate_action": candidate_action,
            "regime": signal.regime.value,
            "trend_state": trend_state,
            "risk_state": risk_state,
            "rsi14": current.rsi14 if current else None,
            "volume_ratio": current.volume_ratio if current else None,
            "oi_change": current.oi_change if current else None,
            "long_short_ratio": current.long_short_ratio if current else None,
            "funding_rate": current.funding_rate if current else None,
            "price": live_price,
            "score": signal.score,
            "reasons": signal.reasons,
            "vetoes": signal.vetoes,
            "smart_money_phase": cycle.phase,
            "setup_type": signal.setup_type,
        }
        self.latest_signals[symbol] = _apply_multi_timeframe_context(payload, mtf_context)
        self.account.latest_signals[symbol] = dict(self.latest_signals[symbol])
        self._signal_updated_at[symbol] = datetime.now(UTC)
        return True

    async def _multi_timeframe_context(
        self,
        symbol: str,
        base_candles: list[Candle],
        base_indicators: list[IndicatorSnapshot],
    ) -> tuple[dict[str, list[IndicatorSnapshot]], dict[str, object]]:
        timeframe_candles = self._timeframe_candles.get(symbol, {self.interval: base_candles})
        timeframe_indicators = self.latest_timeframe_indicators.get(symbol, {self.interval: base_indicators})
        return timeframe_indicators, _build_multi_timeframe_context(timeframe_candles, timeframe_indicators, self.settings)

    async def _price(self, symbol: str) -> float:
        symbol = symbol.upper()
        if symbol not in self.latest_prices:
            mark_prices = getattr(self.market_data, "mark_prices", None)
            if mark_prices is not None:
                prices = await mark_prices([symbol])
                if symbol in prices:
                    self._remember_mark_price(symbol, float(prices[symbol]))
            if symbol not in self.latest_prices:
                candles = await self.market_data.klines(
                    symbol,
                    self.interval,
                    limit=2,
                )
                if not candles:
                    raise ValueError(f"cannot fetch price for {symbol}")
                self._remember_mark_price(symbol, float(candles[-1].close))
        return self.latest_prices[symbol]

    def _remember_mark_price(self, symbol: str, price: float, *, fresh: bool = True) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        symbol = symbol.upper()
        self.latest_prices[symbol] = price
        now = datetime.now(UTC)
        if not fresh:
            position = self.account.positions.get(symbol)
            if position is not None:
                position.metadata["last_mark_price"] = price
            return
        self._price_updated_at[symbol] = now
        self._price_update_event.set()
        slot = _current_candle_slot("15m", now)
        live = self._live_m15_candles.get(symbol)
        if live is None or live.timestamp != slot:
            previous_close = (
                self._timeframe_candles.get(symbol, {}).get("15m", [])[-1].close
                if self._timeframe_candles.get(symbol, {}).get("15m")
                else price
            )
            self._live_m15_candles[symbol] = Candle(
                timestamp=slot,
                open=previous_close,
                high=max(previous_close, price),
                low=min(previous_close, price),
                close=price,
                volume=0.0,
            )
        else:
            self._live_m15_candles[symbol] = Candle(
                timestamp=slot,
                open=live.open,
                high=max(live.high, price),
                low=min(live.low, price),
                close=price,
                volume=live.volume,
            )
        position = self.account.positions.get(symbol)
        if position is not None:
            position.metadata["last_mark_price"] = price

    async def _auto_trade_once(self) -> None:
        auto_entry_symbols = self._auto_main_symbols()
        baseline_signals = {
            symbol: dict(signal)
            for symbol, signal in self.latest_signals.items()
            if (
                symbol in auto_entry_symbols
                or symbol in self.account.positions
            )
        }
        evaluated_signals = {
            symbol: dict(signal)
            for symbol, signal in baseline_signals.items()
        }
        for signal in evaluated_signals.values():
            _update_entry_position_fields(signal)
        if await self._btc_4h_extreme_volatility():
            self.last_error = "BTC 4h extreme volatility; pause new altcoin entries"
            for symbol, signal in evaluated_signals.items():
                if symbol not in self.account.positions and signal.get("action") in {
                    SignalAction.ENTRY_LONG.value,
                    SignalAction.ENTRY_SHORT.value,
                }:
                    _record_auto_entry_block(signal, "BTC 4h extreme volatility; pause new altcoin entries")
            self._commit_auto_entry_signal_snapshot(
                baseline_signals,
                evaluated_signals,
            )
            return
        max_positions = min(self.settings.risk.max_open_positions, 5)
        candidates: list[tuple[str, dict[str, object]]] = []
        for symbol, signal in evaluated_signals.items():
            signal.pop("exit_plan_error", None)
            signal.pop("exit_plan_status", None)
            _clear_transient_auto_entry_blocks(signal)
            if symbol in self.account.positions:
                continue
            reentry_reason = self._reentry_block_reason(symbol)
            if reentry_reason:
                _record_auto_entry_block(signal, reentry_reason)
                continue
            for reason in _auto_entry_prerequisite_blocks(
                signal,
                include_entry_position=False,
            ):
                _record_auto_entry_block(signal, reason)
            entry_position_block = _auto_entry_position_block(signal)
            if self.running:
                for reason in self._data_freshness_blocks(symbol, signal):
                    _record_auto_entry_block(signal, reason)
            if signal.get("vetoes") or entry_position_block:
                continue
            candidates.append((symbol, signal))
        candidates.sort(key=lambda item: int(item[1].get("score") or 0), reverse=True)
        if len(self.account.positions) >= max_positions:
            await self._rebalance_for_better_candidate(candidates)
        slots = max_positions - len(self.account.positions)
        if slots <= 0:
            for _, signal in candidates:
                _record_auto_entry_block(signal, f"position capacity full: {max_positions} open positions")
            self._commit_auto_entry_signal_snapshot(
                baseline_signals,
                evaluated_signals,
            )
            return
        opened_count = 0
        for symbol, signal in candidates:
            if opened_count >= slots:
                _record_auto_entry_block(signal, f"position capacity full: {max_positions} open positions")
                continue
            if symbol in self.account.positions:
                continue
            action = signal.get("action")
            if action not in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}:
                continue
            side = "LONG" if action == SignalAction.ENTRY_LONG.value else "SHORT"
            status = self.status()
            available = float(status["available_balance"])
            equity = float(status["equity"])
            used_margin = float(status["used_margin"])
            score = int(signal.get("score") or 0)
            max_total_margin = equity * INITIAL_ENTRY_MARGIN_CAP
            remaining_total_margin = max(max_total_margin - used_margin, 0.0)
            try:
                indicators = self.latest_indicators.get(symbol, [])
                precision = self.latest_timeframe_contexts.get(symbol, {}).get("m15_precision", {})
                mtf_context = self.latest_timeframe_contexts.get(symbol, {})
                trend_state = str(signal.get("trend_state") or "CHOP")
                side_enum = PositionSide(side)
                entry_price = self.latest_prices.get(symbol) or await self._price(symbol)
                entry_timeframe = _entry_timeframe_for_signal(
                    signal,
                    side_enum,
                    entry_price,
                )
                timeframe_indicators = self.latest_timeframe_indicators.get(symbol, {})
                preferred_indicator = _indicator_for_entry_timeframe(
                    timeframe_indicators,
                    entry_timeframe,
                    indicators,
                )
                trend_stage = str(signal.get("trend_stage_phase") or _trend_stage_from_signal(signal, preferred_indicator))
                initial_leverage = min(
                    _leverage_for_signal(
                        score,
                        self.settings.risk.leverage_max,
                        trend_state,
                        preferred_indicator,
                        trend_stage,
                    ),
                    int(signal.get("leverage_cap") or self.settings.risk.leverage_max),
                )
                stop_loss, take_profit_1, take_profit_2 = _adaptive_exits(
                    side_enum,
                    entry_price,
                    initial_leverage,
                    trend_state,
                    preferred_indicator,
                )
                stop_basis = "volatility_fallback"
                precision_entry_only = (
                    entry_timeframe == "15m"
                )
                if precision_entry_only:
                    if not _precision_stop_allowed(
                        side_enum,
                        trend_state,
                        str(signal.get("risk_state") or "NORMAL"),
                        score,
                        precision,
                        str(signal.get("smart_money_phase") or ""),
                    ) or not _m15_precision_stop_distance_ok(
                        side_enum,
                        entry_price,
                        precision,
                        preferred_indicator,
                    ):
                        _record_auto_entry_block(
                            signal,
                            "15m tactical entry lacks a valid 15m structure stop or 1h/4h entry zone",
                        )
                        continue
                    refined_stop = _refine_stop_with_precision(
                        side_enum,
                        stop_loss,
                        precision,
                        entry_price,
                        preferred_indicator,
                    )
                    stop_loss = refined_stop
                    stop_basis = "15m_precision_structure"
                else:
                    structure_stop, structure_basis = _refine_stop_with_setup_structure(
                        side_enum,
                        stop_loss,
                        entry_price,
                        signal,
                        mtf_context,
                        preferred_indicator,
                        timeframe=entry_timeframe,
                    )
                    if structure_stop != stop_loss:
                        stop_loss = structure_stop
                        stop_basis = structure_basis
                staged_stop = _refine_stop_with_distribution_stage(
                    side_enum,
                    stop_loss,
                    entry_price,
                    signal,
                    preferred_indicator,
                )
                if staged_stop != stop_loss:
                    stop_loss = staged_stop
                    stop_basis = (
                        f"{signal.get('distribution_short_stage')}"
                        "_structure"
                    )
                take_profit_1, take_profit_2 = _take_profits_for_final_stop(
                    signal,
                    side_enum,
                    entry_price,
                    stop_loss,
                    timeframe=entry_timeframe,
                )
                try:
                    _assert_valid_exit_plan(
                        side_enum,
                        entry_price,
                        stop_loss,
                        take_profit_1,
                        take_profit_2,
                    )
                except ExitPlanAssertionError as exc:
                    signal["exit_plan_status"] = "ERROR"
                    signal["exit_plan_error"] = str(exc)
                    self.last_error = f"{symbol} exit plan assertion failed: {exc}"
                    continue
                reward_r = _entry_reward_r(
                    signal,
                    side_enum,
                    entry_price,
                    stop_loss,
                    timeframe=entry_timeframe,
                    planned_target=take_profit_2,
                    round_trip_cost_rate=2 * (
                        self.settings.execution.taker_fee_rate
                        + self.settings.execution.slippage_rate
                    ),
                )
                reward_r_unavailable = reward_r is None
                if reward_r_unavailable:
                    low_reward_note = "实际盈亏比无法计算"
                    signal_reasons = list(signal.get("reasons") or ())
                    if low_reward_note not in signal_reasons:
                        signal_reasons.append(low_reward_note)
                    signal["reasons"] = tuple(signal_reasons)
                    reward_r = 0.0
                if not reward_r_unavailable and reward_r < MIN_ENTRY_REWARD_R:
                    low_reward_note = f"实际盈亏比低于 {MIN_ENTRY_REWARD_R:.2f}R"
                    signal_reasons = list(signal.get("reasons") or ())
                    if low_reward_note not in signal_reasons:
                        signal_reasons.append(low_reward_note)
                    signal["reasons"] = tuple(signal_reasons)
                signal["exit_plan_status"] = "NORMAL"
                stop_pct = _entry_stop_pct(entry_price, stop_loss)
                entry_quality = _entry_quality_grade(
                    score=score,
                    reward_r=reward_r,
                    stop_pct=stop_pct,
                    setup_type=str(signal.get("setup_type") or ""),
                )
                leverage = _leverage_for_entry_quality(
                    entry_quality,
                    stop_pct=stop_pct,
                    leverage_max=self.settings.risk.leverage_max,
                    leverage_cap=int(
                        signal.get("leverage_cap")
                        or self.settings.risk.leverage_max
                    ),
                )
                if leverage <= 0:
                    _record_auto_entry_block(
                        signal,
                        "entry stop distance too wide for minimum 5x leverage safety",
                    )
                    continue
                margin_factor = _daily_bias_margin_factor(side, signal)
                margin = _quality_sized_margin(
                    equity=equity,
                    available=available,
                    remaining_total_margin=remaining_total_margin,
                    quality=entry_quality,
                    margin_factor=margin_factor,
                )
                planned_risk_usdt = margin * leverage * stop_pct
                if margin < 5:
                    _record_auto_entry_block(
                        signal,
                        f"available entry margin {margin:.2f} USDT below minimum 5.00 USDT",
                    )
                    continue
                entry_context = _entry_context_from_signal(signal)
                entry_context["stop_basis"] = stop_basis
                entry_context["stop_timeframe"] = entry_timeframe
                entry_context["entry_setup"] = f"{entry_timeframe}_entry"
                if signal.get("setup_type"):
                    entry_context["setup_type"] = signal["setup_type"]
                entry_context["trend_stage_phase"] = trend_stage
                entry_context["entry_reward_r"] = reward_r
                entry_context["planned_risk_usdt"] = planned_risk_usdt
                entry_context["entry_quality"] = entry_quality
                entry_context["entry_quality_units"] = _entry_quality_units(
                    entry_quality
                )
                entry_context["entry_stop_pct"] = stop_pct
                entry_context["entry_margin_factor"] = margin_factor
                await self.open_position(
                    symbol,
                    side,
                    margin_usdt=margin,
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    reason=f"auto strategy score={signal.get('score')}; state={trend_state}",
                    entry_reasons=tuple(str(reason) for reason in (signal.get("reasons") or ())),
                    entry_context=entry_context,
                )
                opened_count += 1
            except Exception as exc:  # noqa: BLE001 - one rejected candidate must not block later candidates
                signal["exit_plan_status"] = "ERROR"
                _record_auto_entry_block(signal, f"auto entry execution failed: {exc}")
        self._commit_auto_entry_signal_snapshot(
            baseline_signals,
            evaluated_signals,
        )
        self._add_to_strong_positions()

    def _commit_auto_entry_signal_snapshot(
        self,
        baseline_signals: dict[str, dict[str, object]],
        evaluated_signals: dict[str, dict[str, object]],
    ) -> None:
        """Publish one completed auto-entry pass without exposing partial veto states."""
        for symbol, evaluated in evaluated_signals.items():
            baseline = baseline_signals.get(symbol)
            current = self.latest_signals.get(symbol)
            if baseline is None or current != baseline:
                continue
            committed = dict(evaluated)
            self.latest_signals[symbol] = committed
            self.account.latest_signals[symbol] = dict(committed)

    async def _rebalance_for_better_candidate(self, candidates: list[tuple[str, dict[str, object]]]) -> None:
        if not candidates or not self.account.positions:
            return
        best_symbol, best_signal = candidates[0]
        best_indicators = self.latest_indicators.get(best_symbol, [])
        best_indicator = best_indicators[-1] if best_indicators else None
        if not _rotation_candidate_allowed(best_signal, best_indicator):
            return
        replace_target = self._rotation_replace_target(best_signal, best_indicator)
        if replace_target is None:
            return
        best_score = int(best_signal.get("score") or 0)
        position, current_score, rotation_type = replace_target
        await self.close_position(
            position.symbol,
            reason=f"rotation exit: {rotation_type}; symbol={best_symbol} score={best_score} current_score={current_score}",
        )

    def _rotation_replace_target(
        self,
        candidate_signal: dict[str, object],
        candidate_indicator: IndicatorSnapshot | None,
    ) -> tuple[Position, int, str] | None:
        best_target: tuple[Position, int, str] | None = None
        candidate_score = int(candidate_signal.get("score") or 0)
        candidate_is_strong = _rotation_candidate_strong(candidate_signal)
        for symbol, position in self.account.positions.items():
            signal = self.latest_signals.get(symbol, {})
            current_score = int(signal.get("score") or 0)
            price = self.latest_prices.get(symbol)
            current_indicators = self.latest_indicators.get(symbol, [])
            current_indicator = current_indicators[-1] if current_indicators else None
            if candidate_score - current_score < ROTATION_MIN_SCORE_GAP:
                continue
            trend_failed = _position_structure_failed(position, signal)
            efficiency_stalled = (
                candidate_is_strong
                and price is not None
                and _position_reached_initial_r(position, price)
                and _position_h4_allows_efficiency_rotation(position, signal)
                and _position_efficiency_stalled(position, signal, current_indicator, price)
            )
            if not trend_failed and not efficiency_stalled:
                continue
            if not _rotation_efficiency_better(candidate_indicator, current_indicator):
                continue
            rotation_type = "trend invalidated" if trend_failed else "efficiency rotation"
            target = (position, current_score, rotation_type)
            if best_target is None or current_score < best_target[1]:
                best_target = target
        return best_target

    async def _btc_4h_extreme_volatility(self) -> bool:
        now = datetime.now(UTC)
        if (
            self._last_btc_extreme_check_at is not None
            and (now - self._last_btc_extreme_check_at).total_seconds() < FULL_DATA_CHECK_SECONDS
        ):
            return self._btc_extreme_cached
        try:
            candles = await self.market_data.klines("BTCUSDT", "4h", limit=2)
        except Exception:  # noqa: BLE001 - do not block trading on missing public filter data
            return self._btc_extreme_cached
        self._last_btc_extreme_check_at = now
        if not candles:
            return self._btc_extreme_cached
        candle = candles[-1]
        amplitude = (candle.high - candle.low) / candle.open if candle.open else 0.0
        body_move = abs(candle.close - candle.open) / candle.open if candle.open else 0.0
        self._btc_extreme_cached = (
            amplitude >= BTC_EXTREME_4H_AMPLITUDE
            or body_move >= BTC_EXTREME_4H_AMPLITUDE * 0.75
        )
        return self._btc_extreme_cached

    def _data_freshness_blocks(
        self,
        symbol: str,
        signal: dict[str, object] | None = None,
    ) -> tuple[str, ...]:
        now = datetime.now(UTC)
        blocks: list[str] = []
        price_updated_at = self._price_updated_at.get(symbol)
        if (
            price_updated_at is None
            or (now - price_updated_at).total_seconds() > MARKET_PRICE_STALE_SECONDS
        ):
            blocks.append("latest price is stale for more than 15 seconds")
        derivatives_updated_at = (
            self._oi_ratio_updated_at.get(symbol)
            or self._derivatives_updated_at.get(symbol)
        )
        if (
            derivatives_updated_at is None
            or (now - derivatives_updated_at).total_seconds() > DERIVATIVES_STALE_SECONDS
        ):
            blocks.append("OI/long-short ratio data is stale for more than 180 seconds")
        if not self._symbol_cache_valid(symbol):
            cache = self._timeframe_candles.get(symbol, {})
            for timeframe in _required_entry_timeframes(signal):
                if not _candles_contiguous(cache.get(timeframe, []), timeframe):
                    blocks.append(f"{timeframe} K-line context is missing or discontinuous")
        return tuple(blocks)

    def _manage_open_positions(self, *, fresh_only: bool = False) -> None:
        now = datetime.now(UTC)
        for position in list(self.account.positions.values()):
            price = self.latest_prices.get(position.symbol)
            if price is None:
                continue
            if fresh_only:
                updated_at = self._price_updated_at.get(position.symbol)
                if (
                    updated_at is None
                    or (now - updated_at).total_seconds() > MARKET_PRICE_STALE_SECONDS
                ):
                    continue
            indicators = self.latest_indicators.get(position.symbol, [])
            tf_indicators = self.latest_timeframe_indicators.get(position.symbol, {})
            trend_state = self.strategy.trend_state(position.side.value, indicators) if indicators else "CHOP"
            risk_state = self.strategy.risk_state(indicators) if indicators else "NORMAL"
            strong_trend = trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}
            signal = self.latest_signals.get(position.symbol, {})
            exit_indicator = _preferred_exit_indicator(tf_indicators, indicators, position)
            _update_position_excursions(position, price)
            profit_management_enabled = _position_profit_management_enabled(
                position,
                price,
            )
            short_support_hold = (
                _short_near_h4_support(position, price, signal, exit_indicator)
                and _short_support_trend_should_hold(position, price, signal, exit_indicator, trend_state)
            )
            if short_support_hold:
                _tighten_short_support_stop(position, price, signal, exit_indicator)
            if strong_trend:
                _protect_confirmed_breakout_position(position, price, signal, exit_indicator)
            if profit_management_enabled:
                _apply_profit_protection(position, price, signal, exit_indicator, self.settings.execution.taker_fee_rate)
            self._update_trailing_stop(position, price, strong_trend, exit_indicator)
            if (
                not position.first_tp_done
                and _take_profit_1_hit(position, price)
                and profit_management_enabled
            ):
                self._close_position_fraction_unlocked(
                    position,
                    price,
                    min(
                        self.settings.risk.first_take_profit_fraction,
                        position.remaining_fraction,
                    ),
                    "take profit: target 1 reached",
                )
                position.first_tp_done = True
                fee_buffer = (
                    self.settings.execution.taker_fee_rate * 2
                    + self.settings.execution.slippage_rate
                )
                if position.side == PositionSide.LONG:
                    position.stop_price = max(
                        position.stop_price,
                        position.entry_price * (1 + fee_buffer),
                    )
                else:
                    position.stop_price = min(
                        position.stop_price,
                        position.entry_price * (1 - fee_buffer),
                    )
                if position.remaining_fraction <= 1e-9:
                    self._mark_symbol_for_post_close_pool_review(
                        position.symbol
                    )
                    self.account.positions.pop(position.symbol, None)
                self._save_state_unlocked()
                continue
            handoff_exit_reason = _high_distribution_handoff_exit_reason(
                position,
                signal,
            ) if profit_management_enabled else None
            if handoff_exit_reason:
                self._close_position_unlocked(
                    position,
                    price,
                    handoff_exit_reason,
                )
                continue
            oi_valley_exit_reason = _oi_valley_short_exit_reason(
                position,
                signal,
            ) if profit_management_enabled else None
            if oi_valley_exit_reason:
                self._close_position_unlocked(
                    position,
                    price,
                    oi_valley_exit_reason,
                )
                continue
            structure_exit_reason = _confirmed_structure_exit_reason(position, price, signal)
            if structure_exit_reason:
                self._close_position_unlocked(position, price, structure_exit_reason)
                continue
            if _stop_hit(position, price):
                self._close_position_unlocked(position, price, _stop_exit_reason(position, signal, price, self.settings.execution.taker_fee_rate))
                continue
            if strong_trend and _strong_trend_invalidated(position, exit_indicator):
                self._close_position_unlocked(position, price, _outcome_exit_reason(position, price, "strong trend EMA50 structure invalidated", self.settings.execution.taker_fee_rate))
                continue
            drawdown_exit_reason = (
                None
                if short_support_hold or not profit_management_enabled
                else _profit_drawdown_exit_reason(position, price, signal, exit_indicator)
            )
            if drawdown_exit_reason:
                self._close_position_unlocked(position, price, drawdown_exit_reason)
                continue
            structure_take_profit_reason = (
                _structure_take_profit_reason(
                    position,
                    price,
                    signal,
                    exit_indicator,
                    trend_state,
                    self.settings.execution.taker_fee_rate,
                )
                if profit_management_enabled
                else None
            )
            if structure_take_profit_reason:
                self._close_position_unlocked(position, price, structure_take_profit_reason)
                continue
            risk_exit_reason = (
                _risk_exit_reason(position.side, trend_state, risk_state)
                if profit_management_enabled
                else None
            )
            if risk_exit_reason:
                self._close_position_unlocked(position, price, _risk_outcome_exit_reason(position, price, risk_exit_reason, self.settings.execution.taker_fee_rate))
                continue
            if _take_profit_hit(position, price) and not strong_trend:
                self._close_position_unlocked(position, price, "take profit: target 2 reached")

    def _update_trailing_stop(self, position: Position, price: float, strong_trend: bool, indicator: IndicatorSnapshot | None) -> None:
        stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
        if stop_distance <= 0:
            return
        profit_distance = price - position.entry_price if position.side == PositionSide.LONG else position.entry_price - price
        if profit_distance < stop_distance * self.settings.risk.trailing_activation_r:
            return
        if not strong_trend:
            return
        lock_distance = stop_distance * self.settings.risk.trailing_lock_r
        if position.side == PositionSide.LONG:
            position.stop_price = max(position.stop_price, position.entry_price + lock_distance)
            if indicator and indicator.ema20 is not None and indicator.atr14 is not None:
                position.stop_price = max(position.stop_price, indicator.ema20 - indicator.atr14 * 0.5)
        else:
            position.stop_price = min(position.stop_price, position.entry_price - lock_distance)
            if indicator and indicator.ema20 is not None and indicator.atr14 is not None:
                position.stop_price = min(position.stop_price, indicator.ema20 + indicator.atr14 * 0.5)

    def _add_to_strong_positions(self) -> None:
        status = self.status()
        equity = float(status["equity"])
        available = float(status["available_balance"])
        used_margin = float(status["used_margin"])
        remaining_total_margin = max(equity * PYRAMID_TOTAL_MARGIN_CAP - used_margin, 0.0)
        if min(available, remaining_total_margin) < 20:
            return
        ranked = sorted(
            self.account.positions.values(),
            key=lambda position: int(self.latest_signals.get(position.symbol, {}).get("score") or 0),
            reverse=True,
        )
        for position in ranked:
            signal = self.latest_signals.get(position.symbol, {})
            indicators = self.latest_indicators.get(position.symbol, [])
            indicator = indicators[-1] if indicators else None
            price = self.latest_prices.get(position.symbol)
            if price is None or indicator is None or not _pyramid_allowed(position, price, signal, indicator):
                continue
            current_margin = float(position.metadata.get("margin_usdt", 0.0))
            leverage = int(position.metadata.get("leverage", self.settings.risk.leverage_default))
            add_margin = min(current_margin * PYRAMID_MARGIN_FRACTION, equity * 0.10, available, remaining_total_margin)
            if add_margin < 20:
                continue
            notional = add_margin * leverage
            add_quantity = notional / price
            fee = notional * self.settings.execution.taker_fee_rate
            old_quantity = position.quantity
            position.quantity += add_quantity
            position.entry_price = ((position.entry_price * old_quantity) + (price * add_quantity)) / position.quantity
            position.metadata["margin_usdt"] = current_margin + add_margin
            position.metadata["adds"] = int(position.metadata.get("adds", 0)) + 1
            self.account.wallet_balance -= fee
            self.account.fees_paid += fee
            trend_state = str(signal.get("trend_state") or "CHOP")
            stop_loss, take_profit_1, take_profit_2 = _adaptive_exits(position.side, price, leverage, trend_state, indicator)
            if position.side == PositionSide.LONG:
                position.stop_price = max(position.stop_price, stop_loss)
                position.take_profit_1 = max(position.take_profit_1, take_profit_1)
                position.take_profit_2 = max(position.take_profit_2, take_profit_2)
            else:
                position.stop_price = min(position.stop_price, stop_loss)
                position.take_profit_1 = min(position.take_profit_1, take_profit_1)
                position.take_profit_2 = min(position.take_profit_2, take_profit_2)
            self.account.fills.append(
                PaperFill(
                    timestamp=datetime.now(UTC),
                    symbol=position.symbol,
                    side=position.side,
                    action="ADD",
                    price=price,
                    entry_price=position.entry_price,
                    quantity=add_quantity,
                    realized_pnl=0.0,
                    fee=fee,
                    reason=f"pyramid add: score={signal.get('score')}; state={trend_state}",
                    leverage=leverage,
                    margin_usdt=add_margin,
                    stop_price=position.stop_price,
                    take_profit_1=position.take_profit_1,
                take_profit_2=position.take_profit_2,
                opened_at=position.opened_at,
                entry_position=str(position.metadata.get("entry_position") or ""),
            )
            )
            self._save_state_unlocked()
            break

    def _close_position_unlocked(self, position: Position, price: float, reason: str) -> Trade:
        self._mark_symbol_for_post_close_pool_review(
            position.symbol
        )
        self.account.positions.pop(position.symbol, None)
        close_quantity = (
            position.quantity * max(position.remaining_fraction, 0.0)
        )
        gross_pnl = _pnl(
            position.side,
            position.entry_price,
            price,
            close_quantity,
        )
        notional = price * close_quantity
        fee = notional * self.settings.execution.taker_fee_rate
        realized = gross_pnl - fee
        reason = _strict_exit_reason_by_realized(reason, realized)
        margin_usdt = float(position.metadata.get("margin_usdt", 0.0))
        leverage = int(position.metadata.get("leverage", self.settings.risk.leverage_default))
        self.account.wallet_balance += realized
        self.account.realized_pnl += realized
        self.account.fees_paid += fee
        timestamp = datetime.now(UTC)
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=close_quantity,
            opened_at=position.opened_at,
            closed_at=timestamp,
            pnl=realized,
            reason=reason,
        )
        self.account.fills.append(
            PaperFill(
                timestamp=timestamp,
                symbol=position.symbol,
                side=position.side,
                action="CLOSE",
                price=price,
                entry_price=position.entry_price,
                quantity=close_quantity,
                realized_pnl=realized,
                fee=fee,
                reason=reason,
                leverage=leverage,
                margin_usdt=margin_usdt,
                stop_price=position.stop_price,
                take_profit_1=position.take_profit_1,
                take_profit_2=position.take_profit_2,
                opened_at=position.opened_at,
                entry_position=str(
                    position.metadata.get("entry_position")
                    or _entry_position_text(
                        position.side,
                        position.entry_price,
                        position.metadata.get("entry_context") if isinstance(position.metadata.get("entry_context"), dict) else {},
                        str(position.metadata.get("reason", "")),
                    )
                ),
                closed_at=timestamp,
                return_pct=realized / margin_usdt if margin_usdt else 0.0,
            )
        )
        if reason.lower().startswith("stop loss"):
            self._record_reentry_barrier(position)
        self._save_state_unlocked()
        return trade

    def _mark_symbol_for_post_close_pool_review(
        self,
        symbol: str,
    ) -> None:
        if not self._auto_universe:
            return
        self._post_close_pool_review.add(symbol)

    def _record_reentry_barrier(self, position: Position) -> None:
        timeframe = _exit_timeframe_from_position(position) or "1h"
        candles = self._timeframe_candles.get(position.symbol, {}).get(
            timeframe,
            [],
        )
        candle_timestamp = (
            candles[-1].timestamp
            if candles
            else datetime.now(UTC)
        )
        self.account.reentry_barriers[position.symbol] = {
            "timeframe": timeframe,
            "candle_timestamp": candle_timestamp.isoformat(),
        }

    def _reentry_block_reason(self, symbol: str) -> str | None:
        barrier = self.account.reentry_barriers.get(symbol)
        if not isinstance(barrier, dict):
            return None
        timeframe = str(barrier.get("timeframe") or "1h")
        try:
            blocked_candle = _parse_datetime(
                str(barrier.get("candle_timestamp") or "")
            )
        except (TypeError, ValueError):
            self.account.reentry_barriers.pop(symbol, None)
            return None
        candles = self._timeframe_candles.get(symbol, {}).get(timeframe, [])
        if candles and candles[-1].timestamp > blocked_candle:
            self.account.reentry_barriers.pop(symbol, None)
            return None
        return f"waiting for new {timeframe} closed candle after stop loss"

    def _close_position_fraction_unlocked(
        self,
        position: Position,
        price: float,
        fraction: float,
        reason: str,
    ) -> Trade:
        close_fraction = min(
            max(fraction, 0.0),
            max(position.remaining_fraction, 0.0),
        )
        close_quantity = position.quantity * close_fraction
        notional = price * close_quantity
        fee = notional * self.settings.execution.taker_fee_rate
        realized = (
            _pnl(
                position.side,
                position.entry_price,
                price,
                close_quantity,
            )
            - fee
        )
        remaining_before = max(position.remaining_fraction, 0.0)
        current_margin = float(position.metadata.get("margin_usdt", 0.0))
        closed_margin = (
            current_margin * close_fraction / remaining_before
            if remaining_before > 0
            else 0.0
        )
        position.remaining_fraction = max(
            remaining_before - close_fraction,
            0.0,
        )
        position.metadata["margin_usdt"] = max(
            current_margin - closed_margin,
            0.0,
        )
        self.account.wallet_balance += realized
        self.account.realized_pnl += realized
        self.account.fees_paid += fee
        timestamp = datetime.now(UTC)
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=close_quantity,
            opened_at=position.opened_at,
            closed_at=timestamp,
            pnl=realized,
            reason=reason,
        )
        self.account.fills.append(
            PaperFill(
                timestamp=timestamp,
                symbol=position.symbol,
                side=position.side,
                action="PARTIAL_CLOSE",
                price=price,
                entry_price=position.entry_price,
                quantity=close_quantity,
                realized_pnl=realized,
                fee=fee,
                reason=reason,
                leverage=int(
                    position.metadata.get(
                        "leverage",
                        self.settings.risk.leverage_default,
                    )
                ),
                margin_usdt=closed_margin,
                stop_price=position.stop_price,
                take_profit_1=position.take_profit_1,
                take_profit_2=position.take_profit_2,
                opened_at=position.opened_at,
                entry_position=str(
                    position.metadata.get("entry_position") or ""
                ),
                closed_at=timestamp,
                return_pct=(
                    realized / closed_margin
                    if closed_margin
                    else 0.0
                ),
            )
        )
        return trade

    def _available_balance_unlocked(self) -> float:
        status = self.status()
        return float(status["available_balance"])

    def _save_state_unlocked(self) -> None:
        if self.state_path is None:
            return
        self.account.latest_signals = {symbol: dict(signal) for symbol, signal in self.latest_signals.items()}
        payload = _paper_account_payload(self.account)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_name(f"{self.state_path.name}.tmp")
        backup_path = self.state_path.with_name(f"{self.state_path.name}.bak")
        backup_temp_path = backup_path.with_name(f"{backup_path.name}.tmp")
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        _write_text_synced(temp_path, serialized)
        if self.state_path.exists():
            try:
                json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                shutil.copy2(self.state_path, backup_temp_path)
                backup_temp_path.replace(backup_path)
        temp_path.replace(self.state_path)


def _pnl(side: PositionSide, entry_price: float, mark_price: float, quantity: float) -> float:
    if side == PositionSide.LONG:
        return (mark_price - entry_price) * quantity
    return (entry_price - mark_price) * quantity


def _stored_mark_price(position: Position) -> float | None:
    try:
        price = float(position.metadata.get("last_mark_price"))
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _score_from_reason(reason: str) -> str:
    marker = "score="
    if marker not in reason:
        return ""
    tail = reason.split(marker, 1)[1]
    score = []
    for char in tail:
        if not char.isdigit():
            break
        score.append(char)
    return "".join(score)


def _entry_reason_text(reason: str, entry_reasons: list[str] | tuple[str, ...] | None = None, score: object | None = None) -> str:
    reasons = [str(item).strip() for item in (entry_reasons or ()) if str(item).strip()]
    if "manual" in reason.lower() and not reasons:
        return "手动"
    score_text = str(score if score is not None else _score_from_reason(reason)).strip()
    prefix = f"自动，评分：{score_text}" if score_text else "自动"
    return f"{prefix}；{'；'.join(reasons)}" if reasons else prefix


def _normalize_entry_context(entry_context: dict[str, object] | None) -> dict[str, object]:
    context = dict(entry_context or {})
    context.setdefault("entry_setup", _entry_setup_from_context(context))
    context.setdefault("stop_timeframe", _stop_timeframe_from_context(context))
    return context


def _entry_position_text(
    side: PositionSide,
    entry_price: float,
    entry_context: dict[str, object] | None,
    reason: str = "",
) -> str:
    if "manual" in reason.lower():
        return f"手动开仓≈{_summary_price(entry_price)}"
    context = entry_context if isinstance(entry_context, dict) else {}
    levels = context.get("entry_levels") if isinstance(context.get("entry_levels"), dict) else {}
    side_key = "long" if side == PositionSide.LONG else "short"
    side_levels = levels.get(side_key) if isinstance(levels.get(side_key), dict) else {}
    labels = (
        {
            "h1_support": "1H支撑回踩",
            "h4_support": "4H支撑回踩",
            "h1_boll_mid": "1H BOLL中轨回踩",
            "h4_boll_mid": "4H BOLL中轨回踩",
            "h1_ema20_ema60": "1H EMA20/EMA60回踩",
            "h4_ema20_ema60": "4H EMA20/EMA60回踩",
            "h1_ema20": "1H EMA20回踩",
            "h1_ema60": "1H EMA60回踩",
            "h4_ema20": "4H EMA20回踩",
            "h4_ema60": "4H EMA60回踩",
            "m15_ema20_ema60": "强单边15m EMA20/EMA60回踩",
            "sweep_reclaim_support": "下插针扫损后收回支撑",
            "ma_cluster_breakout": "上穿均线密集区",
            "ma20_retest": "突破均线密集后回踩MA20",
            "breakout_retest": "前压力突破后回踩确认",
            "vwap_pullback": "VWAP/成交密集区回踩",
        }
        if side == PositionSide.LONG
        else {
            "distribution_range_high": "高位派发箱体上沿",
            "descending_high_trendline": "下降顶点趋势线",
            "h1_resistance": "1H压力反抽",
            "h4_resistance": "4H压力反抽",
            "h1_boll_mid": "1H BOLL中轨反抽",
            "h4_boll_mid": "4H BOLL中轨反抽",
            "h1_ema20_ema60": "1H EMA20/EMA60反抽",
            "h4_ema20_ema60": "4H EMA20/EMA60反抽",
            "h1_ema20": "1H EMA20反抽",
            "h1_ema60": "1H EMA60反抽",
            "h4_ema20": "4H EMA20反抽",
            "h4_ema60": "4H EMA60反抽",
            "sweep_reject_resistance": "上插针扫空后跌回压力",
            "ma_cluster_breakdown": "下穿均线密集区",
            "ma20_retest": "跌破均线密集后反抽MA20",
            "breakdown_retest": "前支撑跌破后反抽确认",
            "vwap_retest": "VWAP/成交密集区反抽",
        }
    )
    matched: dict[tuple[str, str], list[str]] = {}
    for key, label in labels.items():
        level = side_levels.get(key)
        if not _entry_level_hit(entry_price, level) or not isinstance(level, dict):
            continue
        low = _float_or_none(level.get("low"))
        high = _float_or_none(level.get("high"))
        point = _float_or_none(level.get("price"))
        values = [value for value in (low, high, point) if value is not None]
        if not values:
            continue
        zone_key = (_summary_price(min(values)), _summary_price(max(values)))
        matched.setdefault(zone_key, []).append(label)
    details = []
    for (low_text, high_text), names in list(matched.items())[:3]:
        zone_text = low_text if low_text == high_text else f"{low_text}-{high_text}"
        details.append(f"{'/'.join(names)}≈{zone_text}")
    actual = f"实际开仓≈{_summary_price(entry_price)}"
    return f"{'；'.join(details)}；{actual}" if details else actual


def _entry_setup_from_context(context: dict[str, object]) -> str:
    setup_type = str(context.get("setup_type") or "")
    if setup_type == SETUP_M15_SQUEEZE_TACTICAL_LONG:
        return "m15_precision_pullback"
    if setup_type.startswith("H4_"):
        return "h4_pullback"
    if setup_type in {
        SETUP_H1_PULLBACK_LONG,
        SETUP_H1_PULLBACK_SHORT,
        SETUP_OI_VALLEY_REVERSAL_LONG,
    }:
        return "h1_pullback"
    if setup_type in {
        SETUP_H1_STRUCTURE_LONG,
        SETUP_H1_STRUCTURE_SHORT,
        SETUP_DISTRIBUTION_STAGE1_SHORT,
        SETUP_DISTRIBUTION_STAGE2_SHORT,
        SETUP_DISTRIBUTION_STAGE3_SHORT,
    }:
        return "h1_structure"
    stop_basis = str(context.get("stop_basis") or "")
    if stop_basis == "15m_precision_structure":
        return "m15_precision_pullback"
    h1_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    h1_cluster_state = str(h1_cluster.get("state") or "NONE")
    h4_cluster_state = str(h4_cluster.get("state") or "NONE")
    if h1_cluster_state in {"BREAKOUT_UP", "BREAKDOWN_DOWN", "RETEST_MA20_LONG", "RETEST_MA20_SHORT"}:
        return "h1_ma_cluster"
    if h4_cluster_state in {"BREAKOUT_UP", "BREAKDOWN_DOWN", "RETEST_MA20_LONG", "RETEST_MA20_SHORT"}:
        return "h4_ma_cluster"
    h1_trigger = context.get("h1_trigger") if isinstance(context.get("h1_trigger"), dict) else {}
    h1_state = str(h1_trigger.get("state") or "NONE")
    if h1_state in {"BREAKOUT", "BREAKDOWN", "RETEST", "FAKE_BREAKOUT", "FAKE_BREAKDOWN"}:
        return "h1_structure"
    h1_pullback = context.get("h1_pullback") if isinstance(context.get("h1_pullback"), dict) else {}
    if str(h1_pullback.get("state") or "NONE") == "HEALTHY_PULLBACK":
        return "h1_pullback"
    if stop_basis == "1h_4h_retest_structure":
        return "h1_4h_retest"
    if stop_basis == "kc_atr_volatility":
        return "kc_atr_volatility"
    return "volatility_structure"


def _stop_timeframe_from_context(context: dict[str, object]) -> str:
    explicit = str(context.get("stop_timeframe") or "").lower()
    if explicit in {"15m", "1h", "4h", "1d"}:
        return explicit
    setup_type = str(context.get("setup_type") or "")
    if setup_type == SETUP_M15_SQUEEZE_TACTICAL_LONG:
        return "15m"
    if setup_type.startswith("H4_"):
        return "4h"
    stop_basis = str(context.get("stop_basis") or "")
    if stop_basis == "15m_precision_structure":
        return "15m"
    setup = str(context.get("entry_setup") or _entry_setup_from_context(context))
    if setup.startswith("h4_"):
        return "4h"
    if setup.startswith("h1_") or setup == "kc_atr_volatility":
        return "1h"
    h4_structure = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    if str(h4_structure.get("state") or "") in {"BREAKOUT_UP", "BREAKDOWN_DOWN"}:
        return "4h"
    return "1h"


def _entry_context_from_signal(signal: dict[str, object]) -> dict[str, object]:
    keys = (
        "action",
        "signal_timeframe",
        "trend_state",
        "regime",
        "risk_state",
        "rsi14",
        "volume_ratio",
        "oi_change",
        "long_short_ratio",
        "funding_rate",
        "daily_bias",
        "h4_structure",
        "h1_structure",
        "h4_ma_cluster",
        "h1_ma_cluster",
        "h4_oi",
        "h4_oi_valley",
        "h1_trigger",
        "h1_pullback",
        "h1_ema_reliability",
        "high_distribution_handoff",
        "entry_timeframe_override",
        "distribution_short",
        "distribution_short_stage",
        "m15_precision",
        "setup_type",
        "entry_levels",
    )
    return {key: signal[key] for key in keys if key in signal}


def _paper_account_payload(account: PaperAccount) -> dict[str, Any]:
    return {
        "starting_balance": account.starting_balance,
        "wallet_balance": account.wallet_balance,
        "realized_pnl": account.realized_pnl,
        "fees_paid": account.fees_paid,
        "positions": {symbol: _position_payload(position) for symbol, position in account.positions.items()},
        "fills": [_fill_state_payload(fill) for fill in account.fills],
        "daily_pnl_baselines": account.daily_pnl_baselines,
        "pnl_history": account.pnl_history,
        "latest_signals": account.latest_signals,
        "reentry_barriers": account.reentry_barriers,
    }


def _load_paper_account(path: Path | None) -> PaperAccount | None:
    if path is None:
        return None
    backup_path = path.with_name(f"{path.name}.bak")
    if not path.exists() and not backup_path.exists():
        return None
    errors: list[str] = []
    for candidate in (path, backup_path):
        if not candidate.exists():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise TypeError("state root must be a JSON object")
            return _paper_account_from_payload(raw)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate.name}: {exc}")
    detail = "; ".join(errors) or "no readable state or backup"
    raise PaperStateError(f"模拟账户状态损坏，已停止启动以防止资金重置：{detail}")


def _paper_account_from_payload(raw: dict[str, Any]) -> PaperAccount:
    starting_balance = float(raw.get("starting_balance", PAPER_DEFAULT_BALANCE))
    wallet_balance = float(raw.get("wallet_balance", starting_balance))
    if (
        not math.isfinite(starting_balance)
        or starting_balance <= 0
        or not math.isfinite(wallet_balance)
    ):
        raise ValueError("invalid account balance")
    return PaperAccount(
        starting_balance=starting_balance,
        wallet_balance=wallet_balance,
        realized_pnl=float(raw.get("realized_pnl", 0.0)),
        fees_paid=float(raw.get("fees_paid", 0.0)),
        positions={
            symbol: _position_from_payload(payload)
            for symbol, payload in dict(raw.get("positions") or {}).items()
        },
        fills=[
            _fill_from_payload(payload)
            for payload in list(raw.get("fills") or [])
        ],
        daily_pnl_baselines={
            str(key): float(value)
            for key, value in dict(
                raw.get("daily_pnl_baselines") or {}
            ).items()
        },
        pnl_history={
            str(key): float(value)
            for key, value in dict(raw.get("pnl_history") or {}).items()
        },
        latest_signals={
            str(symbol).upper(): dict(signal)
            for symbol, signal in dict(
                raw.get("latest_signals") or {}
            ).items()
            if isinstance(signal, dict)
        },
        reentry_barriers={
            str(symbol).upper(): {
                str(key): str(value)
                for key, value in dict(barrier).items()
            }
            for symbol, barrier in dict(
                raw.get("reentry_barriers") or {}
            ).items()
            if isinstance(barrier, dict)
        },
    )


def _write_text_synced(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _position_payload(position: Position) -> dict[str, Any]:
    return {
        "symbol": position.symbol,
        "side": position.side.value,
        "entry_price": position.entry_price,
        "quantity": position.quantity,
        "opened_at": position.opened_at.isoformat(),
        "stop_price": position.stop_price,
        "take_profit_1": position.take_profit_1,
        "take_profit_2": position.take_profit_2,
        "remaining_fraction": position.remaining_fraction,
        "first_tp_done": position.first_tp_done,
        "second_tp_done": position.second_tp_done,
        "bars_held": position.bars_held,
        "metadata": position.metadata,
    }


def _position_from_payload(payload: dict[str, Any]) -> Position:
    return Position(
        symbol=str(payload["symbol"]).upper(),
        side=PositionSide(str(payload["side"])),
        entry_price=float(payload["entry_price"]),
        quantity=float(payload["quantity"]),
        opened_at=_parse_datetime(payload["opened_at"]),
        stop_price=float(payload["stop_price"]),
        take_profit_1=float(payload["take_profit_1"]),
        take_profit_2=float(payload["take_profit_2"]),
        remaining_fraction=float(payload.get("remaining_fraction", 1.0)),
        first_tp_done=bool(payload.get("first_tp_done", False)),
        second_tp_done=bool(payload.get("second_tp_done", False)),
        bars_held=int(payload.get("bars_held", 0)),
        metadata=dict(payload.get("metadata") or {}),
    )


def _fill_state_payload(fill: PaperFill) -> dict[str, Any]:
    payload = asdict(fill)
    payload["side"] = fill.side.value
    for key in ("timestamp", "opened_at", "closed_at"):
        value = payload.get(key)
        payload[key] = value.isoformat() if isinstance(value, datetime) else None
    return payload


def _fill_from_payload(payload: dict[str, Any]) -> PaperFill:
    return PaperFill(
        timestamp=_parse_datetime(payload["timestamp"]),
        symbol=str(payload["symbol"]).upper(),
        side=PositionSide(str(payload["side"])),
        action=str(payload["action"]),
        price=float(payload["price"]),
        entry_price=float(payload["entry_price"]),
        quantity=float(payload["quantity"]),
        realized_pnl=float(payload["realized_pnl"]),
        fee=float(payload["fee"]),
        reason=str(payload.get("reason", "")),
        leverage=int(payload.get("leverage", 1)),
        margin_usdt=float(payload.get("margin_usdt", 0.0)),
        stop_price=float(payload.get("stop_price", 0.0)),
        take_profit_1=float(payload.get("take_profit_1", 0.0)),
        take_profit_2=float(payload.get("take_profit_2", 0.0)),
        opened_at=_parse_datetime(payload["opened_at"]),
        entry_position=str(payload.get("entry_position", "")),
        closed_at=_parse_datetime(payload["closed_at"]) if payload.get("closed_at") else None,
        return_pct=float(payload.get("return_pct", 0.0)),
    )


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _format_partial_market_errors(errors: list[str]) -> str | None:
    if not errors:
        return None
    sample = "; ".join(errors[:3])
    suffix = f" 等 {len(errors)} 个币种" if len(errors) > 3 else ""
    return f"部分币种行情临时断连，已跳过本轮并继续刷新：{sample}{suffix}"


def _stop_hit(position: Position, price: float) -> bool:
    if position.side == PositionSide.LONG:
        return price <= position.stop_price
    return price >= position.stop_price


def _take_profit_hit(position: Position, price: float) -> bool:
    if position.side == PositionSide.LONG:
        return price >= position.take_profit_2
    return price <= position.take_profit_2


def _take_profit_1_hit(position: Position, price: float) -> bool:
    if position.side == PositionSide.LONG:
        return price >= position.take_profit_1
    return price <= position.take_profit_1


def _strong_trend_invalidated(position: Position, indicator: IndicatorSnapshot | None) -> bool:
    if indicator is None or indicator.ema50 is None:
        return False
    buffer = (
        indicator.atr14 * 0.15
        if indicator.atr14 is not None and indicator.atr14 > 0
        else indicator.close * 0.002
    )
    if position.side == PositionSide.LONG:
        if indicator.close >= indicator.ema50 - buffer:
            return False
        return indicator.boll_mid is None or indicator.close < indicator.boll_mid - buffer
    if indicator.close <= indicator.ema50 + buffer:
        return False
    return indicator.boll_mid is None or indicator.close > indicator.boll_mid + buffer


def _position_reached_initial_r(
    position: Position,
    price: float,
    multiple: float = 1.0,
) -> bool:
    stop_distance = float(
        position.metadata.get("initial_stop_distance")
        or abs(position.entry_price - position.stop_price)
    )
    if stop_distance <= 0:
        return False
    favorable_distance = max(
        _position_current_profit_distance(position, price),
        float(position.metadata.get("max_favorable_distance") or 0.0),
    )
    return favorable_distance >= stop_distance * multiple


def _position_current_profit_distance(position: Position, price: float) -> float:
    if position.side == PositionSide.LONG:
        return price - position.entry_price
    return position.entry_price - price


def _position_profit_management_enabled(
    position: Position,
    price: float,
    multiple: float = 1.0,
) -> bool:
    return (
        _position_reached_initial_r(position, price, multiple)
        and _position_current_profit_distance(position, price) > 0
    )


def _apply_profit_protection(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
    fee_rate: float = 0.0,
) -> None:
    stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
    if stop_distance <= 0:
        return
    profit_distance = price - position.entry_price if position.side == PositionSide.LONG else position.entry_price - price
    if profit_distance <= 0:
        return
    risk_signal = _profit_risk_signal(signal, indicator)
    if profit_distance >= stop_distance * 1.5:
        lock_distance = stop_distance * (0.55 if risk_signal else 0.4)
        lock_distance = max(lock_distance, _profit_lock_price_buffer(position, fee_rate))
    elif profit_distance >= stop_distance:
        lock_distance = stop_distance * (0.25 if risk_signal else 0.05)
        lock_distance = max(lock_distance, _profit_lock_price_buffer(position, fee_rate))
    elif profit_distance >= stop_distance * 0.6:
        lock_distance = -stop_distance * (0.1 if risk_signal else 0.25)
    else:
        return
    if lock_distance > 0:
        min_lock = _profit_lock_price_buffer(position, fee_rate)
        max_lock = profit_distance * 0.85
        if max_lock < min_lock:
            return
        lock_distance = min(max(lock_distance, min_lock), max_lock)
    if position.side == PositionSide.LONG:
        position.stop_price = max(position.stop_price, position.entry_price + lock_distance)
    else:
        position.stop_price = min(position.stop_price, position.entry_price - lock_distance)


def _profit_lock_price_buffer(position: Position, fee_rate: float) -> float:
    return position.entry_price * (max(fee_rate, 0.0) * 2 + PROFIT_LOCK_SLIPPAGE_PCT)


def _profit_risk_signal(signal: dict[str, object], indicator: IndicatorSnapshot | None) -> bool:
    risk_state = str(signal.get("risk_state") or "NORMAL")
    trend_state = str(signal.get("trend_state") or "")
    if risk_state in {"LONG_CROWD", "SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
        return True
    if indicator is None:
        return False
    if indicator.rsi14 is not None:
        if trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}:
            if indicator.rsi14 >= RSI_STRONG_LONG_SEVERE or indicator.rsi14 <= RSI_STRONG_SHORT_SEVERE:
                return True
        elif indicator.rsi14 >= RSI_NORMAL_LONG_MAX or indicator.rsi14 <= RSI_NORMAL_SHORT_MIN:
            return True
    if indicator.oi_change is not None and indicator.oi_change <= -0.03:
        return True
    if indicator.volume_ratio is not None and indicator.volume_ratio >= 2.8:
        return True
    return False


def _update_position_excursions(position: Position, price: float) -> None:
    best_price = _float_or_none(position.metadata.get("best_price"))
    worst_price = _float_or_none(position.metadata.get("worst_price"))
    if position.side == PositionSide.LONG:
        best_price = max(best_price if best_price is not None else position.entry_price, price)
        worst_price = min(worst_price if worst_price is not None else position.entry_price, price)
        favorable = max(0.0, best_price - position.entry_price)
        adverse = max(0.0, position.entry_price - worst_price)
    else:
        best_price = min(best_price if best_price is not None else position.entry_price, price)
        worst_price = max(worst_price if worst_price is not None else position.entry_price, price)
        favorable = max(0.0, position.entry_price - best_price)
        adverse = max(0.0, worst_price - position.entry_price)
    position.metadata["best_price"] = best_price
    position.metadata["worst_price"] = worst_price
    position.metadata["max_favorable_distance"] = max(float(position.metadata.get("max_favorable_distance") or 0.0), favorable)
    position.metadata["max_adverse_distance"] = max(float(position.metadata.get("max_adverse_distance") or 0.0), adverse)


def _stop_exit_reason(
    position: Position,
    signal: dict[str, object],
    exit_price: float | None = None,
    fee_rate: float = 0.0,
) -> str:
    reference_price = position.stop_price if exit_price is None else exit_price
    stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
    if stop_distance > 0:
        if position.side == PositionSide.LONG and position.stop_price >= position.entry_price:
            if _estimated_exit_net_pnl(position, reference_price, fee_rate) > 0:
                return "take profit: protected stop after profit lock"
            return "stop loss: protected stop slipped below entry"
        if position.side == PositionSide.SHORT and position.stop_price <= position.entry_price:
            if _estimated_exit_net_pnl(position, reference_price, fee_rate) > 0:
                return "take profit: protected stop after profit lock"
            return "stop loss: protected stop slipped below entry"
    entry_context = position.metadata.get("entry_context") if isinstance(position.metadata.get("entry_context"), dict) else {}
    if position.side == PositionSide.LONG and str(entry_context.get("stop_basis") or "") == "15m_precision_structure":
        return "stop loss: 15m entry structure stop"
    if position.metadata.get("breakout_protected"):
        return _outcome_exit_reason(position, reference_price, "breakout protection stop", fee_rate)
    if position.metadata.get("short_support_protected"):
        return _outcome_exit_reason(position, reference_price, "short trend support protection stop", fee_rate)
    if _position_structure_failed(position, signal):
        return "stop loss: signal direction or structure failed"
    return "stop loss: ATR volatility hard stop"


def _outcome_exit_reason(position: Position, price: float, detail: str, fee_rate: float = 0.0) -> str:
    prefix = "take profit" if _estimated_exit_net_pnl(position, price, fee_rate) > 0 else "stop loss"
    return f"{prefix}: {detail}"


def _strict_exit_reason_by_realized(reason: str, realized: float) -> str:
    text = (reason or "").strip()
    if not text:
        return "take profit" if realized > 0 else "stop loss"
    lower = text.lower()
    if lower.startswith(("rotation exit:", "pyramid add:", "manual")):
        return text
    if lower.startswith(("take profit:", "stop loss:")):
        detail = text.split(":", 1)[1].strip()
        prefix = "take profit" if realized > 0 else "stop loss"
        return f"{prefix}: {detail}" if detail else prefix
    return text


def _estimated_exit_net_pnl(position: Position, price: float, fee_rate: float = 0.0) -> float:
    gross = _pnl(position.side, position.entry_price, price, position.quantity)
    fees = (position.entry_price + price) * abs(position.quantity) * max(fee_rate, 0.0)
    slippage = price * abs(position.quantity) * PROFIT_LOCK_SLIPPAGE_PCT
    return gross - fees - slippage


def _risk_outcome_exit_reason(position: Position, price: float, risk_reason: str, fee_rate: float = 0.0) -> str:
    detail = risk_reason.replace("risk exit:", "").strip()
    if detail == "LONG_CROWD":
        text = "long crowd risk"
    elif detail == "SHORT_CROWD":
        text = "short crowd risk"
    elif detail == "OI_ABNORMAL":
        text = "OI abnormal risk"
    elif detail == "FUNDING_HOT":
        text = "funding overheated risk"
    else:
        text = detail or "risk exit"
    return _outcome_exit_reason(position, price, text, fee_rate)


def _profit_drawdown_exit_reason(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
) -> str | None:
    stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
    if stop_distance <= 0:
        return None
    current_profit = price - position.entry_price if position.side == PositionSide.LONG else position.entry_price - price
    if current_profit <= 0:
        return None
    max_favorable = float(position.metadata.get("max_favorable_distance") or current_profit)
    if max_favorable < stop_distance:
        return None
    drawdown = max_favorable - current_profit
    if _profit_risk_signal(signal, indicator) and drawdown >= max_favorable * 0.28:
        return _outcome_exit_reason(position, price, f"profit drawdown after {_profit_risk_detail(signal, indicator)}")
    if max_favorable >= stop_distance * 1.5 and current_profit <= max(stop_distance * 0.35, max_favorable * 0.42):
        return _outcome_exit_reason(position, price, "floating profit drawdown protection")
    return None


def _profit_risk_detail(signal: dict[str, object], indicator: IndicatorSnapshot | None) -> str:
    risk_state = str(signal.get("risk_state") or "NORMAL")
    if risk_state == "LONG_CROWD":
        return "long crowd risk"
    if risk_state == "SHORT_CROWD":
        return "short crowd risk"
    if risk_state == "OI_ABNORMAL":
        return "OI abnormal risk"
    if risk_state == "FUNDING_HOT":
        return "funding overheated risk"
    if indicator and indicator.oi_change is not None and indicator.oi_change <= -0.03:
        return "OI drop risk"
    if indicator and indicator.volume_ratio is not None and indicator.volume_ratio >= 2.8:
        return "volume blow-off risk"
    if indicator and indicator.rsi14 is not None:
        if indicator.rsi14 >= RSI_NORMAL_LONG_MAX:
            return "RSI overheated risk"
        if indicator.rsi14 <= RSI_NORMAL_SHORT_MIN:
            return "RSI oversold risk"
    return "risk signal"


def _structure_take_profit_reason(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
    trend_state: str = "CHOP",
    fee_rate: float = 0.0,
) -> str | None:
    stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
    if stop_distance <= 0:
        return None
    current_profit = price - position.entry_price if position.side == PositionSide.LONG else position.entry_price - price
    if current_profit < stop_distance:
        return None
    h4 = signal.get("h4_structure") if isinstance(signal.get("h4_structure"), dict) else {}
    buffer = _exit_structure_buffer(price, indicator)
    if position.side == PositionSide.LONG:
        resistance = _float_or_none(h4.get("resistance"))
        if resistance is not None and price >= resistance - buffer:
            if _profit_risk_signal(signal, indicator) or current_profit >= stop_distance * 1.2:
                return _outcome_exit_reason(position, price, "near 4h resistance with profit protection")
    else:
        support = _float_or_none(h4.get("support"))
        if support is not None and price <= support + buffer:
            if _short_support_trend_should_hold(position, price, signal, indicator, trend_state):
                _tighten_short_support_stop(position, price, signal, indicator)
                return None
            if _short_support_exhaustion_confirmed(price, signal, indicator):
                h4_oi_valley = (
                    signal.get("h4_oi_valley")
                    if isinstance(signal.get("h4_oi_valley"), dict)
                    else {}
                )
                if str(h4_oi_valley.get("state") or "") == "CONFIRMED":
                    return _outcome_exit_reason(
                        position,
                        price,
                        "4h OI valley formed; downside trend exhausted",
                        fee_rate,
                    )
                return _outcome_exit_reason(position, price, "4h support plus short exhaustion confirmed", fee_rate)
            if trend_state not in {"TREND_SHORT", "ONE_WAY_DOWN"} and _profit_risk_signal(signal, indicator):
                return _outcome_exit_reason(position, price, "near 4h support with profit protection", fee_rate)
    return None


def _exit_structure_buffer(price: float, indicator: IndicatorSnapshot | None) -> float:
    if indicator and indicator.atr14:
        return max(indicator.atr14 * 0.8, price * 0.003)
    return price * 0.006


def _short_support_trend_should_hold(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
    trend_state: str,
) -> bool:
    if position.side != PositionSide.SHORT:
        return False
    if trend_state not in {"TREND_SHORT", "ONE_WAY_DOWN"}:
        return False
    return not _short_support_exhaustion_confirmed(price, signal, indicator)


def _short_near_h4_support(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
) -> bool:
    if position.side != PositionSide.SHORT:
        return False
    h4 = signal.get("h4_structure") if isinstance(signal.get("h4_structure"), dict) else {}
    support = _float_or_none(h4.get("support"))
    if support is None:
        return False
    return price <= support + _exit_structure_buffer(price, indicator)


def _short_support_exhaustion_confirmed(
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
) -> bool:
    risk_state = str(signal.get("risk_state") or "NORMAL")
    h4_oi_valley = signal.get("h4_oi_valley") if isinstance(signal.get("h4_oi_valley"), dict) else {}
    reasons_text = " | ".join(str(reason).lower() for reason in (signal.get("reasons") or ()))
    rsi_extreme = bool(indicator and indicator.rsi14 is not None and indicator.rsi14 <= RSI_STRONG_SHORT_SEVERE)
    short_crowded = risk_state == "SHORT_CROWD" or "short crowd" in reasons_text or "空头拥挤" in reasons_text
    oi_valley = str(h4_oi_valley.get("state") or "") == "CONFIRMED"
    h1_reclaimed = False
    if indicator:
        levels = [value for value in (indicator.ema20, indicator.boll_mid) if value is not None]
        if levels:
            h1_reclaimed = price >= min(levels)
    downside_reclaim = any(
        token in reasons_text
        for token in (
            "lower wick sweep reclaimed",
            "downside sweep reclaimed",
            "capitulation absorption",
            "support held",
            "下插针",
            "收回支撑",
            "支撑收回",
        )
    )
    return oi_valley or (
        (rsi_extreme or short_crowded)
        and (h1_reclaimed or downside_reclaim)
    )


def _tighten_short_support_stop(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
) -> None:
    if position.side != PositionSide.SHORT or indicator is None or not indicator.atr14:
        return
    initial_stop_distance = float(
        position.metadata.get("initial_stop_distance")
        or abs(position.entry_price - position.stop_price)
    )
    max_favorable_distance = float(
        position.metadata.get("max_favorable_distance") or 0.0
    )
    if (
        not position.first_tp_done
        and (
            initial_stop_distance <= 0
            or max_favorable_distance < initial_stop_distance
        )
    ):
        return
    candidates: list[float] = []
    if indicator.ema20 is not None:
        candidates.append(indicator.ema20 + indicator.atr14 * 0.45)
    if indicator.ema50 is not None:
        candidates.append(indicator.ema50 + indicator.atr14 * 0.35)
    h1 = signal.get("h1_trigger") if isinstance(signal.get("h1_trigger"), dict) else {}
    h4 = signal.get("h4_structure") if isinstance(signal.get("h4_structure"), dict) else {}
    for key in ("resistance_zone_low", "resistance", "resistance_zone_high"):
        value = _float_or_none(h1.get(key))
        if value is not None:
            candidates.append(value + indicator.atr14 * 0.25)
    for key in ("resistance_zone_low", "resistance", "resistance_zone_high"):
        value = _float_or_none(h4.get(key))
        if value is not None:
            candidates.append(value + indicator.atr14 * 0.35)
    worst_price = _float_or_none(position.metadata.get("worst_price"))
    if worst_price is not None and worst_price > price:
        candidates.append(worst_price + indicator.atr14 * 0.2)
    valid = [candidate for candidate in candidates if candidate > price]
    if not valid:
        return
    protected_stop = min(valid)
    if protected_stop < position.stop_price:
        position.stop_price = protected_stop
        position.metadata["short_support_protected"] = True


def _confirmed_structure_exit_reason(position: Position, price: float, signal: dict[str, object]) -> str | None:
    """Exit only when 1h/4h structure is lost by close/body, not by a single wick."""
    h4 = signal.get("h4_structure") if isinstance(signal.get("h4_structure"), dict) else {}
    h1 = signal.get("h1_trigger") if isinstance(signal.get("h1_trigger"), dict) else {}
    h4_state = str(h4.get("state") or "UNKNOWN")
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    if position.side == PositionSide.LONG:
        h4_failed = h4_state == "BREAKDOWN_DOWN"
        h1_failed = h1_direction == "SHORT" and h1_state in {"BREAKDOWN", "RETEST"}
        if h4_failed or h1_failed:
            return _outcome_exit_reason(position, price, "1h/4h body closed below support or EMA/BOLL zone")
    else:
        h4_failed = h4_state == "BREAKOUT_UP"
        h1_failed = h1_direction == "LONG" and h1_state in {"BREAKOUT", "RETEST"}
        if h4_failed or h1_failed:
            return _outcome_exit_reason(position, price, "1h/4h body closed above resistance or EMA/BOLL zone")
    return None


def _pyramid_allowed(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot,
) -> bool:
    if position.first_tp_done:
        return False
    if int(position.metadata.get("adds", 0)) >= PYRAMID_MAX_ADDS:
        return False
    if int(signal.get("score") or 0) < PYRAMID_MIN_SCORE:
        return False
    if str(signal.get("risk_state") or "NORMAL") != "NORMAL":
        return False
    if _trend_stage_from_signal(signal, indicator) == TREND_STAGE_LATE:
        return False
    trend_state = str(signal.get("trend_state") or "")
    if position.side == PositionSide.LONG and trend_state != "ONE_WAY_UP":
        return False
    if position.side == PositionSide.SHORT and trend_state != "ONE_WAY_DOWN":
        return False
    if indicator.ema20 is None or indicator.atr14 is None or indicator.atr14 <= 0:
        return False
    if not _pyramid_structure_confirmed(position.side, signal):
        return False
    stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
    if stop_distance <= 0:
        return False
    profit_distance = price - position.entry_price if position.side == PositionSide.LONG else position.entry_price - price
    if profit_distance < stop_distance * 0.8:
        return False
    if position.side == PositionSide.LONG:
        near_support = price >= indicator.ema20 and (price - indicator.ema20) <= indicator.atr14 * 1.2
    else:
        near_support = price <= indicator.ema20 and (indicator.ema20 - price) <= indicator.atr14 * 1.2
    volume_ok = indicator.volume_ratio is None or indicator.volume_ratio >= 0.9
    return near_support and volume_ok


def _pyramid_structure_confirmed(side: PositionSide, signal: dict[str, object]) -> bool:
    h4_oi = signal.get("h4_oi") if isinstance(signal.get("h4_oi"), dict) else {}
    h1_pullback = signal.get("h1_pullback") if isinstance(signal.get("h1_pullback"), dict) else {}
    h1_trigger = signal.get("h1_trigger") if isinstance(signal.get("h1_trigger"), dict) else {}
    reasons = " | ".join(str(reason) for reason in signal.get("reasons") or ())
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    h1_direction = str(h1_trigger.get("direction") or "NONE")
    h1_state = str(h1_trigger.get("state") or "UNKNOWN")
    h4_oi_state = str(h4_oi.get("state") or "UNKNOWN")
    if side == PositionSide.LONG:
        pullback_ok = pullback_direction == "LONG" and pullback_state in {"HEALTHY_PULLBACK", "HIGH_PULLBACK"}
        washout_ok = "washout confirmed: downside wick swept support" in reasons or "downside sweep reclaimed support" in reasons
        breakout_ok = h4_oi_state == "REBUILD_BREAKOUT_LONG" or (
            h1_direction == "LONG"
            and h1_state in {"BREAKOUT", "RETEST"}
            and "market structure: resistance grind broke upward" in reasons
        )
        deleverage_ok = h4_oi_state in {"NORMAL", "DELEVERAGE_HOLD_LONG", "DELEVERAGE_CROWD_HOLD_LONG", "REBUILD_BREAKOUT_LONG"}
        return (pullback_ok or washout_ok or breakout_ok) and deleverage_ok
    pullback_ok = pullback_direction == "SHORT" and pullback_state == "HEALTHY_PULLBACK"
    washout_ok = "washout confirmed: upside wick swept resistance" in reasons or "upside sweep rejected resistance" in reasons
    failed_retest_ok = h1_direction == "SHORT" and h1_state in {"RETEST", "FAKE_BREAKOUT"}
    deleverage_ok = h4_oi_state not in {"DELEVERAGE_HOLD_LONG", "DELEVERAGE_CROWD_HOLD_LONG"}
    return (pullback_ok or washout_ok or failed_retest_ok) and deleverage_ok


def _protect_confirmed_breakout_position(position: Position, price: float, signal: dict[str, object], indicator: IndicatorSnapshot | None) -> None:
    if indicator is None or indicator.atr14 is None or indicator.atr14 <= 0:
        return
    stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
    if stop_distance <= 0:
        return
    profit_distance = price - position.entry_price if position.side == PositionSide.LONG else position.entry_price - price
    if profit_distance < stop_distance * 0.6:
        return
    h4 = signal.get("h4_structure") if isinstance(signal.get("h4_structure"), dict) else {}
    h1 = signal.get("h1_trigger") if isinstance(signal.get("h1_trigger"), dict) else {}
    reasons = " | ".join(str(reason) for reason in signal.get("reasons") or ())
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    h4_state = str(h4.get("state") or "UNKNOWN")
    if position.side == PositionSide.LONG:
        confirmed = (
            h4_state == "BREAKOUT_UP"
            or (h1_direction == "LONG" and h1_state in {"BREAKOUT", "RETEST"})
            or "market structure: resistance grind broke upward" in reasons
        )
        if not confirmed:
            return
        resistance = _float_or_none(h4.get("resistance"))
        candidates = []
        if resistance:
            candidates.append(resistance - indicator.atr14 * 0.6)
        if indicator.ema20 is not None:
            candidates.append(indicator.ema20 - indicator.atr14 * 0.5)
        if candidates:
            position.stop_price = max(position.stop_price, min(candidates))
            position.metadata["breakout_protected"] = True
    else:
        confirmed = h4_state == "BREAKDOWN_DOWN" or (h1_direction == "SHORT" and h1_state in {"BREAKDOWN", "RETEST", "FAKE_BREAKOUT"})
        if not confirmed:
            return
        support = _float_or_none(h4.get("support"))
        candidates = []
        if support:
            candidates.append(support + indicator.atr14 * 0.6)
        if indicator.ema20 is not None:
            candidates.append(indicator.ema20 + indicator.atr14 * 0.5)
        if candidates:
            position.stop_price = min(position.stop_price, max(candidates))
            position.metadata["breakout_protected"] = True


def _adaptive_exits(
    side: PositionSide,
    price: float,
    leverage: int,
    trend_state: str,
    indicators: IndicatorSnapshot | None,
) -> tuple[float, float, float]:
    if trend_state == "CHOP":
        stop_pct, tp1_r, tp2_r = 0.018, 0.8, 1.2
    elif trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}:
        stop_pct, tp1_r, tp2_r = 0.030, 2.0, 3.4
    else:
        stop_pct, tp1_r, tp2_r = 0.024, 1.0, 1.8

    min_stop_pct, tp_boost = _volatility_exit_profile(indicators)
    stop_pct = max(stop_pct, min_stop_pct)
    tp1_r *= tp_boost
    tp2_r *= tp_boost
    leverage_cap = _stop_pct_for_leverage(leverage) * 4.0
    stop_pct = min(stop_pct, leverage_cap)
    stop = _stop_from_pct(side, price, stop_pct)
    if indicators and indicators.atr14:
        atr_stop = _stop_from_atr(side, price, indicators.atr14, _atr_stop_multiple(indicators))
        if side == PositionSide.LONG:
            stop = min(stop, atr_stop)
        else:
            stop = max(stop, atr_stop)
    tp1 = _take_profit_from_r(side, price, stop, tp1_r)
    tp2 = _take_profit_from_r(side, price, stop, tp2_r)
    return stop, tp1, tp2


def _volatility_exit_profile(indicator: IndicatorSnapshot | None) -> tuple[float, float]:
    if indicator is None or not indicator.close or not indicator.atr14:
        return 0.0, 1.0
    atr_pct = indicator.atr14 / indicator.close
    if atr_pct < 0.008:
        return 0.015, 0.9
    if atr_pct < 0.02:
        return 0.025, 1.0
    if atr_pct < 0.04:
        return 0.040, 1.2
    return 0.045, 1.5


def _atr_stop_multiple(indicator: IndicatorSnapshot) -> float:
    if not indicator.close or not indicator.atr14:
        return 1.2
    atr_pct = indicator.atr14 / indicator.close
    if atr_pct < 0.008:
        return 1.4
    if atr_pct < 0.02:
        return 1.6
    if atr_pct < 0.04:
        return 1.8
    return 2.0


def _risk_exit_reason(side: PositionSide, trend_state: str, risk_state: str) -> str | None:
    if trend_state == "ONE_WAY_UP" and side == PositionSide.LONG and risk_state in {"LONG_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
        return f"risk exit: {risk_state}"
    if trend_state == "ONE_WAY_DOWN" and side == PositionSide.SHORT and risk_state in {"SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
        return f"risk exit: {risk_state}"
    return None


def _high_distribution_handoff_exit_reason(
    position: Position,
    signal: dict[str, object],
) -> str | None:
    if position.side != PositionSide.LONG:
        return None
    handoff = (
        signal.get("high_distribution_handoff")
        if isinstance(signal.get("high_distribution_handoff"), dict)
        else {}
    )
    if str(handoff.get("state") or "") != "CONFIRMED":
        return None
    stop_distance = float(
        position.metadata.get("initial_stop_distance")
        or abs(position.entry_price - position.stop_price)
    )
    favorable_distance = float(
        position.metadata.get("max_favorable_distance") or 0.0
    )
    if stop_distance <= 0 or favorable_distance < stop_distance:
        return None
    return "take profit: high distribution handoff complete"


def _oi_valley_short_exit_reason(
    position: Position,
    signal: dict[str, object],
) -> str | None:
    if position.side != PositionSide.SHORT:
        return None
    stop_distance = float(
        position.metadata.get("initial_stop_distance")
        or abs(position.entry_price - position.stop_price)
    )
    favorable_distance = float(
        position.metadata.get("max_favorable_distance") or 0.0
    )
    if stop_distance <= 0 or favorable_distance < stop_distance:
        return None
    h4_oi_valley = (
        signal.get("h4_oi_valley")
        if isinstance(signal.get("h4_oi_valley"), dict)
        else {}
    )
    if str(h4_oi_valley.get("state") or "") != "CONFIRMED":
        return None
    return "take profit: 4h OI valley formed; downside trend exhausted"


def _trend_stage_from_signal(signal: dict[str, object], indicator: IndicatorSnapshot | None = None) -> str:
    stored = str(signal.get("trend_stage_phase") or "")
    if stored in {TREND_STAGE_EARLY, TREND_STAGE_MID, TREND_STAGE_LATE, TREND_STAGE_NEUTRAL}:
        return stored
    side = _signal_side(signal)
    if side is None:
        return TREND_STAGE_NEUTRAL
    trend_state = str(signal.get("trend_state") or signal.get("regime") or "")
    risk_state = str(signal.get("risk_state") or "NORMAL")
    rsi14 = _float_or_none(signal.get("rsi14"))
    price = _float_or_none(signal.get("price"))
    reasons_text = " | ".join(str(reason).lower() for reason in signal.get("reasons") or ())
    smart_money_phase = str(signal.get("smart_money_phase") or "")
    if indicator is not None:
        rsi14 = rsi14 if rsi14 is not None else indicator.rsi14
        price = price if price is not None else indicator.close
    if _late_stage_risk(
        side,
        risk_state,
        rsi14,
        price,
        indicator,
        reasons_text,
        allow_short_squeeze=(
            side == PositionSide.LONG
            and smart_money_phase == "SHORT_SQUEEZE_MARKUP"
        ),
    ):
        return TREND_STAGE_LATE
    h1 = signal.get("h1_trigger") if isinstance(signal.get("h1_trigger"), dict) else {}
    h1_pullback = signal.get("h1_pullback") if isinstance(signal.get("h1_pullback"), dict) else {}
    h4_oi = signal.get("h4_oi") if isinstance(signal.get("h4_oi"), dict) else {}
    h1_state = str(h1.get("state") or "UNKNOWN")
    h1_direction = str(h1.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    oi_state = str(h4_oi.get("state") or "UNKNOWN")
    expected = "LONG" if side == PositionSide.LONG else "SHORT"
    if h1_direction == expected and h1_state in {"BREAKOUT", "BREAKDOWN", "RETEST", "FAKE_BREAKDOWN", "FAKE_BREAKOUT"}:
        return TREND_STAGE_EARLY
    if oi_state in {"REBUILD_BREAKOUT_LONG", "DELEVERAGE_HOLD_LONG", "DELEVERAGE_CROWD_HOLD_LONG"}:
        return TREND_STAGE_EARLY if side == PositionSide.LONG else TREND_STAGE_NEUTRAL
    if trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}:
        if pullback_direction == expected and pullback_state in {"HEALTHY_PULLBACK", "HIGH_PULLBACK", "LOW_PULLBACK"}:
            return TREND_STAGE_MID
        return TREND_STAGE_MID
    return TREND_STAGE_NEUTRAL


def _signal_side(signal: dict[str, object]) -> PositionSide | None:
    action = str(signal.get("action") or "")
    if action == SignalAction.ENTRY_LONG.value:
        return PositionSide.LONG
    if action == SignalAction.ENTRY_SHORT.value:
        return PositionSide.SHORT
    return None


def _late_stage_risk(
    side: PositionSide,
    risk_state: str,
    rsi14: float | None,
    price: float | None,
    indicator: IndicatorSnapshot | None,
    reasons_text: str,
    *,
    allow_short_squeeze: bool = False,
) -> bool:
    if risk_state in {"LONG_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
        return True
    if risk_state == "SHORT_CROWD" and not allow_short_squeeze:
        return True
    if side == PositionSide.LONG and rsi14 is not None and rsi14 >= RSI_STRONG_LONG_HARD:
        return True
    if side == PositionSide.SHORT and rsi14 is not None and rsi14 <= RSI_STRONG_SHORT_HARD:
        return True
    if indicator is not None and price is not None and indicator.ema20 and indicator.atr14:
        extension = abs(price - indicator.ema20)
        if extension >= indicator.atr14 * 2.4:
            return True
    if side == PositionSide.LONG and any(token in reasons_text for token in ("upper wick", "far above vwap", "long crowd")):
        return True
    if side == PositionSide.SHORT and any(token in reasons_text for token in ("lower wick", "far below vwap", "short crowd")):
        return True
    return False


def _entry_reward_r(
    signal: dict[str, object],
    side: PositionSide,
    price: float,
    stop: float,
    *,
    timeframe: str = "1h",
    planned_target: float | None = None,
    round_trip_cost_rate: float = 0.0,
) -> float | None:
    trading_cost = price * max(round_trip_cost_rate, 0.0)
    risk = abs(price - stop) + trading_cost
    if risk <= 0:
        return None
    target = (
        None
        if timeframe == "15m"
        else _nearest_structure_target(
            signal,
            side,
            price,
            timeframe=timeframe,
        )
    )
    if target is None and _target_is_profitable(side, price, planned_target):
        target = planned_target
    if target is None:
        return None
    reward = (
        target - price
        if side == PositionSide.LONG
        else price - target
    ) - trading_cost
    if reward <= 0:
        return 0.0
    return reward / risk


def _take_profits_for_final_stop(
    signal: dict[str, object],
    side: PositionSide,
    entry_price: float,
    stop_price: float,
    *,
    timeframe: str,
) -> tuple[float, float]:
    risk_distance = abs(entry_price - stop_price)
    if risk_distance <= 0:
        return entry_price, entry_price
    target = (
        None
        if timeframe == "15m"
        else _nearest_structure_target(
            signal,
            side,
            entry_price,
            timeframe=timeframe,
        )
    )
    if target is None:
        target = _take_profit_from_r(
            side,
            entry_price,
            stop_price,
            2.0,
        )
    target_r = abs(target - entry_price) / risk_distance
    first_r = 1.0 if target_r > 1.0 else target_r * 0.5
    take_profit_1 = _take_profit_from_r(
        side,
        entry_price,
        stop_price,
        first_r,
    )
    return take_profit_1, target


def _nearest_structure_target(
    signal: dict[str, object],
    side: PositionSide,
    price: float,
    *,
    timeframe: str = "1h",
) -> float | None:
    primary_timeframe = "4h" if timeframe == "4h" else "1h"
    candidates = _timeframe_target_candidates(
        signal,
        side,
        price,
        primary_timeframe,
    )
    if not candidates:
        return None
    return min(candidates) if side == PositionSide.LONG else max(candidates)


def _timeframe_target_candidates(
    signal: dict[str, object],
    side: PositionSide,
    price: float,
    timeframe: str,
) -> list[float]:
    candidates: list[float] = []
    structure_keys = (
        ("h4_structure",)
        if timeframe == "4h"
        else ("h1_structure", "h1_trigger")
    )
    for key in structure_keys:
        mapping = signal.get(key) if isinstance(signal.get(key), dict) else {}
        candidates.extend(_structure_target_candidates(mapping, side, price))
    cluster_key = "h4_ma_cluster" if timeframe == "4h" else "h1_ma_cluster"
    for key in (cluster_key,):
        cluster = signal.get(key) if isinstance(signal.get(key), dict) else {}
        target_key = "target_up" if side == PositionSide.LONG else "target_down"
        target = _float_or_none(cluster.get(target_key))
        if target is not None and ((side == PositionSide.LONG and target > price) or (side == PositionSide.SHORT and target < price)):
            candidates.append(target)
    return candidates


def _structure_target_candidates(mapping: dict[str, object], side: PositionSide, price: float) -> list[float]:
    if side == PositionSide.LONG:
        keys = ("resistance", "resistance_zone_high")
        return [value for value in (_float_or_none(mapping.get(key)) for key in keys) if value is not None and value > price]
    keys = ("support", "support_zone_low")
    return [value for value in (_float_or_none(mapping.get(key)) for key in keys) if value is not None and value < price]


def _auto_signal_allowed(signal: dict[str, object]) -> bool:
    if signal.get("vetoes"):
        return False
    return not _auto_entry_prerequisite_blocks(signal)


def _auto_entry_prerequisite_blocks(
    signal: dict[str, object],
    *,
    include_entry_position: bool = True,
) -> tuple[str, ...]:
    blocks: list[str] = []
    action = str(signal.get("action") or "")
    score = int(signal.get("score") or 0)
    is_entry = action in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}
    if not is_entry:
        blocks.append("directional entry signal not established")
    if score < AUTO_ENTRY_MIN_SCORE:
        blocks.append(f"final score {score} below auto-entry minimum {AUTO_ENTRY_MIN_SCORE}")
    entry_position_block = (
        _auto_entry_position_block(signal)
        if include_entry_position
        else None
    )
    if entry_position_block:
        existing_vetoes = {str(reason) for reason in (signal.get("vetoes") or ())}
        if entry_position_block not in existing_vetoes:
            blocks.append(entry_position_block)
    return tuple(dict.fromkeys(blocks))


def _auto_entry_position_block(signal: dict[str, object]) -> str | None:
    action = str(signal.get("action") or "")
    if action not in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}:
        return None
    entry_position, entry_reason = _signal_entry_timing(signal)
    if entry_position == ENTRY_TIMING_GOOD:
        return None
    return entry_reason or entry_position


def _auto_entry_status_signal(
    symbol: str,
    signal: dict[str, object],
    *,
    auto_trade: bool,
    has_position: bool,
) -> dict[str, object]:
    payload = dict(signal)
    _clear_transient_auto_entry_blocks(
        payload,
        preserve_evaluation_results=True,
    )
    if has_position:
        payload["vetoes"] = ("symbol already has an open position",)
        return payload
    _update_entry_position_fields(payload)
    for reason in _auto_entry_prerequisite_blocks(
        payload,
        include_entry_position=False,
    ):
        _record_auto_entry_block(payload, reason)
    if not auto_trade:
        _record_auto_entry_block(payload, "auto strategy disabled; new entries are paused")
    return payload


def _record_auto_entry_block(signal: dict[str, object], reason: str) -> None:
    vetoes = list(signal.get("vetoes") or ())
    if reason not in vetoes:
        vetoes.append(reason)
    signal["vetoes"] = tuple(vetoes)


def _update_entry_position_fields(signal: dict[str, object]) -> None:
    entry_position, reason = _signal_entry_timing(signal)
    signal["entry_timing"] = entry_position
    signal["entry_timing_reason"] = reason
    signal["suggested_entry_text"] = _suggested_entry_text(signal)


_TRANSIENT_AUTO_ENTRY_BLOCK_PREFIXES = (
    "directional entry signal not established",
    "final score ",
    "late trend stage",
    "current entry position is not excellent",
    "等待 ",
    "已进入建议区",
    "暂无有效建议入场区",
    "symbol already has an open position",
    "symbol excluded from automatic universe",
    "auto strategy disabled",
    "position capacity full",
    "available entry margin ",
    "15m tactical entry lacks ",
    "invalid entry stop",
    "entry stop distance too wide",
    "entry reward/risk ",
    "auto entry execution failed",
    "BTC 4h extreme volatility",
    "market warm-up",
    "latest price is stale",
    "OI/long-short ratio data is stale",
    "current funding rate data is stale",
    "required multi-timeframe K-line context",
    "15m K-line context",
    "1h K-line context",
    "4h K-line context",
    "waiting for new ",
)


_AUTO_ENTRY_EVALUATION_RESULT_PREFIXES = (
    "position capacity full",
    "available entry margin ",
    "15m tactical entry lacks ",
    "invalid entry stop",
    "entry stop distance too wide",
    "entry reward/risk ",
    "auto entry execution failed",
)


def _clear_transient_auto_entry_blocks(
    signal: dict[str, object],
    *,
    preserve_evaluation_results: bool = False,
) -> None:
    def should_clear(reason: str) -> bool:
        if reason == "entry reward/risk target unavailable" or reason.startswith("invalid entry stop"):
            return True
        if not reason.startswith(_TRANSIENT_AUTO_ENTRY_BLOCK_PREFIXES):
            return False
        return not (
            preserve_evaluation_results
            and reason.startswith(_AUTO_ENTRY_EVALUATION_RESULT_PREFIXES)
        )

    vetoes = [
        str(reason)
        for reason in (signal.get("vetoes") or ())
        if not should_clear(str(reason))
    ]
    signal["vetoes"] = tuple(vetoes)


def _entry_stop_error(side: PositionSide, entry_price: float, stop_price: float) -> str | None:
    if not math.isfinite(entry_price) or entry_price <= 0:
        return "entry price is not positive and finite"
    if not math.isfinite(stop_price) or stop_price <= 0:
        return "stop price is not positive and finite"
    if side == PositionSide.LONG and stop_price >= entry_price:
        return "long stop must be below entry price"
    if side == PositionSide.SHORT and stop_price <= entry_price:
        return "short stop must be above entry price"
    if abs(entry_price - stop_price) <= entry_price * 1e-8:
        return "stop distance is effectively zero"
    return None


def _exit_plan_error(
    side: PositionSide,
    entry_price: float,
    stop_price: float,
    take_profit_1: float,
    take_profit_2: float,
) -> str | None:
    stop_error = _entry_stop_error(side, entry_price, stop_price)
    if stop_error:
        return f"invalid stop loss: {stop_error}"
    for label, target in (
        ("take profit 1", take_profit_1),
        ("take profit 2", take_profit_2),
    ):
        if not math.isfinite(target) or target <= 0:
            return f"{label} is not positive and finite"
        if not _target_is_profitable(side, entry_price, target):
            direction = "above" if side == PositionSide.LONG else "below"
            return f"{label} must be {direction} entry price"
    if side == PositionSide.LONG and take_profit_2 <= take_profit_1:
        return "take profit 2 must be above take profit 1"
    if side == PositionSide.SHORT and take_profit_2 >= take_profit_1:
        return "take profit 2 must be below take profit 1"
    return None


def _assert_valid_exit_plan(
    side: PositionSide,
    entry_price: float,
    stop_price: float,
    take_profit_1: float,
    take_profit_2: float,
) -> None:
    error = _exit_plan_error(
        side,
        entry_price,
        stop_price,
        take_profit_1,
        take_profit_2,
    )
    if error:
        raise ExitPlanAssertionError(error)


def _rotation_candidate_allowed(signal: dict[str, object], indicator: IndicatorSnapshot | None) -> bool:
    if str(signal.get("risk_state") or "NORMAL") != "NORMAL":
        return False
    if _trend_stage_from_signal(signal, indicator) == TREND_STAGE_LATE:
        return False
    entry_timing, _ = _signal_entry_timing(signal)
    if entry_timing != ENTRY_TIMING_GOOD:
        return False
    action = str(signal.get("action") or "")
    trend = str(signal.get("trend_state") or signal.get("regime") or "")
    if action == SignalAction.ENTRY_LONG.value and trend != "ONE_WAY_UP":
        return False
    if action == SignalAction.ENTRY_SHORT.value and trend != "ONE_WAY_DOWN":
        return False
    if action not in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}:
        return False
    if indicator is None or not indicator.close or not indicator.atr14:
        return False
    atr_pct = indicator.atr14 / indicator.close
    volume_ratio = indicator.volume_ratio or 0.0
    return atr_pct >= ROTATION_MIN_ATR_PCT and volume_ratio >= ROTATION_MIN_VOLUME_RATIO


def _signal_entry_timing(signal: dict[str, object]) -> tuple[str, str]:
    action = str(signal.get("action") or "")
    if action == SignalAction.WATCH.value:
        return ENTRY_TIMING_BLOCK, "entry position blocked: directional entry signal not established"
    if action == SignalAction.NO_TRADE.value:
        return ENTRY_TIMING_BLOCK, "entry position blocked: no trade signal"
    if action not in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}:
        return ENTRY_TIMING_BLOCK, "entry position blocked: not an entry signal"
    price = _float_or_none(signal.get("price"))
    entry_levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    if not entry_levels:
        return ENTRY_TIMING_BLOCK, "暂无有效建议入场区"
    if price is None:
        return ENTRY_TIMING_WAIT, "等待最新价格"
    side = PositionSide.LONG if action == SignalAction.ENTRY_LONG.value else PositionSide.SHORT
    return _side_entry_timing(side, price, signal)


def _side_entry_timing(side: PositionSide, price: float, signal: dict[str, object]) -> tuple[str, str]:
    score = int(signal.get("score") or 0)
    risk_state = str(signal.get("risk_state") or "NORMAL")
    trend_state = str(signal.get("trend_state") or signal.get("regime") or "")
    levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    side_key = "long" if side == PositionSide.LONG else "short"
    side_levels = levels.get(side_key) if isinstance(levels.get(side_key), dict) else {}
    if not side_levels:
        return ENTRY_TIMING_BLOCK, "暂无有效建议入场区"
    m15_precision = signal.get("m15_precision") if isinstance(signal.get("m15_precision"), dict) else {}
    smart_money_phase = str(signal.get("smart_money_phase") or "")
    include_m15 = _strong_m15_pullback_allowed(
        side,
        trend_state,
        risk_state,
        score,
        m15_precision,
        smart_money_phase,
    )
    entry_guardrails = (
        signal.get("entry_guardrails")
        if isinstance(signal.get("entry_guardrails"), dict)
        else {}
    )
    h1_boll_lower = _float_or_none(
        entry_guardrails.get("h1_boll_lower")
    )
    if (
        side == PositionSide.SHORT
        and h1_boll_lower is not None
        and price <= h1_boll_lower
    ):
        return (
            ENTRY_TIMING_WAIT,
            "等待 1H/4H压力反抽区",
        )
    if _entry_level_group_hit(signal, price, side, include_m15=include_m15):
        return ENTRY_TIMING_GOOD, _entry_zone_reason(side)
    return ENTRY_TIMING_WAIT, _entry_wait_hint(
        signal,
        price,
        side,
        include_m15=include_m15,
    )


def _entry_zone_reason(side: PositionSide) -> str:
    return "已到优势入场区"


def _pullback_reason(side: PositionSide) -> str:
    if side == PositionSide.LONG:
        return "entry timing good: 1h pullback held support/EMA/BOLL"
    return "entry timing good: 1h bounce rejected resistance/EMA/BOLL"


def _trigger_reason(side: PositionSide) -> str:
    if side == PositionSide.LONG:
        return "entry timing good: 1h retest or fake breakdown reclaimed support"
    return "entry timing good: 1h retest or fake breakout rejected resistance"


def _breakout_wait_reason(side: PositionSide) -> str:
    if side == PositionSide.LONG:
        return "entry timing wait: breakout needs pullback confirmation before fresh long"
    return "entry timing wait: breakdown needs resistance retest before fresh short"


def _wait_retest_reason(side: PositionSide) -> str:
    if side == PositionSide.LONG:
        return "entry position wait: live price has not reached a scored long entry zone"
    return "entry position wait: live price has not reached a scored short entry zone"


def _entry_level_display_labels(side: PositionSide) -> dict[str, str]:
    if side == PositionSide.LONG:
        return {
            "h1_support": "1H支撑回踩区",
            "h4_support": "4H支撑回踩区",
            "h1_boll_mid": "1H BOLL中轨回踩区",
            "h4_boll_mid": "4H BOLL中轨回踩区",
            "h1_ema20_ema60": "1H EMA20/EMA60回踩区",
            "h4_ema20_ema60": "4H EMA20/EMA60回踩区",
            "h1_ema20": "1H EMA20回踩区",
            "h1_ema60": "1H EMA60回踩区",
            "h4_ema20": "4H EMA20回踩区",
            "h4_ema60": "4H EMA60回踩区",
            "m15_ema20_ema60": "强单边15m EMA20/EMA60回踩区",
            "sweep_reclaim_support": "下插针收回支撑确认区",
            "ma_cluster_breakout": "均线密集突破区",
            "ma20_retest": "突破均线密集后回踩MA20区",
            "breakout_retest": "前压力突破后回踩确认区",
            "vwap_pullback": "VWAP/成交密集区回踩区",
        }
    return {
        "distribution_range_high": "高位派发箱体上沿区",
        "descending_high_trendline": "下降顶点趋势线反抽区",
        "h1_resistance": "1H压力反抽区",
        "h4_resistance": "4H压力反抽区",
        "h1_boll_mid": "1H BOLL中轨反抽区",
        "h4_boll_mid": "4H BOLL中轨反抽区",
        "h1_ema20_ema60": "1H EMA20/EMA60反抽区",
        "h4_ema20_ema60": "4H EMA20/EMA60反抽区",
        "h1_ema20": "1H EMA20反抽区",
        "h1_ema60": "1H EMA60反抽区",
        "h4_ema20": "4H EMA20反抽区",
        "h4_ema60": "4H EMA60反抽区",
        "sweep_reject_resistance": "上插针跌回压力确认区",
        "ma_cluster_breakdown": "均线密集跌破区",
        "ma20_retest": "跌破均线密集后反抽MA20区",
        "breakdown_retest": "前支撑跌破后反抽确认区",
        "vwap_retest": "VWAP/成交密集区反抽区",
    }


def _entry_wait_label(side: PositionSide, key: str) -> str:
    labels = _entry_level_display_labels(side)
    return labels.get(key, "建议入场区")


def _entry_level_zone_text(level: object) -> str:
    if not isinstance(level, dict):
        return ""
    low = _float_or_none(level.get("low"))
    high = _float_or_none(level.get("high"))
    price = _float_or_none(level.get("price"))
    values = [value for value in (low, high, price) if value is not None]
    if not values:
        return ""
    zone_low = min(values)
    zone_high = max(values)
    if abs(zone_high - zone_low) <= max(abs(zone_low), 1.0) * 1e-9:
        return _summary_price(zone_low)
    return f"{_summary_price(zone_low)}-{_summary_price(zone_high)}"


def _suggested_entry_text(signal: dict[str, object]) -> str:
    side = _signal_side(signal)
    if side is None:
        return "暂无有效建议入场区"
    levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    side_key = "long" if side == PositionSide.LONG else "short"
    side_levels = levels.get(side_key) if isinstance(levels.get(side_key), dict) else {}
    if not side_levels:
        return "暂无有效建议入场区"
    labels = _entry_level_display_labels(side)
    ordered_keys = [key for key in _entry_level_keys(side) if key in side_levels]
    ordered_keys.extend(key for key in side_levels if key not in ordered_keys)
    grouped: dict[str, list[str]] = {}
    for key in ordered_keys:
        zone = _entry_level_zone_text(side_levels.get(key))
        if not zone:
            continue
        label = labels.get(key, "建议入场区")
        grouped.setdefault(zone, [])
        if label not in grouped[zone]:
            grouped[zone].append(label)
    if not grouped:
        return "暂无有效建议入场区"
    parts = [
        f"{'/'.join(labels_for_zone)}≈{zone}"
        for zone, labels_for_zone in grouped.items()
    ]
    return "；".join(parts[:3])


def _entry_wait_hint(
    signal: dict[str, object],
    price: float,
    side: PositionSide,
    *,
    include_m15: bool,
) -> str:
    levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    side_key = "long" if side == PositionSide.LONG else "short"
    side_levels = levels.get(side_key) if isinstance(levels.get(side_key), dict) else {}
    if not side_levels:
        return "暂无有效建议入场区"
    keys = _entry_level_keys(side)
    if include_m15:
        keys = (*keys, "m15_ema20_ema60")
    matched_any_zone = any(
        isinstance(side_levels.get(key), dict)
        and _entry_level_hit(price, side_levels.get(key))
        for key in keys
    )
    if matched_any_zone:
        return "已进入建议区，但未处于优势侧"
    for key in keys:
        if _entry_level_has_values(side_levels.get(key)):
            return f"等待 {_entry_wait_label(side, key)}"
    return "暂无有效建议入场区"


def _entry_level_group_hit(signal: dict[str, object], price: float, side: PositionSide, *, include_m15: bool) -> bool:
    return _entry_zone_bounds(
        signal,
        price,
        side,
        include_m15=include_m15,
    ) is not None


def _entry_zone_bounds(
    signal: dict[str, object],
    price: float,
    side: PositionSide,
    *,
    include_m15: bool,
) -> tuple[float, float] | None:
    levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    side_key = "long" if side == PositionSide.LONG else "short"
    side_levels = levels.get(side_key) if isinstance(levels.get(side_key), dict) else {}
    if not side_levels:
        return None
    keys = _entry_level_keys(side)
    if include_m15:
        keys = (*keys, "m15_ema20_ema60")
    matched: list[tuple[str, float, float]] = []
    for key in keys:
        level = side_levels.get(key)
        if not isinstance(level, dict) or not _entry_level_hit(price, level):
            continue
        values = [
            value
            for value in (
                _float_or_none(level.get("low")),
                _float_or_none(level.get("high")),
                _float_or_none(level.get("price")),
            )
            if value is not None
        ]
        if values:
            matched.append((key, min(values), max(values)))
    if not matched:
        return None
    indicator_only_keys = {
        "h1_boll_mid",
        "h4_boll_mid",
        "h1_ema20_ema60",
        "h4_ema20_ema60",
        "h1_ema20",
        "h1_ema60",
        "h4_ema20",
        "h4_ema60",
        "vwap_pullback",
        "vwap_retest",
    }
    matched_keys = {key for key, _, _ in matched}
    if (
        all(key in indicator_only_keys for key, _, _ in matched)
        and not _indicator_entry_confirmation_present(
            signal,
            side,
            matched_keys,
        )
        and not _h4_override_indicator_entry_allowed(signal, matched_keys)
    ):
        return None
    zone_low = max(low for _, low, _ in matched)
    zone_high = min(high for _, _, high in matched)
    if not _entry_advantage_zone_hit(
        price,
        side,
        zone_low,
        zone_high,
    ):
        return None
    return zone_low, zone_high


def _entry_advantage_zone_hit(
    price: float,
    side: PositionSide,
    zone_low: float,
    zone_high: float,
) -> bool:
    if zone_low > zone_high:
        return False
    width = zone_high - zone_low
    if width <= 0:
        return price == zone_low
    if side == PositionSide.LONG:
        advantage_ceiling = zone_low + width * ENTRY_ADVANTAGE_ZONE_FRACTION
        return zone_low <= price <= advantage_ceiling
    advantage_floor = zone_high - width * ENTRY_ADVANTAGE_ZONE_FRACTION
    return advantage_floor <= price <= zone_high


def _h4_override_indicator_entry_allowed(
    signal: dict[str, object],
    matched_keys: set[str],
) -> bool:
    if str(signal.get("entry_timeframe_override") or "") != "4h":
        return False
    return bool(matched_keys) and matched_keys <= {
        "h4_boll_mid",
        "h4_ema20_ema60",
        "h4_ema20",
        "h4_ema60",
    }


def _indicator_entry_confirmation_present(
    signal: dict[str, object],
    side: PositionSide,
    matched_keys: set[str],
) -> bool:
    if not matched_keys & {
        "h1_boll_mid",
        "h1_ema20_ema60",
        "h1_ema20",
        "h1_ema60",
        "h4_boll_mid",
        "h4_ema20_ema60",
        "h4_ema20",
        "h4_ema60",
        "vwap_pullback",
        "vwap_retest",
    }:
        return False
    setup_type = str(signal.get("setup_type") or "")
    h1_pullback = signal.get("h1_pullback") if isinstance(signal.get("h1_pullback"), dict) else {}
    h1_ma_cluster = signal.get("h1_ma_cluster") if isinstance(signal.get("h1_ma_cluster"), dict) else {}
    h4_ma_cluster = signal.get("h4_ma_cluster") if isinstance(signal.get("h4_ma_cluster"), dict) else {}
    h1_state = str(h1_pullback.get("state") or "")
    h1_direction = str(h1_pullback.get("direction") or "")
    h1_cluster_state = str(h1_ma_cluster.get("state") or "")
    h4_cluster_state = str(h4_ma_cluster.get("state") or "")
    h1_indicator_hit = bool(matched_keys & {
        "h1_boll_mid",
        "h1_ema20_ema60",
        "h1_ema20",
        "h1_ema60",
        "vwap_pullback",
        "vwap_retest",
    })
    h4_indicator_hit = bool(matched_keys & {
        "h4_boll_mid",
        "h4_ema20_ema60",
        "h4_ema20",
        "h4_ema60",
    })
    if side == PositionSide.LONG:
        if h1_indicator_hit and (
            setup_type == SETUP_H1_PULLBACK_LONG
            or (h1_direction == "LONG" and h1_state == "HEALTHY_PULLBACK")
            or h1_cluster_state == "RETEST_UP"
        ):
            return True
        if h4_indicator_hit and (
            setup_type == SETUP_H4_PULLBACK_LONG
            or h4_cluster_state == "RETEST_UP"
        ):
            return True
    else:
        if h1_indicator_hit and (
            setup_type == SETUP_H1_PULLBACK_SHORT
            or (h1_direction == "SHORT" and h1_state == "HEALTHY_PULLBACK")
            or h1_cluster_state == "RETEST_DOWN"
        ):
            return True
        if h4_indicator_hit and (
            setup_type == SETUP_H4_PULLBACK_SHORT
            or h4_cluster_state == "RETEST_DOWN"
        ):
            return True
    if h4_indicator_hit and not h1_indicator_hit:
        return False
    reason_text = " | ".join(
        str(reason).lower()
        for reason in signal.get("reasons") or ()
    )
    tokens = (
        (
            "close confirmed near ema20/boll mid",
            "1h boll/ema pullback held",
            "vwap pullback held",
        )
        if side == PositionSide.LONG
        else (
            "close confirmed failed retest",
            "1h boll/ema pullback rejected",
            "vwap retest rejected",
        )
    )
    return any(token in reason_text for token in tokens)


def _entry_level_keys(side: PositionSide) -> tuple[str, ...]:
    return (
        (
            "h1_support",
            "h4_support",
            "h1_boll_mid",
            "h4_boll_mid",
            "h1_ema20_ema60",
            "h4_ema20_ema60",
            "h1_ema20",
            "h1_ema60",
            "h4_ema20",
            "h4_ema60",
            "sweep_reclaim_support",
            "ma_cluster_breakout",
            "ma20_retest",
            "breakout_retest",
            "vwap_pullback",
        )
        if side == PositionSide.LONG
        else (
            "distribution_range_high",
            "descending_high_trendline",
            "h1_resistance",
            "h4_resistance",
            "h1_boll_mid",
            "h4_boll_mid",
            "h1_ema20_ema60",
            "h4_ema20_ema60",
            "h1_ema20",
            "h1_ema60",
            "h4_ema20",
            "h4_ema60",
            "sweep_reject_resistance",
            "ma_cluster_breakdown",
            "ma20_retest",
            "breakdown_retest",
            "vwap_retest",
        )
    )


def _core_scored_entry_zone_hit(
    signal: dict[str, object],
    all_entry_levels: dict[str, object],
    side: PositionSide,
    *,
    reasons: Sequence[object],
    entry_timeframe_override: str | None,
    setup_type: str | None = None,
    action: str | None = None,
) -> bool:
    price = _float_or_none(signal.get("price"))
    if price is None or not isinstance(all_entry_levels, dict):
        return False
    probe = dict(signal)
    probe["reasons"] = tuple(reasons)
    if action:
        probe["action"] = action
    if entry_timeframe_override:
        probe["entry_timeframe_override"] = entry_timeframe_override
    else:
        probe.pop("entry_timeframe_override", None)
    if setup_type:
        probe["setup_type"] = setup_type
    scored_levels = _scored_entry_levels(probe, all_entry_levels)
    if not scored_levels:
        return False
    probe["entry_levels"] = scored_levels
    return (
        _entry_zone_bounds(
            probe,
            price,
            side,
            include_m15=False,
        )
        is not None
    )


def _setup_priority(setup_type: str) -> int:
    return {
        SETUP_M15_SQUEEZE_TACTICAL_LONG: 95,
        SETUP_OI_VALLEY_REVERSAL_LONG: 95,
        SETUP_DISTRIBUTION_STAGE1_SHORT: 95,
        SETUP_DISTRIBUTION_STAGE2_SHORT: 95,
        SETUP_DISTRIBUTION_STAGE3_SHORT: 90,
        SETUP_H4_PULLBACK_LONG: 85,
        SETUP_H4_PULLBACK_SHORT: 85,
        SETUP_H1_PULLBACK_LONG: 75,
        SETUP_H1_PULLBACK_SHORT: 75,
        SETUP_H1_STRUCTURE_LONG: 65,
        SETUP_H1_STRUCTURE_SHORT: 65,
    }.get(setup_type, 0)


def _prefer_setup_type(current: str, candidate: str) -> str:
    if not candidate:
        return current
    if not current:
        return candidate
    if _setup_priority(candidate) >= _setup_priority(current):
        return candidate
    return current


def _ma_cluster_setup_type(
    side: PositionSide,
    h4_cluster: dict[str, object],
    h1_cluster: dict[str, object],
) -> str:
    h1_state = str(h1_cluster.get("state") or "UNKNOWN")
    h4_state = str(h4_cluster.get("state") or "UNKNOWN")
    if side == PositionSide.LONG:
        if h1_state == "RETEST_UP":
            return SETUP_H1_PULLBACK_LONG
        if h1_state == "BREAKOUT_UP":
            return SETUP_H1_STRUCTURE_LONG
        if h4_state == "RETEST_UP":
            return SETUP_H4_PULLBACK_LONG
    else:
        if h1_state == "RETEST_DOWN":
            return SETUP_H1_PULLBACK_SHORT
        if h1_state == "BREAKDOWN_DOWN":
            return SETUP_H1_STRUCTURE_SHORT
        if h4_state == "RETEST_DOWN":
            return SETUP_H4_PULLBACK_SHORT
    return ""


def _scored_entry_levels(
    signal: dict[str, object],
    all_levels: dict[str, object],
) -> dict[str, object]:
    """Keep only entry zones supported by conditions that contributed to the signal."""
    side = _signal_side(signal)
    if side is None:
        return {}
    side_key = "long" if side == PositionSide.LONG else "short"
    source = all_levels.get(side_key) if isinstance(all_levels.get(side_key), dict) else {}
    if str(signal.get("entry_timeframe_override") or "") == "4h":
        keys = [
            "h4_support" if side == PositionSide.LONG else "h4_resistance",
            "h4_boll_mid",
        ]
        keys.extend(
            ["h4_ema20_ema60"]
            if _entry_level_has_values(source.get("h4_ema20_ema60"))
            else ["h4_ema20", "h4_ema60"]
        )
        selected = {
            key: source[key]
            for key in keys
            if _entry_level_has_values(source.get(key))
        }
        return {side_key: selected} if selected else {}
    distribution_stage = str(
        signal.get("distribution_short_stage") or ""
    )
    if side == PositionSide.SHORT and distribution_stage:
        return _distribution_stage_entry_levels(
            signal,
            source,
            distribution_stage,
        )
    selected: dict[str, object] = {}
    reason_text = " | ".join(str(reason).lower() for reason in signal.get("reasons") or ())
    setup_type = str(signal.get("setup_type") or "")

    def add(*keys: str) -> None:
        for key in keys:
            level = source.get(key)
            if _entry_level_has_values(level):
                selected[key] = level

    def add_ema(timeframe: str) -> None:
        merged_key = f"{timeframe}_ema20_ema60"
        if _entry_level_has_values(source.get(merged_key)):
            add(merged_key)
            return
        add(f"{timeframe}_ema20", f"{timeframe}_ema60")

    if side == PositionSide.LONG:
        if setup_type == SETUP_M15_SQUEEZE_TACTICAL_LONG:
            add("m15_ema20_ema60")
        elif setup_type == SETUP_H4_PULLBACK_LONG:
            add("h4_support", "h4_boll_mid")
            add_ema("h4")
        elif setup_type == SETUP_OI_VALLEY_REVERSAL_LONG:
            add("sweep_reclaim_support", "h1_support", "h4_support")
        elif setup_type == SETUP_H1_PULLBACK_LONG:
            add("h1_support", "h1_boll_mid", "vwap_pullback", "ma20_retest")
            add_ema("h1")
        elif setup_type == SETUP_H1_STRUCTURE_LONG:
            add("breakout_retest", "h1_support", "sweep_reclaim_support", "ma_cluster_breakout")
        precision = signal.get("m15_precision") if isinstance(signal.get("m15_precision"), dict) else {}
        if _strong_m15_pullback_allowed(
            side,
            str(signal.get("trend_state") or ""),
            str(signal.get("risk_state") or "NORMAL"),
            int(signal.get("score") or 0),
            precision,
            str(signal.get("smart_money_phase") or ""),
        ):
            add("m15_ema20_ema60")
        if "vwap pullback held" in reason_text:
            add("vwap_pullback")
        if "close confirmed near ema20/boll mid" in reason_text or "1h boll/ema pullback held" in reason_text:
            add("h1_support", "h1_boll_mid")
            add_ema("h1")
        if "1h retest confirms long trigger" in reason_text:
            add("breakout_retest")
        if "1h fake_breakdown confirms long trigger" in reason_text:
            add("h1_support", "sweep_reclaim_support")
        if any(token in reason_text for token in ("market structure confirms long", "resistance grind broke upward", "volume pattern confirms long", "volume breakout above resistance", "breakout retest held quietly")):
            add("breakout_retest", "h1_support")
        if any(token in reason_text for token in ("washout confirmed: downside", "downside sweep reclaimed")):
            add("sweep_reclaim_support")
        if "ma cluster breakout up" in reason_text:
            add("ma_cluster_breakout")
        if "ma cluster retest held" in reason_text:
            add("ma20_retest")
        if "4h structure supports upside" in reason_text:
            add("h4_support")
            add_ema("h4")
    else:
        if setup_type == SETUP_H4_PULLBACK_SHORT:
            add("h4_resistance", "h4_boll_mid")
            add_ema("h4")
        elif setup_type == SETUP_H1_PULLBACK_SHORT:
            add("h1_resistance", "h1_boll_mid", "vwap_retest", "ma20_retest")
            add_ema("h1")
        elif setup_type == SETUP_H1_STRUCTURE_SHORT:
            add("breakdown_retest", "h1_resistance", "sweep_reject_resistance", "ma_cluster_breakdown")
        if "vwap retest rejected" in reason_text:
            add("vwap_retest")
        if "close confirmed failed retest" in reason_text or "1h boll/ema pullback rejected" in reason_text:
            add("h1_resistance", "h1_boll_mid")
            add_ema("h1")
        if "1h retest confirms short trigger" in reason_text:
            add("breakdown_retest")
        if "1h fake_breakout confirms short trigger" in reason_text:
            add("h1_resistance", "sweep_reject_resistance")
        if any(token in reason_text for token in ("market structure confirms short", "support grind broke downward", "volume pattern confirms short", "volume breakdown below support", "breakdown retest rejected quietly")):
            add("breakdown_retest", "h1_resistance")
        if any(token in reason_text for token in ("washout confirmed: upside", "upside sweep rejected")):
            add("sweep_reject_resistance")
        if "ma cluster retest rejected" in reason_text:
            add("ma20_retest")
        if "4h structure supports downside" in reason_text:
            add("h4_resistance")
            add_ema("h4")
    return {side_key: selected} if selected else {}


def _distribution_stage_entry_levels(
    signal: dict[str, object],
    source: dict[str, object],
    stage: str,
) -> dict[str, object]:
    selected: dict[str, object] = {}

    def add(*keys: str) -> None:
        for key in keys:
            level = source.get(key)
            if _entry_level_has_values(level):
                selected[key] = level

    def add_ema(timeframe: str) -> None:
        merged_key = f"{timeframe}_ema20_ema60"
        if _entry_level_has_values(source.get(merged_key)):
            add(merged_key)
            return
        add(f"{timeframe}_ema20", f"{timeframe}_ema60")

    if stage == DISTRIBUTION_STAGE_RANGE:
        reasons = " | ".join(
            str(reason).lower()
            for reason in signal.get("reasons") or ()
        )
        explicit_rejection = (
            str(signal.get("smart_money_phase") or "")
            == "DISTRIBUTION_EXIT"
            or "upside wick swept resistance" in reasons
            or "upside sweep rejected resistance" in reasons
        )
        if explicit_rejection:
            add("distribution_range_high")
    elif stage == DISTRIBUTION_STAGE_DESCENDING:
        add("descending_high_trendline")
    elif stage == DISTRIBUTION_STAGE_MARKDOWN:
        add(
            "h1_boll_mid",
            "h4_boll_mid",
            "ma20_retest",
            "breakdown_retest",
            "vwap_retest",
        )
        add_ema("h1")
        add_ema("h4")
    return {"short": selected} if selected else {}


def _entry_level_has_values(level: object) -> bool:
    if not isinstance(level, dict):
        return False
    return any(
        _float_or_none(level.get(key)) is not None
        for key in ("low", "high", "price")
    )


def _required_entry_timeframes(
    signal: dict[str, object] | None,
) -> tuple[str, ...]:
    if not isinstance(signal, dict):
        return ()
    side = _signal_side(signal)
    if side is None:
        return ()
    price = _float_or_none(signal.get("price"))
    if price is None:
        return (
            str(
                signal.get("entry_timeframe_override")
                or signal.get("signal_timeframe")
                or "1h"
            ),
        )
    return (_entry_timeframe_for_signal(signal, side, price),)


def _entry_timeframe_for_signal(
    signal: dict[str, object],
    side: PositionSide,
    price: float,
) -> str:
    """Resolve the stop/target timeframe from the entry zone actually being traded."""
    levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    side_key = "long" if side == PositionSide.LONG else "short"
    side_levels = levels.get(side_key) if isinstance(levels.get(side_key), dict) else {}
    precision = signal.get("m15_precision") if isinstance(signal.get("m15_precision"), dict) else {}
    m15_hit = (
        _strong_m15_pullback_allowed(
            side,
            str(signal.get("trend_state") or signal.get("regime") or ""),
            str(signal.get("risk_state") or "NORMAL"),
            int(signal.get("score") or 0),
            precision,
            str(signal.get("smart_money_phase") or ""),
        )
        and _entry_level_hit(price, side_levels.get("m15_ema20_ema60"))
    )
    h1_keys = (
        ("h1_support", "h1_boll_mid", "h1_ema20_ema60", "h1_ema20", "h1_ema60", "vwap_pullback")
        if side == PositionSide.LONG
        else (
            "distribution_range_high",
            "descending_high_trendline",
            "h1_resistance",
            "h1_boll_mid",
            "h1_ema20_ema60",
            "h1_ema20",
            "h1_ema60",
            "vwap_retest",
        )
    )
    h4_keys = (
        ("h4_support", "h4_boll_mid", "h4_ema20_ema60", "h4_ema20", "h4_ema60")
        if side == PositionSide.LONG
        else ("h4_resistance", "h4_boll_mid", "h4_ema20_ema60", "h4_ema20", "h4_ema60")
    )
    h1_hit = any(_entry_level_hit(price, side_levels.get(key)) for key in h1_keys)
    h4_hit = any(_entry_level_hit(price, side_levels.get(key)) for key in h4_keys)
    if m15_hit and not h1_hit and not h4_hit:
        return "15m"
    if str(signal.get("entry_timeframe_override") or "") == "4h":
        return "4h"
    if h1_hit:
        return "1h"
    if h4_hit:
        return "4h"
    generic_keys = (
        ("sweep_reclaim_support", "ma_cluster_breakout", "ma20_retest", "breakout_retest")
        if side == PositionSide.LONG
        else ("sweep_reject_resistance", "ma_cluster_breakdown", "ma20_retest", "breakdown_retest")
    )
    generic_hit = any(
        _entry_level_hit(price, side_levels.get(key))
        for key in generic_keys
    )
    h1_cluster = signal.get("h1_ma_cluster") if isinstance(signal.get("h1_ma_cluster"), dict) else {}
    h4_cluster = signal.get("h4_ma_cluster") if isinstance(signal.get("h4_ma_cluster"), dict) else {}
    if (
        generic_hit
        and str(h1_cluster.get("state") or "UNKNOWN") == "UNKNOWN"
        and str(h4_cluster.get("state") or "UNKNOWN") != "UNKNOWN"
    ):
        return "4h"
    return "1h"


def _indicator_for_entry_timeframe(
    timeframe_indicators: dict[str, list[IndicatorSnapshot]],
    timeframe: str,
    fallback: list[IndicatorSnapshot],
) -> IndicatorSnapshot | None:
    indicators = timeframe_indicators.get(timeframe) or []
    if indicators:
        return indicators[-1]
    return _preferred_exit_indicator(timeframe_indicators, fallback)


def _m15_precision_stop_distance_ok(
    side: PositionSide,
    price: float,
    precision: dict[str, object],
    indicator: IndicatorSnapshot | None = None,
) -> bool:
    minimum_distance = _minimum_precision_stop_distance(price, indicator)
    if side == PositionSide.LONG:
        anchor = _float_or_none(precision.get("long_stop_anchor"))
        return anchor is not None and anchor < price and price - anchor >= minimum_distance
    anchor = _float_or_none(precision.get("short_stop_anchor"))
    return anchor is not None and anchor > price and anchor - price >= minimum_distance


def _entry_level_hit(price: float, level: object) -> bool:
    if not isinstance(level, dict):
        return False
    low = _float_or_none(level.get("low"))
    high = _float_or_none(level.get("high"))
    point = _float_or_none(level.get("price"))
    values = [value for value in (low, high, point) if value is not None]
    if not values:
        return False
    zone_low = min(values)
    zone_high = max(values)
    return zone_low <= price <= zone_high


def _position_structure_failed(position: Position, signal: dict[str, object]) -> bool:
    action = str(signal.get("action") or "")
    trend = str(signal.get("trend_state") or signal.get("regime") or "CHOP")
    risk_state = str(signal.get("risk_state") or "NORMAL")
    vetoes = tuple(signal.get("vetoes") or ())
    if position.side == PositionSide.LONG:
        opposite_signal = action == SignalAction.ENTRY_SHORT.value or trend == "ONE_WAY_DOWN"
        trend_lost = trend not in {"TREND_LONG", "ONE_WAY_UP"} and int(signal.get("score") or 0) < 75
        risk_break = risk_state in {"LONG_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}
    else:
        opposite_signal = action == SignalAction.ENTRY_LONG.value or trend == "ONE_WAY_UP"
        trend_lost = trend not in {"TREND_SHORT", "ONE_WAY_DOWN"} and int(signal.get("score") or 0) < 75
        risk_break = risk_state in {"SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}
    return opposite_signal or trend_lost or risk_break or bool(vetoes)


def _rotation_candidate_strong(signal: dict[str, object]) -> bool:
    action = str(signal.get("action") or "")
    trend = str(signal.get("trend_state") or signal.get("regime") or "")
    risk_state = str(signal.get("risk_state") or "NORMAL")
    entry_timing, _ = _signal_entry_timing(signal)
    return (
        action in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}
        and trend in {"ONE_WAY_UP", "ONE_WAY_DOWN"}
        and int(signal.get("score") or 0) >= 95
        and risk_state == "NORMAL"
        and entry_timing == ENTRY_TIMING_GOOD
        and not signal.get("vetoes")
    )


def _position_h4_allows_efficiency_rotation(position: Position, signal: dict[str, object]) -> bool:
    h4 = signal.get("h4_structure") if isinstance(signal.get("h4_structure"), dict) else {}
    h4_state = str(h4.get("state") or "UNKNOWN")
    trend = str(signal.get("trend_state") or signal.get("regime") or "CHOP")
    score = int(signal.get("score") or 0)
    if position.side == PositionSide.LONG:
        if h4_state == "BREAKOUT_UP" or trend == "ONE_WAY_UP":
            return False
        if trend == "TREND_LONG" and score >= 75 and h4_state in {"BOX_UPPER_HALF", "BOX_LOWER_HALF"}:
            return False
        return h4_state in {"UNKNOWN", "BREAKDOWN_DOWN", "RANGE_MID"} or trend not in {"TREND_LONG", "ONE_WAY_UP"}
    if h4_state == "BREAKDOWN_DOWN" or trend == "ONE_WAY_DOWN":
        return False
    if trend == "TREND_SHORT" and score >= 75 and h4_state in {"BOX_UPPER_HALF", "BOX_LOWER_HALF"}:
        return False
    return h4_state in {"UNKNOWN", "BREAKOUT_UP", "RANGE_MID"} or trend not in {"TREND_SHORT", "ONE_WAY_DOWN"}


def _position_efficiency_stalled(
    position: Position,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
    price: float,
) -> bool:
    opened_at = position.opened_at
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=UTC)
    if datetime.now(UTC) - opened_at < timedelta(hours=1):
        return False
    if _position_structure_failed(position, signal):
        return False
    stop_distance = float(position.metadata.get("initial_stop_distance") or abs(position.entry_price - position.stop_price))
    if stop_distance <= 0:
        return False
    current_profit = price - position.entry_price if position.side == PositionSide.LONG else position.entry_price - price
    if current_profit >= stop_distance * 0.8:
        return False
    trend = str(signal.get("trend_state") or signal.get("regime") or "CHOP")
    score = int(signal.get("score") or 0)
    risk_state = str(signal.get("risk_state") or "NORMAL")
    if trend in {"ONE_WAY_UP", "ONE_WAY_DOWN"} and score >= 95 and risk_state == "NORMAL":
        return False
    if indicator is None:
        return True
    weak_volume = (indicator.volume_ratio or 0.0) < 1.05
    weak_oi = (indicator.oi_change or 0.0) <= 0.0
    return trend not in {"ONE_WAY_UP", "ONE_WAY_DOWN"} or score < 85 or weak_volume or weak_oi


def _rotation_efficiency_better(candidate: IndicatorSnapshot | None, current: IndicatorSnapshot | None) -> bool:
    if candidate is None or not candidate.close or not candidate.atr14:
        return False
    candidate_atr_pct = candidate.atr14 / candidate.close
    candidate_volume = candidate.volume_ratio or 0.0
    candidate_oi = candidate.oi_change or 0.0
    if current is None or not current.close or not current.atr14:
        return candidate_atr_pct >= ROTATION_MIN_ATR_PCT and candidate_volume >= ROTATION_MIN_VOLUME_RATIO
    current_atr_pct = current.atr14 / current.close
    current_volume = current.volume_ratio or 0.0
    current_oi = current.oi_change or 0.0
    advantages = 0
    if candidate_atr_pct >= current_atr_pct * 1.15 and candidate_atr_pct >= ROTATION_MIN_ATR_PCT:
        advantages += 1
    if candidate_volume >= max(ROTATION_MIN_VOLUME_RATIO, current_volume + 0.25):
        advantages += 1
    if candidate_oi >= current_oi + 0.005:
        advantages += 1
    return advantages >= 2


def _build_price_only_indicators(candles: list[Candle], settings: AppSettings) -> list[IndicatorSnapshot]:
    return _build_indicators(candles, [], settings)


def _build_indicators(
    candles: list[Candle],
    derivatives: list[DerivativesSnapshot],
    settings: AppSettings,
) -> list[IndicatorSnapshot]:
    return build_indicators(
        candles,
        derivatives,
        ema_fast=settings.strategy.ema_fast,
        ema_slow=settings.strategy.ema_slow,
        ma_trend=settings.strategy.ma_trend,
        bollinger_window=settings.strategy.bollinger_window,
        bollinger_stddev=settings.strategy.bollinger_stddev,
        rsi_window=settings.strategy.rsi_window,
        atr_window=settings.strategy.atr_window,
        volume_window=settings.strategy.volume_window,
        keltner_window=settings.strategy.keltner_window,
        keltner_atr_multiplier=settings.strategy.keltner_atr_multiplier,
        qps_window=settings.strategy.qps_window,
    )


_TIMEFRAME_SECONDS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


def _timeframe_seconds(timeframe: str) -> int:
    try:
        return _TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe: {timeframe}") from exc


def _latest_closed_slot(timeframe: str, now: datetime) -> datetime:
    seconds = _timeframe_seconds(timeframe)
    timestamp = max(now.timestamp() - TIMEFRAME_CLOSE_GRACE_SECONDS, 0.0)
    return datetime.fromtimestamp(int(timestamp // seconds) * seconds, tz=UTC)


def _current_candle_slot(timeframe: str, now: datetime) -> datetime:
    seconds = _timeframe_seconds(timeframe)
    return datetime.fromtimestamp(int(now.timestamp() // seconds) * seconds, tz=UTC)


def _closed_candles(candles: list[Candle], timeframe: str, now: datetime | None = None) -> list[Candle]:
    if not candles:
        return []
    cutoff = (now or datetime.now(UTC)) - timedelta(seconds=TIMEFRAME_CLOSE_GRACE_SECONDS)
    duration = timedelta(seconds=_timeframe_seconds(timeframe))
    closed = [candle for candle in candles if candle.timestamp + duration <= cutoff]
    return closed or candles[:-1]


def _merge_candles(existing: list[Candle], incoming: list[Candle], *, max_length: int) -> list[Candle]:
    merged = {candle.timestamp: candle for candle in existing}
    merged.update({candle.timestamp: candle for candle in incoming})
    return [merged[timestamp] for timestamp in sorted(merged)][-max_length:]


def _derivatives_for_candles(
    derivatives: list[DerivativesSnapshot],
    candles: list[Candle],
) -> list[DerivativesSnapshot]:
    wanted = {candle.timestamp for candle in candles}
    return [snapshot for snapshot in derivatives if snapshot.timestamp in wanted]


def _derivatives_complete(derivatives: list[DerivativesSnapshot]) -> bool:
    recent = derivatives[-12:]
    return bool(recent) and any(snapshot.open_interest is not None for snapshot in recent) and any(
        snapshot.long_short_ratio is not None for snapshot in recent
    )


def _merge_derivatives(
    existing: list[DerivativesSnapshot],
    incoming: list[DerivativesSnapshot],
    *,
    preserve_funding: bool,
) -> list[DerivativesSnapshot]:
    existing_by_time = {snapshot.timestamp: snapshot for snapshot in existing}
    incoming_by_time = {snapshot.timestamp: snapshot for snapshot in incoming}
    latest_funding: float | None = None
    merged: dict[datetime, DerivativesSnapshot] = dict(existing_by_time)
    for timestamp in sorted(set(existing_by_time) | set(incoming_by_time)):
        old = existing_by_time.get(timestamp)
        new = incoming_by_time.get(timestamp)
        if old and old.funding_rate is not None:
            latest_funding = old.funding_rate
        funding = (
            new.funding_rate
            if new and new.funding_rate is not None
            else old.funding_rate
            if old and old.funding_rate is not None
            else latest_funding
        )
        if new is None:
            continue
        merged[timestamp] = DerivativesSnapshot(
            timestamp=timestamp,
            open_interest=new.open_interest if new.open_interest is not None else old.open_interest if old else None,
            long_short_ratio=(
                new.long_short_ratio
                if new.long_short_ratio is not None
                else old.long_short_ratio
                if old
                else None
            ),
            funding_rate=funding if preserve_funding else new.funding_rate,
        )
    return [merged[timestamp] for timestamp in sorted(merged)][-500:]


def _with_current_funding(
    snapshots: list[DerivativesSnapshot],
    funding_rate: float,
) -> list[DerivativesSnapshot]:
    return [
        DerivativesSnapshot(
            timestamp=snapshot.timestamp,
            open_interest=snapshot.open_interest,
            long_short_ratio=snapshot.long_short_ratio,
            funding_rate=funding_rate,
        )
        for snapshot in snapshots
    ]


def _candles_contiguous(candles: list[Candle], timeframe: str) -> bool:
    if len(candles) < 50:
        return False
    expected = timedelta(seconds=_timeframe_seconds(timeframe))
    recent = candles[-12:]
    return all(
        current.timestamp - previous.timestamp == expected
        for previous, current in zip(recent, recent[1:])
    )


def _build_multi_timeframe_context(
    timeframe_candles: dict[str, list[Candle]],
    timeframe_indicators: dict[str, list[IndicatorSnapshot]],
    settings: AppSettings,
) -> dict[str, object]:
    d1 = _daily_direction(timeframe_indicators.get("1d", []))
    h4 = _four_hour_structure(timeframe_candles.get("4h", []), timeframe_indicators.get("4h", []), settings)
    h1_structure = _one_hour_structure(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), settings)
    h1 = _one_hour_trigger(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), h4, h1_structure, settings)
    h1_pullback = _one_hour_pullback(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), h4, settings)
    h1_ema_reliability = _one_hour_ema_reliability(
        timeframe_candles.get("1h", []),
        timeframe_indicators.get("1h", []),
    )
    high_handoff = _high_distribution_handoff(
        timeframe_candles.get("1h", []),
        timeframe_indicators.get("1h", []),
    )
    h4_ma_cluster = _moving_average_cluster(timeframe_candles.get("4h", []), timeframe_indicators.get("4h", []), settings)
    h1_ma_cluster = _moving_average_cluster(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), settings)
    h4_oi = _four_hour_oi_state(timeframe_indicators.get("4h", []), h4, h1, h1_pullback)
    h4_oi_valley = _four_hour_oi_valley(
        timeframe_candles.get("4h", []),
        timeframe_indicators.get("4h", []),
        settings,
    )
    distribution_short = _distribution_short_structure(
        timeframe_candles.get("1h", []),
        timeframe_indicators.get("1h", []),
        h4_oi,
        settings,
    )
    m15 = _fifteen_minute_precision(timeframe_candles.get("15m", []), timeframe_indicators.get("15m", []), settings)
    large_wick_rejections = _large_timeframe_wick_rejections(timeframe_candles)
    entry_levels = _entry_level_context(
        h1_structure,
        h4,
        h1_ma_cluster,
        h4_ma_cluster,
        timeframe_indicators,
        m15,
        distribution_short,
    )
    h1_indicator = _latest_indicator(
        timeframe_indicators.get("1h", [])
    )
    entry_guardrails = {
        "h1_boll_lower": (
            h1_indicator.boll_lower
            if h1_indicator is not None
            else None
        ),
    }
    return {
        "daily_bias": d1,
        "h4_structure": h4,
        "h1_structure": h1_structure,
        "h4_ma_cluster": h4_ma_cluster,
        "h1_ma_cluster": h1_ma_cluster,
        "h4_oi": h4_oi,
        "h4_oi_valley": h4_oi_valley,
        "h1_trigger": h1,
        "h1_pullback": h1_pullback,
        "h1_ema_reliability": h1_ema_reliability,
        "high_distribution_handoff": high_handoff,
        "distribution_short": distribution_short,
        "m15_precision": m15,
        "large_wick_rejections": large_wick_rejections,
        "entry_levels": entry_levels,
        "entry_guardrails": entry_guardrails,
        "summary": _multi_timeframe_summary(d1, h4, h1, h1_pullback, h4_oi, h4_ma_cluster, h1_ma_cluster),
    }


def _large_timeframe_wick_rejections(
    timeframe_candles: dict[str, list[Candle]],
) -> dict[str, dict[str, object]]:
    rejections: dict[str, dict[str, object]] = {}
    for timeframe in ("1h", "4h"):
        candles = timeframe_candles.get(timeframe, [])
        if not candles:
            continue
        candle = candles[-1]
        close = float(candle.close or 0.0)
        if close <= 0:
            continue
        upper_ratio = max((float(candle.high) - close) / close, 0.0)
        lower_ratio = max((close - float(candle.low)) / close, 0.0)
        if upper_ratio >= LARGE_TIMEFRAME_WICK_REJECTION_PCT:
            _remember_largest_wick_rejection(
                rejections,
                "upper",
                timeframe,
                upper_ratio,
            )
        if lower_ratio >= LARGE_TIMEFRAME_WICK_REJECTION_PCT:
            _remember_largest_wick_rejection(
                rejections,
                "lower",
                timeframe,
                lower_ratio,
            )
    return rejections


def _remember_largest_wick_rejection(
    rejections: dict[str, dict[str, object]],
    key: str,
    timeframe: str,
    ratio: float,
) -> None:
    current = rejections.get(key)
    if current is not None and float(current.get("ratio") or 0.0) >= ratio:
        return
    rejections[key] = {
        "timeframe": timeframe,
        "ratio": ratio,
    }


def _large_wick_rejection_veto_text(
    side: PositionSide,
    rejection: dict[str, object],
) -> str:
    timeframe = str(rejection.get("timeframe") or "").upper()
    ratio = float(rejection.get("ratio") or 0.0)
    pct = ratio * 100
    if side == PositionSide.LONG:
        return f"{timeframe}上插针收实{pct:.1f}%，禁止追多"
    return f"{timeframe}下插针收实{pct:.1f}%，禁止追空"


def _daily_direction(indicators: list[IndicatorSnapshot]) -> str:
    if not indicators:
        return "NEUTRAL"
    current = indicators[-1]
    if current.ema20 is None or current.ema50 is None or current.ema200 is None:
        return "NEUTRAL"
    slope = current.ema50_slope or 0.0
    if current.close >= current.ema50 and current.ema20 >= current.ema50 and slope >= 0:
        return "BULL"
    if current.close <= current.ema50 and current.ema20 <= current.ema50 and slope <= 0:
        return "BEAR"
    return "NEUTRAL"


def _four_hour_structure(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    settings: AppSettings,
) -> dict[str, object]:
    return _timeframe_structure(candles, indicators, settings)


def _one_hour_structure(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    settings: AppSettings,
) -> dict[str, object]:
    return _timeframe_structure(candles, indicators, settings)


def _timeframe_structure(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    settings: AppSettings,
) -> dict[str, object]:
    lookback = min(settings.strategy.structure_lookback, len(candles) - 1, len(indicators) - 1)
    if lookback < 6:
        return {"state": "UNKNOWN", "support": None, "resistance": None, "box_width_pct": 0.0}
    current = candles[-1]
    indicator = indicators[-1]
    prior = candles[-lookback - 1 : -1]
    support = min(candle.low for candle in prior)
    resistance = max(candle.high for candle in prior)
    mid = (support + resistance) / 2
    box_width_pct = (resistance - support) / mid if mid else 0.0
    buffer = (indicator.atr14 or current.close * 0.005) * settings.strategy.structure_buffer_atr
    if current.close > resistance + buffer:
        state = "BREAKOUT_UP"
    elif current.close < support - buffer:
        state = "BREAKDOWN_DOWN"
    elif current.close >= mid:
        state = "BOX_UPPER_HALF"
    else:
        state = "BOX_LOWER_HALF"
    return {
        "state": state,
        "support": support,
        "resistance": resistance,
        "support_zone_low": support - buffer,
        "support_zone_high": support + buffer,
        "resistance_zone_low": resistance - buffer,
        "resistance_zone_high": resistance + buffer,
        "box_mid": mid,
        "box_width_pct": box_width_pct,
    }


def _one_hour_trigger(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    h4: dict[str, object],
    h1_structure: dict[str, object],
    settings: AppSettings,
) -> dict[str, object]:
    if len(candles) < 3 or not indicators:
        return {"direction": "NONE", "state": "UNKNOWN"}
    current = candles[-1]
    previous = candles[-2]
    indicator = indicators[-1]
    structure = h1_structure if _float_or_none(h1_structure.get("support")) is not None else h4
    support = _float_or_none(structure.get("support"))
    resistance = _float_or_none(structure.get("resistance"))
    buffer = (indicator.atr14 or current.close * 0.004) * settings.strategy.structure_buffer_atr
    support_zone_low = _float_or_none(structure.get("support_zone_low"))
    support_zone_high = _float_or_none(structure.get("support_zone_high"))
    resistance_zone_low = _float_or_none(structure.get("resistance_zone_low"))
    resistance_zone_high = _float_or_none(structure.get("resistance_zone_high"))
    if support is not None:
        support_zone_low = support_zone_low if support_zone_low is not None else support - buffer
        support_zone_high = support_zone_high if support_zone_high is not None else support + buffer
    if resistance is not None:
        resistance_zone_low = resistance_zone_low if resistance_zone_low is not None else resistance - buffer
        resistance_zone_high = resistance_zone_high if resistance_zone_high is not None else resistance + buffer
    direction = "NONE"
    state = "WAIT"
    if resistance is not None and resistance_zone_low is not None and resistance_zone_high is not None:
        breakout = current.close > resistance_zone_high
        retest = previous.close > resistance_zone_high and current.low <= resistance_zone_high and current.close > resistance_zone_low
        fake_breakout = previous.high > resistance_zone_high and current.close < resistance_zone_low
        if breakout or retest:
            direction, state = "LONG", "BREAKOUT" if breakout else "RETEST"
        elif fake_breakout:
            direction, state = "SHORT", "FAKE_BREAKOUT"
    if support is not None and support_zone_low is not None and support_zone_high is not None:
        breakdown = current.close < support_zone_low
        retest = previous.close < support_zone_low and current.high >= support_zone_low and current.close < support_zone_high
        fake_breakdown = previous.low < support_zone_low and current.close > support_zone_high
        if breakdown or retest:
            direction, state = "SHORT", "BREAKDOWN" if breakdown else "RETEST"
        elif fake_breakdown:
            direction, state = "LONG", "FAKE_BREAKDOWN"
    return {
        "direction": direction,
        "state": state,
        "support": support,
        "resistance": resistance,
        "support_zone_low": support_zone_low,
        "support_zone_high": support_zone_high,
        "resistance_zone_low": resistance_zone_low,
        "resistance_zone_high": resistance_zone_high,
    }


def _moving_average_cluster(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    settings: AppSettings,
) -> dict[str, object]:
    if len(candles) < 130 or not indicators:
        return {"state": "UNKNOWN", "price": None, "width_pct": 0.0}
    closes = [candle.close for candle in candles]
    ema5_values = ema(closes, 5)
    ema20_values = ema(closes, 20)
    ema60_values = ema(closes, 60)
    ema120_values = ema(closes, 120)
    ma20_values = sma(closes, 20)
    ma60_values = sma(closes, 60)
    ma120_values = sma(closes, 120)
    current = candles[-1]
    previous = candles[-2]
    indicator = indicators[-1]
    values = [
        ema5_values[-1],
        ema20_values[-1],
        ema60_values[-1],
        ema120_values[-1],
        ma20_values[-1],
        ma60_values[-1],
        ma120_values[-1],
    ]
    if any(value is None for value in values):
        return {"state": "UNKNOWN", "price": None, "width_pct": 0.0}
    averages = [float(value) for value in values if value is not None]
    lower = min(averages)
    upper = max(averages)
    price = sum(averages) / len(averages)
    width_pct = (upper - lower) / current.close if current.close else 0.0
    atr_pct = (indicator.atr14 or current.close * 0.01) / current.close if current.close else 0.01
    dense_threshold = min(0.03, max(0.012, atr_pct * 1.2))
    buffer = (indicator.atr14 or current.close * 0.004) * settings.strategy.structure_buffer_atr
    previous_close = previous.close
    state = "DENSE" if width_pct <= dense_threshold else "SPREAD"
    if width_pct <= dense_threshold:
        if previous_close <= upper and current.close > upper + buffer:
            state = "BREAKOUT_UP"
        elif previous_close >= lower and current.close < lower - buffer:
            state = "BREAKDOWN_DOWN"
    elif current.low <= max(ema20_values[-1] or price, upper) + buffer and current.close > max(ema20_values[-1] or price, upper):
        state = "RETEST_UP"
    elif current.high >= min(ema20_values[-1] or price, lower) - buffer and current.close < min(ema20_values[-1] or price, lower):
        state = "RETEST_DOWN"
    target_up, target_down = _previous_cluster_targets(candles, dense_threshold)
    return {
        "state": state,
        "price": price,
        "lower": lower,
        "upper": upper,
        "width_pct": width_pct,
        "ema5": ema5_values[-1],
        "ema20": ema20_values[-1],
        "ema60": ema60_values[-1],
        "ema120": ema120_values[-1],
        "ma20": ma20_values[-1],
        "ma60": ma60_values[-1],
        "ma120": ma120_values[-1],
        "target_up": target_up,
        "target_down": target_down,
    }


def _previous_cluster_targets(candles: list[Candle], dense_threshold: float) -> tuple[float | None, float | None]:
    if len(candles) < 150:
        return None, None
    closes = [candle.close for candle in candles]
    series = [ema(closes, 5), ema(closes, 20), ema(closes, 60), ema(closes, 120), sma(closes, 20), sma(closes, 60), sma(closes, 120)]
    current_close = candles[-1].close
    target_up: float | None = None
    target_down: float | None = None
    for idx in range(len(candles) - 12, 130, -1):
        values = [items[idx] for items in series]
        if any(value is None for value in values):
            continue
        averages = [float(value) for value in values if value is not None]
        price = sum(averages) / len(averages)
        width_pct = (max(averages) - min(averages)) / candles[idx].close if candles[idx].close else 0.0
        if width_pct > dense_threshold:
            continue
        if price > current_close and target_up is None:
            target_up = price
        if price < current_close and target_down is None:
            target_down = price
        if target_up is not None and target_down is not None:
            break
    return target_up, target_down


def _four_hour_oi_state(
    indicators: list[IndicatorSnapshot],
    h4: dict[str, object],
    h1: dict[str, object],
    h1_pullback: dict[str, object],
) -> dict[str, object]:
    values = [item.open_interest for item in indicators[-24:] if item.open_interest is not None]
    ratios = [item.long_short_ratio for item in indicators[-8:] if item.long_short_ratio is not None]
    if len(values) < 8 or not values[-1]:
        return {"state": "UNKNOWN", "drop_from_high_pct": 0.0, "rebound_pct": 0.0}
    current = values[-1]
    high = max(values)
    previous = values[-2] if len(values) >= 2 else current
    drop_from_high = (current - high) / high if high else 0.0
    rebound = (current - previous) / previous if previous else 0.0
    ratio_change = (ratios[-1] - ratios[0]) / ratios[0] if len(ratios) >= 2 and ratios[0] else 0.0
    h4_state = str(h4.get("state") or "UNKNOWN")
    h1_direction = str(h1.get("direction") or "NONE")
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    big_drop = drop_from_high <= -0.20
    mild_rebound = rebound >= 0.003
    held_long_structure = pullback_direction == "LONG" and pullback_state in {"HEALTHY_PULLBACK", "HIGH_PULLBACK"}
    failed_long_structure = h4_state == "BREAKDOWN_DOWN" or h1_direction == "SHORT"
    long_crowd_after_drop = ratio_change >= 0.05
    if not big_drop:
        return {"state": "NORMAL", "drop_from_high_pct": drop_from_high, "rebound_pct": rebound, "ratio_change_pct": ratio_change}
    if mild_rebound and h4_state == "BREAKOUT_UP" and not long_crowd_after_drop:
        state = "REBUILD_BREAKOUT_LONG"
    elif failed_long_structure:
        state = "DELEVERAGE_CROWD_BREAKDOWN" if long_crowd_after_drop else "DELEVERAGE_BREAKDOWN"
    elif held_long_structure:
        state = "DELEVERAGE_CROWD_HOLD_LONG" if long_crowd_after_drop else "DELEVERAGE_HOLD_LONG"
    else:
        state = "DELEVERAGE_CROWD_WAIT" if long_crowd_after_drop else "DELEVERAGE_WAIT"
    return {"state": state, "drop_from_high_pct": drop_from_high, "rebound_pct": rebound, "ratio_change_pct": ratio_change}


def _four_hour_oi_valley(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    settings: AppSettings,
) -> dict[str, object]:
    """Confirm retail-long carry, capitulation OI flush, then OI rebuilding."""
    usable = min(len(candles), len(indicators))
    if usable < 8:
        return {"state": "UNKNOWN"}
    observations = [
        (candle, indicator)
        for candle, indicator in zip(
            candles[-24:],
            indicators[-24:],
        )
        if indicator.open_interest is not None
        and indicator.long_short_ratio is not None
        and indicator.open_interest > 0
    ]
    if len(observations) < 8:
        return {"state": "UNKNOWN"}
    flush_candidates: list[tuple[float, int]] = []
    for idx in range(1, len(observations) - 1):
        previous_oi = observations[idx - 1][1].open_interest
        current_oi = observations[idx][1].open_interest
        if previous_oi is None or current_oi is None or previous_oi <= 0:
            continue
        drop = (previous_oi - current_oi) / previous_oi
        flush_candidates.append((drop, idx))
    if not flush_candidates:
        return {"state": "WATCH"}
    flush_pct, valley_idx = max(flush_candidates)
    valley_candle, valley_indicator = observations[valley_idx]
    pre_flush_candle, pre_flush_indicator = observations[valley_idx - 1]
    valley_oi = float(valley_indicator.open_interest or 0.0)
    current_oi = float(observations[-1][1].open_interest or 0.0)
    rebuild_pct = (
        (current_oi - valley_oi) / valley_oi
        if valley_oi > 0
        else 0.0
    )
    carry_start = max(0, valley_idx - 6)
    prior_carry = observations[carry_start:valley_idx]
    pre_flush_oi = float(pre_flush_indicator.open_interest or 0.0)
    pre_flush_ratio = float(pre_flush_indicator.long_short_ratio or 0.0)
    carry_first_candle, carry_first_indicator = (
        prior_carry[0]
        if prior_carry
        else (pre_flush_candle, pre_flush_indicator)
    )
    carry_first_oi = float(carry_first_indicator.open_interest or 0.0)
    carry_first_ratio = float(carry_first_indicator.long_short_ratio or 0.0)
    carry_price_change = (
        (pre_flush_candle.close - carry_first_candle.close) / carry_first_candle.close
        if carry_first_candle.close > 0
        else 0.0
    )
    carry_oi_change = (
        (pre_flush_oi - carry_first_oi) / carry_first_oi
        if carry_first_oi > 0
        else 0.0
    )
    carry_ratio_change = (
        (pre_flush_ratio - carry_first_ratio) / carry_first_ratio
        if carry_first_ratio > 0
        else 0.0
    )
    retail_long_carry = (
        len(prior_carry) >= 3
        and carry_price_change < 0
        and carry_oi_change < 0
        and carry_ratio_change > 0
    )
    valley_is_local_low = valley_oi <= min(
        float(indicator.open_interest or valley_oi)
        for _, indicator in observations[carry_start : valley_idx + 1]
    )
    confirmed = (
        retail_long_carry
        and valley_is_local_low
        and flush_pct >= settings.strategy.smart_money_oi_flush
        and rebuild_pct >= settings.strategy.smart_money_oi_rebuild
    )
    return {
        "state": "CONFIRMED" if confirmed else "WATCH",
        "retail_long_carry": retail_long_carry,
        "valley_is_local_low": valley_is_local_low,
        "flush_timestamp": valley_candle.timestamp.isoformat(),
        "pre_flush_oi": pre_flush_oi,
        "valley_oi": valley_oi,
        "current_oi": current_oi,
        "flush_pct": flush_pct,
        "rebuild_pct": rebuild_pct,
        "pre_flush_long_short_ratio": pre_flush_ratio,
        "carry_price_change_pct": carry_price_change,
        "carry_oi_change_pct": carry_oi_change,
        "carry_long_short_ratio_change_pct": carry_ratio_change,
    }


def _distribution_short_structure(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    h4_oi: dict[str, object],
    settings: AppSettings,
) -> dict[str, object]:
    """Build the price anchors used while a post-markup distribution unwinds."""
    if len(candles) < 12 or not indicators:
        return {
            "active": False,
            "range_high_zone": None,
            "descending_trendline_zone": None,
        }
    indicator = indicators[-1]
    buffer = (
        indicator.atr14 or candles[-1].close * 0.005
    ) * settings.strategy.structure_buffer_atr
    start = max(2, len(candles) - 48)
    swing_highs: list[tuple[int, float]] = []
    for idx in range(start, len(candles) - 2):
        high = candles[idx].high
        neighboring = (
            candles[idx - 2].high,
            candles[idx - 1].high,
            candles[idx + 1].high,
            candles[idx + 2].high,
        )
        if high >= max(neighboring):
            swing_highs.append((idx, high))
    if not swing_highs:
        recent = candles[-12:-1]
        if not recent:
            return {
                "active": False,
                "range_high_zone": None,
                "descending_trendline_zone": None,
            }
        relative_idx, candle = max(
            enumerate(recent),
            key=lambda item: item[1].high,
        )
        swing_highs.append(
            (len(candles) - len(recent) + relative_idx, candle.high)
        )
    latest_idx, latest_high = swing_highs[-1]
    range_high_zone = {
        "low": latest_high - buffer,
        "high": latest_high + buffer,
        "price": latest_high,
    }
    descending_zone: dict[str, float] | None = None
    descending_anchors: tuple[tuple[int, float], tuple[int, float]] | None = None
    for earlier, later in zip(reversed(swing_highs[:-1]), reversed(swing_highs[1:])):
        if later[0] <= earlier[0] or later[1] >= earlier[1]:
            continue
        slope = (later[1] - earlier[1]) / (later[0] - earlier[0])
        projected = later[1] + slope * ((len(candles) - 1) - later[0])
        if projected <= 0:
            continue
        descending_zone = {
            "low": projected - buffer,
            "high": projected + buffer,
            "price": projected,
        }
        descending_anchors = (earlier, later)
        break
    oi_drop = _float_or_none(h4_oi.get("drop_from_high_pct")) or 0.0
    active = oi_drop <= -settings.strategy.smart_money_oi_flush
    return {
        "active": active,
        "oi_drop_from_high_pct": oi_drop,
        "range_high_zone": range_high_zone,
        "descending_trendline_zone": descending_zone,
        "descending_anchors": descending_anchors,
        "latest_swing_high_index": latest_idx,
    }


def _distribution_short_stage(
    signal: dict[str, object],
    context: dict[str, object],
) -> str | None:
    structure = (
        context.get("distribution_short")
        if isinstance(context.get("distribution_short"), dict)
        else {}
    )
    handoff = (
        context.get("high_distribution_handoff")
        if isinstance(context.get("high_distribution_handoff"), dict)
        else {}
    )
    handoff_confirmed = str(handoff.get("state") or "") == "CONFIRMED"
    smart_money_phase = str(signal.get("smart_money_phase") or "")
    if (
        not structure.get("active")
        and smart_money_phase != "DISTRIBUTION_EXIT"
        and not handoff_confirmed
    ):
        return None
    h4 = (
        context.get("h4_structure")
        if isinstance(context.get("h4_structure"), dict)
        else {}
    )
    h1 = (
        context.get("h1_trigger")
        if isinstance(context.get("h1_trigger"), dict)
        else {}
    )
    h4_oi = (
        context.get("h4_oi")
        if isinstance(context.get("h4_oi"), dict)
        else {}
    )
    h4_state = str(h4.get("state") or "UNKNOWN")
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    oi_state = str(h4_oi.get("state") or "UNKNOWN")
    trend_state = str(signal.get("trend_state") or "")
    markdown_confirmed = (
        oi_state in {
            "DELEVERAGE_BREAKDOWN",
            "DELEVERAGE_CROWD_BREAKDOWN",
        }
        and (
            h4_state == "BREAKDOWN_DOWN"
            or (
                trend_state == "ONE_WAY_DOWN"
                and h1_direction == "SHORT"
                and h1_state == "BREAKDOWN"
            )
        )
    )
    if markdown_confirmed:
        return DISTRIBUTION_STAGE_MARKDOWN
    if handoff_confirmed:
        return DISTRIBUTION_STAGE_DESCENDING
    if isinstance(structure.get("descending_trendline_zone"), dict):
        return DISTRIBUTION_STAGE_DESCENDING
    return DISTRIBUTION_STAGE_RANGE


def _one_hour_ema_reliability(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
) -> dict[str, object]:
    """Detect when 1h EMA support/resistance is too easy to sweep."""
    usable = min(len(candles), len(indicators))
    if usable < 60:
        return {
            "state": "UNKNOWN",
            "long_state": "UNKNOWN",
            "short_state": "UNKNOWN",
            "ema20_breach_episodes": 0,
            "ema60_breach_episodes": 0,
            "ema20_resistance_breach_episodes": 0,
            "ema60_resistance_breach_episodes": 0,
        }
    candles = candles[-usable:]
    indicators = indicators[-usable:]
    ema60_values = ema([candle.close for candle in candles], 60)
    start = max(0, usable - 24)
    ema20_support_breaches: list[bool] = []
    ema60_support_breaches: list[bool] = []
    ema20_resistance_breaches: list[bool] = []
    ema60_resistance_breaches: list[bool] = []
    for idx in range(start, usable):
        indicator = indicators[idx]
        ema20_value = indicator.ema20
        ema60_value = ema60_values[idx]
        atr = indicator.atr14 or candles[idx].close * 0.01
        tolerance = atr * 0.10
        ema20_support_breaches.append(
            ema20_value is not None
            and candles[idx].low < float(ema20_value) - tolerance
        )
        ema60_support_breaches.append(
            ema60_value is not None
            and candles[idx].low < float(ema60_value) - tolerance
        )
        ema20_resistance_breaches.append(
            ema20_value is not None
            and candles[idx].high > float(ema20_value) + tolerance
        )
        ema60_resistance_breaches.append(
            ema60_value is not None
            and candles[idx].high > float(ema60_value) + tolerance
        )

    def episodes(values: list[bool]) -> int:
        return sum(
            1
            for idx, value in enumerate(values)
            if value and (idx == 0 or not values[idx - 1])
        )

    ema20_episodes = episodes(ema20_support_breaches)
    ema60_episodes = episodes(ema60_support_breaches)
    ema20_resistance_episodes = episodes(ema20_resistance_breaches)
    ema60_resistance_episodes = episodes(ema60_resistance_breaches)
    long_unreliable = ema20_episodes >= 2 or ema60_episodes >= 1
    short_unreliable = (
        ema20_resistance_episodes >= 2
        or ema60_resistance_episodes >= 1
    )
    long_state = "UNRELIABLE" if long_unreliable else "RELIABLE"
    short_state = "UNRELIABLE" if short_unreliable else "RELIABLE"
    return {
        # Keep state as the legacy alias for long-side support reliability.
        "state": long_state,
        "long_state": long_state,
        "short_state": short_state,
        "ema20_breach_episodes": ema20_episodes,
        "ema60_breach_episodes": ema60_episodes,
        "ema20_resistance_breach_episodes": ema20_resistance_episodes,
        "ema60_resistance_breach_episodes": ema60_resistance_episodes,
        "lookback_hours": len(ema20_support_breaches),
    }


def _high_distribution_handoff(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
) -> dict[str, object]:
    """Confirm that post-high OI rebuilt into weak price while longs crowded in."""
    usable = min(len(candles), len(indicators))
    if usable < 16:
        return {"state": "UNKNOWN"}
    candles = candles[-48:]
    indicators = indicators[-len(candles):]
    current_idx = len(candles) - 1
    peak_candidates = range(
        max(0, len(candles) - 24),
        max(0, current_idx - 1),
    )
    if not peak_candidates:
        return {"state": "UNKNOWN"}
    peak_idx = max(peak_candidates, key=lambda idx: candles[idx].high)
    peak_indicator = indicators[peak_idx]
    current_indicator = indicators[current_idx]
    peak_oi = peak_indicator.open_interest
    current_oi = current_indicator.open_interest
    peak_ratio = peak_indicator.long_short_ratio
    current_ratio = current_indicator.long_short_ratio
    if (
        peak_oi is None
        or current_oi is None
        or peak_ratio is None
        or current_ratio is None
        or peak_oi <= 0
        or peak_idx >= current_idx - 1
    ):
        return {"state": "UNKNOWN"}
    oi_changes = [
        abs(current.open_interest - previous.open_interest)
        / previous.open_interest
        for previous, current in zip(indicators, indicators[1:])
        if previous.open_interest is not None
        and current.open_interest is not None
        and previous.open_interest > 0
    ]
    if not oi_changes:
        return {"state": "UNKNOWN"}
    typical_oi_change = median(oi_changes)
    post_peak_oi = [
        indicator.open_interest
        for indicator in indicators[peak_idx + 1 :]
        if indicator.open_interest is not None
    ]
    if not post_peak_oi:
        return {"state": "UNKNOWN"}
    minimum_oi = min(post_peak_oi)
    flush_pct = (
        max(peak_oi - minimum_oi, 0.0) / peak_oi
        if peak_oi
        else 0.0
    )
    meaningful_flush = (
        typical_oi_change > 0
        and flush_pct >= typical_oi_change
    )
    oi_rebuilt = current_oi >= peak_oi
    price_failed = (
        candles[current_idx].close < candles[peak_idx].low
    )
    ratio_shifted_to_longs = current_ratio > peak_ratio
    confirmed = (
        meaningful_flush
        and oi_rebuilt
        and price_failed
        and ratio_shifted_to_longs
    )
    return {
        "state": "CONFIRMED" if confirmed else "WATCH",
        "peak_timestamp": candles[peak_idx].timestamp.isoformat(),
        "peak_price": candles[peak_idx].high,
        "peak_candle_low": candles[peak_idx].low,
        "peak_oi": peak_oi,
        "minimum_oi": minimum_oi,
        "current_oi": current_oi,
        "oi_flush_pct": flush_pct,
        "typical_oi_change_pct": typical_oi_change,
        "peak_long_short_ratio": peak_ratio,
        "current_long_short_ratio": current_ratio,
        "oi_rebuilt": oi_rebuilt,
        "price_failed": price_failed,
        "ratio_shifted_to_longs": ratio_shifted_to_longs,
    }


def _one_hour_pullback(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    h4: dict[str, object],
    settings: AppSettings,
) -> dict[str, object]:
    if len(candles) < 12 or len(indicators) < 12:
        return {"direction": "NONE", "state": "UNKNOWN", "extension_pct": 0.0}
    current = candles[-1]
    previous = candles[-2]
    indicator = indicators[-1]
    if indicator.ema20 is None or indicator.boll_mid is None or indicator.atr14 is None:
        return {"direction": "NONE", "state": "UNKNOWN", "extension_pct": 0.0}
    recent = candles[-12:]
    recent_low = min(candle.low for candle in recent)
    recent_high = max(candle.high for candle in recent)
    extension_up = (recent_high - recent_low) / recent_low if recent_low else 0.0
    extension_down = (recent_high - recent_low) / recent_high if recent_high else 0.0
    buffer = indicator.atr14 * settings.strategy.structure_buffer_atr
    touched_long_zone = current.low <= max(indicator.ema20, indicator.boll_mid) + buffer
    reclaimed_long_zone = current.close >= indicator.boll_mid and current.close >= indicator.ema20
    touched_short_zone = current.high >= min(indicator.ema20, indicator.boll_mid) - buffer
    rejected_short_zone = current.close <= indicator.boll_mid and current.close <= indicator.ema20
    previous_was_above = previous.close >= indicator.boll_mid or previous.close >= indicator.ema20
    previous_was_below = previous.close <= indicator.boll_mid or previous.close <= indicator.ema20
    h4_state = str(h4.get("state") or "UNKNOWN")
    near_h4_resistance = _near_level(current.close, _float_or_none(h4.get("resistance")), indicator.atr14 * 1.5)
    near_h4_support = _near_level(current.close, _float_or_none(h4.get("support")), indicator.atr14 * 1.5)
    high_area = h4_state in {"BREAKOUT_UP", "BOX_UPPER_HALF"} or near_h4_resistance
    low_area = h4_state in {"BREAKDOWN_DOWN", "BOX_LOWER_HALF"} or near_h4_support
    if touched_long_zone and reclaimed_long_zone and previous_was_above:
        state = "HIGH_PULLBACK" if high_area and extension_up >= 0.08 else "HEALTHY_PULLBACK"
        return {"direction": "LONG", "state": state, "extension_pct": extension_up}
    if touched_short_zone and rejected_short_zone and previous_was_below:
        state = "LOW_PULLBACK" if low_area and extension_down >= 0.08 else "HEALTHY_PULLBACK"
        return {"direction": "SHORT", "state": state, "extension_pct": extension_down}
    return {"direction": "NONE", "state": "WAIT", "extension_pct": max(extension_up, extension_down)}


def _fifteen_minute_precision(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    settings: AppSettings,
) -> dict[str, object]:
    lookback = min(8, len(candles), len(indicators))
    if lookback < 3:
        return {"long_stop_anchor": None, "short_stop_anchor": None, "pullback": "UNKNOWN", "trend": "UNKNOWN"}
    recent = candles[-lookback:]
    indicator = indicators[-1]
    buffer = (indicator.atr14 or recent[-1].close * 0.004) * settings.strategy.structure_buffer_atr
    ema9_values = ema([candle.close for candle in candles], 9)
    ema60_values = ema([candle.close for candle in candles], 60)
    ema9_value = ema9_values[-1] if ema9_values else None
    ema60_value = ema60_values[-1] if ema60_values else None
    current = candles[-1]
    mid = indicator.boll_mid
    strong_up = (
        ema9_value is not None
        and indicator.ema20 is not None
        and ema60_value is not None
        and ema9_value > indicator.ema20 > ema60_value
        and current.close >= indicator.ema20
        and sum(1 for candle in recent[-5:] if candle.close >= ema60_value) >= 4
    )
    strong_down = (
        ema9_value is not None
        and indicator.ema20 is not None
        and ema60_value is not None
        and ema9_value < indicator.ema20 < ema60_value
        and current.close <= indicator.ema20
        and sum(1 for candle in recent[-5:] if candle.close <= ema60_value) >= 4
    )
    m15_trend = "UP" if strong_up else "DOWN" if strong_down else "CHOP"
    long_ref = max(value for value in (ema9_value, mid) if value is not None) if ema9_value is not None or mid is not None else None
    short_ref = min(value for value in (ema9_value, mid) if value is not None) if ema9_value is not None or mid is not None else None
    pullback = "WAIT"
    if strong_up and long_ref is not None and current.low <= long_ref + buffer and current.close >= long_ref - buffer * 0.25:
        pullback = "M15_LONG_PULLBACK"
    elif strong_down and short_ref is not None and current.high >= short_ref - buffer and current.close <= short_ref + buffer * 0.25:
        pullback = "M15_SHORT_PULLBACK"
    return {
        "long_stop_anchor": min(candle.low for candle in recent) - buffer,
        "short_stop_anchor": max(candle.high for candle in recent) + buffer,
        "ema9": ema9_value,
        "ema20": indicator.ema20,
        "ema60": ema60_value,
        "boll_mid": mid,
        "trend": m15_trend,
        "long_pullback_zone": _range_around_values([indicator.ema20, ema60_value, mid], buffer),
        "short_retest_zone": _range_around_values([indicator.ema20, ema60_value, mid], buffer),
        "pullback": pullback,
    }


def _entry_level_context(
    h1_structure: dict[str, object],
    h4_structure: dict[str, object],
    h1_ma_cluster: dict[str, object],
    h4_ma_cluster: dict[str, object],
    timeframe_indicators: dict[str, list[IndicatorSnapshot]],
    m15: dict[str, object],
    distribution_short: dict[str, object],
) -> dict[str, object]:
    h1_indicator = _latest_indicator(timeframe_indicators.get("1h", []))
    h4_indicator = _latest_indicator(timeframe_indicators.get("4h", []))
    m15_indicator = _latest_indicator(timeframe_indicators.get("15m", []))
    return {
        "long": {
            "h1_support": _zone_from_mapping(h1_structure, "support"),
            "h4_support": _zone_from_mapping(h4_structure, "support"),
            "h1_boll_mid": _indicator_band(h1_indicator, h1_indicator.boll_mid if h1_indicator else None),
            "h4_boll_mid": _indicator_band(h4_indicator, h4_indicator.boll_mid if h4_indicator else None),
            "h1_ema20_ema60": _ema20_ema60_band(h1_indicator, h1_ma_cluster),
            "h4_ema20_ema60": _ema20_ema60_band(h4_indicator, h4_ma_cluster),
            "h1_ema20": _ema_level_band(h1_indicator, "ema20", h1_ma_cluster),
            "h1_ema60": _ema_level_band(h1_indicator, "ema60", h1_ma_cluster),
            "h4_ema20": _ema_level_band(h4_indicator, "ema20", h4_ma_cluster),
            "h4_ema60": _ema_level_band(h4_indicator, "ema60", h4_ma_cluster),
            "m15_ema20_ema60": _zone_from_object(m15.get("long_pullback_zone")),
            "sweep_reclaim_support": _zone_from_mapping(h1_structure, "support") or _zone_from_mapping(h4_structure, "support"),
            "ma_cluster_breakout": _cluster_band(h1_ma_cluster) or _cluster_band(h4_ma_cluster),
            "ma20_retest": _ma20_band(h1_ma_cluster, h1_indicator) or _ma20_band(h4_ma_cluster, h4_indicator),
            "breakout_retest": _zone_from_mapping(h1_structure, "resistance") or _zone_from_mapping(h4_structure, "resistance"),
            "vwap_pullback": _indicator_band(h1_indicator, h1_indicator.vwap if h1_indicator else None),
        },
        "short": {
            "distribution_range_high": _zone_from_object(
                distribution_short.get("range_high_zone")
            ),
            "descending_high_trendline": _zone_from_object(
                distribution_short.get("descending_trendline_zone")
            ),
            "h1_resistance": _zone_from_mapping(h1_structure, "resistance"),
            "h4_resistance": _zone_from_mapping(h4_structure, "resistance"),
            "h1_boll_mid": _indicator_band(h1_indicator, h1_indicator.boll_mid if h1_indicator else None),
            "h4_boll_mid": _indicator_band(h4_indicator, h4_indicator.boll_mid if h4_indicator else None),
            "h1_ema20_ema60": _ema20_ema60_band(h1_indicator, h1_ma_cluster),
            "h4_ema20_ema60": _ema20_ema60_band(h4_indicator, h4_ma_cluster),
            "h1_ema20": _ema_level_band(h1_indicator, "ema20", h1_ma_cluster),
            "h1_ema60": _ema_level_band(h1_indicator, "ema60", h1_ma_cluster),
            "h4_ema20": _ema_level_band(h4_indicator, "ema20", h4_ma_cluster),
            "h4_ema60": _ema_level_band(h4_indicator, "ema60", h4_ma_cluster),
            "sweep_reject_resistance": _zone_from_mapping(h1_structure, "resistance") or _zone_from_mapping(h4_structure, "resistance"),
            "ma_cluster_breakdown": _cluster_band(h1_ma_cluster) or _cluster_band(h4_ma_cluster),
            "ma20_retest": _ma20_band(h1_ma_cluster, h1_indicator) or _ma20_band(h4_ma_cluster, h4_indicator),
            "breakdown_retest": _zone_from_mapping(h1_structure, "support") or _zone_from_mapping(h4_structure, "support"),
            "vwap_retest": _indicator_band(h1_indicator, h1_indicator.vwap if h1_indicator else None),
        },
        "chop": {
            "long": _zone_from_mapping(h1_structure, "support") or _zone_from_mapping(h4_structure, "support"),
            "short": _zone_from_mapping(h1_structure, "resistance") or _zone_from_mapping(h4_structure, "resistance"),
            "mid": _indicator_band(h1_indicator, h1_indicator.boll_mid if h1_indicator else None),
        },
    }


def _latest_indicator(indicators: list[IndicatorSnapshot]) -> IndicatorSnapshot | None:
    return indicators[-1] if indicators else None


def _zone_from_mapping(source: dict[str, object], key: str) -> dict[str, float] | None:
    low = _float_or_none(source.get(f"{key}_zone_low"))
    high = _float_or_none(source.get(f"{key}_zone_high"))
    price = _float_or_none(source.get(key))
    if low is None and high is None and price is None:
        return None
    if low is None:
        low = price
    if high is None:
        high = price
    values = [value for value in (low, high, price) if value is not None]
    if not values:
        return None
    return {"low": min(values), "high": max(values), "price": price if price is not None else sum(values) / len(values)}


def _zone_from_object(source: object) -> dict[str, float] | None:
    if not isinstance(source, dict):
        return None
    low = _float_or_none(source.get("low"))
    high = _float_or_none(source.get("high"))
    price = _float_or_none(source.get("price"))
    values = [value for value in (low, high, price) if value is not None]
    if not values:
        return None
    return {"low": min(values), "high": max(values), "price": price if price is not None else sum(values) / len(values)}


def _range_around_values(values: list[float | None], buffer: float) -> dict[str, float] | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return {"low": min(valid) - buffer, "high": max(valid) + buffer, "price": sum(valid) / len(valid)}


def _indicator_buffer(indicator: IndicatorSnapshot | None, price: float | None) -> float:
    if indicator and indicator.atr14:
        return indicator.atr14 * 0.30
    if price:
        return price * 0.003
    return 0.0


def _indicator_band(indicator: IndicatorSnapshot | None, price: float | None) -> dict[str, float] | None:
    value = _float_or_none(price)
    if value is None:
        return None
    buffer = _indicator_buffer(indicator, value)
    return {"low": value - buffer, "high": value + buffer, "price": value}


def _ema_level_band(
    indicator: IndicatorSnapshot | None,
    ema_key: str,
    cluster: dict[str, object],
) -> dict[str, float] | None:
    value = (
        indicator.ema20
        if ema_key == "ema20" and indicator is not None
        else _float_or_none(cluster.get("ema60"))
        if ema_key == "ema60"
        else None
    )
    return _indicator_band(indicator, value)


def _ema20_ema60_band(indicator: IndicatorSnapshot | None, cluster: dict[str, object]) -> dict[str, float] | None:
    ema60 = _float_or_none(cluster.get("ema60"))
    values = [indicator.ema20 if indicator else None, ema60]
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    reference = sum(valid) / len(valid)
    buffer = _indicator_buffer(indicator, reference)
    if (
        indicator is not None
        and indicator.ema20 is not None
        and ema60 is not None
        and abs(float(indicator.ema20) - ema60) > buffer * 2
    ):
        return None
    return {"low": min(valid) - buffer, "high": max(valid) + buffer, "price": reference}


def _cluster_band(cluster: dict[str, object]) -> dict[str, float] | None:
    low = _float_or_none(cluster.get("lower"))
    high = _float_or_none(cluster.get("upper"))
    price = _float_or_none(cluster.get("price"))
    if low is None and high is None and price is None:
        return None
    values = [value for value in (low, high, price) if value is not None]
    return {"low": min(values), "high": max(values), "price": price if price is not None else sum(values) / len(values)}


def _ma20_band(cluster: dict[str, object], indicator: IndicatorSnapshot | None) -> dict[str, float] | None:
    return _indicator_band(indicator, _float_or_none(cluster.get("ma20")))


def _weak_deleverage_rebound(
    signal: dict[str, object],
    h4_oi: dict[str, object],
    h1_pullback: dict[str, object],
) -> bool:
    """OI flushes need volume/OI recovery before a rebound can be trusted as long support."""
    state = str(h4_oi.get("state") or "UNKNOWN")
    drop_from_high = _float_or_none(h4_oi.get("drop_from_high_pct")) or 0.0
    rebound = _float_or_none(h4_oi.get("rebound_pct")) or 0.0
    volume_ratio = _float_or_none(signal.get("volume_ratio"))
    current_oi_change = _float_or_none(signal.get("oi_change"))

    deleveraged = state in {
        "DELEVERAGE_HOLD_LONG",
        "DELEVERAGE_CROWD_HOLD_LONG",
        "DELEVERAGE_WAIT",
        "DELEVERAGE_CROWD_WAIT",
    } or drop_from_high <= -0.16
    if not deleveraged:
        return False
    oi_not_recovered = rebound < 0.003 and (current_oi_change is None or current_oi_change < 0.01)
    volume_weak = volume_ratio is not None and volume_ratio < 1.0
    return oi_not_recovered and volume_weak


def _oi_valley_long_reversal_confirmed(
    h4_oi_valley: dict[str, object],
    h1_trigger: dict[str, object],
    h1_pullback: dict[str, object],
) -> bool:
    if str(h4_oi_valley.get("state") or "") != "CONFIRMED":
        return False
    h1_direction = str(h1_trigger.get("direction") or "NONE")
    h1_state = str(h1_trigger.get("state") or "UNKNOWN")
    return (
        h1_direction == "LONG"
        and h1_state == "FAKE_BREAKDOWN"
    )


def _apply_multi_timeframe_context(signal: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    out = dict(signal)
    reasons = list(out.get("reasons") or [])
    vetoes = list(out.get("vetoes") or [])
    score = int(out.get("score") or 0)
    action = str(out.get("candidate_action") or out.get("action") or "")
    daily_bias = str(context.get("daily_bias") or "NEUTRAL")
    h4 = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    h1_structure = context.get("h1_structure") if isinstance(context.get("h1_structure"), dict) else {}
    h4_ma_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    h1_ma_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_oi = context.get("h4_oi") if isinstance(context.get("h4_oi"), dict) else {}
    h4_oi_valley = context.get("h4_oi_valley") if isinstance(context.get("h4_oi_valley"), dict) else {}
    h1 = context.get("h1_trigger") if isinstance(context.get("h1_trigger"), dict) else {}
    h1_pullback = context.get("h1_pullback") if isinstance(context.get("h1_pullback"), dict) else {}
    h1_ema_reliability = context.get("h1_ema_reliability") if isinstance(context.get("h1_ema_reliability"), dict) else {}
    high_distribution_handoff = context.get("high_distribution_handoff") if isinstance(context.get("high_distribution_handoff"), dict) else {}
    distribution_short = context.get("distribution_short") if isinstance(context.get("distribution_short"), dict) else {}
    m15_precision = context.get("m15_precision") if isinstance(context.get("m15_precision"), dict) else {}
    large_wick_rejections = (
        context.get("large_wick_rejections")
        if isinstance(context.get("large_wick_rejections"), dict)
        else {}
    )
    entry_levels = context.get("entry_levels") if isinstance(context.get("entry_levels"), dict) else {}
    entry_guardrails = (
        context.get("entry_guardrails")
        if isinstance(context.get("entry_guardrails"), dict)
        else {}
    )
    h4_state = str(h4.get("state") or "UNKNOWN")
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    h4_oi_state = str(h4_oi.get("state") or "UNKNOWN")
    risk_state = str(out.get("risk_state") or "NORMAL")
    trend_state = str(out.get("trend_state") or "")
    smart_money_phase = str(out.get("smart_money_phase") or "")
    distribution_short_stage = _distribution_short_stage(out, context)
    setup_type = str(out.get("setup_type") or "")
    handoff_confirmed = (
        str(high_distribution_handoff.get("state") or "")
        == "CONFIRMED"
    )
    if handoff_confirmed:
        reasons.append(
            "high distribution handoff complete: OI rebuilt while price failed and long/short ratio rose"
        )
        if action == SignalAction.ENTRY_LONG.value:
            vetoes.append(
                "high distribution handoff complete; avoid new long"
            )
    if action == SignalAction.ENTRY_LONG.value:
        upper_rejection = large_wick_rejections.get("upper")
        if isinstance(upper_rejection, dict):
            vetoes.append(
                _large_wick_rejection_veto_text(
                    PositionSide.LONG,
                    upper_rejection,
                )
            )
    elif action == SignalAction.ENTRY_SHORT.value:
        lower_rejection = large_wick_rejections.get("lower")
        if isinstance(lower_rejection, dict):
            vetoes.append(
                _large_wick_rejection_veto_text(
                    PositionSide.SHORT,
                    lower_rejection,
                )
            )
    entry_timeframe_override: str | None = None
    m15_long_exception = _strong_m15_pullback_allowed(
        PositionSide.LONG,
        trend_state,
        risk_state,
        score,
        m15_precision,
        smart_money_phase,
    )
    long_ema_unreliable = (
        str(
            h1_ema_reliability.get("long_state")
            or h1_ema_reliability.get("state")
            or "UNKNOWN"
        )
        == "UNRELIABLE"
    )
    short_ema_unreliable = (
        str(h1_ema_reliability.get("short_state") or "UNKNOWN")
        == "UNRELIABLE"
    )
    if (
        action == SignalAction.ENTRY_LONG.value
        and long_ema_unreliable
        and not m15_long_exception
    ):
        entry_timeframe_override = "4h"
        setup_type = _prefer_setup_type(
            setup_type,
            SETUP_H4_PULLBACK_LONG,
        )
        reasons.append(
            "1h EMA20/EMA60 repeatedly breached; promote entry, stop, and target to 4h"
        )
    elif (
        action == SignalAction.ENTRY_SHORT.value
        and short_ema_unreliable
    ):
        entry_timeframe_override = "4h"
        setup_type = _prefer_setup_type(
            setup_type,
            SETUP_H4_PULLBACK_SHORT,
        )
        reasons.append(
            "1h EMA20/EMA60 repeatedly breached as resistance; promote entry, stop, and target to 4h"
        )
    rsi14 = _float_or_none(out.get("rsi14"))
    weak_deleverage_rebound = _weak_deleverage_rebound(out, h4_oi, h1_pullback)
    oi_valley_long_reversal = _oi_valley_long_reversal_confirmed(
        h4_oi_valley,
        h1,
        h1_pullback,
    )
    if action == SignalAction.ENTRY_LONG.value:
        if m15_long_exception:
            setup_type = _prefer_setup_type(
                setup_type,
                SETUP_M15_SQUEEZE_TACTICAL_LONG,
            )
        if daily_bias == "BEAR":
            score -= 10
            reasons.append("1d bearish bias; long position size reduced")
        elif daily_bias == "BULL":
            score += 5
            reasons.append("1d bullish bias supports long")
        if h4_state in {"BREAKOUT_UP", "BOX_UPPER_HALF"}:
            score += 6
            reasons.append("4h structure supports upside")
        ma_score, ma_reasons, ma_vetoes = _ma_cluster_signal_adjustment(PositionSide.LONG, h4_ma_cluster, h1_ma_cluster)
        score += ma_score
        reasons.extend(ma_reasons)
        vetoes.extend(ma_vetoes)
        setup_type = _prefer_setup_type(
            setup_type,
            _ma_cluster_setup_type(
                PositionSide.LONG,
                h4_ma_cluster,
                h1_ma_cluster,
            ),
        )
        if h1_direction == "LONG":
            score += 10
            setup_type = _prefer_setup_type(
                setup_type,
                SETUP_H1_STRUCTURE_LONG,
            )
            reasons.append(f"1h {h1_state.lower()} confirms long trigger")
        elif h1_direction == "SHORT":
            vetoes.append("1h trigger opposes long entry")
        if pullback_direction == "LONG":
            if weak_deleverage_rebound:
                score -= 18
                out["leverage_cap"] = min(int(out.get("leverage_cap") or 99), 3)
                out["margin_factor"] = min(float(out.get("margin_factor") or 1.0), 0.3)
                vetoes.append("4h OI drained and volume is weak; EMA/BOLL bounce is not a clean long pullback")
            elif pullback_state == "HEALTHY_PULLBACK" and risk_state == "NORMAL":
                score += 12
                setup_type = _prefer_setup_type(
                    setup_type,
                    SETUP_H1_PULLBACK_LONG,
                )
                reasons.append("1h BOLL/EMA pullback held with clean risk")
            elif pullback_state == "HIGH_PULLBACK" and risk_state in {"LONG_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
                score -= 20
                vetoes.append("high pullback with OI/funding/crowd risk; avoid long entry")
        elif _high_area_needs_long_pullback(h4_state, h1_direction, h1_state):
            if _core_scored_entry_zone_hit(
                out,
                entry_levels,
                PositionSide.LONG,
                reasons=reasons,
                entry_timeframe_override=entry_timeframe_override,
                setup_type=setup_type,
                action=action,
            ):
                pass
            elif _strong_m15_pullback_allowed(
                PositionSide.LONG,
                trend_state,
                risk_state,
                score,
                m15_precision,
                smart_money_phase,
            ):
                score += 6
                setup_type = _prefer_setup_type(
                    setup_type,
                    SETUP_M15_SQUEEZE_TACTICAL_LONG,
                )
                reasons.append("one-way uptrend 15m BOLL/EMA9 pullback confirmed; allow tactical long")
            else:
                score -= 12
                vetoes.append("high area without pullback confirmation; wait for 1h/4h pullback before long")
        if h4_oi_state in {"DELEVERAGE_BREAKDOWN", "DELEVERAGE_CROWD_BREAKDOWN"}:
            score -= 25
            vetoes.append("4h OI deleverage with price breakdown; avoid long entry")
        elif weak_deleverage_rebound:
            reasons.append("wait for OI/volume recovery before treating the rebound as accumulation")
        elif h4_oi_state == "DELEVERAGE_CROWD_HOLD_LONG":
            score -= 8
            out["leverage_cap"] = min(int(out.get("leverage_cap") or 99), 3)
            out["margin_factor"] = min(float(out.get("margin_factor") or 1.0), 0.3)
            vetoes.append(
                "4h OI dropped while long/short ratio rose; retail longs are carrying the decline"
            )
        elif h4_oi_state == "DELEVERAGE_HOLD_LONG":
            reasons.append(
                "4h OI sharp drop is an event, not a confirmed OI valley; wait for OI rebuilding and downside-wick reclaim"
            )
        elif h4_oi_state == "REBUILD_BREAKOUT_LONG":
            score += 12
            reasons.append("4h OI rebounds after deleverage and price breaks out; strong long restored")
        if oi_valley_long_reversal:
            score += 12
            setup_type = _prefer_setup_type(
                setup_type,
                SETUP_OI_VALLEY_REVERSAL_LONG,
            )
            reasons.append(
                "4h OI valley formed after retail capitulation; downside wick reclaimed support"
            )
        elif str(h4_oi_valley.get("state") or "") == "CONFIRMED":
            reasons.append(
                "4h OI valley formed; wait for a downside-wick support reclaim before long"
            )
        rsi_score, rsi_reasons, rsi_vetoes = _rsi_entry_adjustment(
            PositionSide.LONG,
            rsi14,
            trend_state,
            risk_state,
            score,
            pullback_direction,
            pullback_state,
            m15_precision,
            smart_money_phase,
        )
        score += rsi_score
        reasons.extend(rsi_reasons)
        vetoes.extend(rsi_vetoes)
    elif action == SignalAction.ENTRY_SHORT.value:
        if distribution_short_stage == DISTRIBUTION_STAGE_RANGE:
            setup_type = _prefer_setup_type(
                setup_type,
                SETUP_DISTRIBUTION_STAGE1_SHORT,
            )
            score -= 8
            out["leverage_cap"] = min(
                int(out.get("leverage_cap") or 99),
                3,
            )
            out["margin_factor"] = min(
                float(out.get("margin_factor") or 1.0),
                0.25,
            )
            reasons.append(
                "stage 1 high distribution: only a tiny short at a confirmed prior-high rejection"
            )
        elif distribution_short_stage == DISTRIBUTION_STAGE_DESCENDING:
            setup_type = _prefer_setup_type(
                setup_type,
                SETUP_DISTRIBUTION_STAGE2_SHORT,
            )
            reasons.append(
                "stage 2 descending distribution: short only at the projected lower-high trendline"
            )
        elif distribution_short_stage == DISTRIBUTION_STAGE_MARKDOWN:
            setup_type = _prefer_setup_type(
                setup_type,
                SETUP_DISTRIBUTION_STAGE3_SHORT,
            )
            reasons.append(
                "stage 3 markdown acceleration: 1h/4h EMA/BOLL bounce rejection is eligible"
            )
        if daily_bias == "BULL":
            score -= 10
            reasons.append("1d bullish bias; short position size reduced")
        elif daily_bias == "BEAR":
            score += 5
            reasons.append("1d bearish bias supports short")
        if h4_state in {"BREAKDOWN_DOWN", "BOX_LOWER_HALF"}:
            score += 6
            reasons.append("4h structure supports downside")
        ma_score, ma_reasons, ma_vetoes = _ma_cluster_signal_adjustment(PositionSide.SHORT, h4_ma_cluster, h1_ma_cluster)
        score += ma_score
        reasons.extend(ma_reasons)
        vetoes.extend(ma_vetoes)
        setup_type = _prefer_setup_type(
            setup_type,
            _ma_cluster_setup_type(
                PositionSide.SHORT,
                h4_ma_cluster,
                h1_ma_cluster,
            ),
        )
        if h1_direction == "SHORT":
            score += 10
            setup_type = _prefer_setup_type(
                setup_type,
                SETUP_H1_STRUCTURE_SHORT,
            )
            reasons.append(f"1h {h1_state.lower()} confirms short trigger")
        elif h1_direction == "LONG":
            vetoes.append("1h trigger opposes short entry")
        if pullback_direction == "SHORT":
            if pullback_state == "HEALTHY_PULLBACK" and risk_state == "NORMAL":
                score += 12
                setup_type = _prefer_setup_type(
                    setup_type,
                    SETUP_H1_PULLBACK_SHORT,
                )
                reasons.append("1h BOLL/EMA pullback rejected with clean risk")
            elif pullback_state == "LOW_PULLBACK" and risk_state in {"SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
                score -= 20
                vetoes.append("low pullback with OI/funding/crowd risk; avoid short entry")
        elif _low_area_needs_short_pullback(h4_state, h1_direction, h1_state):
            if _core_scored_entry_zone_hit(
                out,
                entry_levels,
                PositionSide.SHORT,
                reasons=reasons,
                entry_timeframe_override=entry_timeframe_override,
                setup_type=setup_type,
                action=action,
            ):
                pass
            else:
                score -= 12
                vetoes.append("low area without 1h/4h resistance retest; wait for higher-timeframe bounce before short")
        if str(h4_oi_valley.get("state") or "") == "CONFIRMED":
            vetoes.append(
                "4h OI valley confirmed; downside trend exhaustion blocks new short"
            )
        elif h4_oi_state in {
            "DELEVERAGE_BREAKDOWN",
            "DELEVERAGE_CROWD_BREAKDOWN",
            "DELEVERAGE_HOLD_LONG",
            "DELEVERAGE_CROWD_HOLD_LONG",
            "DELEVERAGE_WAIT",
            "DELEVERAGE_CROWD_WAIT",
        }:
            reasons.append(
                "4h OI sharp drop is not an OI valley and does not confirm a short entry"
            )
        rsi_score, rsi_reasons, rsi_vetoes = _rsi_entry_adjustment(
            PositionSide.SHORT,
            rsi14,
            trend_state,
            risk_state,
            score,
            pullback_direction,
            pullback_state,
            m15_precision,
            smart_money_phase,
        )
        score += rsi_score
        reasons.extend(rsi_reasons)
        vetoes.extend(rsi_vetoes)
    reasons.append(str(context.get("summary") or "multi-timeframe context neutral"))
    stage_probe = {
        **out,
        "action": action,
        "score": max(0, score),
        "reasons": tuple(reasons),
        "h1_trigger": h1,
        "h1_pullback": h1_pullback,
        "h1_ema_reliability": h1_ema_reliability,
        "high_distribution_handoff": high_distribution_handoff,
        "entry_timeframe_override": entry_timeframe_override,
        "distribution_short": distribution_short,
        "distribution_short_stage": distribution_short_stage,
        "h4_oi": h4_oi,
        "h4_oi_valley": h4_oi_valley,
        "large_wick_rejections": large_wick_rejections,
        "h1_structure": h1_structure,
        "h4_structure": h4,
        "h1_ma_cluster": h1_ma_cluster,
        "h4_ma_cluster": h4_ma_cluster,
        "m15_precision": m15_precision,
        "entry_guardrails": entry_guardrails,
    }
    if setup_type:
        stage_probe["setup_type"] = setup_type
    selected_entry_levels = _scored_entry_levels(stage_probe, entry_levels)
    if (
        action
        in {
            SignalAction.ENTRY_LONG.value,
            SignalAction.ENTRY_SHORT.value,
        }
        and entry_levels
        and not selected_entry_levels
    ):
        score = min(score, CORE_STRUCTURE_SCORE_CAP)
        reasons.append(
            "core entry structure absent; score capped below auto-entry threshold"
        )
        stage_probe["score"] = max(0, score)
        stage_probe["reasons"] = tuple(reasons)
    trend_stage_phase = _trend_stage_from_signal(stage_probe)
    if trend_stage_phase == TREND_STAGE_LATE and action in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}:
        out["leverage_cap"] = min(int(out.get("leverage_cap") or 99), 5)
        reasons.append("trend late stage; wait for a new pullback before fresh entry")
    final_score = max(0, score)
    out["score"] = final_score
    if final_score < AUTO_MAIN_POOL_MIN_SCORE:
        out["action"] = SignalAction.NO_TRADE.value
    elif final_score < AUTO_ENTRY_MIN_SCORE:
        out["action"] = SignalAction.WATCH.value
    elif action in {
        SignalAction.ENTRY_LONG.value,
        SignalAction.ENTRY_SHORT.value,
    }:
        out["action"] = action
    else:
        out["action"] = SignalAction.WATCH.value
    out["reasons"] = tuple(reasons)
    out["vetoes"] = tuple(vetoes)
    out["daily_bias"] = daily_bias
    out["h4_structure"] = h4
    out["h1_structure"] = h1_structure
    out["h4_ma_cluster"] = h4_ma_cluster
    out["h1_ma_cluster"] = h1_ma_cluster
    out["h4_oi"] = h4_oi
    out["h4_oi_valley"] = h4_oi_valley
    out["h1_trigger"] = h1
    out["h1_pullback"] = h1_pullback
    out["h1_ema_reliability"] = h1_ema_reliability
    out["high_distribution_handoff"] = high_distribution_handoff
    if entry_timeframe_override:
        out["entry_timeframe_override"] = entry_timeframe_override
    else:
        out.pop("entry_timeframe_override", None)
    out["distribution_short"] = distribution_short
    out["distribution_short_stage"] = distribution_short_stage
    out["m15_precision"] = m15_precision
    out["large_wick_rejections"] = large_wick_rejections
    out["entry_guardrails"] = entry_guardrails
    if setup_type:
        out["setup_type"] = setup_type
    else:
        out.pop("setup_type", None)
    out["entry_levels"] = selected_entry_levels
    out["trend_stage_phase"] = trend_stage_phase
    out.pop("entry_confirmation_required", None)
    out.pop("entry_zone_confirmation", None)
    _update_entry_position_fields(out)
    return out


def _rsi_entry_adjustment(
    side: PositionSide,
    rsi14: float | None,
    trend_state: str,
    risk_state: str,
    score: int,
    pullback_direction: str,
    pullback_state: str,
    m15_precision: dict[str, object],
    smart_money_phase: str = "",
) -> tuple[int, list[str], list[str]]:
    if rsi14 is None:
        return 0, [], []
    reasons: list[str] = []
    vetoes: list[str] = []
    if side == PositionSide.LONG:
        if trend_state == "ONE_WAY_UP":
            if rsi14 >= RSI_STRONG_LONG_HARD:
                return -18, [], ["one-way uptrend RSI above 92; skip fresh long and protect existing profit"]
            if rsi14 > RSI_STRONG_LONG_PULLBACK:
                pullback_ok = _strong_rsi_pullback_ok(
                    side,
                    risk_state,
                    score,
                    pullback_direction,
                    pullback_state,
                    m15_precision,
                    smart_money_phase,
                )
                if not pullback_ok:
                    return -12, [], ["one-way uptrend RSI hot without 1h/15m pullback; wait before long"]
                if rsi14 >= RSI_STRONG_LONG_SEVERE:
                    reasons.append("one-way uptrend RSI above 90, but pullback confirmed; use tight 15m structure stop")
                    return -6, reasons, vetoes
                reasons.append("one-way uptrend RSI hot, but 1h/15m pullback confirmed")
                return 0, reasons, vetoes
            return 0, reasons, vetoes
        if rsi14 > RSI_NORMAL_LONG_MAX:
            return -12, [], ["normal/chop trend RSI overheated; wait for 1h/4h pullback before long"]
        return 0, reasons, vetoes
    if trend_state == "ONE_WAY_DOWN":
        if rsi14 <= RSI_STRONG_SHORT_HARD:
            return -18, [], ["one-way downtrend RSI below 8; skip fresh short and protect existing profit"]
        if rsi14 < RSI_STRONG_SHORT_PULLBACK:
            pullback_ok = _strong_rsi_pullback_ok(
                side,
                risk_state,
                score,
                pullback_direction,
                pullback_state,
                m15_precision,
                smart_money_phase,
            )
            if not pullback_ok:
                return -12, [], ["one-way downtrend RSI cold without 1h/15m bounce rejection; wait before short"]
            if rsi14 <= RSI_STRONG_SHORT_SEVERE:
                reasons.append("one-way downtrend RSI below 10, but 1h/4h bounce rejection confirmed; use higher-timeframe structure stop")
                return -6, reasons, vetoes
            reasons.append("one-way downtrend RSI cold, but 1h/4h bounce rejection confirmed")
            return 0, reasons, vetoes
        return 0, reasons, vetoes
    if rsi14 < RSI_NORMAL_SHORT_MIN:
        return -12, [], ["normal/chop trend RSI oversold; wait for 1h/4h retest before short"]
    return 0, reasons, vetoes


def _strong_rsi_pullback_ok(
    side: PositionSide,
    risk_state: str,
    score: int,
    pullback_direction: str,
    pullback_state: str,
    m15_precision: dict[str, object],
    smart_money_phase: str = "",
) -> bool:
    if risk_state != "NORMAL":
        return False
    if side == PositionSide.LONG:
        return (
            pullback_direction == "LONG"
            and pullback_state == "HEALTHY_PULLBACK"
        ) or _strong_m15_pullback_allowed(
            side,
            "ONE_WAY_UP",
            risk_state,
            score,
            m15_precision,
            smart_money_phase,
        )
    return (
        pullback_direction == "SHORT"
        and pullback_state == "HEALTHY_PULLBACK"
    )


def _ma_cluster_signal_adjustment(
    side: PositionSide,
    h4_cluster: dict[str, object],
    h1_cluster: dict[str, object],
) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    vetoes: list[str] = []
    h4_state = str(h4_cluster.get("state") or "UNKNOWN")
    h1_state = str(h1_cluster.get("state") or "UNKNOWN")
    h4_price = _float_or_none(h4_cluster.get("price"))
    h1_price = _float_or_none(h1_cluster.get("price"))
    if side == PositionSide.LONG:
        if h4_state == "BREAKOUT_UP" or h1_state == "BREAKOUT_UP":
            score += 12
            reasons.append(_ma_cluster_reason("MA cluster breakout up", h1_price or h4_price))
        if h1_state == "RETEST_UP":
            score += 14
            reasons.append(_ma_cluster_reason("MA cluster retest held near MA20", h1_price or h4_price))
        if h4_state == "DENSE" and h1_state not in {"BREAKOUT_UP", "RETEST_UP"}:
            reasons.append(_ma_cluster_reason("MA cluster dense; wait for breakout or MA20 retest", h1_price or h4_price))
    else:
        if h4_state == "BREAKDOWN_DOWN" or h1_state == "BREAKDOWN_DOWN":
            score += 12
            reasons.append(_ma_cluster_reason("MA cluster breakdown down", h1_price or h4_price))
        if h1_state == "RETEST_DOWN":
            score += 14
            reasons.append(_ma_cluster_reason("MA cluster retest rejected near MA20", h1_price or h4_price))
        if h4_state == "DENSE" and h1_state not in {"BREAKDOWN_DOWN", "RETEST_DOWN"}:
            reasons.append(_ma_cluster_reason("MA cluster dense; wait for breakdown or MA20 retest", h1_price or h4_price))
    return score, reasons, vetoes


def _ma_cluster_reason(prefix: str, price: float | None) -> str:
    if price is None:
        return prefix
    return f"{prefix}: price={price:.6g}"


def _high_area_needs_long_pullback(h4_state: str, h1_direction: str, h1_state: str) -> bool:
    if h4_state not in {"BREAKOUT_UP", "BOX_UPPER_HALF"}:
        return False
    return not (h1_direction == "LONG" and h1_state in {"RETEST", "FAKE_BREAKDOWN"})


def _low_area_needs_short_pullback(h4_state: str, h1_direction: str, h1_state: str) -> bool:
    if h4_state not in {"BREAKDOWN_DOWN", "BOX_LOWER_HALF"}:
        return False
    return not (h1_direction == "SHORT" and h1_state in {"RETEST", "FAKE_BREAKOUT"})


def _strong_m15_pullback_allowed(
    side: PositionSide,
    trend_state: str,
    risk_state: str,
    score: int,
    precision: dict[str, object],
    smart_money_phase: str = "",
) -> bool:
    if side != PositionSide.LONG:
        return False
    if score < 85 or smart_money_phase != "SHORT_SQUEEZE_MARKUP":
        return False
    if risk_state not in {"NORMAL", "SHORT_CROWD"}:
        return False
    pullback = str(precision.get("pullback") or "")
    m15_trend = str(precision.get("trend") or "")
    return (
        trend_state == "ONE_WAY_UP"
        and m15_trend == "UP"
        and pullback == "M15_LONG_PULLBACK"
    )


def _precision_stop_allowed(
    side: PositionSide,
    trend_state: str,
    risk_state: str,
    score: int,
    precision: dict[str, object],
    smart_money_phase: str = "",
) -> bool:
    return _strong_m15_pullback_allowed(
        side,
        trend_state,
        risk_state,
        score,
        precision,
        smart_money_phase,
    )


def _daily_bias_margin_factor(side: str, signal: dict[str, object]) -> float:
    daily_bias = str(signal.get("daily_bias") or "NEUTRAL")
    factor = float(signal.get("margin_factor") or 1.0)
    if side == "LONG" and daily_bias == "BEAR":
        factor = min(factor, 0.5)
    if side == "SHORT" and daily_bias == "BULL":
        factor = min(factor, 0.5)
    return factor


def _preferred_exit_indicator(
    timeframe_indicators: dict[str, list[IndicatorSnapshot]],
    fallback: list[IndicatorSnapshot],
    position: Position | None = None,
) -> IndicatorSnapshot | None:
    preferred_timeframe = _exit_timeframe_from_position(position)
    if preferred_timeframe:
        indicators = timeframe_indicators.get(preferred_timeframe) or []
        if indicators:
            return indicators[-1]
        if preferred_timeframe == "15m" and fallback:
            return fallback[-1]
    for timeframe in ("1h", "4h"):
        indicators = timeframe_indicators.get(timeframe) or []
        if indicators:
            return indicators[-1]
    return fallback[-1] if fallback else None


def _exit_timeframe_from_position(position: Position | None) -> str | None:
    if position is None:
        return None
    context = position.metadata.get("entry_context")
    if not isinstance(context, dict):
        return None
    timeframe = str(context.get("stop_timeframe") or "").lower()
    if timeframe in {"15m", "1h", "4h", "1d"}:
        if timeframe == "15m" and position.side == PositionSide.SHORT:
            return "1h"
        return timeframe
    stop_basis = str(context.get("stop_basis") or "")
    if stop_basis == "15m_precision_structure":
        if position.side == PositionSide.SHORT:
            return "1h"
        return "15m"
    setup = str(context.get("entry_setup") or "")
    if setup.startswith("h4_"):
        return "4h"
    if setup.startswith("h1_") or setup == "kc_atr_volatility":
        return "1h"
    return None


def _refine_stop_with_precision(
    side: PositionSide,
    stop: float,
    precision: object,
    entry_price: float | None = None,
    indicator: IndicatorSnapshot | None = None,
) -> float:
    if not isinstance(precision, dict):
        return stop
    if side == PositionSide.LONG:
        anchor = _float_or_none(precision.get("long_stop_anchor"))
        if anchor is None:
            return stop
        if entry_price is not None:
            if anchor >= entry_price:
                return stop
            min_distance = _minimum_precision_stop_distance(entry_price, indicator)
            if entry_price - anchor < min_distance:
                return stop
            return max(stop, anchor)
        return anchor
    anchor = _float_or_none(precision.get("short_stop_anchor"))
    if anchor is None:
        return stop
    if entry_price is not None:
        if anchor <= entry_price:
            return stop
        min_distance = _minimum_precision_stop_distance(entry_price, indicator)
        if anchor - entry_price < min_distance:
            return stop
        return min(stop, anchor)
    return anchor


def _minimum_precision_stop_distance(entry_price: float, indicator: IndicatorSnapshot | None = None) -> float:
    distance = entry_price * MIN_PRECISION_STOP_PCT
    if indicator is not None and indicator.atr14:
        distance = max(distance, indicator.atr14 * MIN_PRECISION_STOP_ATR_MULTIPLE)
    return distance


def _refine_stop_with_setup_structure(
    side: PositionSide,
    stop: float,
    entry_price: float,
    signal: dict[str, object],
    context: dict[str, object],
    indicator: IndicatorSnapshot | None,
    *,
    timeframe: str = "1h",
) -> tuple[float, str]:
    setup_type = str(signal.get("setup_type") or "")
    stage = str(signal.get("distribution_short_stage") or "")
    structure_context = dict(context)
    for key in (
        "h1_structure",
        "h4_structure",
        "h1_trigger",
        "h1_pullback",
        "h1_ma_cluster",
        "h4_ma_cluster",
    ):
        if key in signal and key not in structure_context:
            structure_context[key] = signal[key]

    if side == PositionSide.SHORT and (
        stage in {DISTRIBUTION_STAGE_RANGE, DISTRIBUTION_STAGE_DESCENDING}
        or setup_type
        in {
            SETUP_DISTRIBUTION_STAGE1_SHORT,
            SETUP_DISTRIBUTION_STAGE2_SHORT,
        }
    ):
        staged_stop = _refine_stop_with_distribution_stage(
            side,
            stop,
            entry_price,
            signal,
            indicator,
        )
        if staged_stop != stop:
            return staged_stop, f"{stage or setup_type}_structure"

    preferred_timeframes = _setup_stop_timeframes(
        setup_type,
        timeframe,
        side,
    )
    for candidate_timeframe in preferred_timeframes:
        refined_stop = _refine_stop_with_retest_structure(
            side,
            stop,
            entry_price,
            structure_context,
            indicator,
            timeframe=candidate_timeframe,
        )
        if refined_stop != stop:
            return refined_stop, f"{candidate_timeframe}_structure"
    return stop, "volatility_fallback"


def _setup_stop_timeframes(
    setup_type: str,
    entry_timeframe: str,
    side: PositionSide,
) -> tuple[str, ...]:
    if setup_type in {SETUP_H4_PULLBACK_LONG, SETUP_H4_PULLBACK_SHORT, SETUP_OI_VALLEY_REVERSAL_LONG}:
        return ("4h", "1h")
    if setup_type in {
        SETUP_DISTRIBUTION_STAGE3_SHORT,
        SETUP_H1_PULLBACK_LONG,
        SETUP_H1_STRUCTURE_LONG,
        SETUP_H1_PULLBACK_SHORT,
        SETUP_H1_STRUCTURE_SHORT,
    }:
        return ("1h", "4h") if entry_timeframe != "4h" else ("4h", "1h")
    if setup_type in {
        SETUP_DISTRIBUTION_STAGE1_SHORT,
        SETUP_DISTRIBUTION_STAGE2_SHORT,
    }:
        return ("1h", "4h")
    if entry_timeframe == "4h":
        return ("4h", "1h")
    if side == PositionSide.SHORT:
        return ("1h", "4h")
    return ("1h", "4h")


def _refine_stop_with_distribution_stage(
    side: PositionSide,
    stop: float,
    entry_price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
) -> float:
    if side != PositionSide.SHORT:
        return stop
    stage = str(signal.get("distribution_short_stage") or "")
    key = (
        "distribution_range_high"
        if stage == DISTRIBUTION_STAGE_RANGE
        else "descending_high_trendline"
        if stage == DISTRIBUTION_STAGE_DESCENDING
        else ""
    )
    if not key:
        return stop
    levels = (
        signal.get("entry_levels")
        if isinstance(signal.get("entry_levels"), dict)
        else {}
    )
    short_levels = (
        levels.get("short")
        if isinstance(levels.get("short"), dict)
        else {}
    )
    zone = (
        short_levels.get(key)
        if isinstance(short_levels.get(key), dict)
        else {}
    )
    zone_high = _float_or_none(zone.get("high"))
    if zone_high is None:
        return stop
    minimum_distance = max(
        entry_price * 0.003,
        (indicator.atr14 * 0.35)
        if indicator is not None and indicator.atr14
        else 0.0,
    )
    structure_stop = max(
        zone_high,
        entry_price + minimum_distance,
    )
    if structure_stop <= entry_price:
        return stop
    return min(stop, structure_stop)


def _refine_stop_with_ma_cluster(side: PositionSide, stop: float, context: dict[str, object]) -> float:
    h1_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    cluster = h1_cluster if str(h1_cluster.get("state") or "UNKNOWN") != "UNKNOWN" else h4_cluster
    state = str(cluster.get("state") or "UNKNOWN")
    if state not in {"BREAKOUT_UP", "BREAKDOWN_DOWN", "RETEST_UP", "RETEST_DOWN"}:
        return stop
    lower = _float_or_none(cluster.get("lower"))
    upper = _float_or_none(cluster.get("upper"))
    ema20_value = _float_or_none(cluster.get("ema20"))
    width = abs((upper or 0.0) - (lower or 0.0))
    buffer = max(width * 0.3, (_float_or_none(cluster.get("price")) or stop) * 0.002)
    if side == PositionSide.LONG:
        if state == "BREAKOUT_UP" and lower is not None:
            return min(stop, lower - buffer)
        if state == "RETEST_UP" and ema20_value is not None:
            return min(stop, ema20_value - buffer)
    else:
        if state == "BREAKDOWN_DOWN" and upper is not None:
            return max(stop, upper + buffer)
        if state == "RETEST_DOWN" and ema20_value is not None:
            return max(stop, ema20_value + buffer)
    return stop


def _refine_stop_with_keltner(
    side: PositionSide,
    stop: float,
    entry: float,
    indicator: IndicatorSnapshot | None,
    trend_state: str,
) -> float:
    if indicator is None or indicator.atr14 is None or indicator.atr14 <= 0 or indicator.kc_mid is None:
        return stop
    buffer = indicator.atr14 * 0.35
    if side == PositionSide.LONG:
        candidates: list[float] = []
        if indicator.kc_lower is not None and indicator.kc_lower < entry:
            candidates.append(indicator.kc_lower - buffer)
        if trend_state in {"TREND_LONG", "ONE_WAY_UP"} and indicator.kc_mid < entry:
            candidates.append(indicator.kc_mid - buffer)
        min_distance = _minimum_precision_stop_distance(entry, indicator)
        valid = [candidate for candidate in candidates if candidate <= entry - min_distance]
        return max(valid) if valid else stop
    candidates: list[float] = []
    if indicator.kc_upper is not None and indicator.kc_upper > entry:
        candidates.append(indicator.kc_upper + buffer)
    if trend_state in {"TREND_SHORT", "ONE_WAY_DOWN"} and indicator.kc_mid > entry:
        candidates.append(indicator.kc_mid + buffer)
    min_distance = _minimum_precision_stop_distance(entry, indicator)
    valid = [candidate for candidate in candidates if candidate >= entry + min_distance]
    return min(valid) if valid else stop


def _refine_stop_with_retest_structure(
    side: PositionSide,
    stop: float,
    entry: float,
    context: dict[str, object],
    indicator: IndicatorSnapshot | None,
    *,
    timeframe: str = "1h",
) -> float:
    h1 = context.get("h1_structure") if isinstance(context.get("h1_structure"), dict) else {}
    h4 = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    h1_trigger = context.get("h1_trigger") if isinstance(context.get("h1_trigger"), dict) else {}
    h1_pullback = context.get("h1_pullback") if isinstance(context.get("h1_pullback"), dict) else {}
    h1_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    if timeframe == "4h":
        h1 = {}
        h1_cluster = {}
        relevant = bool(
            _short_structure_stop_candidates(entry, {}, h4, {}, h4_cluster)
            if side == PositionSide.SHORT
            else _long_structure_stop_candidates(entry, {}, h4, {}, h4_cluster)
        )
    else:
        h4 = {}
        h4_cluster = {}
        relevant = _retest_structure_stop_relevant(
            side,
            h1_trigger,
            h1_pullback,
            h1_cluster,
            {},
        )
    if not relevant:
        return stop
    buffer = _structure_stop_buffer(entry, indicator)
    min_distance = _minimum_precision_stop_distance(entry, indicator)
    if side == PositionSide.SHORT:
        candidates = _short_structure_stop_candidates(entry, h1, h4, h1_cluster, h4_cluster)
        if not candidates:
            return stop
        structure_stop = max(candidates) + buffer
        return max(structure_stop, entry + min_distance)
    candidates = _long_structure_stop_candidates(entry, h1, h4, h1_cluster, h4_cluster)
    if not candidates:
        return stop
    structure_stop = min(candidates) - buffer
    return min(structure_stop, entry - min_distance)


def _retest_structure_stop_relevant(
    side: PositionSide,
    h1: dict[str, object],
    h1_pullback: dict[str, object],
    h1_cluster: dict[str, object],
    h4_cluster: dict[str, object],
) -> bool:
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    cluster_states = {
        str(h1_cluster.get("state") or "UNKNOWN"),
        str(h4_cluster.get("state") or "UNKNOWN"),
    }
    if side == PositionSide.SHORT:
        return (
            (h1_direction == "SHORT" and h1_state in {"RETEST", "FAKE_BREAKOUT", "BREAKDOWN"})
            or (pullback_direction == "SHORT" and pullback_state == "HEALTHY_PULLBACK")
            or bool(cluster_states & {"RETEST_DOWN", "BREAKDOWN_DOWN"})
        )
    return (
        (h1_direction == "LONG" and h1_state in {"RETEST", "FAKE_BREAKDOWN", "BREAKOUT"})
        or (pullback_direction == "LONG" and pullback_state == "HEALTHY_PULLBACK")
        or bool(cluster_states & {"RETEST_UP", "BREAKOUT_UP"})
    )


def _structure_stop_buffer(entry: float, indicator: IndicatorSnapshot | None) -> float:
    atr_buffer = indicator.atr14 * 0.35 if indicator is not None and indicator.atr14 else 0.0
    return max(entry * 0.003, atr_buffer)


def _short_structure_stop_candidates(
    entry: float,
    h1: dict[str, object],
    h4: dict[str, object],
    h1_cluster: dict[str, object],
    h4_cluster: dict[str, object],
) -> list[float]:
    candidates = [
        _max_above(entry, h1.get("resistance_zone_high"), h1.get("resistance")),
        _max_above(entry, h4.get("resistance_zone_high"), h4.get("resistance")),
        _max_above(entry, h1_cluster.get("upper"), h1_cluster.get("ema60")),
        _max_above(entry, h4_cluster.get("upper"), h4_cluster.get("ema60")),
    ]
    return [value for value in candidates if value is not None and value > entry]


def _long_structure_stop_candidates(
    entry: float,
    h1: dict[str, object],
    h4: dict[str, object],
    h1_cluster: dict[str, object],
    h4_cluster: dict[str, object],
) -> list[float]:
    candidates = [
        _min_below(entry, h1.get("support_zone_low"), h1.get("support")),
        _min_below(entry, h4.get("support_zone_low"), h4.get("support")),
        _min_below(entry, h1_cluster.get("lower"), h1_cluster.get("ema60")),
        _min_below(entry, h4_cluster.get("lower"), h4_cluster.get("ema60")),
    ]
    return [value for value in candidates if value is not None and value < entry]


def _max_above(entry: float, *values: object) -> float | None:
    candidates = [_float_or_none(value) for value in values]
    above = [value for value in candidates if value is not None and value > entry]
    return max(above) if above else None


def _min_below(entry: float, *values: object) -> float | None:
    candidates = [_float_or_none(value) for value in values]
    below = [value for value in candidates if value is not None and value < entry]
    return min(below) if below else None


def _refine_take_profit_with_ma_cluster(
    side: PositionSide,
    entry: float,
    stop: float,
    take_profit_1: float,
    take_profit_2: float,
    context: dict[str, object],
    *,
    timeframe: str = "1h",
) -> tuple[float, float]:
    h1 = context.get("h1_structure") if isinstance(context.get("h1_structure"), dict) else {}
    h4 = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    h1_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    structure = h4 if timeframe == "4h" else h1
    cluster = h4_cluster if timeframe == "4h" else h1_cluster
    target = _cluster_target(side, cluster)
    horizontal = _float_or_none(structure.get("resistance" if side == PositionSide.LONG else "support"))
    candidates = [value for value in (target, horizontal) if _target_is_profitable(side, entry, value)]
    if not candidates:
        return take_profit_1, take_profit_2
    first_target = min(candidates) if side == PositionSide.LONG else max(candidates)
    risk_distance = abs(entry - stop)
    min_first = _take_profit_from_r(side, entry, stop, 0.8)
    if side == PositionSide.LONG:
        take_profit_1 = max(min_first, min(take_profit_1, first_target))
        take_profit_2 = max(take_profit_2, first_target + risk_distance * 0.8)
    else:
        take_profit_1 = min(min_first, max(take_profit_1, first_target))
        take_profit_2 = min(take_profit_2, first_target - risk_distance * 0.8)
    return take_profit_1, take_profit_2


def _cluster_target(side: PositionSide, cluster: dict[str, object]) -> float | None:
    return _float_or_none(cluster.get("target_up" if side == PositionSide.LONG else "target_down"))


def _target_is_profitable(side: PositionSide, entry: float, target: float | None) -> bool:
    if target is None:
        return False
    return target > entry if side == PositionSide.LONG else target < entry


def _multi_timeframe_summary(
    daily_bias: str,
    h4: dict[str, object],
    h1: dict[str, object],
    pullback: dict[str, object],
    h4_oi: dict[str, object],
    h4_ma_cluster: dict[str, object],
    h1_ma_cluster: dict[str, object],
) -> str:
    return (
        f"MTF: 1d={daily_bias}; 4h={h4.get('state', 'UNKNOWN')}; "
        f"1h={h1.get('state', 'UNKNOWN')}/{h1.get('direction', 'NONE')}; "
        f"pullback={pullback.get('state', 'UNKNOWN')}/{pullback.get('direction', 'NONE')}; "
        f"oi4h={h4_oi.get('state', 'UNKNOWN')}; "
        f"ma4h={h4_ma_cluster.get('state', 'UNKNOWN')}@{_summary_price(h4_ma_cluster.get('price'))}; "
        f"ma1h={h1_ma_cluster.get('state', 'UNKNOWN')}@{_summary_price(h1_ma_cluster.get('price'))}"
    )


def _summary_price(value: object) -> str:
    price = _float_or_none(value)
    return "--" if price is None else f"{price:.6g}"


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _near_level(price: float, level: float | None, distance: float) -> bool:
    if level is None or distance <= 0:
        return False
    return abs(price - level) <= distance


def _margin_for_signal(
    score: int,
    equity: float,
    remaining_total_margin: float | None = None,
    slots_remaining: int = 1,
) -> float:
    if score >= 85:
        cap = min(280.0, equity * 0.28)
        floor = min(180.0, equity * 0.18)
    elif score >= 78:
        cap = min(230.0, equity * 0.23)
        floor = min(150.0, equity * 0.15)
    else:
        cap = min(180.0, equity * 0.18)
        floor = min(100.0, equity * 0.10)
    if remaining_total_margin is None:
        return cap
    target = remaining_total_margin / max(slots_remaining, 1)
    return min(cap, max(floor, target))


def _entry_stop_pct(entry_price: float, stop_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return abs(entry_price - stop_price) / entry_price


def _entry_quality_grade(
    *,
    score: int,
    reward_r: float,
    stop_pct: float,
    setup_type: str,
) -> str:
    setup_is_core = bool(setup_type) and setup_type != SETUP_DISTRIBUTION_STAGE1_SHORT
    if (
        setup_is_core
        and score >= 95
        and reward_r >= 1.5
        and stop_pct <= 0.05
    ):
        return ENTRY_QUALITY_S
    if (
        setup_is_core
        and score >= 88
        and reward_r >= MIN_ENTRY_REWARD_R
        and stop_pct <= ENTRY_QUALITY_STOP_MAX_10X
    ):
        return ENTRY_QUALITY_A
    return ENTRY_QUALITY_B


def _max_safe_leverage_for_stop(
    stop_pct: float,
    leverage_max: int,
) -> int:
    effective_max = max(5, min(int(leverage_max or 10), 10))
    for leverage, max_stop_pct in (
        (10, ENTRY_QUALITY_STOP_MAX_10X),
        (7, ENTRY_QUALITY_STOP_MAX_7X),
        (5, ENTRY_QUALITY_STOP_MAX_5X),
    ):
        if leverage <= effective_max and stop_pct <= max_stop_pct:
            return leverage
    return 0


def _leverage_for_entry_quality(
    quality: str,
    *,
    stop_pct: float,
    leverage_max: int,
    leverage_cap: int | None = None,
) -> int:
    capped_max = leverage_max
    if leverage_cap is not None:
        capped_max = min(capped_max, max(5, int(leverage_cap)))
    safe_leverage = _max_safe_leverage_for_stop(stop_pct, capped_max)
    if safe_leverage <= 0:
        return 0
    if quality in {ENTRY_QUALITY_S, ENTRY_QUALITY_A}:
        return min(10, safe_leverage)
    return safe_leverage


def _entry_quality_units(quality: str) -> int:
    return 2 if quality == ENTRY_QUALITY_S else 1


def _quality_sized_margin(
    *,
    equity: float,
    available: float,
    remaining_total_margin: float,
    quality: str,
    margin_factor: float = 1.0,
) -> float:
    if equity <= 0:
        return 0.0
    unit_margin = equity * ENTRY_MARGIN_DEPLOYMENT_CAP / ENTRY_MARGIN_BASE_SLOTS
    target = unit_margin * _entry_quality_units(quality)
    target *= max(0.0, min(float(margin_factor), 1.0))
    return min(
        target,
        max(available, 0.0),
        max(remaining_total_margin, 0.0),
    )


def _current_open_risk_usdt(positions: Iterable[Position]) -> float:
    total = 0.0
    for position in positions:
        if position.side == PositionSide.LONG:
            loss_distance = max(
                position.entry_price - position.stop_price,
                0.0,
            )
        else:
            loss_distance = max(
                position.stop_price - position.entry_price,
                0.0,
            )
        total += (
            loss_distance
            * position.quantity
            * max(position.remaining_fraction, 0.0)
        )
    return total


def _risk_sized_margin(
    *,
    equity: float,
    available: float,
    remaining_total_margin: float,
    current_open_risk_usdt: float,
    risk_per_trade: float,
    risk_factor: float,
    entry_price: float,
    stop_price: float,
    leverage: int,
) -> tuple[float, float]:
    stop_pct = (
        abs(entry_price - stop_price) / entry_price
        if entry_price > 0
        else 0.0
    )
    if stop_pct <= 0 or leverage <= 0 or equity <= 0:
        return 0.0, 0.0
    remaining_risk = max(
        equity * TOTAL_OPEN_RISK_CAP - current_open_risk_usdt,
        0.0,
    )
    risk_budget = min(
        equity * max(risk_per_trade, 0.0) * max(risk_factor, 0.0),
        remaining_risk,
    )
    if risk_budget <= 0:
        return 0.0, 0.0
    notional = risk_budget / stop_pct
    margin = notional / leverage
    margin = min(
        margin,
        equity * 0.28,
        max(available, 0.0),
        max(remaining_total_margin, 0.0),
    )
    planned_risk = margin * leverage * stop_pct
    return margin, planned_risk


def _default_stop(side: PositionSide, price: float, leverage: int) -> float:
    stop_pct = _stop_pct_for_leverage(leverage)
    return _stop_from_pct(side, price, stop_pct)


def _default_take_profit(side: PositionSide, price: float, stop: float, multiple: int) -> float:
    distance = abs(price - stop)
    return _take_profit_from_r(side, price, stop, 1.5 if multiple == 1 else 3.0)


def _stop_from_pct(side: PositionSide, price: float, stop_pct: float) -> float:
    if side == PositionSide.LONG:
        return price * (1 - stop_pct)
    return price * (1 + stop_pct)


def _stop_from_atr(side: PositionSide, price: float, atr: float, multiple: float) -> float:
    if side == PositionSide.LONG:
        return price - atr * multiple
    return price + atr * multiple


def _take_profit_from_r(side: PositionSide, price: float, stop: float, r_multiple: float) -> float:
    distance = abs(price - stop)
    if side == PositionSide.LONG:
        return price + distance * r_multiple
    return price - distance * r_multiple


def _stop_pct_for_leverage(leverage: int) -> float:
    if leverage >= 10:
        return 0.01
    if leverage >= 7:
        return 0.015
    return 0.02


def _leverage_for_signal(
    score: int,
    leverage_max: int,
    trend_state: str = "CHOP",
    indicator: IndicatorSnapshot | None = None,
    trend_stage: str | None = None,
) -> int:
    score_leverage = 5
    if score >= 95 and trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}:
        score_leverage = 10
    elif score >= 85:
        score_leverage = 7
    stage_leverage = 10
    if trend_stage == TREND_STAGE_LATE:
        stage_leverage = 5
    elif trend_stage == TREND_STAGE_NEUTRAL and trend_state not in {"ONE_WAY_UP", "ONE_WAY_DOWN"}:
        stage_leverage = 7
    volatility_leverage = 10
    atr_pct = _indicator_atr_pct(indicator)
    if atr_pct >= 0.04:
        volatility_leverage = 5
    elif atr_pct >= 0.03:
        volatility_leverage = 7
    leverage = min(score_leverage, stage_leverage, volatility_leverage)
    if trend_state == "CHOP":
        leverage = min(leverage, 5)
    return min(leverage, leverage_max)


def _is_high_volatility(indicator: IndicatorSnapshot | None) -> bool:
    return _indicator_atr_pct(indicator) >= 0.04


def _indicator_atr_pct(indicator: IndicatorSnapshot | None) -> float:
    if indicator is None or not indicator.close or not indicator.atr14:
        return 0.0
    return indicator.atr14 / indicator.close


def _entry_signal_timeframe(configured_interval: str) -> str:
    return configured_interval if configured_interval in {"1h", "4h"} else "1h"


def _bars_for_4h(interval: str) -> int:
    normalized = interval.strip().lower()
    if normalized.endswith("m"):
        minutes = int(normalized[:-1] or "15")
        return max(1, round(240 / minutes))
    if normalized.endswith("h"):
        hours = int(normalized[:-1] or "1")
        return max(1, round(4 / hours))
    return 1


def _fill_payload(fill: PaperFill) -> dict[str, object]:
    payload = asdict(fill)
    payload["timestamp"] = fill.timestamp.isoformat()
    payload["side"] = fill.side.value
    payload["opened_at"] = fill.opened_at.isoformat()
    payload["closed_at"] = fill.closed_at.isoformat() if fill.closed_at else None
    return payload


CN_TZ = timezone(timedelta(hours=8))


def _trading_day_key(timestamp: datetime) -> str:
    local = timestamp.astimezone(CN_TZ)
    if local.hour < 8:
        local -= timedelta(days=1)
    return local.date().isoformat()


def _next_trading_day_key(day: str) -> str:
    return (datetime.fromisoformat(day).date() + timedelta(days=1)).isoformat()


def _daily_pnl_payload(baselines: dict[str, float], total_pnl: float, now: datetime | None = None) -> dict[str, object]:
    current_time = now or datetime.now(UTC)
    today_key = _trading_day_key(current_time)
    if today_key not in baselines:
        baselines[today_key] = total_pnl
    today_baseline = baselines[today_key]

    days: list[dict[str, object]] = []
    for day in sorted(baselines):
        next_day = _next_trading_day_key(day)
        end_pnl = baselines.get(next_day, baselines[day])
        net_pnl = end_pnl - baselines[day]
        days.append(
            {
                "date": day,
                "profit": max(net_pnl, 0.0),
                "loss": min(net_pnl, 0.0),
                "net_pnl": net_pnl,
                "fees": 0.0,
                "closed_trades": 0,
                "open_orders": 0,
            }
        )

    total_profit = sum(float(day["profit"]) for day in days)
    total_loss = sum(float(day["loss"]) for day in days)
    total_net = sum(float(day["net_pnl"]) for day in days)
    return {
        "timezone": "Asia/Shanghai",
        "day_start_hour": 8,
        "today": today_key,
        "today_baseline": today_baseline,
        "today_pnl": total_pnl - today_baseline,
        "days": days,
        "summary": {
            "profit": total_profit,
            "loss": total_loss,
            "net_pnl": total_net,
            "fees": 0.0,
        },
    }


def _quarter_hour_start(timestamp: datetime) -> datetime:
    ts = timestamp.astimezone(UTC)
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


def _pnl_history_payload(
    samples: dict[str, float],
    total_pnl: float,
    now: datetime | None = None,
) -> list[dict[str, object]]:
    current_time = now or datetime.now(UTC)
    sample_time = _quarter_hour_start(current_time)
    key = sample_time.isoformat()
    if key not in samples:
        samples[key] = total_pnl

    return [
        {
            "timestamp": timestamp,
            "total_pnl": samples[timestamp],
        }
        for timestamp in sorted(samples)
    ]

