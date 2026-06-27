from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, dataclass, field
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from typing import Literal

from ai_trading.binance import BinanceFuturesMarketData
from ai_trading.config import AppSettings
from ai_trading.indicators import build_indicators, ema, sma
from ai_trading.models import Candle, IndicatorSnapshot, Position, PositionSide, SignalAction, Trade
from ai_trading.strategy import CompositeStrategy


PaperSide = Literal["LONG", "SHORT"]
AUTO_UNIVERSE_EXCLUDED_SYMBOLS = {"BTCUSDT", "BTCUSDC", "ETHUSDT", "SOLUSDT", "XAUUSDT"}
CANDIDATE_MIN_QUOTE_VOLUME = 50_000_000
CANDIDATE_MAX_ABS_24H_CHANGE = 60.0
CANDIDATE_HIGH_CHANGE_THRESHOLD = 40.0
CANDIDATE_MAX_UPPER_WICK_RATIO = 0.45
CANDIDATE_MAX_RETRACE_FROM_HIGH_PCT = 22.0
CANDIDATE_MAX_CENTER_OFFSET_PCT = 24.0
BTC_EXTREME_4H_AMPLITUDE = 0.08
ROTATION_MIN_SCORE_GAP = 20
ROTATION_MIN_ATR_PCT = 0.008
ROTATION_MIN_VOLUME_RATIO = 1.2
PYRAMID_MIN_SCORE = 85
PYRAMID_MARGIN_FRACTION = 0.35
PYRAMID_MAX_ADDS = 1
PAPER_DEFAULT_BALANCE = 1200.0
INITIAL_ENTRY_MARGIN_CAP = 0.78
PYRAMID_TOTAL_MARGIN_CAP = 0.95
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
MIN_ENTRY_REWARD_R = 1.2
PROFIT_LOCK_SLIPPAGE_PCT = 0.0008
MIN_PRECISION_STOP_PCT = 0.012
MIN_PRECISION_STOP_ATR_MULTIPLE = 0.65


def _candidate_prefilter_allowed(item: object) -> bool:
    quote_volume = float(getattr(item, "quote_volume", 0.0) or 0.0)
    if quote_volume < CANDIDATE_MIN_QUOTE_VOLUME:
        return False
    change = _float_or_none(getattr(item, "price_change_percent", None))
    last_price = _float_or_none(getattr(item, "last_price", None))
    high_price = _float_or_none(getattr(item, "high_price", None))
    low_price = _float_or_none(getattr(item, "low_price", None))
    open_price = _float_or_none(getattr(item, "open_price", None))
    if change is not None and abs(change) > CANDIDATE_MAX_ABS_24H_CHANGE:
        return False
    if last_price and high_price and low_price and high_price > low_price:
        center = (high_price + low_price) / 2
        center_offset = abs(last_price - center) / last_price * 100
        if change is not None and abs(change) > 35 and center_offset > CANDIDATE_MAX_CENTER_OFFSET_PCT:
            return False
        if change is not None and change > CANDIDATE_HIGH_CHANGE_THRESHOLD and open_price:
            range_size = high_price - low_price
            upper_wick_ratio = max(high_price - max(open_price, last_price), 0.0) / range_size
            retrace_from_high_pct = (high_price - last_price) / last_price * 100
            if upper_wick_ratio >= CANDIDATE_MAX_UPPER_WICK_RATIO:
                return False
            if retrace_from_high_pct >= CANDIDATE_MAX_RETRACE_FROM_HIGH_PCT:
                return False
    return True


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
        interval: str = "15m",
        market_data: BinanceFuturesMarketData | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.state_path = Path(state_path) if state_path else None
        self.account = _load_paper_account(self.state_path) or PaperAccount(starting_balance=starting_balance, wallet_balance=starting_balance)
        self.symbols = [symbol.upper() for symbol in (symbols or ["AUTO_TOP30"])]
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
        self._poll_seconds = 20
        self._lock = asyncio.Lock()

    async def start(self, *, auto_trade: bool = False, poll_seconds: int = 20) -> None:
        async with self._lock:
            self.auto_trade = auto_trade
            self._poll_seconds = max(poll_seconds, 5)
            if self.running:
                return
            self.running = True
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
            self._task = None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def reset(self, starting_balance: float = PAPER_DEFAULT_BALANCE) -> None:
        await self.stop()
        async with self._lock:
            self.account = PaperAccount(starting_balance=starting_balance, wallet_balance=starting_balance)
            self.latest_prices.clear()
            self.latest_signals.clear()
            self.latest_indicators.clear()
            self.latest_timeframe_contexts.clear()
            self.latest_timeframe_indicators.clear()
            self.last_error = None
            self.last_market_update_at = None
            self.auto_trade = False
            self._save_state_unlocked()

    async def close(self) -> None:
        await self.shutdown()
        close = getattr(self.market_data, "aclose", None)
        if close is not None:
            await close()

    async def refresh_once(self) -> None:
        await self.refresh_universe_if_needed()
        semaphore = asyncio.Semaphore(6)
        errors: list[str] = []
        refreshed = 0

        async def refresh_with_limit(symbol: str) -> None:
            nonlocal refreshed
            async with semaphore:
                try:
                    if await self._refresh_symbol(symbol):
                        refreshed += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{symbol}: {exc}")

        await asyncio.gather(*(refresh_with_limit(symbol) for symbol in self.symbols))
        if refreshed:
            self.last_market_update_at = datetime.now(UTC)
            self.last_error = _format_partial_market_errors(errors) if errors else None
            self._save_state_unlocked()
        elif errors:
            self.last_error = f"行情刷新失败，等待网络恢复：{'; '.join(errors[:3])}"

    async def refresh_open_position_prices(self) -> int:
        """Refresh persisted positions without starting the strategy loop."""
        symbols = list(self.account.positions)
        if not symbols:
            return 0

        semaphore = asyncio.Semaphore(5)

        async def fetch(symbol: str) -> tuple[str, float] | None:
            async with semaphore:
                try:
                    candles = await self.market_data.klines(symbol, self.interval, limit=2)
                except Exception:  # noqa: BLE001 - keep the persisted mark when the network is unavailable
                    return None
                if not candles:
                    return None
                return symbol, float(candles[-1].close)

        results = await asyncio.gather(*(fetch(symbol) for symbol in symbols))
        refreshed = [result for result in results if result is not None]
        if not refreshed:
            return 0

        async with self._lock:
            for symbol, price in refreshed:
                self._remember_mark_price(symbol, price)
            self.last_market_update_at = datetime.now(UTC)
            self._save_state_unlocked()
        return len(refreshed)

    async def refresh_universe_if_needed(self) -> None:
        if self.symbols and self.symbols != ["AUTO_TOP30"]:
            return
        top_symbols = await self.market_data.top_usdt_perpetuals(limit=80)
        eligible = [item for item in top_symbols if item.symbol.upper() not in AUTO_UNIVERSE_EXCLUDED_SYMBOLS]
        filtered = [item.symbol for item in eligible if _candidate_prefilter_allowed(item)]
        if len(filtered) < 30:
            seen = set(filtered)
            for item in eligible:
                if item.symbol not in seen:
                    filtered.append(item.symbol)
                    seen.add(item.symbol)
                if len(filtered) >= 30:
                    break
        self.symbols = filtered[:30] or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

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

    def _status_unlocked(self) -> dict[str, object]:
        positions = []
        unrealized = 0.0
        used_margin = 0.0
        latest_prices = dict(self.latest_prices)
        latest_signals = {symbol: dict(signal) for symbol, signal in self.latest_signals.items()}
        latest_timeframe_contexts = {symbol: dict(context) for symbol, context in self.latest_timeframe_contexts.items()}
        positions_snapshot = list(self.account.positions.values())
        fills_snapshot = list(self.account.fills[-100:])
        for position in positions_snapshot:
            price = latest_prices.get(position.symbol) or _stored_mark_price(position) or position.entry_price
            pnl = _pnl(position.side, position.entry_price, price, position.quantity)
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
                    "quantity": position.quantity,
                    "notional": price * position.quantity,
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
        while self.running:
            try:
                await self.refresh_once()
                self._manage_open_positions()
                if self.auto_trade:
                    await self._auto_trade_once()
            except Exception as exc:  # noqa: BLE001 - user-facing paper loop should keep running
                self.last_error = str(exc)
            self._record_account_snapshots()
            await asyncio.sleep(self._poll_seconds)

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
        candles, derivatives = await self.market_data.historical_bundle(symbol, self.interval, limit=history_limit)
        if not candles:
            return False
        price = candles[-1].close
        self._remember_mark_price(symbol, price)
        indicators = build_indicators(
            candles,
            derivatives,
            ema_fast=self.settings.strategy.ema_fast,
            ema_slow=self.settings.strategy.ema_slow,
            ma_trend=self.settings.strategy.ma_trend,
            bollinger_window=self.settings.strategy.bollinger_window,
            bollinger_stddev=self.settings.strategy.bollinger_stddev,
            rsi_window=self.settings.strategy.rsi_window,
            atr_window=self.settings.strategy.atr_window,
            volume_window=self.settings.strategy.volume_window,
            keltner_window=self.settings.strategy.keltner_window,
            keltner_atr_multiplier=self.settings.strategy.keltner_atr_multiplier,
            qps_window=self.settings.strategy.qps_window,
        )
        self.latest_indicators[symbol] = indicators
        timeframe_indicators, mtf_context = await self._multi_timeframe_context(symbol, candles, indicators)
        self.latest_timeframe_indicators[symbol] = timeframe_indicators
        self.latest_timeframe_contexts[symbol] = mtf_context
        strategy = CompositeStrategy(replace(self.settings.strategy, smart_money_window=_bars_for_4h(self.interval)))
        signal = strategy.generate_signal(symbol, candles, indicators)
        cycle = strategy.smart_money_cycle(candles, indicators)
        signal_side = "LONG" if signal.action == SignalAction.ENTRY_LONG else "SHORT" if signal.action == SignalAction.ENTRY_SHORT else None
        trend_state = strategy.trend_state(signal_side, indicators)
        risk_state = strategy.risk_state(indicators)
        current = indicators[-1] if indicators else None
        payload = {
            "timestamp": signal.timestamp.isoformat(),
            "action": signal.action.value,
            "regime": signal.regime.value,
            "trend_state": trend_state,
            "risk_state": risk_state,
            "rsi14": current.rsi14 if current else None,
            "volume_ratio": current.volume_ratio if current else None,
            "oi_change": current.oi_change if current else None,
            "long_short_ratio": current.long_short_ratio if current else None,
            "funding_rate": current.funding_rate if current else None,
            "price": current.close if current else price,
            "score": signal.score,
            "reasons": signal.reasons,
            "vetoes": signal.vetoes,
            "smart_money_phase": cycle.phase,
        }
        self.latest_signals[symbol] = _apply_multi_timeframe_context(payload, mtf_context)
        self.account.latest_signals[symbol] = dict(self.latest_signals[symbol])
        return True

    async def _multi_timeframe_context(
        self,
        symbol: str,
        base_candles: list[Candle],
        base_indicators: list[IndicatorSnapshot],
    ) -> tuple[dict[str, list[IndicatorSnapshot]], dict[str, object]]:
        timeframe_candles: dict[str, list[Candle]] = {}
        timeframe_indicators: dict[str, list[IndicatorSnapshot]] = {}
        timeframes = ("15m", "1h", "4h", "1d")

        async def fetch_timeframe(timeframe: str) -> tuple[str, list[Candle], list[IndicatorSnapshot]] | Exception:
            if timeframe == self.interval:
                return timeframe, base_candles, base_indicators
            if timeframe == "4h":
                candles, derivatives = await self.market_data.historical_bundle(symbol, timeframe, limit=max(240, self.settings.strategy.ma_trend + 40))
                return timeframe, candles, build_indicators(
                    candles,
                    derivatives,
                    ema_fast=self.settings.strategy.ema_fast,
                    ema_slow=self.settings.strategy.ema_slow,
                    ma_trend=self.settings.strategy.ma_trend,
                    bollinger_window=self.settings.strategy.bollinger_window,
                    bollinger_stddev=self.settings.strategy.bollinger_stddev,
                    rsi_window=self.settings.strategy.rsi_window,
                    atr_window=self.settings.strategy.atr_window,
                    volume_window=self.settings.strategy.volume_window,
                    keltner_window=self.settings.strategy.keltner_window,
                    keltner_atr_multiplier=self.settings.strategy.keltner_atr_multiplier,
                    qps_window=self.settings.strategy.qps_window,
                )
            candles = await self.market_data.klines(symbol, timeframe, limit=max(240, self.settings.strategy.ma_trend + 40))
            return timeframe, candles, _build_price_only_indicators(candles, self.settings)

        results = await asyncio.gather(*(fetch_timeframe(timeframe) for timeframe in timeframes), return_exceptions=True)
        previous_indicators = self.latest_timeframe_indicators.get(symbol, {})
        for result in results:
            if isinstance(result, Exception):
                continue
            timeframe, candles, indicators = result
            timeframe_candles[timeframe] = candles
            timeframe_indicators[timeframe] = indicators
        for timeframe in timeframes:
            if timeframe not in timeframe_indicators and timeframe in previous_indicators:
                timeframe_indicators[timeframe] = previous_indicators[timeframe]
        if not timeframe_candles:
            return previous_indicators, self.latest_timeframe_contexts.get(symbol, {})
        return timeframe_indicators, _build_multi_timeframe_context(timeframe_candles, timeframe_indicators, self.settings)

    async def _price(self, symbol: str) -> float:
        symbol = symbol.upper()
        if symbol not in self.latest_prices:
            candles = await self.market_data.klines(symbol, self.interval, limit=2)
            if not candles:
                raise ValueError(f"cannot fetch price for {symbol}")
            self._remember_mark_price(symbol, float(candles[-1].close))
        return self.latest_prices[symbol]

    def _remember_mark_price(self, symbol: str, price: float) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        symbol = symbol.upper()
        self.latest_prices[symbol] = price
        position = self.account.positions.get(symbol)
        if position is not None:
            position.metadata["last_mark_price"] = price

    async def _auto_trade_once(self) -> None:
        if await self._btc_4h_extreme_volatility():
            self.last_error = "BTC 4h extreme volatility; pause new altcoin entries"
            return
        max_positions = min(self.settings.risk.max_open_positions, 5)
        candidates = [
            (symbol, signal)
            for symbol, signal in self.latest_signals.items()
            if symbol not in self.account.positions
            and symbol.upper() not in AUTO_UNIVERSE_EXCLUDED_SYMBOLS
            and signal.get("action") in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}
            and _auto_signal_allowed(signal)
        ]
        candidates.sort(key=lambda item: int(item[1].get("score") or 0), reverse=True)
        if len(self.account.positions) >= max_positions:
            await self._rebalance_for_better_candidate(candidates)
        slots = max_positions - len(self.account.positions)
        if slots <= 0:
            return
        for index, (symbol, signal) in enumerate(candidates[:slots]):
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
            slots_remaining = max(slots - index, 1)
            margin = _margin_for_signal(score, equity, remaining_total_margin, slots_remaining)
            margin *= _daily_bias_margin_factor(side, signal)
            margin = min(margin, available, remaining_total_margin)
            if margin >= 20:
                indicators = self.latest_indicators.get(symbol, [])
                precision = self.latest_timeframe_contexts.get(symbol, {}).get("m15_precision", {})
                mtf_context = self.latest_timeframe_contexts.get(symbol, {})
                trend_state = str(signal.get("trend_state") or "CHOP")
                preferred_indicator = _preferred_exit_indicator(self.latest_timeframe_indicators.get(symbol, {}), indicators)
                trend_stage = str(signal.get("trend_stage_phase") or _trend_stage_from_signal(signal, preferred_indicator))
                leverage = min(
                    _leverage_for_signal(
                        score,
                        self.settings.risk.leverage_max,
                        trend_state,
                        preferred_indicator,
                        trend_stage,
                    ),
                    int(signal.get("leverage_cap") or self.settings.risk.leverage_max),
                )
                entry_price = self.latest_prices.get(symbol) or await self._price(symbol)
                stop_loss, take_profit_1, take_profit_2 = _adaptive_exits(
                    PositionSide(side),
                    entry_price,
                    leverage,
                    trend_state,
                    preferred_indicator,
                )
                stop_basis = "volatility_structure"
                kc_stop = _refine_stop_with_keltner(
                    PositionSide(side),
                    stop_loss,
                    entry_price,
                    preferred_indicator,
                    trend_state,
                )
                if kc_stop != stop_loss:
                    stop_loss = kc_stop
                    stop_basis = "kc_atr_volatility"
                stop_loss = _refine_stop_with_ma_cluster(PositionSide(side), stop_loss, mtf_context)
                structure_stop = _refine_stop_with_retest_structure(
                    PositionSide(side),
                    stop_loss,
                    entry_price,
                    mtf_context,
                    preferred_indicator,
                )
                if structure_stop != stop_loss:
                    stop_loss = structure_stop
                    stop_basis = "1h_4h_retest_structure"
                precision_entry_only = (
                    _entry_level_group_hit(signal, entry_price, PositionSide(side), include_m15=True)
                    and not _entry_level_group_hit(signal, entry_price, PositionSide(side), include_m15=False)
                )
                if _precision_stop_allowed(PositionSide(side), trend_state, str(signal.get("risk_state") or "NORMAL"), score, precision):
                    refined_stop = _refine_stop_with_precision(
                        PositionSide(side),
                        stop_loss,
                        precision,
                        entry_price,
                        preferred_indicator,
                    )
                    if refined_stop != stop_loss:
                        stop_loss = refined_stop
                        stop_basis = "15m_precision_structure"
                    elif precision_entry_only:
                        signal["entry_timing"] = ENTRY_TIMING_WAIT
                        signal["entry_timing_reason"] = "entry timing wait: 15m tactical entry needs a valid 15m structure stop or 1h/4h entry zone"
                        continue
                take_profit_1, take_profit_2 = _refine_take_profit_with_ma_cluster(
                    PositionSide(side),
                    entry_price,
                    stop_loss,
                    take_profit_1,
                    take_profit_2,
                    mtf_context,
                )
                reward_r = _entry_reward_r(signal, PositionSide(side), entry_price, stop_loss)
                if reward_r is not None and reward_r < MIN_ENTRY_REWARD_R:
                    signal["entry_timing"] = ENTRY_TIMING_BLOCK
                    signal["entry_timing_reason"] = "entry timing blocked: reward/risk too low"
                    continue
                entry_context = _entry_context_from_signal(signal)
                entry_context["stop_basis"] = stop_basis
                entry_context["trend_stage_phase"] = trend_stage
                entry_context["entry_reward_r"] = reward_r
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
        self._add_to_strong_positions()

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
        try:
            candles = await self.market_data.klines("BTCUSDT", "4h", limit=2)
        except Exception:  # noqa: BLE001 - do not block trading on missing public filter data
            return False
        if not candles:
            return False
        candle = candles[-1]
        amplitude = (candle.high - candle.low) / candle.open if candle.open else 0.0
        body_move = abs(candle.close - candle.open) / candle.open if candle.open else 0.0
        return amplitude >= BTC_EXTREME_4H_AMPLITUDE or body_move >= BTC_EXTREME_4H_AMPLITUDE * 0.75

    def _manage_open_positions(self) -> None:
        for position in list(self.account.positions.values()):
            price = self.latest_prices.get(position.symbol)
            if price is None:
                continue
            indicators = self.latest_indicators.get(position.symbol, [])
            tf_indicators = self.latest_timeframe_indicators.get(position.symbol, {})
            trend_state = self.strategy.trend_state(position.side.value, indicators) if indicators else "CHOP"
            risk_state = self.strategy.risk_state(indicators) if indicators else "NORMAL"
            strong_trend = trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}
            signal = self.latest_signals.get(position.symbol, {})
            exit_indicator = _preferred_exit_indicator(tf_indicators, indicators, position)
            _update_position_excursions(position, price)
            short_support_hold = (
                _short_near_h4_support(position, price, signal, exit_indicator)
                and _short_support_trend_should_hold(position, price, signal, exit_indicator, trend_state)
            )
            if short_support_hold:
                _tighten_short_support_stop(position, price, signal, exit_indicator)
            if strong_trend:
                _protect_confirmed_breakout_position(position, price, signal, exit_indicator)
            _apply_profit_protection(position, price, signal, exit_indicator, self.settings.execution.taker_fee_rate)
            self._update_trailing_stop(position, price, strong_trend, exit_indicator)
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
            drawdown_exit_reason = None if short_support_hold else _profit_drawdown_exit_reason(position, price, signal, exit_indicator)
            if drawdown_exit_reason:
                self._close_position_unlocked(position, price, drawdown_exit_reason)
                continue
            structure_take_profit_reason = _structure_take_profit_reason(
                position,
                price,
                signal,
                exit_indicator,
                trend_state,
                self.settings.execution.taker_fee_rate,
            )
            if structure_take_profit_reason:
                self._close_position_unlocked(position, price, structure_take_profit_reason)
                continue
            risk_exit_reason = _risk_exit_reason(position.side, trend_state, risk_state)
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
        self.account.positions.pop(position.symbol, None)
        gross_pnl = _pnl(position.side, position.entry_price, price, position.quantity)
        notional = price * position.quantity
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
            quantity=position.quantity,
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
                quantity=position.quantity,
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
        self._save_state_unlocked()
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
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
            "h1_ema20_ema60": "1H EMA20/EMA60回踩",
            "h4_ema20_ema60": "4H EMA20/EMA60回踩",
            "m15_ema20_ema60": "强单边15m EMA20/EMA60回踩",
            "sweep_reclaim_support": "下插针扫损后收回支撑",
            "oi_valley_recovery": "OI洼地止跌回稳",
            "ma_cluster_breakout": "上穿均线密集区",
            "ma20_retest": "突破均线密集后回踩MA20",
            "breakout_retest": "前压力突破后回踩确认",
            "vwap_pullback": "VWAP/成交密集区回踩",
        }
        if side == PositionSide.LONG
        else {
            "h1_resistance": "1H压力反抽",
            "h4_resistance": "4H压力反抽",
            "h1_boll_mid": "1H BOLL中轨反抽",
            "h1_ema20_ema60": "1H EMA20/EMA60反抽",
            "h4_ema20_ema60": "4H EMA20/EMA60反抽",
            "sweep_reject_resistance": "上插针扫空后跌回压力",
            "oi_distribution": "高位OI下降横盘滞涨",
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
        "h1_trigger",
        "h1_pullback",
        "m15_precision",
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
    }


def _load_paper_account(path: Path | None) -> PaperAccount | None:
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PaperAccount(
            starting_balance=float(raw.get("starting_balance", PAPER_DEFAULT_BALANCE)),
            wallet_balance=float(raw.get("wallet_balance", raw.get("starting_balance", PAPER_DEFAULT_BALANCE))),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            fees_paid=float(raw.get("fees_paid", 0.0)),
            positions={
                symbol: _position_from_payload(payload)
                for symbol, payload in dict(raw.get("positions") or {}).items()
            },
            fills=[_fill_from_payload(payload) for payload in list(raw.get("fills") or [])],
            daily_pnl_baselines={str(key): float(value) for key, value in dict(raw.get("daily_pnl_baselines") or {}).items()},
            pnl_history={str(key): float(value) for key, value in dict(raw.get("pnl_history") or {}).items()},
            latest_signals={
                str(symbol).upper(): dict(signal)
                for symbol, signal in dict(raw.get("latest_signals") or {}).items()
                if isinstance(signal, dict)
            },
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


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


def _strong_trend_invalidated(position: Position, indicator: IndicatorSnapshot | None) -> bool:
    if indicator is None or indicator.ema50 is None:
        return False
    if position.side == PositionSide.LONG:
        return indicator.close < indicator.ema50
    return indicator.close > indicator.ema50


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
    if current_profit < stop_distance * 0.8:
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
    h4_oi = signal.get("h4_oi") if isinstance(signal.get("h4_oi"), dict) else {}
    drop_from_high = _float_or_none(h4_oi.get("drop_from_high_pct")) or 0.0
    reasons_text = " | ".join(str(reason).lower() for reason in (signal.get("reasons") or ()))
    rsi_extreme = bool(indicator and indicator.rsi14 is not None and indicator.rsi14 <= RSI_STRONG_SHORT_SEVERE)
    short_crowded = risk_state == "SHORT_CROWD" or "short crowd" in reasons_text or "空头拥挤" in reasons_text
    oi_valley = drop_from_high <= -0.16 or any(
        token in reasons_text
        for token in (
            "oi valley",
            "oi flush",
            "oi drained",
            "deleverag",
            "oi洼地",
            "oi 去杠杆",
        )
    )
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
    return ((rsi_extreme or short_crowded) and (h1_reclaimed or downside_reclaim)) or (oi_valley and downside_reclaim)


def _tighten_short_support_stop(
    position: Position,
    price: float,
    signal: dict[str, object],
    indicator: IndicatorSnapshot | None,
) -> None:
    if position.side != PositionSide.SHORT or indicator is None or not indicator.atr14:
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
    if indicator is not None:
        rsi14 = rsi14 if rsi14 is not None else indicator.rsi14
        price = price if price is not None else indicator.close
    if _late_stage_risk(side, risk_state, rsi14, price, indicator, reasons_text):
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
) -> bool:
    if risk_state in {"LONG_CROWD", "SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
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


def _entry_reward_r(signal: dict[str, object], side: PositionSide, price: float, stop: float) -> float | None:
    risk = abs(price - stop)
    if risk <= 0:
        return None
    target = _nearest_structure_target(signal, side, price)
    if target is None:
        return None
    reward = target - price if side == PositionSide.LONG else price - target
    if reward <= 0:
        return 0.0
    return reward / risk


def _nearest_structure_target(signal: dict[str, object], side: PositionSide, price: float) -> float | None:
    candidates: list[float] = []
    for key in ("h1_structure", "h4_structure", "h1_trigger"):
        mapping = signal.get(key) if isinstance(signal.get(key), dict) else {}
        candidates.extend(_structure_target_candidates(mapping, side, price))
    for key in ("h1_ma_cluster", "h4_ma_cluster"):
        cluster = signal.get(key) if isinstance(signal.get(key), dict) else {}
        target_key = "target_up" if side == PositionSide.LONG else "target_down"
        target = _float_or_none(cluster.get(target_key))
        if target is not None and ((side == PositionSide.LONG and target > price) or (side == PositionSide.SHORT and target < price)):
            candidates.append(target)
    if not candidates:
        return None
    return min(candidates) if side == PositionSide.LONG else max(candidates)


def _structure_target_candidates(mapping: dict[str, object], side: PositionSide, price: float) -> list[float]:
    if side == PositionSide.LONG:
        keys = ("resistance_zone_low", "resistance", "resistance_zone_high")
        return [value for value in (_float_or_none(mapping.get(key)) for key in keys) if value is not None and value > price]
    keys = ("support_zone_high", "support", "support_zone_low")
    return [value for value in (_float_or_none(mapping.get(key)) for key in keys) if value is not None and value < price]


def _auto_signal_allowed(signal: dict[str, object]) -> bool:
    score = int(signal.get("score") or 0)
    if signal.get("vetoes"):
        return False
    entry_timing, _ = _signal_entry_timing(signal)
    if _trend_stage_from_signal(signal) == TREND_STAGE_LATE:
        return False
    if score >= 82 and entry_timing == ENTRY_TIMING_GOOD:
        return True
    return False


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
    stored = str(signal.get("entry_timing") or "")
    if stored in {ENTRY_TIMING_GOOD, ENTRY_TIMING_WAIT, ENTRY_TIMING_BLOCK}:
        return stored, str(signal.get("entry_timing_reason") or "")
    action = str(signal.get("action") or "")
    if action == SignalAction.WATCH.value:
        return ENTRY_TIMING_WAIT, "entry timing wait: watch signal needs entry confirmation"
    if action == SignalAction.NO_TRADE.value:
        return ENTRY_TIMING_BLOCK, "entry timing blocked: no trade signal"
    if action not in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}:
        return ENTRY_TIMING_BLOCK, "entry timing blocked: not an entry signal"
    price = _float_or_none(signal.get("price"))
    entry_levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    if not entry_levels:
        return ENTRY_TIMING_WAIT, "entry timing wait: suggested entry zone unavailable"
    if price is None:
        return ENTRY_TIMING_WAIT, "entry timing wait: latest price unavailable"
    side = PositionSide.LONG if action == SignalAction.ENTRY_LONG.value else PositionSide.SHORT
    return _side_entry_timing(side, price, signal)


def _side_entry_timing(side: PositionSide, price: float, signal: dict[str, object]) -> tuple[str, str]:
    score = int(signal.get("score") or 0)
    risk_state = str(signal.get("risk_state") or "NORMAL")
    trend_state = str(signal.get("trend_state") or signal.get("regime") or "")
    h1 = signal.get("h1_trigger") if isinstance(signal.get("h1_trigger"), dict) else {}
    m15_precision = signal.get("m15_precision") if isinstance(signal.get("m15_precision"), dict) else {}
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    if _trend_stage_from_signal(signal) == TREND_STAGE_LATE:
        return ENTRY_TIMING_BLOCK, "entry timing blocked: late trend stage needs a new pullback"
    if risk_state in {"LONG_CROWD", "SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
        return ENTRY_TIMING_BLOCK, "entry timing blocked: crowding/OI/funding risk not clean"
    include_m15 = _strong_m15_pullback_allowed(
        side,
        trend_state,
        risk_state,
        score,
        m15_precision,
    ) and _m15_precision_stop_distance_ok(side, price, m15_precision)
    if _entry_level_group_hit(signal, price, side, include_m15=include_m15):
        return ENTRY_TIMING_GOOD, _entry_zone_reason(side)
    expected_direction = "LONG" if side == PositionSide.LONG else "SHORT"
    if h1_direction == expected_direction and h1_state in {"BREAKOUT", "BREAKDOWN"}:
        return ENTRY_TIMING_WAIT, _breakout_wait_reason(side)
    return ENTRY_TIMING_WAIT, _wait_retest_reason(side)


def _entry_zone_reason(side: PositionSide) -> str:
    if side == PositionSide.LONG:
        return "entry timing good: price is inside 1h/4h support, EMA/BOLL, VWAP, or retest entry zone"
    return "entry timing good: price is inside 1h/4h resistance, EMA/BOLL, VWAP, or retest entry zone"


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
        return "entry timing wait: long needs 1h/4h support, EMA/BOLL, VWAP, or breakout retest"
    return "entry timing wait: short needs 1h/4h resistance, EMA/BOLL, VWAP, or breakdown retest"


def _entry_level_group_hit(signal: dict[str, object], price: float, side: PositionSide, *, include_m15: bool) -> bool:
    levels = signal.get("entry_levels") if isinstance(signal.get("entry_levels"), dict) else {}
    side_key = "long" if side == PositionSide.LONG else "short"
    side_levels = levels.get(side_key) if isinstance(levels.get(side_key), dict) else {}
    if not side_levels:
        return False
    keys = (
        (
            "h1_support",
            "h4_support",
            "h1_boll_mid",
            "h1_ema20_ema60",
            "h4_ema20_ema60",
            "sweep_reclaim_support",
            "oi_valley_recovery",
            "ma_cluster_breakout",
            "ma20_retest",
            "breakout_retest",
            "vwap_pullback",
        )
        if side == PositionSide.LONG
        else (
            "h1_resistance",
            "h4_resistance",
            "h1_boll_mid",
            "h1_ema20_ema60",
            "h4_ema20_ema60",
            "sweep_reject_resistance",
            "oi_distribution",
            "ma_cluster_breakdown",
            "ma20_retest",
            "breakdown_retest",
            "vwap_retest",
        )
    )
    if include_m15:
        keys = (*keys, "m15_ema20_ema60")
    return any(_entry_level_hit(price, side_levels.get(key)) for key in keys)


def _m15_precision_stop_distance_ok(side: PositionSide, price: float, precision: dict[str, object]) -> bool:
    if side == PositionSide.LONG:
        anchor = _float_or_none(precision.get("long_stop_anchor"))
        return anchor is not None and anchor < price and price - anchor >= price * MIN_PRECISION_STOP_PCT
    anchor = _float_or_none(precision.get("short_stop_anchor"))
    return anchor is not None and anchor > price and anchor - price >= price * MIN_PRECISION_STOP_PCT


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
    width = max(zone_high - zone_low, 0.0)
    pad = max(width * 0.25, price * 0.0025)
    return zone_low - pad <= price <= zone_high + pad


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
    return build_indicators(
        candles,
        [],
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
    h4_ma_cluster = _moving_average_cluster(timeframe_candles.get("4h", []), timeframe_indicators.get("4h", []), settings)
    h1_ma_cluster = _moving_average_cluster(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), settings)
    h4_oi = _four_hour_oi_state(timeframe_indicators.get("4h", []), h4, h1, h1_pullback)
    m15 = _fifteen_minute_precision(timeframe_candles.get("15m", []), timeframe_indicators.get("15m", []), settings)
    entry_levels = _entry_level_context(h1_structure, h4, h1_ma_cluster, h4_ma_cluster, timeframe_indicators, m15)
    return {
        "daily_bias": d1,
        "h4_structure": h4,
        "h1_structure": h1_structure,
        "h4_ma_cluster": h4_ma_cluster,
        "h1_ma_cluster": h1_ma_cluster,
        "h4_oi": h4_oi,
        "h1_trigger": h1,
        "h1_pullback": h1_pullback,
        "m15_precision": m15,
        "entry_levels": entry_levels,
        "summary": _multi_timeframe_summary(d1, h4, h1, h1_pullback, h4_oi, h4_ma_cluster, h1_ma_cluster),
    }


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
) -> dict[str, object]:
    h1_indicator = _latest_indicator(timeframe_indicators.get("1h", []))
    h4_indicator = _latest_indicator(timeframe_indicators.get("4h", []))
    m15_indicator = _latest_indicator(timeframe_indicators.get("15m", []))
    return {
        "long": {
            "h1_support": _zone_from_mapping(h1_structure, "support"),
            "h4_support": _zone_from_mapping(h4_structure, "support"),
            "h1_boll_mid": _indicator_band(h1_indicator, h1_indicator.boll_mid if h1_indicator else None),
            "h1_ema20_ema60": _ema20_ema60_band(h1_indicator, h1_ma_cluster),
            "h4_ema20_ema60": _ema20_ema60_band(h4_indicator, h4_ma_cluster),
            "m15_ema20_ema60": _zone_from_object(m15.get("long_pullback_zone")),
            "sweep_reclaim_support": _zone_from_mapping(h1_structure, "support") or _zone_from_mapping(h4_structure, "support"),
            "oi_valley_recovery": _recovery_band(h1_indicator) or _zone_from_mapping(h1_structure, "support"),
            "ma_cluster_breakout": _cluster_band(h1_ma_cluster) or _cluster_band(h4_ma_cluster),
            "ma20_retest": _ma20_band(h1_ma_cluster, h1_indicator) or _ma20_band(h4_ma_cluster, h4_indicator),
            "breakout_retest": _zone_from_mapping(h1_structure, "resistance") or _zone_from_mapping(h4_structure, "resistance"),
            "vwap_pullback": _indicator_band(h1_indicator, h1_indicator.vwap if h1_indicator else None),
        },
        "short": {
            "h1_resistance": _zone_from_mapping(h1_structure, "resistance"),
            "h4_resistance": _zone_from_mapping(h4_structure, "resistance"),
            "h1_boll_mid": _indicator_band(h1_indicator, h1_indicator.boll_mid if h1_indicator else None),
            "h1_ema20_ema60": _ema20_ema60_band(h1_indicator, h1_ma_cluster),
            "h4_ema20_ema60": _ema20_ema60_band(h4_indicator, h4_ma_cluster),
            "m15_ema20_ema60": _zone_from_object(m15.get("short_retest_zone")),
            "sweep_reject_resistance": _zone_from_mapping(h1_structure, "resistance") or _zone_from_mapping(h4_structure, "resistance"),
            "oi_distribution": _recovery_band(h1_indicator) or _zone_from_mapping(h1_structure, "resistance"),
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
        return indicator.atr14 * 0.45
    if price:
        return price * 0.003
    return 0.0


def _indicator_band(indicator: IndicatorSnapshot | None, price: float | None) -> dict[str, float] | None:
    value = _float_or_none(price)
    if value is None:
        return None
    buffer = _indicator_buffer(indicator, value)
    return {"low": value - buffer, "high": value + buffer, "price": value}


def _ema20_ema60_band(indicator: IndicatorSnapshot | None, cluster: dict[str, object]) -> dict[str, float] | None:
    ema60 = _float_or_none(cluster.get("ema60"))
    values = [indicator.ema20 if indicator else None, ema60]
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    reference = sum(valid) / len(valid)
    buffer = _indicator_buffer(indicator, reference)
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


def _recovery_band(indicator: IndicatorSnapshot | None) -> dict[str, float] | None:
    if indicator is None:
        return None
    return _indicator_band(indicator, indicator.close)


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


def _distribution_short_retest_setup(
    signal: dict[str, object],
    h4_oi: dict[str, object],
    h1: dict[str, object],
    h1_pullback: dict[str, object],
    h1_ma_cluster: dict[str, object],
    h4_ma_cluster: dict[str, object],
) -> bool:
    if not _weak_deleverage_rebound(signal, h4_oi, h1_pullback):
        return False
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    cluster_states = {
        str(h1_ma_cluster.get("state") or "UNKNOWN"),
        str(h4_ma_cluster.get("state") or "UNKNOWN"),
    }
    resistance_rejected = h1_direction == "SHORT" and h1_state in {"RETEST", "FAKE_BREAKOUT"}
    rebound_failed = pullback_direction == "SHORT" and pullback_state == "HEALTHY_PULLBACK"
    ma_pressure_rejected = bool(cluster_states & {"RETEST_DOWN", "BREAKDOWN_DOWN"})
    return resistance_rejected or rebound_failed or ma_pressure_rejected


def _apply_multi_timeframe_context(signal: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    out = dict(signal)
    reasons = list(out.get("reasons") or [])
    vetoes = list(out.get("vetoes") or [])
    score = int(out.get("score") or 0)
    action = str(out.get("action") or "")
    daily_bias = str(context.get("daily_bias") or "NEUTRAL")
    h4 = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    h1_structure = context.get("h1_structure") if isinstance(context.get("h1_structure"), dict) else {}
    h4_ma_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    h1_ma_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_oi = context.get("h4_oi") if isinstance(context.get("h4_oi"), dict) else {}
    h1 = context.get("h1_trigger") if isinstance(context.get("h1_trigger"), dict) else {}
    h1_pullback = context.get("h1_pullback") if isinstance(context.get("h1_pullback"), dict) else {}
    m15_precision = context.get("m15_precision") if isinstance(context.get("m15_precision"), dict) else {}
    entry_levels = context.get("entry_levels") if isinstance(context.get("entry_levels"), dict) else {}
    h4_state = str(h4.get("state") or "UNKNOWN")
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    h4_oi_state = str(h4_oi.get("state") or "UNKNOWN")
    risk_state = str(out.get("risk_state") or "NORMAL")
    trend_state = str(out.get("trend_state") or "")
    rsi14 = _float_or_none(out.get("rsi14"))
    weak_deleverage_rebound = _weak_deleverage_rebound(out, h4_oi, h1_pullback)
    distribution_short_retest_setup = _distribution_short_retest_setup(
        out,
        h4_oi,
        h1,
        h1_pullback,
        h1_ma_cluster,
        h4_ma_cluster,
    )
    if action == SignalAction.ENTRY_LONG.value:
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
        if h1_direction == "LONG":
            score += 10
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
                reasons.append("1h BOLL/EMA pullback held with clean risk")
            elif pullback_state == "HIGH_PULLBACK" and risk_state in {"LONG_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
                score -= 20
                vetoes.append("high pullback with OI/funding/crowd risk; avoid long entry")
        elif _high_area_needs_long_pullback(h4_state, h1_direction, h1_state):
            if _strong_m15_pullback_allowed(PositionSide.LONG, trend_state, risk_state, score, m15_precision):
                score += 6
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
            reasons.append("4h OI deleveraged while long/short ratio rose; 1h support held, only tiny long allowed")
        elif h4_oi_state == "DELEVERAGE_HOLD_LONG":
            score += 6
            out["leverage_cap"] = min(int(out.get("leverage_cap") or 99), 5)
            out["margin_factor"] = min(float(out.get("margin_factor") or 1.0), 0.5)
            reasons.append("4h OI deleveraged but 1h BOLL/EMA held; allow small long only")
        elif h4_oi_state == "REBUILD_BREAKOUT_LONG":
            score += 12
            reasons.append("4h OI rebounds after deleverage and price breaks out; strong long restored")
        rsi_score, rsi_reasons, rsi_vetoes = _rsi_entry_adjustment(
            PositionSide.LONG,
            rsi14,
            trend_state,
            risk_state,
            score,
            pullback_direction,
            pullback_state,
            m15_precision,
        )
        score += rsi_score
        reasons.extend(rsi_reasons)
        vetoes.extend(rsi_vetoes)
    elif action == SignalAction.ENTRY_SHORT.value:
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
        if h1_direction == "SHORT":
            score += 10
            reasons.append(f"1h {h1_state.lower()} confirms short trigger")
        elif h1_direction == "LONG":
            vetoes.append("1h trigger opposes short entry")
        if pullback_direction == "SHORT":
            if pullback_state == "HEALTHY_PULLBACK" and risk_state == "NORMAL":
                score += 12
                reasons.append("1h BOLL/EMA pullback rejected with clean risk")
            elif pullback_state == "LOW_PULLBACK" and risk_state in {"SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
                score -= 20
                vetoes.append("low pullback with OI/funding/crowd risk; avoid short entry")
        elif _low_area_needs_short_pullback(h4_state, h1_direction, h1_state):
            score -= 12
            vetoes.append("low area without 1h/4h resistance retest; wait for higher-timeframe bounce before short")
        if h4_oi_state in {"DELEVERAGE_BREAKDOWN", "DELEVERAGE_CROWD_BREAKDOWN"}:
            if _deleverage_short_failure_confirmed(h1_direction, h1_state, pullback_direction, pullback_state, reasons):
                score += 8
                reasons.append("4h OI deleverage breakdown with failed bounce; short candidate improved")
            else:
                score -= 8
                vetoes.append("4h OI deleverage breakdown; wait for resistance retest or upper-wick rejection before short")
        elif h4_oi_state in {"DELEVERAGE_HOLD_LONG", "DELEVERAGE_CROWD_HOLD_LONG"}:
            if distribution_short_retest_setup:
                score += 8
                reasons.append("OI drained with weak rebound and resistance rejection; short candidate improved")
            else:
                vetoes.append("4h OI deleveraged but 1h support held; avoid chasing short")
        elif h4_oi_state in {"DELEVERAGE_WAIT", "DELEVERAGE_CROWD_WAIT"}:
            if distribution_short_retest_setup:
                score += 8
                reasons.append("OI drained, rebound volume weak, and 1h/MA resistance rejected; short candidate improved")
            else:
                score -= 6
                vetoes.append("4h OI drained; wait for 1h resistance retest or upper-wick rejection before short")
        rsi_score, rsi_reasons, rsi_vetoes = _rsi_entry_adjustment(
            PositionSide.SHORT,
            rsi14,
            trend_state,
            risk_state,
            score,
            pullback_direction,
            pullback_state,
            m15_precision,
        )
        score += rsi_score
        reasons.extend(rsi_reasons)
        vetoes.extend(rsi_vetoes)
    reasons.append(str(context.get("summary") or "multi-timeframe context neutral"))
    stage_probe = {
        **out,
        "reasons": tuple(reasons),
        "h1_trigger": h1,
        "h1_pullback": h1_pullback,
        "h4_oi": h4_oi,
        "h1_structure": h1_structure,
        "h4_structure": h4,
        "h1_ma_cluster": h1_ma_cluster,
        "h4_ma_cluster": h4_ma_cluster,
    }
    trend_stage_phase = _trend_stage_from_signal(stage_probe)
    if trend_stage_phase == TREND_STAGE_LATE and action in {SignalAction.ENTRY_LONG.value, SignalAction.ENTRY_SHORT.value}:
        out["leverage_cap"] = min(int(out.get("leverage_cap") or 99), 5)
        reasons.append("trend late stage; wait for a new pullback before fresh entry")
    out["score"] = max(0, score)
    out["reasons"] = tuple(reasons)
    out["vetoes"] = tuple(vetoes)
    out["daily_bias"] = daily_bias
    out["h4_structure"] = h4
    out["h1_structure"] = h1_structure
    out["h4_ma_cluster"] = h4_ma_cluster
    out["h1_ma_cluster"] = h1_ma_cluster
    out["h4_oi"] = h4_oi
    out["h1_trigger"] = h1
    out["h1_pullback"] = h1_pullback
    out["m15_precision"] = m15_precision
    out["entry_levels"] = entry_levels
    out["trend_stage_phase"] = trend_stage_phase
    entry_timing, entry_timing_reason = _signal_entry_timing(out)
    out["entry_timing"] = entry_timing
    out["entry_timing_reason"] = entry_timing_reason
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
                pullback_ok = _strong_rsi_pullback_ok(side, risk_state, score, pullback_direction, pullback_state, m15_precision)
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
            pullback_ok = _strong_rsi_pullback_ok(side, risk_state, score, pullback_direction, pullback_state, m15_precision)
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
) -> bool:
    if risk_state != "NORMAL":
        return False
    if side == PositionSide.LONG:
        return (
            pullback_direction == "LONG"
            and pullback_state == "HEALTHY_PULLBACK"
        ) or _strong_m15_pullback_allowed(side, "ONE_WAY_UP", risk_state, score, m15_precision)
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
            vetoes.append(_ma_cluster_reason("MA cluster dense; wait for breakout or MA20 retest", h1_price or h4_price))
    else:
        if h4_state == "BREAKDOWN_DOWN" or h1_state == "BREAKDOWN_DOWN":
            score += 12
            reasons.append(_ma_cluster_reason("MA cluster breakdown down", h1_price or h4_price))
        if h1_state == "RETEST_DOWN":
            score += 14
            reasons.append(_ma_cluster_reason("MA cluster retest rejected near MA20", h1_price or h4_price))
        if h4_state == "DENSE" and h1_state not in {"BREAKDOWN_DOWN", "RETEST_DOWN"}:
            vetoes.append(_ma_cluster_reason("MA cluster dense; wait for breakdown or MA20 retest", h1_price or h4_price))
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
) -> bool:
    if score < 85 or risk_state != "NORMAL":
        return False
    pullback = str(precision.get("pullback") or "")
    m15_trend = str(precision.get("trend") or "")
    if side == PositionSide.LONG:
        return trend_state == "ONE_WAY_UP" and m15_trend == "UP" and pullback == "M15_LONG_PULLBACK"
    return False


def _precision_stop_allowed(
    side: PositionSide,
    trend_state: str,
    risk_state: str,
    score: int,
    precision: dict[str, object],
) -> bool:
    return _strong_m15_pullback_allowed(side, trend_state, risk_state, score, precision)


def _deleverage_short_failure_confirmed(
    h1_direction: str,
    h1_state: str,
    pullback_direction: str,
    pullback_state: str,
    reasons: list[object],
) -> bool:
    if h1_direction == "SHORT" and h1_state in {"RETEST", "FAKE_BREAKOUT"}:
        return True
    if pullback_direction == "SHORT" and pullback_state == "HEALTHY_PULLBACK":
        return True
    reason_text = " | ".join(str(reason) for reason in reasons)
    rejection_markers = (
        "washout confirmed: upside wick swept resistance",
        "upside sweep rejected resistance",
        "market structure confirms short: breakdown or retest failed",
    )
    return any(marker in reason_text for marker in rejection_markers)


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
        candidates = [stop]
        if indicator.kc_lower is not None and indicator.kc_lower < entry:
            candidates.append(indicator.kc_lower - buffer)
        if trend_state in {"TREND_LONG", "ONE_WAY_UP"} and indicator.kc_mid < entry:
            candidates.append(indicator.kc_mid - buffer)
        valid = [candidate for candidate in candidates if candidate < entry]
        return min(valid) if valid else stop
    candidates = [stop]
    if indicator.kc_upper is not None and indicator.kc_upper > entry:
        candidates.append(indicator.kc_upper + buffer)
    if trend_state in {"TREND_SHORT", "ONE_WAY_DOWN"} and indicator.kc_mid > entry:
        candidates.append(indicator.kc_mid + buffer)
    valid = [candidate for candidate in candidates if candidate > entry]
    return max(valid) if valid else stop


def _refine_stop_with_retest_structure(
    side: PositionSide,
    stop: float,
    entry: float,
    context: dict[str, object],
    indicator: IndicatorSnapshot | None,
) -> float:
    h1 = context.get("h1_structure") if isinstance(context.get("h1_structure"), dict) else {}
    h4 = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    h1_trigger = context.get("h1_trigger") if isinstance(context.get("h1_trigger"), dict) else {}
    h1_pullback = context.get("h1_pullback") if isinstance(context.get("h1_pullback"), dict) else {}
    h1_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    relevant = _retest_structure_stop_relevant(side, h1_trigger, h1_pullback, h1_cluster, h4_cluster)
    if not relevant:
        return stop
    buffer = _structure_stop_buffer(entry, indicator)
    min_distance = _minimum_precision_stop_distance(entry, indicator)
    if side == PositionSide.SHORT:
        candidates = _short_structure_stop_candidates(entry, h1, h4, h1_cluster, h4_cluster)
        if not candidates:
            return stop
        structure_stop = min(candidates) + buffer
        return max(stop, structure_stop, entry + min_distance)
    candidates = _long_structure_stop_candidates(entry, h1, h4, h1_cluster, h4_cluster)
    if not candidates:
        return stop
    structure_stop = max(candidates) - buffer
    return min(stop, structure_stop, entry - min_distance)


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
) -> tuple[float, float]:
    h4 = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    h1_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    target = _cluster_target(side, h1_cluster) or _cluster_target(side, h4_cluster)
    horizontal = _float_or_none(h4.get("resistance" if side == PositionSide.LONG else "support"))
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
