from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Literal

from ai_trading.binance import BinanceFuturesMarketData
from ai_trading.config import AppSettings
from ai_trading.indicators import build_indicators, ema, sma
from ai_trading.models import Candle, IndicatorSnapshot, Position, PositionSide, SignalAction, Trade
from ai_trading.strategy import CompositeStrategy


PaperSide = Literal["LONG", "SHORT"]
AUTO_UNIVERSE_EXCLUDED_SYMBOLS = {"BTCUSDT", "BTCUSDC", "ETHUSDT", "SOLUSDT", "XAUUSDT"}
BTC_EXTREME_4H_AMPLITUDE = 0.08
ROTATION_MIN_SCORE = 90
ROTATION_MIN_SCORE_GAP = 18
ROTATION_MIN_ATR_PCT = 0.008
ROTATION_MIN_VOLUME_RATIO = 1.2
ROTATION_MAX_PROFIT_TO_REPLACE = 0.01
ROTATION_MIN_HOLD_SECONDS = 30 * 60
PYRAMID_MIN_SCORE = 85
PYRAMID_MARGIN_FRACTION = 0.35
PYRAMID_MAX_ADDS = 1
PAPER_DEFAULT_BALANCE = 1200.0
INITIAL_ENTRY_MARGIN_CAP = 0.78
PYRAMID_TOTAL_MARGIN_CAP = 0.95


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
    ) -> None:
        self.settings = settings
        self.account = PaperAccount(starting_balance=starting_balance, wallet_balance=starting_balance)
        self.symbols = [symbol.upper() for symbol in (symbols or ["AUTO_TOP30"])]
        self.interval = interval
        self.market_data = market_data or BinanceFuturesMarketData()
        self.strategy = CompositeStrategy(settings.strategy)
        self.latest_prices: dict[str, float] = {}
        self.latest_signals: dict[str, dict[str, object]] = {}
        self.latest_indicators: dict[str, list[IndicatorSnapshot]] = {}
        self.latest_timeframe_contexts: dict[str, dict[str, object]] = {}
        self.latest_timeframe_indicators: dict[str, dict[str, list[IndicatorSnapshot]]] = {}
        self.running = False
        self.auto_trade = False
        self.last_error: str | None = None
        self.last_market_update_at: datetime | None = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self, *, auto_trade: bool = False, poll_seconds: int = 20) -> None:
        async with self._lock:
            self.auto_trade = auto_trade
            if self.running:
                return
            self.running = True
            self._task = asyncio.create_task(self._run_loop(poll_seconds=poll_seconds))

    async def stop(self) -> None:
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
        elif errors:
            self.last_error = f"行情刷新失败，等待网络恢复：{'; '.join(errors[:3])}"

    async def refresh_universe_if_needed(self) -> None:
        if self.symbols and self.symbols != ["AUTO_TOP30"]:
            return
        top_symbols = await self.market_data.top_usdt_perpetuals(limit=45)
        self.symbols = [item.symbol for item in top_symbols if item.symbol.upper() not in AUTO_UNIVERSE_EXCLUDED_SYMBOLS][:30]

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
                    "adds": 0,
                    "initial_stop_distance": abs(price - stop_loss),
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
                )
            )
            return position

    async def close_position(self, symbol: str, *, reason: str = "manual close") -> Trade:
        symbol = symbol.upper()
        price = await self._price(symbol)
        async with self._lock:
            position = self.account.positions.get(symbol)
            if position is None:
                raise ValueError(f"{symbol} has no open paper position")
            return self._close_position_unlocked(position, price, reason)

    def status(self) -> dict[str, object]:
        positions = []
        unrealized = 0.0
        used_margin = 0.0
        for position in self.account.positions.values():
            price = self.latest_prices.get(position.symbol, position.entry_price)
            pnl = _pnl(position.side, position.entry_price, price, position.quantity)
            margin = float(position.metadata.get("margin_usdt", 0.0))
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
                }
            )
        equity = self.account.wallet_balance + unrealized
        total_pnl = equity - self.account.starting_balance
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
            "realized_pnl": self.account.realized_pnl,
            "unrealized_pnl": unrealized,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.account.starting_balance,
            "fees_paid": self.account.fees_paid,
            "latest_prices": self.latest_prices,
            "latest_signals": self.latest_signals,
            "latest_timeframe_contexts": self.latest_timeframe_contexts,
            "positions": positions,
            "fills": [_fill_payload(fill) for fill in self.account.fills[-100:]],
            "daily_pnl": _daily_pnl_payload(self.account.daily_pnl_baselines, total_pnl),
            "last_error": self.last_error,
            "market_updated_at": self.last_market_update_at.isoformat() if self.last_market_update_at else None,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    async def _run_loop(self, *, poll_seconds: int) -> None:
        while self.running:
            try:
                await self.refresh_once()
                self._manage_open_positions()
                if self.auto_trade:
                    await self._auto_trade_once()
            except Exception as exc:  # noqa: BLE001 - user-facing paper loop should keep running
                self.last_error = str(exc)
            await asyncio.sleep(max(poll_seconds, 5))

    async def _refresh_symbol(self, symbol: str) -> bool:
        history_limit = max(240, self.settings.strategy.ma_trend + 40)
        candles, derivatives = await self.market_data.historical_bundle(symbol, self.interval, limit=history_limit)
        if not candles:
            return False
        price = candles[-1].close
        self.latest_prices[symbol] = price
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
        payload = {
            "timestamp": signal.timestamp.isoformat(),
            "action": signal.action.value,
            "regime": signal.regime.value,
            "trend_state": trend_state,
            "risk_state": risk_state,
            "score": signal.score,
            "reasons": signal.reasons,
            "vetoes": signal.vetoes,
            "smart_money_phase": cycle.phase,
        }
        self.latest_signals[symbol] = _apply_multi_timeframe_context(payload, mtf_context)
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
            self.latest_prices[symbol] = candles[-1].close
        return self.latest_prices[symbol]

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
                leverage = min(_leverage_for_signal(score, self.settings.risk.leverage_max), int(signal.get("leverage_cap") or self.settings.risk.leverage_max))
                indicators = self.latest_indicators.get(symbol, [])
                precision = self.latest_timeframe_contexts.get(symbol, {}).get("m15_precision", {})
                mtf_context = self.latest_timeframe_contexts.get(symbol, {})
                trend_state = str(signal.get("trend_state") or "CHOP")
                stop_loss, take_profit_1, take_profit_2 = _adaptive_exits(
                    PositionSide(side),
                    self.latest_prices.get(symbol) or await self._price(symbol),
                    leverage,
                    trend_state,
                    _preferred_exit_indicator(self.latest_timeframe_indicators.get(symbol, {}), indicators),
                )
                stop_loss = _refine_stop_with_precision(PositionSide(side), stop_loss, precision)
                stop_loss = _refine_stop_with_ma_cluster(PositionSide(side), stop_loss, mtf_context)
                take_profit_1, take_profit_2 = _refine_take_profit_with_ma_cluster(
                    PositionSide(side),
                    self.latest_prices.get(symbol) or await self._price(symbol),
                    stop_loss,
                    take_profit_1,
                    take_profit_2,
                    mtf_context,
                )
                await self.open_position(
                    symbol,
                    side,
                    margin_usdt=margin,
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    reason=f"auto strategy score={signal.get('score')}; state={trend_state}",
                )
        self._add_to_strong_positions()

    async def _rebalance_for_better_candidate(self, candidates: list[tuple[str, dict[str, object]]]) -> None:
        if not candidates or not self.account.positions:
            return
        best_symbol, best_signal = candidates[0]
        best_indicators = self.latest_indicators.get(best_symbol, [])
        if not _rotation_candidate_allowed(best_signal, best_indicators[-1] if best_indicators else None):
            return
        weakest = self._weakest_position()
        if weakest is None:
            return
        best_score = int(best_signal.get("score") or 0)
        weakest_symbol, weakest_score, weakest_pnl_pct, weakest_trend = weakest
        score_gap = best_score - weakest_score
        if score_gap < ROTATION_MIN_SCORE_GAP:
            return
        if weakest_pnl_pct > ROTATION_MAX_PROFIT_TO_REPLACE:
            return
        if weakest_trend in {"ONE_WAY_UP", "ONE_WAY_DOWN"} and weakest_score >= 78:
            return
        await self.close_position(weakest_symbol, reason=f"rotation exit: symbol={best_symbol} score={best_score}")

    def _weakest_position(self) -> tuple[str, int, float, str] | None:
        weakest: tuple[str, int, float, str] | None = None
        now = datetime.now(UTC)
        for symbol, position in self.account.positions.items():
            if (now - position.opened_at).total_seconds() < ROTATION_MIN_HOLD_SECONDS:
                continue
            signal = self.latest_signals.get(symbol, {})
            score = int(signal.get("score") or 0)
            trend = str(signal.get("trend_state") or signal.get("regime") or "CHOP")
            price = self.latest_prices.get(symbol, position.entry_price)
            pnl = _pnl(position.side, position.entry_price, price, position.quantity)
            margin = float(position.metadata.get("margin_usdt", 0.0))
            pnl_pct = pnl / margin if margin else 0.0
            candidate = (symbol, score, pnl_pct, trend)
            if weakest is None or (score, pnl_pct) < (weakest[1], weakest[2]):
                weakest = candidate
        return weakest

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
            latest_indicator = indicators[-1] if indicators else None
            trend_state = self.strategy.trend_state(position.side.value, indicators) if indicators else "CHOP"
            risk_state = self.strategy.risk_state(indicators) if indicators else "NORMAL"
            strong_trend = trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}
            signal = self.latest_signals.get(position.symbol, {})
            if strong_trend:
                _protect_confirmed_breakout_position(position, signal, _preferred_exit_indicator(tf_indicators, indicators))
            self._update_trailing_stop(position, price, strong_trend, _preferred_exit_indicator(tf_indicators, indicators))
            if _stop_hit(position, price):
                self._close_position_unlocked(position, price, "stop loss")
                continue
            if strong_trend and _strong_trend_invalidated(position, latest_indicator):
                self._close_position_unlocked(position, price, "strong trend invalidated")
                continue
            risk_exit_reason = _risk_exit_reason(position.side, trend_state, risk_state)
            if risk_exit_reason:
                self._close_position_unlocked(position, price, risk_exit_reason)
                continue
            if _take_profit_hit(position, price) and not strong_trend:
                self._close_position_unlocked(position, price, "take profit")

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
                )
            )
            break

    def _close_position_unlocked(self, position: Position, price: float, reason: str) -> Trade:
        self.account.positions.pop(position.symbol, None)
        gross_pnl = _pnl(position.side, position.entry_price, price, position.quantity)
        notional = price * position.quantity
        fee = notional * self.settings.execution.taker_fee_rate
        realized = gross_pnl - fee
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
                closed_at=timestamp,
                return_pct=realized / margin_usdt if margin_usdt else 0.0,
            )
        )
        return trade

    def _available_balance_unlocked(self) -> float:
        status = self.status()
        return float(status["available_balance"])


def _pnl(side: PositionSide, entry_price: float, mark_price: float, quantity: float) -> float:
    if side == PositionSide.LONG:
        return (mark_price - entry_price) * quantity
    return (entry_price - mark_price) * quantity


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


def _protect_confirmed_breakout_position(position: Position, signal: dict[str, object], indicator: IndicatorSnapshot | None) -> None:
    if indicator is None or indicator.atr14 is None or indicator.atr14 <= 0:
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
        stop_pct, tp1_r, tp2_r = 0.018, 0.5, 0.8
    elif trend_state in {"ONE_WAY_UP", "ONE_WAY_DOWN"}:
        stop_pct, tp1_r, tp2_r = 0.030, 0.8, 1.8
    else:
        stop_pct, tp1_r, tp2_r = 0.024, 0.7, 1.1

    min_stop_pct, tp_boost = _volatility_exit_profile(indicators)
    stop_pct = max(stop_pct, min_stop_pct)
    tp1_r *= tp_boost
    tp2_r *= tp_boost
    leverage_cap = _stop_pct_for_leverage(leverage) * 2.5
    stop_pct = min(stop_pct, leverage_cap)
    stop = _stop_from_pct(side, price, stop_pct)
    if indicators and indicators.atr14:
        atr_stop = _stop_from_atr(side, price, indicators.atr14, 1.2)
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
        return 0.018, 0.9
    if atr_pct < 0.02:
        return 0.024, 1.0
    if atr_pct < 0.04:
        return 0.032, 1.25
    return 0.045, 1.5


def _risk_exit_reason(side: PositionSide, trend_state: str, risk_state: str) -> str | None:
    if trend_state == "ONE_WAY_UP" and side == PositionSide.LONG and risk_state in {"LONG_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
        return f"risk exit: {risk_state}"
    if trend_state == "ONE_WAY_DOWN" and side == PositionSide.SHORT and risk_state in {"SHORT_CROWD", "OI_ABNORMAL", "FUNDING_HOT"}:
        return f"risk exit: {risk_state}"
    return None


def _auto_signal_allowed(signal: dict[str, object]) -> bool:
    score = int(signal.get("score") or 0)
    risk_state = str(signal.get("risk_state") or "NORMAL")
    if signal.get("vetoes"):
        return False
    if score >= 78:
        return True
    if 75 <= score <= 77:
        return risk_state == "NORMAL"
    return False


def _rotation_candidate_allowed(signal: dict[str, object], indicator: IndicatorSnapshot | None) -> bool:
    score = int(signal.get("score") or 0)
    if score < ROTATION_MIN_SCORE:
        return False
    if str(signal.get("risk_state") or "NORMAL") != "NORMAL":
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
    )


def _build_multi_timeframe_context(
    timeframe_candles: dict[str, list[Candle]],
    timeframe_indicators: dict[str, list[IndicatorSnapshot]],
    settings: AppSettings,
) -> dict[str, object]:
    d1 = _daily_direction(timeframe_indicators.get("1d", []))
    h4 = _four_hour_structure(timeframe_candles.get("4h", []), timeframe_indicators.get("4h", []), settings)
    h1 = _one_hour_trigger(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), h4, settings)
    h1_pullback = _one_hour_pullback(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), h4, settings)
    h4_ma_cluster = _moving_average_cluster(timeframe_candles.get("4h", []), timeframe_indicators.get("4h", []), settings)
    h1_ma_cluster = _moving_average_cluster(timeframe_candles.get("1h", []), timeframe_indicators.get("1h", []), settings)
    h4_oi = _four_hour_oi_state(timeframe_indicators.get("4h", []), h4, h1, h1_pullback)
    m15 = _fifteen_minute_precision(timeframe_candles.get("15m", []), timeframe_indicators.get("15m", []), settings)
    return {
        "daily_bias": d1,
        "h4_structure": h4,
        "h4_ma_cluster": h4_ma_cluster,
        "h1_ma_cluster": h1_ma_cluster,
        "h4_oi": h4_oi,
        "h1_trigger": h1,
        "h1_pullback": h1_pullback,
        "m15_precision": m15,
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
        "box_mid": mid,
        "box_width_pct": box_width_pct,
    }


def _one_hour_trigger(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot],
    h4: dict[str, object],
    settings: AppSettings,
) -> dict[str, object]:
    if len(candles) < 3 or not indicators:
        return {"direction": "NONE", "state": "UNKNOWN"}
    current = candles[-1]
    previous = candles[-2]
    indicator = indicators[-1]
    support = _float_or_none(h4.get("support"))
    resistance = _float_or_none(h4.get("resistance"))
    buffer = (indicator.atr14 or current.close * 0.004) * settings.strategy.structure_buffer_atr
    direction = "NONE"
    state = "WAIT"
    if resistance is not None:
        breakout = current.close > resistance + buffer
        retest = previous.close > resistance and current.low <= resistance + buffer and current.close > resistance
        fake_breakout = previous.high > resistance and current.close < resistance - buffer
        if breakout or retest:
            direction, state = "LONG", "BREAKOUT" if breakout else "RETEST"
        elif fake_breakout:
            direction, state = "SHORT", "FAKE_BREAKOUT"
    if support is not None:
        breakdown = current.close < support - buffer
        retest = previous.close < support and current.high >= support - buffer and current.close < support
        fake_breakdown = previous.low < support and current.close > support + buffer
        if breakdown or retest:
            direction, state = "SHORT", "BREAKDOWN" if breakdown else "RETEST"
        elif fake_breakdown:
            direction, state = "LONG", "FAKE_BREAKDOWN"
    return {"direction": direction, "state": state}


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
        return {"long_stop_anchor": None, "short_stop_anchor": None, "pullback": "UNKNOWN"}
    recent = candles[-lookback:]
    indicator = indicators[-1]
    buffer = (indicator.atr14 or recent[-1].close * 0.004) * settings.strategy.structure_buffer_atr
    ema9_values = ema([candle.close for candle in candles], 9)
    ema9_value = ema9_values[-1] if ema9_values else None
    current = candles[-1]
    mid = indicator.boll_mid
    long_ref = max(value for value in (ema9_value, mid) if value is not None) if ema9_value is not None or mid is not None else None
    short_ref = min(value for value in (ema9_value, mid) if value is not None) if ema9_value is not None or mid is not None else None
    pullback = "WAIT"
    if long_ref is not None and current.low <= long_ref + buffer and current.close >= long_ref - buffer * 0.25:
        pullback = "M15_LONG_PULLBACK"
    elif short_ref is not None and current.high >= short_ref - buffer and current.close <= short_ref + buffer * 0.25:
        pullback = "M15_SHORT_PULLBACK"
    return {
        "long_stop_anchor": min(candle.low for candle in recent) - buffer,
        "short_stop_anchor": max(candle.high for candle in recent) + buffer,
        "ema9": ema9_value,
        "boll_mid": mid,
        "pullback": pullback,
    }


def _apply_multi_timeframe_context(signal: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    out = dict(signal)
    reasons = list(out.get("reasons") or [])
    vetoes = list(out.get("vetoes") or [])
    score = int(out.get("score") or 0)
    action = str(out.get("action") or "")
    daily_bias = str(context.get("daily_bias") or "NEUTRAL")
    h4 = context.get("h4_structure") if isinstance(context.get("h4_structure"), dict) else {}
    h4_ma_cluster = context.get("h4_ma_cluster") if isinstance(context.get("h4_ma_cluster"), dict) else {}
    h1_ma_cluster = context.get("h1_ma_cluster") if isinstance(context.get("h1_ma_cluster"), dict) else {}
    h4_oi = context.get("h4_oi") if isinstance(context.get("h4_oi"), dict) else {}
    h1 = context.get("h1_trigger") if isinstance(context.get("h1_trigger"), dict) else {}
    h1_pullback = context.get("h1_pullback") if isinstance(context.get("h1_pullback"), dict) else {}
    m15_precision = context.get("m15_precision") if isinstance(context.get("m15_precision"), dict) else {}
    h4_state = str(h4.get("state") or "UNKNOWN")
    h1_direction = str(h1.get("direction") or "NONE")
    h1_state = str(h1.get("state") or "UNKNOWN")
    pullback_direction = str(h1_pullback.get("direction") or "NONE")
    pullback_state = str(h1_pullback.get("state") or "UNKNOWN")
    h4_oi_state = str(h4_oi.get("state") or "UNKNOWN")
    risk_state = str(out.get("risk_state") or "NORMAL")
    trend_state = str(out.get("trend_state") or "")
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
            if pullback_state == "HEALTHY_PULLBACK" and risk_state == "NORMAL":
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
            if _strong_m15_pullback_allowed(PositionSide.SHORT, trend_state, risk_state, score, m15_precision):
                score += 6
                reasons.append("one-way downtrend 15m BOLL/EMA9 bounce rejected; allow tactical short")
            else:
                score -= 12
                vetoes.append("low area without bounce confirmation; wait for 1h/4h retest before short")
        if h4_oi_state in {"DELEVERAGE_BREAKDOWN", "DELEVERAGE_CROWD_BREAKDOWN"}:
            if _deleverage_short_failure_confirmed(h1_direction, h1_state, pullback_direction, pullback_state, reasons):
                score += 8
                reasons.append("4h OI deleverage breakdown with failed bounce; short candidate improved")
            else:
                score -= 8
                vetoes.append("4h OI deleverage breakdown; wait for resistance retest or upper-wick rejection before short")
        elif h4_oi_state in {"DELEVERAGE_HOLD_LONG", "DELEVERAGE_CROWD_HOLD_LONG"}:
            vetoes.append("4h OI deleveraged but 1h support held; avoid chasing short")
    reasons.append(str(context.get("summary") or "multi-timeframe context neutral"))
    out["score"] = max(0, score)
    out["reasons"] = tuple(reasons)
    out["vetoes"] = tuple(vetoes)
    out["daily_bias"] = daily_bias
    out["h4_structure"] = h4
    out["h4_ma_cluster"] = h4_ma_cluster
    out["h1_ma_cluster"] = h1_ma_cluster
    out["h4_oi"] = h4_oi
    out["h1_trigger"] = h1
    out["h1_pullback"] = h1_pullback
    out["m15_precision"] = m15_precision
    return out


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
    if side == PositionSide.LONG:
        return trend_state == "ONE_WAY_UP" and pullback == "M15_LONG_PULLBACK"
    return trend_state == "ONE_WAY_DOWN" and pullback == "M15_SHORT_PULLBACK"


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
) -> IndicatorSnapshot | None:
    m15 = timeframe_indicators.get("15m") or []
    if m15:
        return m15[-1]
    return fallback[-1] if fallback else None


def _refine_stop_with_precision(side: PositionSide, stop: float, precision: object) -> float:
    if not isinstance(precision, dict):
        return stop
    if side == PositionSide.LONG:
        anchor = _float_or_none(precision.get("long_stop_anchor"))
        return min(stop, anchor) if anchor else stop
    anchor = _float_or_none(precision.get("short_stop_anchor"))
    return max(stop, anchor) if anchor else stop


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


def _leverage_for_signal(score: int, leverage_max: int) -> int:
    if score >= 85:
        return min(10, leverage_max)
    if score >= 78:
        return min(7, leverage_max)
    return min(5, leverage_max)


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
        "days": days,
        "summary": {
            "profit": total_profit,
            "loss": total_loss,
            "net_pnl": total_net,
            "fees": 0.0,
        },
    }
