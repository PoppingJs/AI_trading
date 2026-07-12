from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Iterable

from ai_trading.config import AppSettings, ExecutionSettings, RiskSettings, StrategySettings
from ai_trading.indicators import build_indicators
from ai_trading.models import Candle, DerivativesSnapshot, Position, PositionSide, SignalAction, Trade
from ai_trading.risk import RiskManager
from ai_trading.strategy import CompositeStrategy
from ai_trading.paper import PaperFill, PaperTradingEngine, SUPPORTED_TIMEFRAMES


@dataclass(frozen=True)
class BacktestResult:
    starting_equity: float
    ending_equity: float
    trades: tuple[Trade, ...]
    max_drawdown: float
    win_rate: float
    total_return: float
    notes: tuple[str, ...] = ()


@dataclass
class LegacyBacktestEngine:
    symbol: str
    starting_equity: float = 10_000.0
    strategy_settings: StrategySettings = field(default_factory=StrategySettings)
    risk_settings: RiskSettings = field(default_factory=RiskSettings)
    execution_settings: ExecutionSettings = field(default_factory=ExecutionSettings)

    def run(self, candles: list[Candle], derivatives: list[DerivativesSnapshot] | None = None) -> BacktestResult:
        warmup_bars = max(self.strategy_settings.ma_trend, 200)
        if len(candles) < warmup_bars + 5:
            return BacktestResult(
                starting_equity=self.starting_equity,
                ending_equity=self.starting_equity,
                trades=(),
                max_drawdown=0.0,
                win_rate=0.0,
                total_return=0.0,
                notes=(
                    "legacy single-timeframe baseline",
                    "not enough data for backtest",
                ),
            )

        indicators = build_indicators(
            candles,
            derivatives,
            ema_fast=self.strategy_settings.ema_fast,
            ema_slow=self.strategy_settings.ema_slow,
            ma_trend=self.strategy_settings.ma_trend,
            bollinger_window=self.strategy_settings.bollinger_window,
            bollinger_stddev=self.strategy_settings.bollinger_stddev,
            rsi_window=self.strategy_settings.rsi_window,
            atr_window=self.strategy_settings.atr_window,
            volume_window=self.strategy_settings.volume_window,
        )
        strategy = CompositeStrategy(self.strategy_settings)
        risk = RiskManager(self.risk_settings)
        equity = self.starting_equity
        peak_equity = equity
        max_drawdown = 0.0
        daily_pnl = 0.0
        consecutive_losses = 0
        position: Position | None = None
        trades: list[Trade] = []

        warmup = warmup_bars + 1
        for idx in range(warmup, len(candles)):
            candle = candles[idx]
            current_indicator = indicators[idx]
            if position is not None:
                position.bars_held += 1
                equity, closed_trades, position = self._manage_position(
                    position,
                    candle,
                    current_indicator,
                    strategy,
                    risk,
                    equity,
                )
                if closed_trades:
                    trades.extend(closed_trades)
                    pnl = sum(trade.pnl for trade in closed_trades)
                    daily_pnl += pnl
                    consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0

            if position is None:
                signal = strategy.generate_signal(self.symbol, candles[: idx + 1], indicators[: idx + 1])
                decision = risk.plan_entry(
                    signal,
                    candles[: idx + 1],
                    equity,
                    [],
                    daily_pnl,
                    consecutive_losses,
                )
                if decision.allowed:
                    side = PositionSide.LONG if signal.action == SignalAction.ENTRY_LONG else PositionSide.SHORT
                    entry_price = _slipped(candle.close, side, self.execution_settings.slippage_rate, entering=True)
                    first_leg_quantity = decision.quantity * 0.45
                    equity -= entry_price * first_leg_quantity * self.execution_settings.taker_fee_rate
                    position = Position(
                        symbol=self.symbol,
                        side=side,
                        entry_price=entry_price,
                        quantity=first_leg_quantity,
                        opened_at=candle.timestamp,
                        stop_price=decision.stop_price,
                        take_profit_1=decision.take_profit_1,
                        take_profit_2=decision.take_profit_2,
                        metadata={"entry_score": signal.score, "entry_reasons": signal.reasons},
                    )

            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
            max_drawdown = max(max_drawdown, drawdown)

        if position is not None:
            final_candle = candles[-1]
            exit_price = _slipped(final_candle.close, position.side, self.execution_settings.slippage_rate, entering=False)
            pnl = _position_pnl(position, exit_price)
            fee = exit_price * position.quantity * position.remaining_fraction * self.execution_settings.taker_fee_rate
            equity += pnl - fee
            trades.append(
                Trade(
                    symbol=position.symbol,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity * position.remaining_fraction,
                    opened_at=position.opened_at,
                    closed_at=final_candle.timestamp,
                    pnl=pnl - fee,
                    reason="final close",
                )
            )

        winning = [trade for trade in trades if trade.pnl > 0]
        return BacktestResult(
            starting_equity=self.starting_equity,
            ending_equity=equity,
            trades=tuple(trades),
            max_drawdown=max_drawdown,
            win_rate=len(winning) / len(trades) if trades else 0.0,
            total_return=(equity - self.starting_equity) / self.starting_equity,
            notes=("legacy single-timeframe baseline",),
        )

    def _manage_position(
        self,
        position: Position,
        candle: Candle,
        indicator,
        strategy: CompositeStrategy,
        risk: RiskManager,
        equity: float,
    ) -> tuple[float, list[Trade], Position | None]:
        closed: list[Trade] = []
        exit_signal = strategy.exit_signal(position.side.value, indicator)
        stop_hit = candle.low <= position.stop_price if position.side == PositionSide.LONG else candle.high >= position.stop_price
        time_stop = risk.should_time_stop(position, candle.close)

        if not position.first_tp_done and _tp_hit(position, candle, target=1):
            equity, trade = self._close_fraction(position, position.take_profit_1, self.risk_settings.first_take_profit_fraction, candle.timestamp, "take profit 1", equity)
            closed.append(trade)
            position.remaining_fraction -= self.risk_settings.first_take_profit_fraction
            position.first_tp_done = True
            position.stop_price = position.entry_price

        if not position.second_tp_done and _tp_hit(position, candle, target=2):
            equity, trade = self._close_fraction(position, position.take_profit_2, self.risk_settings.second_take_profit_fraction, candle.timestamp, "take profit 2", equity)
            closed.append(trade)
            position.remaining_fraction -= self.risk_settings.second_take_profit_fraction
            position.second_tp_done = True
            position.stop_price = _trail_stop(position, indicator)

        if stop_hit or time_stop or exit_signal is not None:
            reason = "stop loss" if stop_hit else "time stop" if time_stop else "; ".join(exit_signal.reasons if exit_signal else ())
            exit_price = position.stop_price if stop_hit else candle.close
            equity, trade = self._close_fraction(position, exit_price, position.remaining_fraction, candle.timestamp, reason, equity)
            closed.append(trade)
            return equity, closed, None

        if position.second_tp_done:
            position.stop_price = _trail_stop(position, indicator)
        return equity, closed, position

    def _close_fraction(
        self,
        position: Position,
        price: float,
        fraction: float,
        timestamp,
        reason: str,
        equity: float,
    ) -> tuple[float, Trade]:
        quantity = position.quantity * max(fraction, 0.0)
        exit_price = _slipped(price, position.side, self.execution_settings.slippage_rate, entering=False)
        pnl = _raw_pnl(position.side, position.entry_price, exit_price, quantity)
        fee = exit_price * quantity * self.execution_settings.taker_fee_rate
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=quantity,
            opened_at=position.opened_at,
            closed_at=timestamp,
            pnl=pnl - fee,
            reason=reason,
        )
        return equity + trade.pnl, trade


@dataclass(frozen=True)
class PortfolioBacktestResult:
    starting_equity: float
    ending_equity: float
    trades: tuple[Trade, ...]
    fills: tuple[PaperFill, ...]
    equity_curve: tuple[tuple[datetime, float], ...]
    max_drawdown: float
    win_rate: float
    total_return: float
    per_symbol_pnl: dict[str, float]
    notes: tuple[str, ...] = ()


@dataclass
class _ReplayClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


class _ReplayMarketData:
    def __init__(self) -> None:
        self.candles: dict[str, dict[str, list[Candle]]] = {}
        self.prices: dict[str, float] = {}

    async def klines(
        self,
        symbol: str,
        interval: str = "15m",
        *,
        limit: int = 500,
        **_: object,
    ) -> list[Candle]:
        return list(self.candles.get(symbol, {}).get(interval, []))[-limit:]

    async def mark_prices(self, symbols: Iterable[str] | None = None) -> dict[str, float]:
        wanted = set(symbols or self.prices)
        return {
            symbol: price
            for symbol, price in self.prices.items()
            if symbol in wanted
        }

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True)
class _PreparedHistory:
    base_timeframe: str
    base_seconds: int
    candles: dict[str, list[Candle]]
    derivatives: dict[str, list[DerivativesSnapshot]]


@dataclass
class ProductionPortfolioBacktestEngine:
    """Point-in-time replay of the production paper-trading decision path."""

    starting_equity: float = 10_000.0
    settings: AppSettings = field(default_factory=AppSettings)

    def run(
        self,
        market_data: dict[
            str,
            tuple[list[Candle], list[DerivativesSnapshot] | None],
        ],
    ) -> PortfolioBacktestResult:
        return asyncio.run(self._run_async(market_data))

    async def _run_async(
        self,
        market_data: dict[
            str,
            tuple[list[Candle], list[DerivativesSnapshot] | None],
        ],
    ) -> PortfolioBacktestResult:
        prepared = {
            symbol.upper(): _prepare_history(candles, derivatives or [])
            for symbol, (candles, derivatives) in market_data.items()
            if candles
        }
        if not prepared:
            return _empty_portfolio_result(
                self.starting_equity,
                "not enough historical market data",
            )
        base_intervals = {
            history.base_seconds for history in prepared.values()
        }
        if len(base_intervals) != 1:
            return _empty_portfolio_result(
                self.starting_equity,
                "portfolio replay requires one aligned base timeframe",
            )

        timeline = sorted(
            {
                candle.timestamp
                for history in prepared.values()
                for candle in history.candles[history.base_timeframe]
            }
        )
        if not timeline:
            return _empty_portfolio_result(
                self.starting_equity,
                "not enough historical market data",
            )

        initial_time = _as_utc(timeline[0])
        clock = _ReplayClock(initial_time)
        replay_market = _ReplayMarketData()
        symbols = sorted(prepared)
        paper = PaperTradingEngine(
            self.settings,
            starting_balance=self.starting_equity,
            symbols=symbols,
            interval="1h",
            market_data=replay_market,  # type: ignore[arg-type]
            clock=clock,
            fill_price_resolver=lambda price, side, entering: _slipped(
                price,
                side,
                self.settings.execution.slippage_rate,
                entering=entering,
            ),
        )
        paper.auto_trade = True
        equity_curve: list[tuple[datetime, float]] = []
        base_by_time = {
            symbol: {
                _as_utc(candle.timestamp): candle
                for candle in history.candles[history.base_timeframe]
            }
            for symbol, history in prepared.items()
        }
        charged_funding: set[tuple[str, datetime]] = set()

        for timestamp in timeline:
            event_time = _as_utc(timestamp)
            active = {
                symbol: candles[event_time]
                for symbol, candles in base_by_time.items()
                if event_time in candles
            }
            if not active:
                continue
            clock.value = event_time
            for symbol, candle in active.items():
                paper.latest_prices[symbol] = candle.open
                replay_market.prices[symbol] = candle.open
            paper._account_risk_snapshot(paper.status(), now=clock.value)

            # Existing stops are processed before a pending signal may enter at
            # this bar's open. Signals built at the prior close therefore never
            # fill retroactively at that same close.
            paper._manage_open_positions()
            suspended_signals = {
                symbol: paper.latest_signals.pop(symbol)
                for symbol in list(paper.latest_signals)
                if symbol not in active
            }
            try:
                paper._refresh_live_entry_timing()
                await paper._auto_trade_once()
            finally:
                paper.latest_signals.update(suspended_signals)
            _append_equity_point(paper, clock.value, equity_curve)

            for symbol, candle in active.items():
                position = paper.account.positions.get(symbol)
                side = position.side if position is not None else None
                for offset, price in _conservative_intrabar_path(candle, side):
                    clock.value = event_time + timedelta(
                        seconds=prepared[symbol].base_seconds * offset,
                    )
                    _replay_intrabar_price(
                        paper,
                        replay_market,
                        symbol,
                        price,
                    )
                    _append_equity_point(paper, clock.value, equity_curve)

            close_time = max(
                event_time
                + timedelta(seconds=prepared[symbol].base_seconds)
                for symbol in active
            )
            clock.value = close_time
            for symbol, candle in active.items():
                paper.latest_prices[symbol] = candle.close
                replay_market.prices[symbol] = candle.close
            paper._manage_open_positions()

            for symbol in active:
                history = prepared[symbol]
                visible_candles = {
                    timeframe: _visible_candles(
                        candles,
                        timeframe,
                        close_time,
                    )
                    for timeframe, candles in history.candles.items()
                }
                visible_derivatives = {
                    timeframe: _visible_derivatives(
                        history.derivatives.get(timeframe, []),
                        visible_candles.get(timeframe, []),
                    )
                    for timeframe in visible_candles
                }
                paper._timeframe_candles[symbol] = visible_candles
                paper._timeframe_derivatives[symbol] = visible_derivatives
                replay_market.candles[symbol] = visible_candles
                paper._publish_symbol_from_cache(symbol)
                _apply_replay_funding(
                    paper,
                    symbol,
                    visible_derivatives,
                    close_time,
                    charged_funding,
                )

            # Closed-timeframe structure exits may act at this close. New
            # entries remain pending until the next event open.
            paper._manage_open_positions()
            _append_equity_point(paper, clock.value, equity_curve)

        final_time = max(
            _as_utc(candle.timestamp)
            + timedelta(seconds=history.base_seconds)
            for history in prepared.values()
            for candle in history.candles[history.base_timeframe][-1:]
        )
        clock.value = final_time
        for position in list(paper.account.positions.values()):
            price = paper.latest_prices.get(position.symbol, position.entry_price)
            paper._close_position_unlocked(
                position,
                price,
                "final backtest close",
            )
        _append_equity_point(paper, clock.value, equity_curve)
        return _portfolio_result_from_paper(
            paper,
            self.starting_equity,
            equity_curve,
            notes=(
                "production-parity strategy path",
                "signals execute no earlier than the next market event",
                "OHLC ambiguity uses adverse-first conservative replay",
                "fixed historical universe may contain survivorship bias",
            ),
        )


@dataclass
class BacktestEngine:
    """Single-symbol production replay with the legacy engine still available."""

    symbol: str
    starting_equity: float = 10_000.0
    strategy_settings: StrategySettings = field(default_factory=StrategySettings)
    risk_settings: RiskSettings = field(default_factory=RiskSettings)
    execution_settings: ExecutionSettings = field(default_factory=ExecutionSettings)
    mode: str = "production"

    def run(
        self,
        candles: list[Candle],
        derivatives: list[DerivativesSnapshot] | None = None,
    ) -> BacktestResult:
        if self.mode.lower() == "legacy":
            return LegacyBacktestEngine(
                symbol=self.symbol,
                starting_equity=self.starting_equity,
                strategy_settings=self.strategy_settings,
                risk_settings=self.risk_settings,
                execution_settings=self.execution_settings,
            ).run(candles, derivatives)
        settings = AppSettings(
            strategy=self.strategy_settings,
            risk=self.risk_settings,
            execution=self.execution_settings,
            timeframes=list(SUPPORTED_TIMEFRAMES),
        )
        result = ProductionPortfolioBacktestEngine(
            starting_equity=self.starting_equity,
            settings=settings,
        ).run({self.symbol: (candles, derivatives)})
        return BacktestResult(
            starting_equity=result.starting_equity,
            ending_equity=result.ending_equity,
            trades=result.trades,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            total_return=result.total_return,
            notes=result.notes,
        )


def run_portfolio_backtest(
    market_data: dict[
        str,
        tuple[list[Candle], list[DerivativesSnapshot] | None],
    ],
    starting_equity: float = 10_000.0,
    *,
    settings: AppSettings | None = None,
) -> PortfolioBacktestResult:
    """Run one shared account across every symbol on a common event clock."""

    return ProductionPortfolioBacktestEngine(
        starting_equity=starting_equity,
        settings=settings or AppSettings(),
    ).run(market_data)


def run_batch_single_symbol_backtests(
    market_data: dict[
        str,
        tuple[list[Candle], list[DerivativesSnapshot] | None],
    ],
    starting_equity: float = 10_000.0,
) -> dict[str, BacktestResult]:
    """Compatibility helper for the old independent-per-symbol research view."""

    return {
        symbol: LegacyBacktestEngine(
            symbol=symbol,
            starting_equity=starting_equity,
        ).run(candles, derivatives)
        for symbol, (candles, derivatives) in market_data.items()
    }


_TIMEFRAME_SECONDS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


def _prepare_history(
    candles: list[Candle],
    derivatives: list[DerivativesSnapshot],
) -> _PreparedHistory:
    normalized = sorted(
        (
            Candle(
                timestamp=_as_utc(candle.timestamp),
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            for candle in candles
        ),
        key=lambda candle: candle.timestamp,
    )
    base_seconds = _infer_interval_seconds(normalized)
    base_timeframe = _timeframe_for_seconds(base_seconds)
    by_timeframe: dict[str, list[Candle]] = {base_timeframe: normalized}
    for timeframe, seconds in _TIMEFRAME_SECONDS.items():
        if seconds < base_seconds or timeframe == base_timeframe:
            continue
        by_timeframe[timeframe] = _aggregate_candles(
            normalized,
            base_seconds=base_seconds,
            target_seconds=seconds,
        )

    normalized_derivatives = sorted(
        (
            DerivativesSnapshot(
                timestamp=_as_utc(snapshot.timestamp),
                open_interest=snapshot.open_interest,
                long_short_ratio=snapshot.long_short_ratio,
                funding_rate=snapshot.funding_rate,
            )
            for snapshot in derivatives
        ),
        key=lambda snapshot: snapshot.timestamp,
    )
    derivatives_by_timeframe: dict[str, list[DerivativesSnapshot]] = {}
    for timeframe, timeframe_candles in by_timeframe.items():
        derivatives_by_timeframe[timeframe] = _aggregate_derivatives(
            normalized_derivatives,
            timeframe_candles,
            _TIMEFRAME_SECONDS[timeframe],
        )
    return _PreparedHistory(
        base_timeframe=base_timeframe,
        base_seconds=base_seconds,
        candles=by_timeframe,
        derivatives=derivatives_by_timeframe,
    )


def _infer_interval_seconds(candles: list[Candle]) -> int:
    if len(candles) < 2:
        return _TIMEFRAME_SECONDS["15m"]
    differences = [
        int((current.timestamp - previous.timestamp).total_seconds())
        for previous, current in zip(candles, candles[1:])
        if current.timestamp > previous.timestamp
    ]
    if not differences:
        return _TIMEFRAME_SECONDS["15m"]
    observed = int(median(differences))
    return min(
        _TIMEFRAME_SECONDS.values(),
        key=lambda seconds: abs(seconds - observed),
    )


def _timeframe_for_seconds(seconds: int) -> str:
    return min(
        _TIMEFRAME_SECONDS,
        key=lambda timeframe: abs(_TIMEFRAME_SECONDS[timeframe] - seconds),
    )


def _aggregate_candles(
    candles: list[Candle],
    *,
    base_seconds: int,
    target_seconds: int,
) -> list[Candle]:
    if target_seconds == base_seconds:
        return list(candles)
    expected = max(target_seconds // base_seconds, 1)
    groups: dict[datetime, list[Candle]] = {}
    for candle in candles:
        bucket = datetime.fromtimestamp(
            int(candle.timestamp.timestamp() // target_seconds)
            * target_seconds,
            tz=UTC,
        )
        groups.setdefault(bucket, []).append(candle)
    aggregated: list[Candle] = []
    for bucket in sorted(groups):
        values = sorted(groups[bucket], key=lambda candle: candle.timestamp)
        if len(values) != expected:
            continue
        if any(
            current.timestamp - previous.timestamp
            != timedelta(seconds=base_seconds)
            for previous, current in zip(values, values[1:])
        ):
            continue
        aggregated.append(
            Candle(
                timestamp=bucket,
                open=values[0].open,
                high=max(candle.high for candle in values),
                low=min(candle.low for candle in values),
                close=values[-1].close,
                volume=sum(candle.volume for candle in values),
            )
        )
    return aggregated


def _aggregate_derivatives(
    derivatives: list[DerivativesSnapshot],
    candles: list[Candle],
    timeframe_seconds: int,
) -> list[DerivativesSnapshot]:
    if not derivatives or not candles:
        return []
    snapshots: list[DerivativesSnapshot] = []
    cursor = 0
    latest: DerivativesSnapshot | None = None
    for candle in candles:
        close_time = candle.timestamp + timedelta(seconds=timeframe_seconds)
        while cursor < len(derivatives) and derivatives[cursor].timestamp < close_time:
            latest = derivatives[cursor]
            cursor += 1
        if latest is None or latest.timestamp < candle.timestamp:
            continue
        snapshots.append(
            DerivativesSnapshot(
                timestamp=candle.timestamp,
                open_interest=latest.open_interest,
                long_short_ratio=latest.long_short_ratio,
                funding_rate=latest.funding_rate,
            )
        )
    return snapshots


def _visible_candles(
    candles: list[Candle],
    timeframe: str,
    event_time: datetime,
) -> list[Candle]:
    seconds = _TIMEFRAME_SECONDS[timeframe]
    return [
        candle
        for candle in candles
        if candle.timestamp + timedelta(seconds=seconds) <= event_time
    ][-500:]


def _visible_derivatives(
    derivatives: list[DerivativesSnapshot],
    candles: list[Candle],
) -> list[DerivativesSnapshot]:
    wanted = {candle.timestamp for candle in candles}
    return [
        snapshot
        for snapshot in derivatives
        if snapshot.timestamp in wanted
    ][-500:]


def _conservative_intrabar_path(
    candle: Candle,
    side: PositionSide | None,
) -> tuple[tuple[float, float], ...]:
    prices = (
        (candle.high, candle.low)
        if side == PositionSide.SHORT
        else (candle.low, candle.high)
    )
    unique: list[tuple[float, float]] = []
    for offset, price in zip((1 / 3, 2 / 3), prices):
        if price in {candle.open, candle.close} or any(
            existing_price == price for _, existing_price in unique
        ):
            continue
        unique.append((offset, price))
    return tuple(unique)


def _replay_intrabar_price(
    paper: PaperTradingEngine,
    replay_market: _ReplayMarketData,
    symbol: str,
    extreme_price: float,
) -> None:
    position = paper.account.positions.get(symbol)
    if position is None:
        paper.latest_prices[symbol] = extreme_price
        replay_market.prices[symbol] = extreme_price
        return

    trigger_prices: list[float] = []
    if position.side == PositionSide.LONG:
        if extreme_price <= position.stop_price:
            trigger_prices.append(position.stop_price)
        else:
            if (
                not position.first_tp_done
                and extreme_price >= position.take_profit_1
            ):
                trigger_prices.append(position.take_profit_1)
            if extreme_price >= position.take_profit_2:
                trigger_prices.append(position.take_profit_2)
            trigger_prices.sort()
    else:
        if extreme_price >= position.stop_price:
            trigger_prices.append(position.stop_price)
        else:
            if (
                not position.first_tp_done
                and extreme_price <= position.take_profit_1
            ):
                trigger_prices.append(position.take_profit_1)
            if extreme_price <= position.take_profit_2:
                trigger_prices.append(position.take_profit_2)
            trigger_prices.sort(reverse=True)

    for trigger_price in trigger_prices:
        if symbol not in paper.account.positions:
            break
        paper.latest_prices[symbol] = trigger_price
        replay_market.prices[symbol] = trigger_price
        paper._manage_open_positions()
    if symbol in paper.account.positions:
        paper.latest_prices[symbol] = extreme_price
        replay_market.prices[symbol] = extreme_price
        paper._manage_open_positions()


def _append_equity_point(
    paper: PaperTradingEngine,
    timestamp: datetime,
    curve: list[tuple[datetime, float]],
) -> None:
    equity = float(paper.status()["equity"])
    point = (_as_utc(timestamp), equity)
    if curve and curve[-1][0] == point[0]:
        curve[-1] = point
    else:
        curve.append(point)


def _apply_replay_funding(
    paper: PaperTradingEngine,
    symbol: str,
    derivatives: dict[str, list[DerivativesSnapshot]],
    event_time: datetime,
    charged: set[tuple[str, datetime]],
) -> None:
    if event_time.minute != 0 or event_time.hour % 8 != 0:
        return
    key = (symbol, event_time)
    position = paper.account.positions.get(symbol)
    snapshots = derivatives.get("1h") or derivatives.get("15m") or []
    if key in charged or position is None or not snapshots:
        return
    rate = snapshots[-1].funding_rate
    if rate is None:
        return
    mark = paper.latest_prices.get(symbol, position.entry_price)
    remaining_quantity = position.quantity * max(position.remaining_fraction, 0.0)
    notional = mark * remaining_quantity
    realized = (
        -notional * rate
        if position.side == PositionSide.LONG
        else notional * rate
    )
    paper.account.wallet_balance += realized
    paper.account.realized_pnl += realized
    paper.account.fills.append(
        PaperFill(
            timestamp=event_time,
            symbol=symbol,
            side=position.side,
            action="FUNDING",
            price=mark,
            entry_price=position.entry_price,
            quantity=remaining_quantity,
            realized_pnl=realized,
            fee=0.0,
            reason=f"funding settlement rate={rate}",
            leverage=int(position.metadata.get("leverage", 1)),
            margin_usdt=0.0,
            stop_price=position.stop_price,
            take_profit_1=position.take_profit_1,
            take_profit_2=position.take_profit_2,
            opened_at=position.opened_at,
            closed_at=event_time,
        )
    )
    charged.add(key)


def _portfolio_result_from_paper(
    paper: PaperTradingEngine,
    starting_equity: float,
    equity_curve: list[tuple[datetime, float]],
    *,
    notes: tuple[str, ...],
) -> PortfolioBacktestResult:
    trades = tuple(
        Trade(
            symbol=fill.symbol,
            side=fill.side,
            entry_price=fill.entry_price,
            exit_price=fill.price,
            quantity=fill.quantity,
            opened_at=fill.opened_at,
            closed_at=fill.closed_at or fill.timestamp,
            pnl=fill.realized_pnl,
            reason=fill.reason,
        )
        for fill in paper.account.fills
        if fill.action in {"CLOSE", "PARTIAL_CLOSE"}
    )
    lifecycle_pnl: dict[tuple[str, datetime], float] = {}
    completed: set[tuple[str, datetime]] = set()
    per_symbol: dict[str, float] = {}
    for fill in paper.account.fills:
        key = (fill.symbol, fill.opened_at)
        value = (
            -fill.fee
            if fill.action in {"OPEN", "ADD"}
            else fill.realized_pnl
        )
        lifecycle_pnl[key] = lifecycle_pnl.get(key, 0.0) + value
        per_symbol[fill.symbol] = per_symbol.get(fill.symbol, 0.0) + value
        if fill.action == "CLOSE":
            completed.add(key)
    outcomes = [lifecycle_pnl[key] for key in completed]
    peak = starting_equity
    max_drawdown = 0.0
    for _, equity in equity_curve:
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    ending_equity = paper.account.wallet_balance
    return PortfolioBacktestResult(
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        trades=trades,
        fills=tuple(paper.account.fills),
        equity_curve=tuple(equity_curve),
        max_drawdown=max_drawdown,
        win_rate=(
            sum(outcome > 0 for outcome in outcomes) / len(outcomes)
            if outcomes
            else 0.0
        ),
        total_return=(ending_equity - starting_equity) / starting_equity,
        per_symbol_pnl=per_symbol,
        notes=notes,
    )


def _empty_portfolio_result(
    starting_equity: float,
    note: str,
) -> PortfolioBacktestResult:
    return PortfolioBacktestResult(
        starting_equity=starting_equity,
        ending_equity=starting_equity,
        trades=(),
        fills=(),
        equity_curve=(),
        max_drawdown=0.0,
        win_rate=0.0,
        total_return=0.0,
        per_symbol_pnl={},
        notes=(note,),
    )


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(UTC)
        if value.tzinfo is not None
        else value.replace(tzinfo=UTC)
    )


def _tp_hit(position: Position, candle: Candle, target: int) -> bool:
    price = position.take_profit_1 if target == 1 else position.take_profit_2
    if position.side == PositionSide.LONG:
        return candle.high >= price
    return candle.low <= price


def _trail_stop(position: Position, indicator) -> float:
    if indicator.ema20 is None:
        return position.stop_price
    if position.side == PositionSide.LONG:
        return max(position.stop_price, indicator.ema20)
    return min(position.stop_price, indicator.ema20)


def _position_pnl(position: Position, exit_price: float) -> float:
    return _raw_pnl(position.side, position.entry_price, exit_price, position.quantity * position.remaining_fraction)


def _raw_pnl(side: PositionSide, entry_price: float, exit_price: float, quantity: float) -> float:
    if side == PositionSide.LONG:
        return (exit_price - entry_price) * quantity
    return (entry_price - exit_price) * quantity


def _slipped(price: float, side: PositionSide, slippage_rate: float, *, entering: bool) -> float:
    if side == PositionSide.LONG:
        return price * (1 + slippage_rate if entering else 1 - slippage_rate)
    return price * (1 - slippage_rate if entering else 1 + slippage_rate)
