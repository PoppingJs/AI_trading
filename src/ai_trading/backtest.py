from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ai_trading.config import ExecutionSettings, RiskSettings, StrategySettings
from ai_trading.indicators import build_indicators
from ai_trading.models import Candle, DerivativesSnapshot, Position, PositionSide, SignalAction, Trade
from ai_trading.risk import RiskManager
from ai_trading.strategy import CompositeStrategy


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
class BacktestEngine:
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
                notes=("not enough data for backtest",),
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


def run_portfolio_backtest(
    market_data: dict[str, tuple[list[Candle], list[DerivativesSnapshot] | None]],
    starting_equity: float = 10_000.0,
) -> dict[str, BacktestResult]:
    return {
        symbol: BacktestEngine(symbol=symbol, starting_equity=starting_equity).run(candles, derivatives)
        for symbol, (candles, derivatives) in market_data.items()
    }


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
