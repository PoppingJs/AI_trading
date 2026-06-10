from __future__ import annotations

from collections.abc import Sequence

from ai_trading.config import RiskSettings
from ai_trading.models import Candle, IndicatorSnapshot, Position, PositionSide, RiskDecision, SignalAction, StrategySignal


class RiskManager:
    def __init__(self, settings: RiskSettings | None = None) -> None:
        self.settings = settings or RiskSettings()

    def plan_entry(
        self,
        signal: StrategySignal,
        candles: Sequence[Candle],
        equity: float,
        open_positions: Sequence[Position],
        daily_pnl: float,
        consecutive_losses: int,
        leverage: int | None = None,
    ) -> RiskDecision:
        leverage = min(leverage or self.settings.leverage_default, self.settings.leverage_max)
        reasons: list[str] = []

        if signal.action not in {SignalAction.ENTRY_LONG, SignalAction.ENTRY_SHORT}:
            return _blocked("signal is not an entry")
        if signal.indicators is None or signal.indicators.atr14 is None:
            return _blocked("missing ATR for stop buffer")
        if len(open_positions) >= self.settings.max_open_positions:
            return _blocked("max open positions reached")
        if daily_pnl <= -equity * self.settings.daily_loss_limit:
            return _blocked("daily loss limit reached")
        if consecutive_losses >= self.settings.max_consecutive_losses:
            return _blocked("consecutive loss cooldown active")

        side = PositionSide.LONG if signal.action == SignalAction.ENTRY_LONG else PositionSide.SHORT
        entry = signal.indicators.close
        stop = self._initial_stop(side, candles, signal.indicators, leverage)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return _blocked("invalid stop distance")

        stop_pct = stop_distance / entry
        risk_amount = equity * self.settings.risk_per_trade
        notional = risk_amount / stop_pct
        margin_required = notional / leverage

        symbol_margin = sum(position.notional / leverage for position in open_positions if position.symbol == signal.symbol)
        total_margin = sum(position.notional / leverage for position in open_positions)
        if symbol_margin + margin_required > equity * self.settings.single_symbol_margin_limit:
            margin_required = max(equity * self.settings.single_symbol_margin_limit - symbol_margin, 0.0)
            notional = margin_required * leverage
            reasons.append("sized down to single-symbol margin limit")
        if total_margin + margin_required > equity * self.settings.total_margin_limit:
            margin_required = max(equity * self.settings.total_margin_limit - total_margin, 0.0)
            notional = min(notional, margin_required * leverage)
            reasons.append("sized down to total margin limit")
        if notional <= 0:
            return _blocked("margin limits leave no tradable size")

        quantity = notional / entry
        take_profit_1, take_profit_2 = self._take_profit_prices(side, entry, stop_distance)
        return RiskDecision(
            allowed=True,
            quantity=quantity,
            notional=notional,
            margin_required=margin_required,
            stop_price=stop,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            reasons=tuple(reasons),
        )

    def should_time_stop(self, position: Position, current_price: float) -> bool:
        if position.bars_held < self.settings.time_stop_bars:
            return False
        if position.side == PositionSide.LONG:
            return current_price <= position.entry_price
        return current_price >= position.entry_price

    def _initial_stop(self, side: PositionSide, candles: Sequence[Candle], indicators: IndicatorSnapshot, leverage: int) -> float:
        lookback = candles[-10:]
        buffer = (indicators.atr14 or 0.0) * self.settings.atr_stop_buffer
        fixed_distance = _stop_pct_for_leverage(leverage)
        if side == PositionSide.LONG:
            structure_low = min(candle.low for candle in lookback)
            candidates = [indicators.close * (1 - fixed_distance), structure_low - buffer]
            if indicators.boll_lower is not None:
                candidates.append(indicators.boll_lower)
            below_entry = [price for price in candidates if price < indicators.close]
            return max(below_entry) if below_entry else indicators.close * (1 - fixed_distance)
        structure_high = max(candle.high for candle in lookback)
        candidates = [indicators.close * (1 + fixed_distance), structure_high + buffer]
        if indicators.boll_upper is not None:
            candidates.append(indicators.boll_upper)
        above_entry = [price for price in candidates if price > indicators.close]
        return min(above_entry) if above_entry else indicators.close * (1 + fixed_distance)

    def _take_profit_prices(self, side: PositionSide, entry: float, stop_distance: float) -> tuple[float, float]:
        if side == PositionSide.LONG:
            return (
                entry + stop_distance * self.settings.first_take_profit_r,
                entry + stop_distance * self.settings.second_take_profit_r,
            )
        return (
            entry - stop_distance * self.settings.first_take_profit_r,
            entry - stop_distance * self.settings.second_take_profit_r,
        )


def _blocked(reason: str) -> RiskDecision:
    return RiskDecision(
        allowed=False,
        quantity=0.0,
        notional=0.0,
        margin_required=0.0,
        stop_price=0.0,
        take_profit_1=0.0,
        take_profit_2=0.0,
        reasons=(reason,),
    )


def _stop_pct_for_leverage(leverage: int) -> float:
    if leverage >= 10:
        return 0.01
    if leverage >= 7:
        return 0.015
    return 0.02
