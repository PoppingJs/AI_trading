from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ai_trading.config import RiskSettings
from ai_trading.models import Candle, IndicatorSnapshot, Position, PositionSide, RiskDecision, SignalAction, StrategySignal


@dataclass(frozen=True)
class TradePlan:
    """A strategy-owned plan whose stop and targets are already final."""

    symbol: str
    side: PositionSide
    entry_price: float
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    leverage: int
    risk_factor: float = 1.0
    is_addition: bool = False


@dataclass(frozen=True)
class AccountRiskSnapshot:
    equity: float
    available_balance: float
    used_margin: float
    open_positions: tuple[Position, ...] = ()
    day_start_equity: float | None = None
    week_start_equity: float | None = None
    peak_equity: float | None = None
    daily_loss_locked: bool = False
    weekly_loss_locked: bool = False
    drawdown_locked: bool = False
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None


@dataclass(frozen=True)
class PortfolioRiskDecision:
    allowed: bool
    quantity: float = 0.0
    notional: float = 0.0
    margin_required: float = 0.0
    planned_risk_usdt: float = 0.0
    risk_budget_usdt: float = 0.0
    open_risk_before_usdt: float = 0.0
    open_risk_after_usdt: float = 0.0
    blocked_code: str = ""
    reasons: tuple[str, ...] = ()


class PortfolioRiskGate:
    """Single account-level entry and sizing authority.

    Strategies own entry, stop and target selection. This gate only decides
    whether new risk may be added and how large that risk may be.
    """

    def __init__(self, settings: RiskSettings | None = None) -> None:
        self.settings = settings or RiskSettings()

    def evaluate(
        self,
        plan: TradePlan,
        account: AccountRiskSnapshot,
        *,
        now: datetime | None = None,
    ) -> PortfolioRiskDecision:
        now = now or datetime.now(UTC)
        open_risk = current_open_risk_usdt(account.open_positions)

        invalid = _trade_plan_error(plan)
        if invalid:
            return _portfolio_blocked("INVALID_TRADE_PLAN", invalid, open_risk)
        if account.daily_loss_locked or _loss_limit_hit(
            account.equity,
            account.day_start_equity,
            self.settings.daily_loss_limit,
        ):
            return _portfolio_blocked("DAILY_LOSS_LIMIT", "daily loss limit reached", open_risk)
        if account.weekly_loss_locked or _loss_limit_hit(
            account.equity,
            account.week_start_equity,
            self.settings.weekly_loss_limit,
        ):
            return _portfolio_blocked("WEEKLY_LOSS_LIMIT", "weekly loss limit reached", open_risk)
        if account.drawdown_locked or _loss_limit_hit(
            account.equity,
            account.peak_equity,
            self.settings.max_drawdown_circuit_breaker,
        ):
            return _portfolio_blocked("MAX_DRAWDOWN", "maximum drawdown circuit breaker active", open_risk)
        if account.cooldown_until is not None and now < account.cooldown_until:
            return _portfolio_blocked("LOSS_COOLDOWN", "consecutive-loss cooldown active", open_risk)
        if account.consecutive_losses >= self.settings.max_consecutive_losses:
            return _portfolio_blocked("CONSECUTIVE_LOSSES", "consecutive loss limit reached", open_risk)

        symbol_positions = [
            position for position in account.open_positions
            if position.symbol == plan.symbol
        ]
        if not plan.is_addition and symbol_positions:
            return _portfolio_blocked("SYMBOL_ALREADY_OPEN", "symbol already has an open position", open_risk)
        if not plan.is_addition and len(account.open_positions) >= self.settings.max_open_positions:
            return _portfolio_blocked("MAX_POSITIONS", "max open positions reached", open_risk)

        stop_pct = abs(plan.entry_price - plan.stop_price) / plan.entry_price
        configured_risk = account.equity * max(self.settings.risk_per_trade, 0.0) * max(plan.risk_factor, 0.0)
        remaining_open_risk = max(
            account.equity * max(self.settings.total_open_risk_limit, 0.0) - open_risk,
            0.0,
        )
        risk_budget = min(configured_risk, remaining_open_risk)
        if risk_budget <= 0:
            return _portfolio_blocked("OPEN_RISK_LIMIT", "portfolio open-risk limit reached", open_risk)

        notional = risk_budget / stop_pct
        margin = notional / plan.leverage
        symbol_margin = sum(_position_margin(position) for position in symbol_positions)
        margin = min(
            margin,
            max(account.equity * self.settings.single_symbol_margin_limit - symbol_margin, 0.0),
            max(account.equity * self.settings.total_margin_limit - account.used_margin, 0.0),
            max(account.available_balance, 0.0),
        )
        if margin <= 0:
            return _portfolio_blocked("MARGIN_LIMIT", "margin limits leave no tradable size", open_risk)

        notional = margin * plan.leverage
        planned_risk = notional * stop_pct
        quantity = notional / plan.entry_price
        return PortfolioRiskDecision(
            allowed=True,
            quantity=quantity,
            notional=notional,
            margin_required=margin,
            planned_risk_usdt=planned_risk,
            risk_budget_usdt=risk_budget,
            open_risk_before_usdt=open_risk,
            open_risk_after_usdt=open_risk + planned_risk,
        )


def current_open_risk_usdt(positions: Sequence[Position]) -> float:
    total = 0.0
    for position in positions:
        loss_distance = (
            max(position.entry_price - position.stop_price, 0.0)
            if position.side == PositionSide.LONG
            else max(position.stop_price - position.entry_price, 0.0)
        )
        total += loss_distance * position.quantity * max(position.remaining_fraction, 0.0)
    return total


def risk_factor_for_quality(quality: str) -> float:
    """Quality may reduce risk, but never raise it above the configured budget."""

    return 1.0 if quality in {"S", "A"} else 0.5


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
        if signal.action not in {SignalAction.ENTRY_LONG, SignalAction.ENTRY_SHORT}:
            return _blocked("signal is not an entry")
        if signal.indicators is None or signal.indicators.atr14 is None:
            return _blocked("missing ATR for stop buffer")
        side = PositionSide.LONG if signal.action == SignalAction.ENTRY_LONG else PositionSide.SHORT
        entry = signal.indicators.close
        stop = self._initial_stop(side, candles, signal.indicators, leverage)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return _blocked("invalid stop distance")

        take_profit_1, take_profit_2 = self._take_profit_prices(side, entry, stop_distance)
        used_margin = sum(_position_margin(position) for position in open_positions)
        decision = PortfolioRiskGate(self.settings).evaluate(
            TradePlan(
                symbol=signal.symbol,
                side=side,
                entry_price=entry,
                stop_price=stop,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                leverage=leverage,
            ),
            AccountRiskSnapshot(
                equity=equity,
                available_balance=max(equity - used_margin, 0.0),
                used_margin=used_margin,
                open_positions=tuple(open_positions),
                day_start_equity=max(equity - daily_pnl, 0.0),
                consecutive_losses=consecutive_losses,
            ),
        )
        if not decision.allowed:
            return _blocked(decision.reasons[0] if decision.reasons else decision.blocked_code)
        return RiskDecision(
            allowed=True,
            quantity=decision.quantity,
            notional=decision.notional,
            margin_required=decision.margin_required,
            stop_price=stop,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            reasons=decision.reasons,
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


def _portfolio_blocked(
    code: str,
    reason: str,
    open_risk: float,
) -> PortfolioRiskDecision:
    return PortfolioRiskDecision(
        allowed=False,
        blocked_code=code,
        reasons=(reason,),
        open_risk_before_usdt=open_risk,
        open_risk_after_usdt=open_risk,
    )


def _trade_plan_error(plan: TradePlan) -> str | None:
    if plan.entry_price <= 0:
        return "entry price must be positive"
    if plan.leverage <= 0:
        return "leverage must be positive"
    if plan.side == PositionSide.LONG and plan.stop_price >= plan.entry_price:
        return "long stop must be below entry"
    if plan.side == PositionSide.SHORT and plan.stop_price <= plan.entry_price:
        return "short stop must be above entry"
    return None


def _loss_limit_hit(
    equity: float,
    reference_equity: float | None,
    limit: float,
) -> bool:
    if reference_equity is None or reference_equity <= 0 or limit <= 0:
        return False
    return equity <= reference_equity * (1 - limit)


def _position_margin(position: Position) -> float:
    stored = position.metadata.get("margin_usdt")
    if stored is not None:
        return max(float(stored), 0.0)
    leverage = max(int(position.metadata.get("leverage", 1)), 1)
    return position.notional / leverage


def _stop_pct_for_leverage(leverage: int) -> float:
    if leverage >= 10:
        return 0.01
    if leverage >= 7:
        return 0.015
    return 0.02
