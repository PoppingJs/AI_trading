from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_trading.config import RiskSettings
from ai_trading.models import (
    PLAN_TARGET_MODE_BOUNDED_TARGETS,
    PLAN_TARGET_MODE_LEGACY_BOUNDED_DUAL_TP,
    PLAN_TARGET_MODE_OPEN_SPACE,
    POSITION_SCHEMA_VERSION,
    Position,
    PositionSide,
    RiskDecision,
)
from ai_trading.risk import AccountRiskSnapshot, PortfolioRiskGate, TradePlan


def _account() -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        equity=1000.0,
        available_balance=1000.0,
        used_margin=0.0,
    )


def test_position_and_risk_decision_support_open_space_targets() -> None:
    position = Position(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
        opened_at=datetime(2026, 8, 7, tzinfo=UTC),
        stop_price=98.0,
        take_profit_1=None,
        take_profit_2=None,
        plan_target_mode=PLAN_TARGET_MODE_OPEN_SPACE,
    )
    decision = RiskDecision(
        allowed=True,
        quantity=1.0,
        notional=100.0,
        margin_required=20.0,
        stop_price=98.0,
        take_profit_1=None,
        take_profit_2=None,
        plan_target_mode=PLAN_TARGET_MODE_OPEN_SPACE,
    )

    assert position.position_schema_version == POSITION_SCHEMA_VERSION == 2
    assert position.take_profit_1 is None
    assert position.take_profit_2 is None
    assert decision.take_profit_1 is None
    assert decision.take_profit_2 is None


def test_legacy_dual_tp_is_the_default_target_mode() -> None:
    position = Position(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
        opened_at=datetime(2026, 8, 7, tzinfo=UTC),
        stop_price=98.0,
        take_profit_1=102.0,
        take_profit_2=104.0,
    )
    plan = TradePlan(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        stop_price=98.0,
        take_profit_1=102.0,
        take_profit_2=104.0,
        leverage=5,
    )

    assert position.plan_target_mode == PLAN_TARGET_MODE_LEGACY_BOUNDED_DUAL_TP
    assert plan.plan_target_mode == PLAN_TARGET_MODE_LEGACY_BOUNDED_DUAL_TP
    assert PortfolioRiskGate(RiskSettings()).evaluate(plan, _account()).allowed


def test_open_space_plan_with_empty_targets_keeps_stop_and_cost_risk_sizing() -> None:
    decision = PortfolioRiskGate(RiskSettings()).evaluate(
        TradePlan(
            symbol="TESTUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_1=None,
            take_profit_2=None,
            leverage=5,
            adverse_cost_rate=0.001,
            plan_target_mode=PLAN_TARGET_MODE_OPEN_SPACE,
        ),
        _account(),
    )

    assert decision.allowed
    assert decision.blocked_code == ""
    assert decision.risk_budget_usdt == pytest.approx(5.0)
    assert decision.planned_risk_usdt == pytest.approx(5.0)
    assert decision.notional == pytest.approx(5.0 / 0.021)
    assert decision.margin_required == pytest.approx((5.0 / 0.021) / 5)


@pytest.mark.parametrize(
    ("take_profit_1", "take_profit_2"),
    [(102.0, None), (None, 104.0), (102.0, 104.0)],
)
def test_open_space_plan_rejects_any_take_profit_target(
    take_profit_1: float | None,
    take_profit_2: float | None,
) -> None:
    decision = PortfolioRiskGate(RiskSettings()).evaluate(
        TradePlan(
            symbol="TESTUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            leverage=5,
            plan_target_mode=PLAN_TARGET_MODE_OPEN_SPACE,
        ),
        _account(),
    )

    assert not decision.allowed
    assert decision.blocked_code == "INVALID_TRADE_PLAN"
    assert decision.reasons == (
        "open-space plan must not define take-profit targets",
    )


@pytest.mark.parametrize(
    "target_mode",
    [PLAN_TARGET_MODE_BOUNDED_TARGETS, PLAN_TARGET_MODE_LEGACY_BOUNDED_DUAL_TP],
)
def test_bounded_target_modes_still_require_two_finite_targets(
    target_mode: str,
) -> None:
    decision = PortfolioRiskGate(RiskSettings()).evaluate(
        TradePlan(
            symbol="TESTUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_1=None,
            take_profit_2=None,
            leverage=5,
            plan_target_mode=target_mode,
        ),
        _account(),
    )

    assert not decision.allowed
    assert decision.blocked_code == "INVALID_TRADE_PLAN"


@pytest.mark.parametrize(
    "target_mode",
    [PLAN_TARGET_MODE_BOUNDED_TARGETS, PLAN_TARGET_MODE_LEGACY_BOUNDED_DUAL_TP],
)
@pytest.mark.parametrize(
    ("side", "take_profit_1", "take_profit_2", "expected_reason"),
    [
        (PositionSide.LONG, 99.0, 104.0, "long targets must be ordered above entry"),
        (PositionSide.LONG, 104.0, 102.0, "long targets must be ordered above entry"),
        (PositionSide.SHORT, 101.0, 96.0, "short targets must be ordered below entry"),
        (PositionSide.SHORT, 96.0, 98.0, "short targets must be ordered below entry"),
    ],
)
def test_bounded_target_modes_require_directional_ordering(
    target_mode: str,
    side: PositionSide,
    take_profit_1: float,
    take_profit_2: float,
    expected_reason: str,
) -> None:
    stop_price = 98.0 if side == PositionSide.LONG else 102.0
    decision = PortfolioRiskGate(RiskSettings()).evaluate(
        TradePlan(
            symbol="TESTUSDT",
            side=side,
            entry_price=100.0,
            stop_price=stop_price,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            leverage=5,
            plan_target_mode=target_mode,
        ),
        _account(),
    )

    assert not decision.allowed
    assert decision.blocked_code == "INVALID_TRADE_PLAN"
    assert decision.reasons == (expected_reason,)


@pytest.mark.parametrize(
    "target_mode",
    [PLAN_TARGET_MODE_BOUNDED_TARGETS, PLAN_TARGET_MODE_LEGACY_BOUNDED_DUAL_TP],
)
@pytest.mark.parametrize(
    ("side", "stop_price", "take_profit_1", "take_profit_2"),
    [
        (PositionSide.LONG, 98.0, 102.0, 104.0),
        (PositionSide.LONG, 98.0, 102.0, 102.0),
        (PositionSide.SHORT, 102.0, 98.0, 96.0),
        (PositionSide.SHORT, 102.0, 98.0, 98.0),
    ],
)
def test_bounded_target_modes_accept_ordered_directional_targets(
    target_mode: str,
    side: PositionSide,
    stop_price: float,
    take_profit_1: float,
    take_profit_2: float,
) -> None:
    decision = PortfolioRiskGate(RiskSettings()).evaluate(
        TradePlan(
            symbol="TESTUSDT",
            side=side,
            entry_price=100.0,
            stop_price=stop_price,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            leverage=5,
            plan_target_mode=target_mode,
        ),
        _account(),
    )

    assert decision.allowed


def test_unknown_target_mode_is_rejected() -> None:
    decision = PortfolioRiskGate(RiskSettings()).evaluate(
        TradePlan(
            symbol="TESTUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_1=None,
            take_profit_2=None,
            leverage=5,
            plan_target_mode="UNKNOWN",
        ),
        _account(),
    )

    assert not decision.allowed
    assert decision.blocked_code == "INVALID_TRADE_PLAN"
    assert decision.reasons == ("unsupported plan target mode",)
