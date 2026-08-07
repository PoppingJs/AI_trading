from __future__ import annotations

import pytest

from ai_trading.execution_plan_v7 import (
    EXECUTION_PLAN_FORMULA_VERSION,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_READY,
    PROTECTION_ACTIVE,
    PROTECTION_ELIGIBLE,
    PROTECTION_INACTIVE,
    ExecutionPlanV7Input,
    advance_open_space_protection,
    build_execution_plan_v7,
)
from ai_trading.models import (
    PLAN_TARGET_MODE_BOUNDED_TARGETS,
    PLAN_TARGET_MODE_OPEN_SPACE,
    PositionSide,
)


def _request(**changes: object) -> ExecutionPlanV7Input:
    values: dict[str, object] = {
        "side": PositionSide.LONG,
        "final_fill_price": 100.0,
        "structure_invalidation_price": 98.0,
        "disaster_stop_price": 95.0,
        "tick_size": 0.01,
        "fee_rate": 0.0004,
        "slippage_rate": 0.0003,
        "target_prices": (104.0, 106.0),
        "target_weights": (0.5, 0.5),
    }
    values.update(changes)
    return ExecutionPlanV7Input(**values)  # type: ignore[arg-type]


def test_bounded_plan_uses_per_unit_costs_and_passes() -> None:
    plan = build_execution_plan_v7(_request())

    assert plan.formula_version == EXECUTION_PLAN_FORMULA_VERSION
    assert plan.status == PLAN_STATUS_READY
    assert plan.target_mode == PLAN_TARGET_MODE_BOUNDED_TARGETS
    assert plan.take_profit_1 == 104.0
    assert plan.take_profit_2 == 106.0
    assert plan.risk_unit > 2.0
    assert plan.net_plan_r is not None and plan.net_plan_r >= 1.30


def test_bounded_plan_rejects_low_net_r_without_changing_signal_stage() -> None:
    plan = build_execution_plan_v7(
        _request(target_prices=(101.0, 102.0))
    )

    assert plan.status == PLAN_STATUS_BLOCKED
    assert plan.blocked_code == "NET_PLAN_R_BELOW_MINIMUM"
    assert plan.net_plan_r is not None and plan.net_plan_r < 1.30
    assert f"{plan.net_plan_r:.4f}R" in plan.diagnostic_message
    assert "最低1.30R" in plan.diagnostic_message


def test_minimum_net_r_cannot_be_configured_below_1_30() -> None:
    plan = build_execution_plan_v7(
        _request(
            target_prices=(101.0, 102.0),
            min_net_plan_r=0.10,
        )
    )

    assert plan.status == PLAN_STATUS_BLOCKED
    assert plan.minimum_net_plan_r == pytest.approx(1.30)
    assert "最低1.30R" in plan.diagnostic_message


@pytest.mark.parametrize(
    ("side", "targets"),
    (
        (PositionSide.LONG, (106.0, 104.0)),
        (PositionSide.SHORT, (94.0, 96.0)),
    ),
)
def test_bounded_targets_must_be_ordered_nearest_first(
    side: PositionSide,
    targets: tuple[float, float],
) -> None:
    plan = build_execution_plan_v7(
        _request(
            side=side,
            structure_invalidation_price=(98.0 if side is PositionSide.LONG else 102.0),
            disaster_stop_price=(95.0 if side is PositionSide.LONG else 105.0),
            target_prices=targets,
        )
    )

    assert plan.status == PLAN_STATUS_BLOCKED
    assert plan.blocked_code == "TARGET_ORDER_INVALID"
    assert "排序" in plan.diagnostic_message


def test_bounded_reward_keeps_adverse_slippage_signed() -> None:
    plan = build_execution_plan_v7(
        _request(
            fee_rate=0.0,
            slippage_rate=0.01,
            target_prices=(100.01, 100.02),
        )
    )

    assert plan.status == PLAN_STATUS_BLOCKED
    assert plan.bounded_reward_unit is not None
    assert plan.bounded_reward_unit < 0


def test_open_space_keeps_targets_empty_and_builds_reference() -> None:
    plan = build_execution_plan_v7(
        _request(
            target_prices=(),
            target_weights=(),
            market_state="STRONG_UP",
            reliable_historical_target_absent=True,
        )
    )

    assert plan.status == PLAN_STATUS_READY
    assert plan.target_mode == PLAN_TARGET_MODE_OPEN_SPACE
    assert plan.take_profit_1 is None
    assert plan.take_profit_2 is None
    assert plan.open_space_reference_price is not None
    assert plan.open_space_reference_price > plan.entry_price
    assert plan.fee_adjusted_breakeven_price is not None
    assert plan.fee_adjusted_breakeven_price > plan.entry_price
    assert plan.open_space_capacity_r is not None
    assert plan.open_space_capacity_r >= 1.30
    assert f"{plan.open_space_capacity_r:.4f}R" in plan.diagnostic_message


@pytest.mark.parametrize(
    ("side", "market_state", "structure", "disaster"),
    (
        (PositionSide.LONG, "STRONG_UP", 98.0, 95.0),
        (PositionSide.SHORT, "STRONG_DOWN", 102.0, 105.0),
    ),
)
def test_open_space_reference_still_meets_minimum_after_coarse_tick_rounding(
    side: PositionSide,
    market_state: str,
    structure: float,
    disaster: float,
) -> None:
    plan = build_execution_plan_v7(
        _request(
            side=side,
            structure_invalidation_price=structure,
            disaster_stop_price=disaster,
            tick_size=0.5,
            target_prices=(),
            target_weights=(),
            market_state=market_state,
            reliable_historical_target_absent=True,
        )
    )

    assert plan.allowed
    assert plan.open_space_capacity_r is not None
    assert plan.open_space_capacity_r >= 1.30


def test_open_space_rejects_known_opponent_before_reference() -> None:
    plan = build_execution_plan_v7(
        _request(
            target_prices=(),
            target_weights=(),
            opposing_structure_prices=(102.0,),
            market_state="STRONG_UP",
            reliable_historical_target_absent=True,
        )
    )

    assert plan.status == PLAN_STATUS_BLOCKED
    assert plan.blocked_code == "OPEN_SPACE_OPPOSING_STRUCTURE_BEFORE_1_30R"
    assert plan.take_profit_1 is None
    assert plan.take_profit_2 is None
    assert "最低1.30R" in plan.diagnostic_message
    assert "102" in plan.diagnostic_message


def test_open_space_requires_confirmed_strong_state_and_target_search() -> None:
    not_strong = build_execution_plan_v7(
        _request(
            target_prices=(),
            target_weights=(),
            market_state="CHANNEL_UP",
            reliable_historical_target_absent=True,
        )
    )
    target_search_unconfirmed = build_execution_plan_v7(
        _request(
            target_prices=(),
            target_weights=(),
            market_state="STRONG_UP",
        )
    )

    assert not_strong.blocked_code == "OPEN_SPACE_STRONG_TREND_REQUIRED"
    assert (
        target_search_unconfirmed.blocked_code
        == "OPEN_SPACE_TARGET_ABSENCE_UNCONFIRMED"
    )


@pytest.mark.parametrize(
    ("side", "legacy_state"),
    (
        (PositionSide.LONG, "ONE_WAY_UP"),
        (PositionSide.SHORT, "ONE_WAY_DOWN"),
    ),
)
def test_open_space_accepts_semantically_equivalent_legacy_strong_state(
    side: PositionSide,
    legacy_state: str,
) -> None:
    plan = build_execution_plan_v7(
        _request(
            side=side,
            structure_invalidation_price=(98.0 if side is PositionSide.LONG else 102.0),
            disaster_stop_price=(95.0 if side is PositionSide.LONG else 105.0),
            target_prices=(),
            target_weights=(),
            market_state=legacy_state,
            reliable_historical_target_absent=True,
        )
    )

    assert plan.allowed


@pytest.mark.parametrize(
    ("side", "market_state"),
    (
        (PositionSide.LONG, "STRONG_DOWN"),
        (PositionSide.SHORT, "STRONG_UP"),
    ),
)
def test_open_space_rejects_direction_state_mismatch(
    side: PositionSide,
    market_state: str,
) -> None:
    plan = build_execution_plan_v7(
        _request(
            side=side,
            structure_invalidation_price=(98.0 if side is PositionSide.LONG else 102.0),
            disaster_stop_price=(95.0 if side is PositionSide.LONG else 105.0),
            target_prices=(),
            target_weights=(),
            market_state=market_state,
            reliable_historical_target_absent=True,
        )
    )

    assert plan.blocked_code == "OPEN_SPACE_DIRECTION_MISMATCH"


def test_short_open_space_is_a_price_mirror() -> None:
    long_plan = build_execution_plan_v7(
        _request(
            target_prices=(),
            target_weights=(),
            market_state="STRONG_UP",
            reliable_historical_target_absent=True,
        )
    )
    short_plan = build_execution_plan_v7(
        _request(
            side=PositionSide.SHORT,
            structure_invalidation_price=102.0,
            disaster_stop_price=105.0,
            target_prices=(),
            target_weights=(),
            market_state="STRONG_DOWN",
            reliable_historical_target_absent=True,
        )
    )

    assert short_plan.allowed
    assert short_plan.open_space_reference_price is not None
    assert short_plan.open_space_reference_price < short_plan.entry_price
    assert long_plan.open_space_reference_price is not None
    assert (long_plan.open_space_reference_price - 100.0) == pytest.approx(
        100.0 - short_plan.open_space_reference_price,
        abs=0.03,
    )


def test_short_open_space_rejects_support_before_reference() -> None:
    plan = build_execution_plan_v7(
        _request(
            side=PositionSide.SHORT,
            structure_invalidation_price=102.0,
            disaster_stop_price=105.0,
            target_prices=(),
            target_weights=(),
            opposing_structure_prices=(98.0,),
            market_state="STRONG_DOWN",
            reliable_historical_target_absent=True,
        )
    )

    assert plan.blocked_code == "OPEN_SPACE_OPPOSING_STRUCTURE_BEFORE_1_30R"


def test_tick_rounding_is_adverse_and_directionally_symmetric() -> None:
    long_plan = build_execution_plan_v7(
        _request(
            final_fill_price=100.004,
            structure_invalidation_price=98.006,
            disaster_stop_price=95.006,
            target_prices=(104.009, 106.009),
        )
    )
    short_plan = build_execution_plan_v7(
        _request(
            side=PositionSide.SHORT,
            final_fill_price=99.996,
            structure_invalidation_price=101.994,
            disaster_stop_price=104.994,
            target_prices=(95.991, 93.991),
        )
    )

    assert long_plan.entry_price == 100.01
    assert long_plan.structure_invalidation_price == 98.0
    assert long_plan.take_profit_1 == 104.0
    assert short_plan.entry_price == 99.99
    assert short_plan.structure_invalidation_price == 102.0
    assert short_plan.take_profit_1 == 96.0


def test_disaster_stop_must_be_farther_outside_structure() -> None:
    plan = build_execution_plan_v7(
        _request(disaster_stop_price=99.0)
    )

    assert plan.status == PLAN_STATUS_BLOCKED
    assert plan.blocked_code == "INVALID_STOP_GEOMETRY"


def test_reaching_reference_only_makes_protection_eligible() -> None:
    decision = advance_open_space_protection(
        side=PositionSide.LONG,
        previous_state=PROTECTION_INACTIVE,
        previous_protected_price=None,
        closed_h1_price=103.0,
        reference_price=102.5,
        fee_adjusted_breakeven_price=100.1,
        confirmed_h1_swing_invalidation_price=None,
        tick_size=0.01,
    )

    assert decision.state == PROTECTION_ELIGIBLE
    assert decision.protected_structure_invalidation_price is None


def test_real_h1_swing_is_required_and_protection_only_improves() -> None:
    first = advance_open_space_protection(
        side=PositionSide.LONG,
        previous_state=PROTECTION_ELIGIBLE,
        previous_protected_price=None,
        closed_h1_price=103.0,
        reference_price=102.5,
        fee_adjusted_breakeven_price=100.1,
        confirmed_h1_swing_invalidation_price=101.23,
        tick_size=0.01,
    )
    worse = advance_open_space_protection(
        side=PositionSide.LONG,
        previous_state=first.state,
        previous_protected_price=first.protected_structure_invalidation_price,
        closed_h1_price=104.0,
        reference_price=102.5,
        fee_adjusted_breakeven_price=100.1,
        confirmed_h1_swing_invalidation_price=101.0,
        tick_size=0.01,
    )

    assert first.state == PROTECTION_ACTIVE
    assert first.protected_structure_invalidation_price == 101.23
    assert worse.protected_structure_invalidation_price == 101.23


def test_short_protection_is_a_monotonic_mirror() -> None:
    first = advance_open_space_protection(
        side=PositionSide.SHORT,
        previous_state=PROTECTION_INACTIVE,
        previous_protected_price=None,
        closed_h1_price=97.0,
        reference_price=97.5,
        fee_adjusted_breakeven_price=99.9,
        confirmed_h1_swing_invalidation_price=98.77,
        tick_size=0.01,
    )
    worse = advance_open_space_protection(
        side=PositionSide.SHORT,
        previous_state=first.state,
        previous_protected_price=first.protected_structure_invalidation_price,
        closed_h1_price=96.0,
        reference_price=97.5,
        fee_adjusted_breakeven_price=99.9,
        confirmed_h1_swing_invalidation_price=99.0,
        tick_size=0.01,
    )

    assert first.state == PROTECTION_ACTIVE
    assert first.protected_structure_invalidation_price == 98.77
    assert worse.protected_structure_invalidation_price == 98.77
