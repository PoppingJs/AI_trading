from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Sequence

from ai_trading.models import (
    PLAN_TARGET_MODE_BOUNDED_TARGETS,
    PLAN_TARGET_MODE_OPEN_SPACE,
    PositionSide,
)


EXECUTION_PLAN_FORMULA_VERSION = 2
MIN_NET_PLAN_R = 1.30

PLAN_STATUS_READY = "READY"
PLAN_STATUS_BLOCKED = "BLOCKED"

PROTECTION_INACTIVE = "INACTIVE"
PROTECTION_ELIGIBLE = "PROTECTION_ELIGIBLE"
PROTECTION_ACTIVE = "PROTECTION_ACTIVE"

STRONG_MARKET_STATE_UP = "STRONG_UP"
STRONG_MARKET_STATE_DOWN = "STRONG_DOWN"


@dataclass(frozen=True)
class ExecutionPlanV7Input:
    """Inputs frozen for one executable-price plan review.

    All reward/risk values are calculated per unit.  Quantity is deliberately
    absent: grading and fixed-capital sizing happen only after this plan has
    passed.
    """

    side: PositionSide
    final_fill_price: float
    structure_invalidation_price: float
    disaster_stop_price: float
    tick_size: float
    fee_rate: float
    slippage_rate: float
    target_prices: tuple[float, ...] = ()
    target_weights: tuple[float, ...] = ()
    opposing_structure_prices: tuple[float, ...] = ()
    min_net_plan_r: float = MIN_NET_PLAN_R
    # OPEN_SPACE is a price-discovery exception, not a fallback for missing
    # target data.  Callers must explicitly freeze both facts below.  ``str``
    # is intentional so a persisted enum value can be replayed without
    # importing the market-state classifier into the execution layer.
    market_state: str | None = None
    reliable_historical_target_absent: bool | None = None


@dataclass(frozen=True)
class ExecutionPlanV7:
    formula_version: int
    status: str
    blocked_code: str
    diagnostic_message: str
    side: PositionSide
    target_mode: str
    entry_price: float
    structure_invalidation_price: float
    expected_structure_exit_price: float
    disaster_stop_price: float
    take_profit_1: float | None
    take_profit_2: float | None
    target_weights: tuple[float, ...]
    risk_unit: float
    bounded_reward_unit: float | None
    net_plan_r: float | None
    minimum_net_plan_r: float
    open_space_reference_price: float | None
    open_space_capacity_r: float | None
    fee_adjusted_breakeven_price: float | None
    entry_fee_unit: float
    invalidation_exit_fee_unit: float
    fee_rate: float
    slippage_rate: float
    tick_size: float

    @property
    def allowed(self) -> bool:
        return self.status == PLAN_STATUS_READY


@dataclass(frozen=True)
class OpenSpaceProtectionDecision:
    state: str
    protected_structure_invalidation_price: float | None


def build_execution_plan_v7(request: ExecutionPlanV7Input) -> ExecutionPlanV7:
    """Build the sole execution-price plan for entry pipeline v7.

    Slippage is represented once through the expected executable exit price.
    Fees remain explicit.  OPEN_SPACE creates a 1.30R reference for capacity
    review; it never turns that reference into TP1 or TP2.
    """

    error = _input_error(request)
    if error:
        return _blocked_plan(request, error)

    side = request.side
    tick = float(request.tick_size)
    fee_rate = max(float(request.fee_rate), 0.0)
    slippage_rate = max(float(request.slippage_rate), 0.0)
    entry = _round_price(
        float(request.final_fill_price),
        tick,
        "UP" if side is PositionSide.LONG else "DOWN",
    )
    structure_price = _round_price(
        float(request.structure_invalidation_price),
        tick,
        "DOWN" if side is PositionSide.LONG else "UP",
    )
    disaster_price = _round_price(
        float(request.disaster_stop_price),
        tick,
        "DOWN" if side is PositionSide.LONG else "UP",
    )
    expected_structure_exit = _expected_exit_price(
        side,
        structure_price,
        slippage_rate,
        tick,
        favorable=False,
    )
    entry_fee = entry * fee_rate
    invalidation_fee = expected_structure_exit * fee_rate
    risk_unit = (
        abs(entry - expected_structure_exit)
        + entry_fee
        + invalidation_fee
    )
    if not _valid_stop_geometry(
        side,
        entry,
        structure_price,
        disaster_price,
    ):
        return _blocked_plan(request, "INVALID_STOP_GEOMETRY")
    if not math.isfinite(risk_unit) or risk_unit <= 0:
        return _blocked_plan(request, "INVALID_RISK_UNIT")

    minimum_net_plan_r = max(
        MIN_NET_PLAN_R,
        float(request.min_net_plan_r),
    )
    targets = tuple(float(value) for value in request.target_prices)
    if targets:
        return _bounded_plan(
            request,
            entry=entry,
            structure_price=structure_price,
            expected_structure_exit=expected_structure_exit,
            disaster_price=disaster_price,
            entry_fee=entry_fee,
            invalidation_fee=invalidation_fee,
            risk_unit=risk_unit,
            targets=targets,
            minimum_net_plan_r=minimum_net_plan_r,
        )
    return _open_space_plan(
        request,
        entry=entry,
        structure_price=structure_price,
        expected_structure_exit=expected_structure_exit,
        disaster_price=disaster_price,
        entry_fee=entry_fee,
        invalidation_fee=invalidation_fee,
        risk_unit=risk_unit,
        minimum_net_plan_r=minimum_net_plan_r,
    )


def advance_open_space_protection(
    *,
    side: PositionSide,
    previous_state: str,
    previous_protected_price: float | None,
    closed_h1_price: float,
    reference_price: float,
    fee_adjusted_breakeven_price: float,
    confirmed_h1_swing_invalidation_price: float | None,
    tick_size: float,
) -> OpenSpaceProtectionDecision:
    """Advance OPEN_SPACE protection without inventing a breakeven stop.

    Reaching the reference on a closed 1H candle only grants eligibility.  A
    real, directionally improving 1H swing whose invalidation is already past
    fee-adjusted breakeven is required before a protected structure is frozen.
    """

    values = (
        closed_h1_price,
        reference_price,
        fee_adjusted_breakeven_price,
        tick_size,
    )
    if not all(math.isfinite(float(value)) for value in values) or tick_size <= 0:
        return OpenSpaceProtectionDecision(previous_state, previous_protected_price)

    reached = (
        closed_h1_price >= reference_price
        if side is PositionSide.LONG
        else closed_h1_price <= reference_price
    )
    state = previous_state
    if reached and state == PROTECTION_INACTIVE:
        state = PROTECTION_ELIGIBLE
    if state not in {PROTECTION_ELIGIBLE, PROTECTION_ACTIVE}:
        return OpenSpaceProtectionDecision(state, previous_protected_price)
    if confirmed_h1_swing_invalidation_price is None or not math.isfinite(
        float(confirmed_h1_swing_invalidation_price)
    ):
        return OpenSpaceProtectionDecision(state, previous_protected_price)

    candidate = _round_price(
        float(confirmed_h1_swing_invalidation_price),
        float(tick_size),
        "DOWN" if side is PositionSide.LONG else "UP",
    )
    past_breakeven = (
        candidate >= fee_adjusted_breakeven_price
        if side is PositionSide.LONG
        else candidate <= fee_adjusted_breakeven_price
    )
    improves = (
        previous_protected_price is None
        or (
            candidate > previous_protected_price
            if side is PositionSide.LONG
            else candidate < previous_protected_price
        )
    )
    if not past_breakeven or not improves:
        return OpenSpaceProtectionDecision(state, previous_protected_price)
    return OpenSpaceProtectionDecision(PROTECTION_ACTIVE, candidate)


def _bounded_plan(
    request: ExecutionPlanV7Input,
    *,
    entry: float,
    structure_price: float,
    expected_structure_exit: float,
    disaster_price: float,
    entry_fee: float,
    invalidation_fee: float,
    risk_unit: float,
    targets: tuple[float, ...],
    minimum_net_plan_r: float,
) -> ExecutionPlanV7:
    if len(targets) != 2:
        return _blocked_plan(request, "BOUNDED_TARGET_COUNT_INVALID")
    weights = request.target_weights or (0.5, 0.5)
    if len(weights) != len(targets):
        return _blocked_plan(request, "TARGET_WEIGHT_COUNT_INVALID")
    weights = tuple(float(weight) for weight in weights)
    if (
        not all(math.isfinite(weight) and weight >= 0 for weight in weights)
        or sum(weights) <= 0
        or sum(weights) > 1.0 + 1e-12
    ):
        return _blocked_plan(request, "TARGET_WEIGHTS_INVALID")

    rounded_targets = tuple(
        _round_price(
            target,
            request.tick_size,
            "DOWN" if request.side is PositionSide.LONG else "UP",
        )
        for target in targets
    )
    if not _targets_in_profitable_direction(
        request.side,
        entry,
        rounded_targets,
    ):
        return _blocked_plan(request, "TARGET_DIRECTION_INVALID")
    if not _targets_in_nearest_first_order(request.side, rounded_targets):
        return _blocked_plan(request, "TARGET_ORDER_INVALID")

    reward = -entry_fee
    for weight, target in zip(weights, rounded_targets, strict=True):
        expected_target_exit = _expected_exit_price(
            request.side,
            target,
            request.slippage_rate,
            request.tick_size,
            favorable=True,
        )
        reward += weight * (
            _favorable_distance(request.side, entry, expected_target_exit)
            - expected_target_exit * request.fee_rate
        )
    net_r = reward / risk_unit
    if not math.isfinite(net_r) or net_r < minimum_net_plan_r:
        return _blocked_plan(
            request,
            "NET_PLAN_R_BELOW_MINIMUM",
            target_mode=PLAN_TARGET_MODE_BOUNDED_TARGETS,
            entry=entry,
            structure_price=structure_price,
            expected_structure_exit=expected_structure_exit,
            disaster_price=disaster_price,
            take_profit_1=rounded_targets[0],
            take_profit_2=rounded_targets[1],
            target_weights=weights,
            risk_unit=risk_unit,
            bounded_reward_unit=reward,
            net_plan_r=net_r,
            minimum_net_plan_r=minimum_net_plan_r,
            diagnostic_message=(
                f"执行时预计净计划R为{net_r:.4f}R，"
                f"低于最低{minimum_net_plan_r:.2f}R。"
            ),
            entry_fee=entry_fee,
            invalidation_fee=invalidation_fee,
        )
    return _ready_plan(
        request,
        target_mode=PLAN_TARGET_MODE_BOUNDED_TARGETS,
        entry=entry,
        structure_price=structure_price,
        expected_structure_exit=expected_structure_exit,
        disaster_price=disaster_price,
        take_profit_1=rounded_targets[0],
        take_profit_2=rounded_targets[1],
        target_weights=weights,
        risk_unit=risk_unit,
        bounded_reward_unit=reward,
        net_plan_r=net_r,
        minimum_net_plan_r=minimum_net_plan_r,
        open_space_reference=None,
        open_space_capacity_r=None,
        breakeven=None,
        entry_fee=entry_fee,
        invalidation_fee=invalidation_fee,
        diagnostic_message=(
            f"执行计划通过：执行时预计净计划R为{net_r:.4f}R，"
            f"不低于最低{minimum_net_plan_r:.2f}R。"
        ),
    )


def _open_space_plan(
    request: ExecutionPlanV7Input,
    *,
    entry: float,
    structure_price: float,
    expected_structure_exit: float,
    disaster_price: float,
    entry_fee: float,
    invalidation_fee: float,
    risk_unit: float,
    minimum_net_plan_r: float,
) -> ExecutionPlanV7:
    state = _market_state_value(request.market_state)
    required_state = (
        STRONG_MARKET_STATE_UP
        if request.side is PositionSide.LONG
        else STRONG_MARKET_STATE_DOWN
    )
    if state not in {STRONG_MARKET_STATE_UP, STRONG_MARKET_STATE_DOWN}:
        return _blocked_plan(
            request,
            "OPEN_SPACE_STRONG_TREND_REQUIRED",
            minimum_net_plan_r=minimum_net_plan_r,
            diagnostic_message=(
                "开放空间计划仅允许用于已确认的强单边行情；"
                "当前未提供方向一致的强单边状态。"
            ),
        )
    if state != required_state:
        side_text = "多头" if request.side is PositionSide.LONG else "空头"
        state_text = {
            STRONG_MARKET_STATE_UP: "强单边上涨",
            STRONG_MARKET_STATE_DOWN: "强单边下跌",
        }.get(state, "非强单边行情")
        return _blocked_plan(
            request,
            "OPEN_SPACE_DIRECTION_MISMATCH",
            minimum_net_plan_r=minimum_net_plan_r,
            diagnostic_message=(
                f"开放空间方向不一致：候选方向为{side_text}，"
                f"行情状态为{state_text}。"
            ),
        )
    if request.reliable_historical_target_absent is not True:
        return _blocked_plan(
            request,
            "OPEN_SPACE_TARGET_ABSENCE_UNCONFIRMED",
            minimum_net_plan_r=minimum_net_plan_r,
            diagnostic_message=(
                "未确认历史数据中不存在可靠对手目标，"
                "不能使用开放空间计划。"
            ),
        )

    reference = _price_for_net_reward(
        side=request.side,
        entry=entry,
        required_reward=minimum_net_plan_r * risk_unit,
        fee_rate=request.fee_rate,
        slippage_rate=request.slippage_rate,
        tick_size=request.tick_size,
    )
    breakeven = _price_for_net_reward(
        side=request.side,
        entry=entry,
        required_reward=0.0,
        fee_rate=request.fee_rate,
        slippage_rate=request.slippage_rate,
        tick_size=request.tick_size,
    )
    if reference is None or breakeven is None:
        return _blocked_plan(
            request,
            "OPEN_SPACE_REFERENCE_INVALID",
            minimum_net_plan_r=minimum_net_plan_r,
        )
    obstacle = _nearest_obstacle_between(
        request.side,
        entry,
        reference,
        request.opposing_structure_prices,
    )
    if obstacle is not None:
        return _blocked_plan(
            request,
            "OPEN_SPACE_OPPOSING_STRUCTURE_BEFORE_1_30R",
            target_mode=PLAN_TARGET_MODE_OPEN_SPACE,
            entry=entry,
            structure_price=structure_price,
            expected_structure_exit=expected_structure_exit,
            disaster_price=disaster_price,
            risk_unit=risk_unit,
            minimum_net_plan_r=minimum_net_plan_r,
            open_space_reference=reference,
            breakeven=breakeven,
            diagnostic_message=(
                f"开放空间审核未通过：从成交价到最低"
                f"{minimum_net_plan_r:.2f}R参考价之间存在已知对手结构"
                f"（{obstacle:g}）。"
            ),
            entry_fee=entry_fee,
            invalidation_fee=invalidation_fee,
        )
    expected_reference_exit = _expected_exit_price(
        request.side,
        reference,
        request.slippage_rate,
        request.tick_size,
        favorable=True,
    )
    reference_reward = (
        _favorable_distance(request.side, entry, expected_reference_exit)
        - entry_fee
        - expected_reference_exit * request.fee_rate
    )
    capacity_r = reference_reward / risk_unit
    if not math.isfinite(capacity_r) or capacity_r < minimum_net_plan_r:
        return _blocked_plan(
            request,
            "OPEN_SPACE_REFERENCE_INVALID",
            target_mode=PLAN_TARGET_MODE_OPEN_SPACE,
            entry=entry,
            structure_price=structure_price,
            expected_structure_exit=expected_structure_exit,
            disaster_price=disaster_price,
            risk_unit=risk_unit,
            net_plan_r=capacity_r,
            minimum_net_plan_r=minimum_net_plan_r,
            open_space_reference=reference,
            breakeven=breakeven,
            entry_fee=entry_fee,
            invalidation_fee=invalidation_fee,
            diagnostic_message=(
                f"开放空间参考价取整后预计净计划R为{capacity_r:.4f}R，"
                f"低于最低{minimum_net_plan_r:.2f}R。"
            ),
        )
    return _ready_plan(
        request,
        target_mode=PLAN_TARGET_MODE_OPEN_SPACE,
        entry=entry,
        structure_price=structure_price,
        expected_structure_exit=expected_structure_exit,
        disaster_price=disaster_price,
        take_profit_1=None,
        take_profit_2=None,
        target_weights=(),
        risk_unit=risk_unit,
        bounded_reward_unit=None,
        net_plan_r=capacity_r,
        minimum_net_plan_r=minimum_net_plan_r,
        open_space_reference=reference,
        open_space_capacity_r=capacity_r,
        breakeven=breakeven,
        entry_fee=entry_fee,
        invalidation_fee=invalidation_fee,
        diagnostic_message=(
            f"开放空间审核通过：预计净计划R为{capacity_r:.4f}R，"
            f"最低{minimum_net_plan_r:.2f}R参考价前"
            "没有已知对手结构；该参考价不是固定止盈。"
        ),
    )


def _input_error(request: ExecutionPlanV7Input) -> str:
    numeric = (
        request.final_fill_price,
        request.structure_invalidation_price,
        request.disaster_stop_price,
        request.tick_size,
        request.fee_rate,
        request.slippage_rate,
        request.min_net_plan_r,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        return "PLAN_INPUT_NOT_FINITE"
    if (
        request.final_fill_price <= 0
        or request.structure_invalidation_price <= 0
        or request.disaster_stop_price <= 0
        or request.tick_size <= 0
        or request.fee_rate < 0
        or request.fee_rate >= 1
        or request.slippage_rate < 0
        or request.slippage_rate >= 1
        or request.min_net_plan_r <= 0
    ):
        return "PLAN_INPUT_OUT_OF_RANGE"
    if not all(math.isfinite(float(value)) for value in request.target_prices):
        return "TARGET_NOT_FINITE"
    if not all(
        math.isfinite(float(value))
        for value in request.opposing_structure_prices
    ):
        return "OPPOSING_STRUCTURE_NOT_FINITE"
    return ""


def _valid_stop_geometry(
    side: PositionSide,
    entry: float,
    structure_price: float,
    disaster_price: float,
) -> bool:
    if side is PositionSide.LONG:
        return 0 < disaster_price <= structure_price < entry
    return disaster_price >= structure_price > entry


def _targets_in_profitable_direction(
    side: PositionSide,
    entry: float,
    targets: Sequence[float],
) -> bool:
    if side is PositionSide.LONG:
        return all(target > entry for target in targets)
    return all(target < entry for target in targets)


def _targets_in_nearest_first_order(
    side: PositionSide,
    targets: Sequence[float],
) -> bool:
    if side is PositionSide.LONG:
        return all(left <= right for left, right in zip(targets, targets[1:]))
    return all(left >= right for left, right in zip(targets, targets[1:]))


def _favorable_distance(
    side: PositionSide,
    entry: float,
    exit_price: float,
) -> float:
    return exit_price - entry if side is PositionSide.LONG else entry - exit_price


def _market_state_value(value: object) -> str:
    raw = getattr(value, "value", value)
    normalized = str(raw or "").strip().upper()
    # The execution formula can safely replay legacy persisted strong-state
    # names because these aliases express the same semantic fact.  Channel or
    # generic trend labels are deliberately not promoted to strong states.
    return {
        "ONE_WAY_UP": STRONG_MARKET_STATE_UP,
        "ONE_WAY_DOWN": STRONG_MARKET_STATE_DOWN,
    }.get(normalized, normalized)


def _expected_exit_price(
    side: PositionSide,
    reference_price: float,
    slippage_rate: float,
    tick_size: float,
    *,
    favorable: bool,
) -> float:
    # Both structure and target exits receive adverse execution slippage:
    # long exits sell lower, short exits buy higher.
    del favorable
    raw = (
        reference_price * (1 - slippage_rate)
        if side is PositionSide.LONG
        else reference_price * (1 + slippage_rate)
    )
    return _round_price(
        raw,
        tick_size,
        "DOWN" if side is PositionSide.LONG else "UP",
    )


def _price_for_net_reward(
    *,
    side: PositionSide,
    entry: float,
    required_reward: float,
    fee_rate: float,
    slippage_rate: float,
    tick_size: float,
) -> float | None:
    entry_fee = entry * fee_rate
    if side is PositionSide.LONG:
        exit_net_factor = 1 - fee_rate
        execution_factor = 1 - slippage_rate
        if exit_net_factor <= 0 or execution_factor <= 0:
            return None
        minimum_executable_exit = _round_price(
            (entry + entry_fee + required_reward) / exit_net_factor,
            tick_size,
            "UP",
        )
        return _round_price(
            minimum_executable_exit / execution_factor,
            tick_size,
            "UP",
        )
    exit_gross_factor = 1 + fee_rate
    execution_factor = 1 + slippage_rate
    numerator = entry - entry_fee - required_reward
    if exit_gross_factor <= 0 or execution_factor <= 0 or numerator <= 0:
        return None
    maximum_executable_exit = _round_price(
        numerator / exit_gross_factor,
        tick_size,
        "DOWN",
    )
    return _round_price(
        maximum_executable_exit / execution_factor,
        tick_size,
        "DOWN",
    )


def _nearest_obstacle_between(
    side: PositionSide,
    entry: float,
    reference: float,
    values: Sequence[float],
) -> float | None:
    candidates = [
        float(value)
        for value in values
        if math.isfinite(float(value))
        and (
            entry < float(value) <= reference
            if side is PositionSide.LONG
            else reference <= float(value) < entry
        )
    ]
    if not candidates:
        return None
    return min(candidates) if side is PositionSide.LONG else max(candidates)


def _round_price(price: float, tick_size: float, direction: str) -> float:
    value = Decimal(str(price))
    tick = Decimal(str(tick_size))
    units = value / tick
    rounding = {
        "UP": ROUND_CEILING,
        "DOWN": ROUND_FLOOR,
        "NEAREST": ROUND_HALF_UP,
    }[direction]
    return float(units.to_integral_value(rounding=rounding) * tick)


def _ready_plan(
    request: ExecutionPlanV7Input,
    *,
    target_mode: str,
    entry: float,
    structure_price: float,
    expected_structure_exit: float,
    disaster_price: float,
    take_profit_1: float | None,
    take_profit_2: float | None,
    target_weights: tuple[float, ...],
    risk_unit: float,
    bounded_reward_unit: float | None,
    net_plan_r: float | None,
    open_space_reference: float | None,
    open_space_capacity_r: float | None,
    breakeven: float | None,
    entry_fee: float,
    invalidation_fee: float,
    minimum_net_plan_r: float,
    diagnostic_message: str,
) -> ExecutionPlanV7:
    return ExecutionPlanV7(
        formula_version=EXECUTION_PLAN_FORMULA_VERSION,
        status=PLAN_STATUS_READY,
        blocked_code="",
        diagnostic_message=diagnostic_message,
        side=request.side,
        target_mode=target_mode,
        entry_price=entry,
        structure_invalidation_price=structure_price,
        expected_structure_exit_price=expected_structure_exit,
        disaster_stop_price=disaster_price,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        target_weights=target_weights,
        risk_unit=risk_unit,
        bounded_reward_unit=bounded_reward_unit,
        net_plan_r=net_plan_r,
        minimum_net_plan_r=minimum_net_plan_r,
        open_space_reference_price=open_space_reference,
        open_space_capacity_r=open_space_capacity_r,
        fee_adjusted_breakeven_price=breakeven,
        entry_fee_unit=entry_fee,
        invalidation_exit_fee_unit=invalidation_fee,
        fee_rate=request.fee_rate,
        slippage_rate=request.slippage_rate,
        tick_size=request.tick_size,
    )


def _blocked_plan(
    request: ExecutionPlanV7Input,
    code: str,
    *,
    target_mode: str | None = None,
    entry: float | None = None,
    structure_price: float | None = None,
    expected_structure_exit: float | None = None,
    disaster_price: float | None = None,
    take_profit_1: float | None = None,
    take_profit_2: float | None = None,
    target_weights: tuple[float, ...] = (),
    risk_unit: float = 0.0,
    bounded_reward_unit: float | None = None,
    net_plan_r: float | None = None,
    minimum_net_plan_r: float | None = None,
    open_space_reference: float | None = None,
    breakeven: float | None = None,
    entry_fee: float = 0.0,
    invalidation_fee: float = 0.0,
    diagnostic_message: str = "",
) -> ExecutionPlanV7:
    minimum = max(
        MIN_NET_PLAN_R,
        float(
            request.min_net_plan_r
            if minimum_net_plan_r is None
            else minimum_net_plan_r
        ),
    )
    return ExecutionPlanV7(
        formula_version=EXECUTION_PLAN_FORMULA_VERSION,
        status=PLAN_STATUS_BLOCKED,
        blocked_code=code,
        diagnostic_message=(
            diagnostic_message or _default_blocked_diagnostic(code, minimum)
        ),
        side=request.side,
        target_mode=(
            target_mode
            or (
                PLAN_TARGET_MODE_BOUNDED_TARGETS
                if request.target_prices
                else PLAN_TARGET_MODE_OPEN_SPACE
            )
        ),
        entry_price=float(entry or 0.0),
        structure_invalidation_price=float(structure_price or 0.0),
        expected_structure_exit_price=float(expected_structure_exit or 0.0),
        disaster_stop_price=float(disaster_price or 0.0),
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        target_weights=target_weights,
        risk_unit=risk_unit,
        bounded_reward_unit=bounded_reward_unit,
        net_plan_r=net_plan_r,
        minimum_net_plan_r=minimum,
        open_space_reference_price=open_space_reference,
        open_space_capacity_r=None,
        fee_adjusted_breakeven_price=breakeven,
        entry_fee_unit=entry_fee,
        invalidation_exit_fee_unit=invalidation_fee,
        fee_rate=float(request.fee_rate),
        slippage_rate=float(request.slippage_rate),
        tick_size=float(request.tick_size),
    )


def _default_blocked_diagnostic(code: str, minimum_net_plan_r: float) -> str:
    messages = {
        "PLAN_INPUT_NOT_FINITE": "执行计划输入包含无效数值。",
        "PLAN_INPUT_OUT_OF_RANGE": "执行计划输入超出有效范围。",
        "TARGET_NOT_FINITE": "目标价格包含无效数值。",
        "OPPOSING_STRUCTURE_NOT_FINITE": "对手结构价格包含无效数值。",
        "BOUNDED_TARGET_COUNT_INVALID": "有界目标计划必须包含两个目标。",
        "TARGET_WEIGHT_COUNT_INVALID": "目标权重数量与目标数量不一致。",
        "TARGET_WEIGHTS_INVALID": "目标权重无效。",
        "TARGET_DIRECTION_INVALID": "目标价格方向与候选交易方向不一致。",
        "TARGET_ORDER_INVALID": "目标未按最近对手结构到更远结构排序。",
        "INVALID_STOP_GEOMETRY": "结构失效价或灾难保护价的方向关系无效。",
        "INVALID_RISK_UNIT": "执行计划的预计风险单位无效。",
        "NET_PLAN_R_BELOW_MINIMUM": (
            f"执行时预计净计划R低于最低{minimum_net_plan_r:.2f}R。"
        ),
        "OPEN_SPACE_STRONG_TREND_REQUIRED": (
            "开放空间计划仅允许用于方向一致的强单边行情。"
        ),
        "OPEN_SPACE_DIRECTION_MISMATCH": "开放空间计划与强单边方向不一致。",
        "OPEN_SPACE_TARGET_ABSENCE_UNCONFIRMED": (
            "尚未确认历史数据中不存在可靠对手目标。"
        ),
        "OPEN_SPACE_REFERENCE_INVALID": (
            f"无法生成最低{minimum_net_plan_r:.2f}R开放空间参考价。"
        ),
        "OPEN_SPACE_OPPOSING_STRUCTURE_BEFORE_1_30R": (
            f"最低{minimum_net_plan_r:.2f}R参考价前存在已知对手结构。"
        ),
        "FINAL_PRICE_OUTSIDE_ENTRY_ZONE": "最终成交价已经离开有效入场区间。",
        "DISASTER_STOP_UNAVAILABLE": "无法生成灾难保护价。",
    }
    return messages.get(code, "执行计划输入或结构关系无效。")
