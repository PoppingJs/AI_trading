from __future__ import annotations

import dataclasses

import pytest

from ai_trading.entry_policy_v7 import (
    BLOCK_CATEGORY_DIRECTION_STRUCTURE,
    BLOCK_CATEGORY_MARKET_DATA,
    BLOCK_CATEGORY_SCORE_ENTRY,
    DIAGNOSTIC_DERIVATIVES_DATA_INCOMPLETE,
    ENTRY_LONG,
    ENTRY_MODE_NONE,
    ENTRY_MODE_STANDARD,
    ENTRY_MODE_TREND_RESEARCH,
    ENTRY_MODE_TREND_STARTUP,
    ENTRY_ROUTE_H4_ALIGNED,
    ENTRY_ROUTE_H4_CLEAN_NEUTRAL,
    ENTRY_ROUTE_M15_EARLY_RESEARCH,
    ENTRY_SHORT,
    ENTRY_STATE_DATA_BLOCKED,
    ENTRY_STATE_DIRECTION_BLOCKED,
    ENTRY_STATE_DIRECTION_PENDING,
    ENTRY_STATE_ENTRY_ZONE_PENDING,
    ENTRY_STATE_READY,
    ENTRY_STATE_READY_RESEARCH,
    ENTRY_STATE_SCORE_PENDING,
    ENTRY_STATE_STRUCTURE_PENDING,
    LOCATION_STATUS_EXCELLENT,
    LOCATION_STATUS_WAITING,
    WATCH,
    EntryDataConfidenceV7,
    EntryPolicyV7Input,
    decide_entry_policy_v7,
)


def _request(**overrides: object) -> EntryPolicyV7Input:
    values: dict[str, object] = {
        "total_score": 75,
        "candidate_action": ENTRY_LONG,
        "entry_route": ENTRY_ROUTE_H4_ALIGNED,
        "data_confidence": EntryDataConfidenceV7(),
        "semantic_structure_exists": True,
        "setup_valid": True,
        "authorized_entry_zone_exists": True,
        "current_price_inside_zone": True,
        "h4_direction_opposed": False,
    }
    values.update(overrides)
    return EntryPolicyV7Input(**values)  # type: ignore[arg-type]


def test_h4_aligned_standard_signal_opens_at_75() -> None:
    decision = decide_entry_policy_v7(_request())

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_STANDARD
    assert decision.entry_state == ENTRY_STATE_READY
    assert decision.required_score == 75
    assert decision.effective_score == 75
    assert decision.location_status == LOCATION_STATUS_EXCELLENT
    assert decision.blocks == ()
    assert decision.reasons == ()
    assert decision.allowed is True
    assert decision.preliminary_eligible is True


@pytest.mark.parametrize("score", [65, 69, 70, 74])
def test_h4_aligned_65_to_74_uses_executable_research_channel(
    score: int,
) -> None:
    decision = decide_entry_policy_v7(_request(total_score=score))

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_TREND_RESEARCH
    assert decision.entry_state == ENTRY_STATE_READY_RESEARCH
    assert decision.required_score == 65
    assert decision.blocks == ()


@pytest.mark.parametrize("score", [70, 75, 110])
def test_clean_neutral_direction_uses_startup_channel_from_70(
    score: int,
) -> None:
    decision = decide_entry_policy_v7(
        _request(
            total_score=score,
            candidate_action=ENTRY_SHORT,
            entry_route=ENTRY_ROUTE_H4_CLEAN_NEUTRAL,
        )
    )

    assert decision.decision_action == ENTRY_SHORT
    assert decision.entry_mode == ENTRY_MODE_TREND_STARTUP
    assert decision.entry_state == ENTRY_STATE_READY
    assert decision.required_score == 70


def test_clean_neutral_65_to_69_uses_research_channel() -> None:
    decision = decide_entry_policy_v7(
        _request(
            total_score=69,
            entry_route=ENTRY_ROUTE_H4_CLEAN_NEUTRAL,
        )
    )

    assert decision.entry_mode == ENTRY_MODE_TREND_RESEARCH
    assert decision.required_score == 65


@pytest.mark.parametrize("score", [65, 75, 110])
def test_early_m15_candidate_always_uses_research_channel(
    score: int,
) -> None:
    decision = decide_entry_policy_v7(
        _request(
            total_score=score,
            entry_route=ENTRY_ROUTE_M15_EARLY_RESEARCH,
        )
    )

    assert decision.entry_mode == ENTRY_MODE_TREND_RESEARCH
    assert decision.entry_state == ENTRY_STATE_READY_RESEARCH
    assert decision.required_score == 65
    assert decision.effective_score == score


def test_score_below_65_is_a_visible_total_score_block() -> None:
    decision = decide_entry_policy_v7(_request(total_score=64))

    assert decision.decision_action == WATCH
    assert decision.entry_mode == ENTRY_MODE_NONE
    assert decision.entry_state == ENTRY_STATE_SCORE_PENDING
    assert decision.required_score == 65
    assert decision.blocks == ("RESEARCH_SCORE_BELOW_MINIMUM",)
    assert decision.reasons[0].category == BLOCK_CATEGORY_SCORE_ENTRY
    assert decision.reasons[0].detail == "64<65"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "bad"])
def test_invalid_total_score_is_visible_instead_of_raising(
    score: object,
) -> None:
    decision = decide_entry_policy_v7(_request(total_score=score))

    assert decision.entry_state == ENTRY_STATE_SCORE_PENDING
    assert decision.blocks == ("TOTAL_SCORE_INVALID",)
    assert decision.effective_score == 0


def test_missing_derivatives_is_diagnostic_only_without_downgrade() -> None:
    decision = decide_entry_policy_v7(
        _request(
            total_score=80,
            data_confidence=EntryDataConfidenceV7(
                derivatives_data_complete=False
            ),
        )
    )

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_STANDARD
    assert decision.blocks == ()
    assert decision.diagnostics == (
        DIAGNOSTIC_DERIVATIVES_DATA_INCOMPLETE,
    )


@pytest.mark.parametrize(
    ("confidence", "expected_codes"),
    [
        (
            EntryDataConfidenceV7(price_data_contiguous=False),
            ("PRICE_DATA_DISCONTINUOUS",),
        ),
        (
            EntryDataConfidenceV7(price_data_fresh=False),
            ("PRICE_DATA_STALE",),
        ),
        (
            EntryDataConfidenceV7(
                price_data_contiguous=False,
                price_data_fresh=False,
            ),
            ("PRICE_DATA_DISCONTINUOUS", "PRICE_DATA_STALE"),
        ),
    ],
)
def test_broken_price_data_is_a_classified_safety_block(
    confidence: EntryDataConfidenceV7,
    expected_codes: tuple[str, ...],
) -> None:
    decision = decide_entry_policy_v7(
        _request(total_score=110, data_confidence=confidence)
    )

    assert decision.entry_state == ENTRY_STATE_DATA_BLOCKED
    assert decision.blocks == expected_codes
    assert all(
        reason.category == BLOCK_CATEGORY_MARKET_DATA
        for reason in decision.reasons
    )
    assert decision.strategy_hard_vetoes == ()


def test_h4_opposition_is_the_only_strategy_hard_veto() -> None:
    decision = decide_entry_policy_v7(
        _request(total_score=110, h4_direction_opposed=True)
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_DIRECTION_BLOCKED
    assert decision.blocks == ("H4_DIRECTION_OPPOSED",)
    assert decision.strategy_hard_vetoes == ("H4_DIRECTION_OPPOSED",)
    assert decision.reasons[0].strategy_hard_veto is True
    assert (
        decision.reasons[0].category
        == BLOCK_CATEGORY_DIRECTION_STRUCTURE
    )


@pytest.mark.parametrize(
    ("overrides", "expected_code", "expected_state"),
    [
        (
            {"candidate_action": "WATCH"},
            "CANDIDATE_ACTION_NOT_ESTABLISHED",
            ENTRY_STATE_DIRECTION_PENDING,
        ),
        (
            {"entry_route": "UNKNOWN"},
            "CANDIDATE_ROUTE_NOT_ESTABLISHED",
            ENTRY_STATE_DIRECTION_PENDING,
        ),
        (
            {"semantic_structure_exists": False},
            "SEMANTIC_STRUCTURE_UNAVAILABLE",
            ENTRY_STATE_STRUCTURE_PENDING,
        ),
        (
            {"setup_valid": False},
            "SETUP_INVALID",
            ENTRY_STATE_STRUCTURE_PENDING,
        ),
        (
            {"authorized_entry_zone_exists": False},
            "AUTHORIZED_ENTRY_ZONE_UNAVAILABLE",
            ENTRY_STATE_ENTRY_ZONE_PENDING,
        ),
        (
            {"current_price_inside_zone": False},
            "CURRENT_PRICE_OUTSIDE_AUTHORIZED_ZONE",
            ENTRY_STATE_ENTRY_ZONE_PENDING,
        ),
    ],
)
def test_every_non_score_signal_failure_has_a_visible_classified_reason(
    overrides: dict[str, object],
    expected_code: str,
    expected_state: str,
) -> None:
    decision = decide_entry_policy_v7(_request(**overrides))

    assert decision.decision_action == WATCH
    assert decision.entry_state == expected_state
    assert expected_code in decision.blocks
    assert all(reason.category for reason in decision.reasons)
    assert decision.strategy_hard_vetoes == ()


def test_location_excellent_is_only_a_projection_of_zone_hit_and_setup() -> None:
    score_blocked = decide_entry_policy_v7(_request(total_score=64))
    h4_blocked = decide_entry_policy_v7(
        _request(h4_direction_opposed=True)
    )
    setup_invalid = decide_entry_policy_v7(_request(setup_valid=False))
    zone_missing = decide_entry_policy_v7(
        _request(authorized_entry_zone_exists=False)
    )
    outside = decide_entry_policy_v7(
        _request(current_price_inside_zone=False)
    )

    assert score_blocked.location_status == LOCATION_STATUS_EXCELLENT
    assert h4_blocked.location_status == LOCATION_STATUS_EXCELLENT
    assert setup_invalid.location_status == LOCATION_STATUS_WAITING
    assert zone_missing.location_status == LOCATION_STATUS_WAITING
    assert outside.location_status == LOCATION_STATUS_WAITING


def test_missing_semantic_structure_does_not_duplicate_setup_root_cause() -> None:
    decision = decide_entry_policy_v7(
        _request(
            semantic_structure_exists=False,
            setup_valid=False,
        )
    )

    assert decision.blocks == ("SEMANTIC_STRUCTURE_UNAVAILABLE",)


def test_signal_stage_has_no_component_score_stop_target_trigger_or_r_fields() -> None:
    input_fields = {
        item.name for item in dataclasses.fields(EntryPolicyV7Input)
    }

    forbidden_fields = {
        "family_scores",
        "structure_score",
        "location_score",
        "trigger_score",
        "price_progress_score",
        "location_ready",
        "trigger_ready",
        "stop_ready",
        "target_ready",
        "stop_price",
        "take_profit_1",
        "take_profit_2",
        "net_reward_r",
    }
    assert input_fields.isdisjoint(forbidden_fields)

    decision = decide_entry_policy_v7(_request(total_score=110))
    forbidden_reason_prefixes = (
        "LOCATION_SCORE_BELOW_MINIMUM",
        "TRIGGER_SCORE_BELOW_MINIMUM",
        "PRICE_PROGRESS_SCORE_BELOW_MINIMUM",
        "STRUCTURE_SCORE_BELOW_MINIMUM",
        "ENTRY_TRIGGER_UNAVAILABLE",
        "STRUCTURE_STOP_UNAVAILABLE",
        "STRUCTURE_TARGET_UNAVAILABLE",
        "NET_REWARD_R_",
    )
    assert not any(
        code.startswith(forbidden_reason_prefixes)
        for code in decision.blocks
    )


def test_policy_is_frozen_deterministic_and_does_not_mutate_input() -> None:
    request = _request(total_score=72)

    first = decide_entry_policy_v7(request)
    second = decide_entry_policy_v7(request)

    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.total_score = 80  # type: ignore[misc]
