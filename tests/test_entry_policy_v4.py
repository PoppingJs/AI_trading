from __future__ import annotations

import pytest

from ai_trading.entry_policy_v4 import (
    ENTRY_LONG,
    ENTRY_MODE_NONE,
    ENTRY_MODE_STANDARD,
    ENTRY_MODE_TREND_RESEARCH,
    ENTRY_MODE_TREND_STARTUP,
    ENTRY_SHORT,
    ENTRY_STATE_DATA_BLOCKED,
    ENTRY_STATE_DIRECTION_BLOCKED,
    ENTRY_STATE_PLAN_PENDING,
    ENTRY_STATE_READY,
    ENTRY_STATE_READY_RESEARCH,
    ENTRY_STATE_SCORE_PENDING,
    ENTRY_STATE_VETOED,
    WATCH,
    EntryDataConfidenceV4,
    EntryPolicyV4Input,
    decide_entry_policy_v4,
)


CORE_FAMILIES = {
    "STRUCTURE": 25,
    "LOCATION": 15,
    "TRIGGER": 10,
    "PRICE_PROGRESS": 6,
}


def _request(**overrides: object) -> EntryPolicyV4Input:
    values: dict[str, object] = {
        "setup_score": 75,
        "candidate_action": ENTRY_LONG,
        "direction_confirmation_state": "H4_CONFIRMED",
        "family_scores": CORE_FAMILIES,
        "data_confidence": EntryDataConfidenceV4(),
        "structure_ready": True,
        "location_ready": True,
        "trigger_ready": True,
        "stop_ready": True,
        "target_ready": True,
        "net_reward_r": 1.6,
        "vetoes": (),
    }
    values.update(overrides)
    return EntryPolicyV4Input(**values)  # type: ignore[arg-type]


def test_standard_entry_requires_75_and_keeps_compatible_action() -> None:
    decision = decide_entry_policy_v4(_request())

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_STANDARD
    assert decision.entry_state == ENTRY_STATE_READY
    assert decision.required_score == 75
    assert decision.blocks == ()
    assert decision.allowed is True


def test_score_above_100_is_not_normalized_or_blocked() -> None:
    decision = decide_entry_policy_v4(_request(setup_score=110))

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_STANDARD
    assert decision.required_score == 75


def test_temporary_direction_opens_startup_channel_at_70() -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=70,
            candidate_action=ENTRY_SHORT,
            direction_confirmation_state="H4_NEUTRAL_TEMPORARY",
        )
    )

    assert decision.decision_action == ENTRY_SHORT
    assert decision.entry_mode == ENTRY_MODE_TREND_STARTUP
    assert decision.entry_state == ENTRY_STATE_READY
    assert decision.required_score == 70


def test_65_to_69_is_research_only_with_complete_core_families() -> None:
    decision = decide_entry_policy_v4(_request(setup_score=67))

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_TREND_RESEARCH
    assert decision.entry_state == ENTRY_STATE_READY_RESEARCH
    assert decision.required_score == 65


def test_non_price_data_gap_downgrades_eligible_startup_to_research() -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=72,
            direction_confirmation_state="TEMPORARY_CONFIRMED",
            data_confidence=EntryDataConfidenceV4(
                derivatives_data_complete=False
            ),
        )
    )

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_TREND_RESEARCH
    assert decision.entry_state == ENTRY_STATE_READY_RESEARCH
    assert decision.required_score == 70
    assert decision.blocks == ()


def test_non_price_data_gap_downgrades_standard_to_research() -> None:
    decision = decide_entry_policy_v4(
        _request(
            data_confidence=EntryDataConfidenceV4(
                derivatives_data_complete=False
            )
        )
    )

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_TREND_RESEARCH
    assert decision.required_score == 75


def test_formal_70_to_74_with_missing_derivatives_uses_research() -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=72,
            direction_confirmation_state="H4_CONFIRMED",
            data_confidence=EntryDataConfidenceV4(
                derivatives_data_complete=False
            ),
        )
    )

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_mode == ENTRY_MODE_TREND_RESEARCH
    assert decision.entry_state == ENTRY_STATE_READY_RESEARCH
    assert decision.required_score == 70


def test_non_price_data_downgrade_still_requires_research_core_families() -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=90,
            family_scores={**CORE_FAMILIES, "PRICE_PROGRESS": 0},
            data_confidence=EntryDataConfidenceV4(
                derivatives_data_complete=False
            ),
        )
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_PLAN_PENDING
    assert decision.blocks == (
        "PRICE_PROGRESS_SCORE_BELOW_MINIMUM:0<6",
    )


def test_h4_opposition_is_a_hard_block_at_any_score() -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=100,
            direction_confirmation_state="H4_OPPOSED",
        )
    )

    assert decision.decision_action == WATCH
    assert decision.entry_mode == ENTRY_MODE_NONE
    assert decision.entry_state == ENTRY_STATE_DIRECTION_BLOCKED
    assert "H4_DIRECTION_OPPOSED" in decision.blocks


def test_broken_price_history_is_not_downgraded_to_research() -> None:
    decision = decide_entry_policy_v4(
        _request(
            data_confidence=EntryDataConfidenceV4(
                price_data_contiguous=False,
                derivatives_data_complete=False,
            )
        )
    )

    assert decision.decision_action == WATCH
    assert decision.entry_mode == ENTRY_MODE_NONE
    assert decision.entry_state == ENTRY_STATE_DATA_BLOCKED
    assert "PRICE_DATA_DISCONTINUOUS" in decision.blocks


def test_missing_stop_target_and_real_r_are_explicit_plan_blocks() -> None:
    decision = decide_entry_policy_v4(
        _request(
            stop_ready=False,
            target_ready=False,
            net_reward_r=None,
        )
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_PLAN_PENDING
    assert decision.blocks == (
        "STRUCTURE_STOP_UNAVAILABLE",
        "STRUCTURE_TARGET_UNAVAILABLE",
        "NET_REWARD_R_UNAVAILABLE",
    )


def test_trigger_is_a_hard_gate_even_for_a_high_standard_score() -> None:
    decision = decide_entry_policy_v4(
        _request(setup_score=110, trigger_ready=False)
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_PLAN_PENDING
    assert decision.blocks == ("ENTRY_TRIGGER_UNAVAILABLE",)


def test_real_r_below_minimum_cannot_be_overridden_by_score() -> None:
    decision = decide_entry_policy_v4(
        _request(setup_score=100, net_reward_r=1.29)
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_PLAN_PENDING
    assert decision.blocks == ("NET_REWARD_R_BELOW_MINIMUM:1.29<1.3",)


def test_non_finite_real_r_is_a_plan_block() -> None:
    decision = decide_entry_policy_v4(
        _request(setup_score=100, net_reward_r=float("nan"))
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_PLAN_PENDING
    assert decision.blocks == ("NET_REWARD_R_INVALID",)


@pytest.mark.parametrize("net_reward_r", [None, float("nan"), 1.29])
def test_signal_preview_can_defer_net_r_to_execution_layer(
    net_reward_r: float | None,
) -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=100,
            net_reward_r=net_reward_r,
            enforce_net_reward_r=False,
        )
    )

    assert decision.decision_action == ENTRY_LONG
    assert decision.entry_state == ENTRY_STATE_READY
    assert decision.blocks == ()


def test_temporary_direction_cannot_bypass_price_progress_with_high_score() -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=90,
            direction_confirmation_state="PROVISIONAL_CONFIRMED",
            family_scores={**CORE_FAMILIES, "PRICE_PROGRESS": 5},
        )
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_PLAN_PENDING
    assert decision.blocks == ("PRICE_PROGRESS_SCORE_BELOW_MINIMUM:5<6",)


def test_formal_direction_between_70_and_74_still_needs_standard_score() -> None:
    decision = decide_entry_policy_v4(_request(setup_score=74))

    assert decision.decision_action == WATCH
    assert decision.entry_mode == ENTRY_MODE_NONE
    assert decision.entry_state == ENTRY_STATE_SCORE_PENDING
    assert decision.required_score == 75
    assert decision.blocks == ("STANDARD_SCORE_BELOW_MINIMUM:74<75",)


def test_active_veto_has_priority_over_score_and_plan() -> None:
    decision = decide_entry_policy_v4(
        _request(
            setup_score=100,
            stop_ready=False,
            vetoes=("1h trigger opposes long",),
        )
    )

    assert decision.decision_action == WATCH
    assert decision.entry_state == ENTRY_STATE_VETOED
    assert decision.blocks[0] == "ACTIVE_VETO:1h trigger opposes long"


def test_family_aliases_are_supported_without_mutating_input() -> None:
    families = {
        "market_structure": 25,
        "ma_position": 15,
        "entry_trigger": 10,
        "price_efficiency": 6,
    }
    request = _request(
        setup_score=70,
        direction_confirmation_state="TEMPORARY_CONFIRMED",
        family_scores=families,
    )

    first = decide_entry_policy_v4(request)
    second = decide_entry_policy_v4(request)

    assert first == second
    assert first.entry_mode == ENTRY_MODE_TREND_STARTUP
    assert families == {
        "market_structure": 25,
        "ma_position": 15,
        "entry_trigger": 10,
        "price_efficiency": 6,
    }
