from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence


ENTRY_LONG = "ENTRY_LONG"
ENTRY_SHORT = "ENTRY_SHORT"
WATCH = "WATCH"

ENTRY_MODE_STANDARD = "STANDARD"
ENTRY_MODE_TREND_STARTUP = "TREND_STARTUP"
ENTRY_MODE_TREND_RESEARCH = "TREND_RESEARCH"
ENTRY_MODE_NONE = "NONE"

ENTRY_STATE_READY = "READY"
ENTRY_STATE_READY_RESEARCH = "READY_RESEARCH"
ENTRY_STATE_SIGNAL_PENDING = "SIGNAL_PENDING"
ENTRY_STATE_SCORE_PENDING = "SCORE_PENDING"
ENTRY_STATE_DIRECTION_PENDING = "DIRECTION_PENDING"
ENTRY_STATE_DIRECTION_BLOCKED = "DIRECTION_BLOCKED"
ENTRY_STATE_DATA_BLOCKED = "DATA_BLOCKED"
ENTRY_STATE_PLAN_PENDING = "PLAN_PENDING"
ENTRY_STATE_VETOED = "VETOED"

STANDARD_MIN_SCORE = 75
TREND_STARTUP_MIN_SCORE = 70
TREND_RESEARCH_MIN_SCORE = 65
MIN_NET_REWARD_R = 1.3

STARTUP_FAMILY_MINIMUMS: Mapping[str, int] = {
    "STRUCTURE": 25,
    "LOCATION": 15,
    "TRIGGER": 10,
    "PRICE_PROGRESS": 6,
}

_FORMAL_DIRECTION_STATES = frozenset(
    {
        "CONFIRMED",
        "FORMAL_CONFIRMED",
        "H4_CONFIRMED",
        "H4_ALIGNED",
    }
)
_TEMPORARY_DIRECTION_STATES = frozenset(
    {
        "TEMPORARY_CONFIRMED",
        "PROVISIONAL_CONFIRMED",
        "TREND_STARTUP_CONFIRMED",
        "H4_NEUTRAL_TEMPORARY",
        "H4_NEUTRAL_TEMP_CONFIRMED",
        "H4_NEUTRAL_TEMPORARY_CONFIRMED",
    }
)
_OPPOSING_DIRECTION_STATES = frozenset(
    {
        "OPPOSED",
        "OPPOSITE",
        "REVERSED",
        "H4_OPPOSED",
        "H4_OPPOSITE",
        "H4_REVERSED",
    }
)

_FAMILY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "STRUCTURE": ("STRUCTURE", "CORE_STRUCTURE", "MARKET_STRUCTURE"),
    "LOCATION": ("LOCATION", "ENTRY_LOCATION", "MA_POSITION"),
    "TRIGGER": ("TRIGGER", "ENTRY_TRIGGER"),
    "PRICE_PROGRESS": (
        "PRICE_PROGRESS",
        "PRICE_EFFICIENCY",
        "PRICE_ADVANCE_EFFICIENCY",
        "PROGRESS",
    ),
}


@dataclass(frozen=True)
class EntryDataConfidenceV4:
    """Data facts used by the entry policy, separate from opportunity score.

    Broken or stale price history is a hard safety block. Missing non-price
    derivatives evidence only downgrades an otherwise executable setup to the
    paper/research channel; it must not manufacture or reverse direction.
    """

    price_data_contiguous: bool = True
    price_data_fresh: bool = True
    derivatives_data_complete: bool = True


@dataclass(frozen=True)
class EntryPolicyV4Input:
    setup_score: int
    candidate_action: str
    direction_confirmation_state: str
    family_scores: Mapping[str, int] = field(default_factory=dict)
    data_confidence: EntryDataConfidenceV4 = field(
        default_factory=EntryDataConfidenceV4
    )
    structure_ready: bool = False
    location_ready: bool = False
    trigger_ready: bool = False
    stop_ready: bool = False
    target_ready: bool = False
    net_reward_r: float | None = None
    vetoes: Sequence[str] = ()
    min_net_reward_r: float = MIN_NET_REWARD_R


@dataclass(frozen=True)
class EntryPolicyV4Decision:
    decision_action: str
    entry_mode: str
    entry_state: str
    required_score: int
    blocks: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        return self.decision_action in {ENTRY_LONG, ENTRY_SHORT}


def decide_entry_policy_v4(
    request: EntryPolicyV4Input,
) -> EntryPolicyV4Decision:
    """Return the single v4 entry decision without mutating input state.

    Policy summary:
    - Standard entries require score >= 75.
    - A temporary direction under neutral H4 may enter the startup channel at
      score >= 70 only when the four independent startup families are ready.
    - Score 65-69 is research-only and requires the same core family gates.
    - Missing non-price derivatives data downgrades an otherwise eligible
      score >= 70 setup to research; broken price data always blocks.
    - Structure, executable location, stop, target, net R and veto checks are
      gates. They never rewrite the opportunity score.
    """

    action = str(request.candidate_action or "").strip().upper()
    # v4 scores are raw evidence points.  The six independent families can
    # legitimately total 110, so policy evaluation must never silently turn
    # a 101-110 score back into the legacy 100-point scale.
    score = max(0, int(request.setup_score))
    direction_kind = _direction_kind(request.direction_confirmation_state)
    required_score = _required_score(score, direction_kind)

    blocks: list[str] = []
    if action not in {ENTRY_LONG, ENTRY_SHORT}:
        blocks.append("CANDIDATE_ACTION_NOT_ESTABLISHED")
    blocks.extend(
        f"ACTIVE_VETO:{reason}"
        for reason in _normalized_vetoes(request.vetoes)
    )
    if not request.data_confidence.price_data_contiguous:
        blocks.append("PRICE_DATA_DISCONTINUOUS")
    if not request.data_confidence.price_data_fresh:
        blocks.append("PRICE_DATA_STALE")
    if direction_kind == "OPPOSED":
        blocks.append("H4_DIRECTION_OPPOSED")
    elif direction_kind == "PENDING":
        blocks.append("DIRECTION_CONFIRMATION_PENDING")
    if not request.structure_ready:
        blocks.append("CORE_STRUCTURE_UNAVAILABLE")
    if not request.location_ready:
        blocks.append("ENTRY_LOCATION_NOT_READY")
    if not request.trigger_ready:
        blocks.append("ENTRY_TRIGGER_UNAVAILABLE")
    if not request.stop_ready:
        blocks.append("STRUCTURE_STOP_UNAVAILABLE")
    if not request.target_ready:
        blocks.append("STRUCTURE_TARGET_UNAVAILABLE")
    if request.net_reward_r is None:
        blocks.append("NET_REWARD_R_UNAVAILABLE")
    elif not math.isfinite(float(request.net_reward_r)):
        blocks.append("NET_REWARD_R_INVALID")
    elif request.net_reward_r < request.min_net_reward_r:
        blocks.append(
            "NET_REWARD_R_BELOW_MINIMUM:"
            f"{request.net_reward_r:.4g}<{request.min_net_reward_r:.4g}"
        )

    if blocks:
        return _blocked_decision(
            required_score=required_score,
            blocks=blocks,
        )

    family_blocks = _startup_family_blocks(request.family_scores)
    derivatives_complete = (
        request.data_confidence.derivatives_data_complete
    )

    # A formal H4 direction retains the legacy standard channel. A temporary
    # direction can also earn standard status at 75, but it must first satisfy
    # every startup family so score alone cannot bypass provisional-direction
    # quality controls.
    if score >= STANDARD_MIN_SCORE:
        if direction_kind == "TEMPORARY" and family_blocks:
            return _blocked_decision(
                required_score=STANDARD_MIN_SCORE,
                blocks=family_blocks,
            )
        if not derivatives_complete:
            if family_blocks:
                return _blocked_decision(
                    required_score=STANDARD_MIN_SCORE,
                    blocks=family_blocks,
                )
            return _allowed_decision(
                action=action,
                mode=ENTRY_MODE_TREND_RESEARCH,
                required_score=STANDARD_MIN_SCORE,
            )
        return _allowed_decision(
            action=action,
            mode=ENTRY_MODE_STANDARD,
            required_score=STANDARD_MIN_SCORE,
        )

    if score >= TREND_STARTUP_MIN_SCORE:
        if direction_kind == "FORMAL" and not derivatives_complete:
            if family_blocks:
                return _blocked_decision(
                    required_score=TREND_STARTUP_MIN_SCORE,
                    blocks=family_blocks,
                )
            return _allowed_decision(
                action=action,
                mode=ENTRY_MODE_TREND_RESEARCH,
                required_score=TREND_STARTUP_MIN_SCORE,
            )
        if direction_kind != "TEMPORARY":
            return _blocked_decision(
                required_score=STANDARD_MIN_SCORE,
                blocks=(
                    "STANDARD_SCORE_BELOW_MINIMUM:"
                    f"{score}<{STANDARD_MIN_SCORE}",
                ),
            )
        if family_blocks:
            return _blocked_decision(
                required_score=TREND_STARTUP_MIN_SCORE,
                blocks=family_blocks,
            )
        return _allowed_decision(
            action=action,
            mode=(
                ENTRY_MODE_TREND_STARTUP
                if derivatives_complete
                else ENTRY_MODE_TREND_RESEARCH
            ),
            required_score=TREND_STARTUP_MIN_SCORE,
        )

    if score >= TREND_RESEARCH_MIN_SCORE:
        if family_blocks:
            return _blocked_decision(
                required_score=TREND_RESEARCH_MIN_SCORE,
                blocks=family_blocks,
            )
        return _allowed_decision(
            action=action,
            mode=ENTRY_MODE_TREND_RESEARCH,
            required_score=TREND_RESEARCH_MIN_SCORE,
        )

    return _blocked_decision(
        required_score=TREND_RESEARCH_MIN_SCORE,
        blocks=(
            "RESEARCH_SCORE_BELOW_MINIMUM:"
            f"{score}<{TREND_RESEARCH_MIN_SCORE}",
        ),
    )


def _direction_kind(state: str) -> str:
    normalized = str(state or "").strip().upper()
    if normalized in _FORMAL_DIRECTION_STATES:
        return "FORMAL"
    if normalized in _TEMPORARY_DIRECTION_STATES:
        return "TEMPORARY"
    if normalized in _OPPOSING_DIRECTION_STATES:
        return "OPPOSED"
    return "PENDING"


def _required_score(score: int, direction_kind: str) -> int:
    if score < TREND_RESEARCH_MIN_SCORE:
        return TREND_RESEARCH_MIN_SCORE
    if score < TREND_STARTUP_MIN_SCORE:
        return TREND_RESEARCH_MIN_SCORE
    if direction_kind == "TEMPORARY" and score < STANDARD_MIN_SCORE:
        return TREND_STARTUP_MIN_SCORE
    return STANDARD_MIN_SCORE


def _normalized_vetoes(vetoes: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            reason
            for raw_reason in vetoes
            if (reason := str(raw_reason).strip())
        )
    )


def _family_score(
    family_scores: Mapping[str, int],
    canonical_family: str,
) -> int:
    normalized = {
        str(name).strip().upper(): int(value)
        for name, value in family_scores.items()
        if isinstance(value, (int, float))
    }
    return max(
        (
            normalized.get(alias, 0)
            for alias in _FAMILY_ALIASES[canonical_family]
        ),
        default=0,
    )


def _startup_family_blocks(
    family_scores: Mapping[str, int],
) -> tuple[str, ...]:
    blocks: list[str] = []
    for family, minimum in STARTUP_FAMILY_MINIMUMS.items():
        actual = _family_score(family_scores, family)
        if actual < minimum:
            blocks.append(
                f"{family}_SCORE_BELOW_MINIMUM:{actual}<{minimum}"
            )
    return tuple(blocks)


def _blocked_decision(
    *,
    required_score: int,
    blocks: Sequence[str],
) -> EntryPolicyV4Decision:
    normalized_blocks = tuple(dict.fromkeys(str(block) for block in blocks))
    return EntryPolicyV4Decision(
        decision_action=WATCH,
        entry_mode=ENTRY_MODE_NONE,
        entry_state=_entry_state_for_blocks(normalized_blocks),
        required_score=required_score,
        blocks=normalized_blocks,
    )


def _allowed_decision(
    *,
    action: str,
    mode: str,
    required_score: int,
) -> EntryPolicyV4Decision:
    return EntryPolicyV4Decision(
        decision_action=action,
        entry_mode=mode,
        entry_state=(
            ENTRY_STATE_READY_RESEARCH
            if mode == ENTRY_MODE_TREND_RESEARCH
            else ENTRY_STATE_READY
        ),
        required_score=required_score,
        blocks=(),
    )


def _entry_state_for_blocks(blocks: Sequence[str]) -> str:
    if any(block.startswith("ACTIVE_VETO:") for block in blocks):
        return ENTRY_STATE_VETOED
    if any(block.startswith("PRICE_DATA_") for block in blocks):
        return ENTRY_STATE_DATA_BLOCKED
    if "H4_DIRECTION_OPPOSED" in blocks:
        return ENTRY_STATE_DIRECTION_BLOCKED
    if (
        "DIRECTION_CONFIRMATION_PENDING" in blocks
        or "CANDIDATE_ACTION_NOT_ESTABLISHED" in blocks
    ):
        return ENTRY_STATE_DIRECTION_PENDING
    if any(
        block.startswith(
            (
                "CORE_STRUCTURE_",
                "ENTRY_LOCATION_",
                "ENTRY_TRIGGER_",
                "STRUCTURE_STOP_",
                "STRUCTURE_TARGET_",
                "NET_REWARD_R_",
                "STRUCTURE_SCORE_",
                "LOCATION_SCORE_",
                "TRIGGER_SCORE_",
                "PRICE_PROGRESS_SCORE_",
            )
        )
        for block in blocks
    ):
        return ENTRY_STATE_PLAN_PENDING
    if any("SCORE_BELOW_MINIMUM" in block for block in blocks):
        return ENTRY_STATE_SCORE_PENDING
    return ENTRY_STATE_SIGNAL_PENDING
