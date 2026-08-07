from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final


# Keep the action values compatible with the existing entry-policy consumers.
ENTRY_LONG: Final = "ENTRY_LONG"
ENTRY_SHORT: Final = "ENTRY_SHORT"
WATCH: Final = "WATCH"

ENTRY_MODE_STANDARD: Final = "STANDARD"
ENTRY_MODE_TREND_STARTUP: Final = "TREND_STARTUP"
ENTRY_MODE_TREND_RESEARCH: Final = "TREND_RESEARCH"
ENTRY_MODE_NONE: Final = "NONE"

ENTRY_STATE_READY: Final = "READY"
ENTRY_STATE_READY_RESEARCH: Final = "READY_RESEARCH"
ENTRY_STATE_SCORE_PENDING: Final = "SCORE_PENDING"
ENTRY_STATE_DIRECTION_PENDING: Final = "DIRECTION_PENDING"
ENTRY_STATE_DIRECTION_BLOCKED: Final = "DIRECTION_BLOCKED"
ENTRY_STATE_DATA_BLOCKED: Final = "DATA_BLOCKED"
ENTRY_STATE_STRUCTURE_PENDING: Final = "STRUCTURE_PENDING"
ENTRY_STATE_ENTRY_ZONE_PENDING: Final = "ENTRY_ZONE_PENDING"

ENTRY_ROUTE_H4_ALIGNED: Final = "H4_ALIGNED"
ENTRY_ROUTE_H4_CLEAN_NEUTRAL: Final = "H4_CLEAN_NEUTRAL"
ENTRY_ROUTE_M15_EARLY_RESEARCH: Final = "M15_EARLY_RESEARCH"

LOCATION_STATUS_EXCELLENT: Final = "EXCELLENT"
LOCATION_STATUS_WAITING: Final = "WAITING"

STANDARD_MIN_SCORE: Final = 75
TREND_STARTUP_MIN_SCORE: Final = 70
TREND_RESEARCH_MIN_SCORE: Final = 65

BLOCK_CATEGORY_MARKET_DATA: Final = "MARKET_DATA"
BLOCK_CATEGORY_DIRECTION_STRUCTURE: Final = "DIRECTION_STRUCTURE"
BLOCK_CATEGORY_SCORE_ENTRY: Final = "SCORE_ENTRY"

DIAGNOSTIC_DERIVATIVES_DATA_INCOMPLETE: Final = (
    "DERIVATIVES_DATA_INCOMPLETE"
)

_VALID_ACTIONS: Final = frozenset({ENTRY_LONG, ENTRY_SHORT})
_VALID_ROUTES: Final = frozenset(
    {
        ENTRY_ROUTE_H4_ALIGNED,
        ENTRY_ROUTE_H4_CLEAN_NEUTRAL,
        ENTRY_ROUTE_M15_EARLY_RESEARCH,
    }
)


@dataclass(frozen=True)
class EntryDataConfidenceV7:
    """Signal-stage data facts.

    Price continuity and freshness are safety blockers. Derivatives
    completeness is deliberately diagnostic only: missing OI, funding,
    long/short ratio or active-flow evidence contributes no score upstream,
    but does not block or cap an otherwise valid signal.
    """

    price_data_contiguous: bool = True
    price_data_fresh: bool = True
    derivatives_data_complete: bool = True


@dataclass(frozen=True)
class EntryPolicyV7Input:
    """Facts allowed to participate in the v7.5 signal-stage decision.

    ``entry_route`` is the already-authorized direction route produced by the
    4H/1H/15m router. It selects the published 75/70/65 total-score channel;
    this module does not reconstruct direction from component scores.

    Exact stop, target and R values intentionally do not exist in this input.
    They belong to the execution-plan stage and therefore cannot become
    signal-stage blockers by accident.
    """

    total_score: int | float
    candidate_action: str
    entry_route: str
    data_confidence: EntryDataConfidenceV7 = field(
        default_factory=EntryDataConfidenceV7
    )
    semantic_structure_exists: bool = False
    setup_valid: bool = False
    authorized_entry_zone_exists: bool = False
    current_price_inside_zone: bool = False
    h4_direction_opposed: bool = False


@dataclass(frozen=True)
class EntryPolicyV7Reason:
    """Stable, classified reason suitable for audit-ledger projection."""

    code: str
    category: str
    strategy_hard_veto: bool = False
    detail: str | None = None


@dataclass(frozen=True)
class EntryPolicyV7Decision:
    decision_action: str
    entry_mode: str
    entry_state: str
    required_score: int
    effective_score: int
    location_status: str
    reasons: tuple[EntryPolicyV7Reason, ...]
    diagnostics: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Whether the signal may proceed to execution-plan construction."""

        return self.decision_action in _VALID_ACTIONS and not self.reasons

    @property
    def preliminary_eligible(self) -> bool:
        """Compatibility name for signal-stage eligibility.

        It is intentionally not the final atomic-submit eligibility, which is
        recalculated later with the execution plan and final fill price.
        """

        return self.allowed

    @property
    def blocks(self) -> tuple[str, ...]:
        """Legacy-friendly flat reason codes, retaining deterministic order."""

        return tuple(reason.code for reason in self.reasons)

    @property
    def strategy_hard_vetoes(self) -> tuple[str, ...]:
        return tuple(
            reason.code
            for reason in self.reasons
            if reason.strategy_hard_veto
        )


def decide_entry_policy_v7(
    request: EntryPolicyV7Input,
) -> EntryPolicyV7Decision:
    """Evaluate the v7.5 signal stage without execution-plan assumptions.

    The decision is deliberately narrow:

    * price data must be continuous and fresh;
    * a routed candidate direction, semantic price structure, live setup and
      currently-hit authorized entry zone must exist;
    * the route-specific *total* score must pass;
    * explicit closed-4H opposition is the sole strategy hard veto.

    There are no component-score gates, generic trigger gate, stop/target
    requirement or expected-R requirement in this function.
    """

    action = _normalize_token(request.candidate_action)
    route = _normalize_token(request.entry_route)
    score, score_is_valid = _normalize_score(request.total_score)
    location_status = _location_status(request)

    reasons: list[EntryPolicyV7Reason] = []
    if action not in _VALID_ACTIONS:
        reasons.append(
            _reason(
                "CANDIDATE_ACTION_NOT_ESTABLISHED",
                BLOCK_CATEGORY_DIRECTION_STRUCTURE,
            )
        )
    if route not in _VALID_ROUTES:
        reasons.append(
            _reason(
                "CANDIDATE_ROUTE_NOT_ESTABLISHED",
                BLOCK_CATEGORY_DIRECTION_STRUCTURE,
            )
        )
    if not request.data_confidence.price_data_contiguous:
        reasons.append(
            _reason(
                "PRICE_DATA_DISCONTINUOUS",
                BLOCK_CATEGORY_MARKET_DATA,
            )
        )
    if not request.data_confidence.price_data_fresh:
        reasons.append(
            _reason("PRICE_DATA_STALE", BLOCK_CATEGORY_MARKET_DATA)
        )
    if request.h4_direction_opposed:
        reasons.append(
            _reason(
                "H4_DIRECTION_OPPOSED",
                BLOCK_CATEGORY_DIRECTION_STRUCTURE,
                strategy_hard_veto=True,
            )
        )
    if not request.semantic_structure_exists:
        reasons.append(
            _reason(
                "SEMANTIC_STRUCTURE_UNAVAILABLE",
                BLOCK_CATEGORY_DIRECTION_STRUCTURE,
            )
        )
    elif not request.setup_valid:
        reasons.append(
            _reason(
                "SETUP_INVALID",
                BLOCK_CATEGORY_DIRECTION_STRUCTURE,
            )
        )
    if not request.authorized_entry_zone_exists:
        reasons.append(
            _reason(
                "AUTHORIZED_ENTRY_ZONE_UNAVAILABLE",
                BLOCK_CATEGORY_SCORE_ENTRY,
            )
        )
    elif not request.current_price_inside_zone:
        reasons.append(
            _reason(
                "CURRENT_PRICE_OUTSIDE_AUTHORIZED_ZONE",
                BLOCK_CATEGORY_SCORE_ENTRY,
            )
        )

    required_score = _minimum_score_for_route(route)
    if not score_is_valid:
        reasons.append(
            _reason("TOTAL_SCORE_INVALID", BLOCK_CATEGORY_SCORE_ENTRY)
        )
    elif route in _VALID_ROUTES and score < required_score:
        reasons.append(
            _reason(
                "RESEARCH_SCORE_BELOW_MINIMUM",
                BLOCK_CATEGORY_SCORE_ENTRY,
                detail=f"{score}<{TREND_RESEARCH_MIN_SCORE}",
            )
        )

    diagnostics = (
        ()
        if request.data_confidence.derivatives_data_complete
        else (DIAGNOSTIC_DERIVATIVES_DATA_INCOMPLETE,)
    )
    normalized_reasons = _deduplicate_reasons(reasons)

    if normalized_reasons:
        return EntryPolicyV7Decision(
            decision_action=WATCH,
            entry_mode=ENTRY_MODE_NONE,
            entry_state=_entry_state_for_reasons(normalized_reasons),
            required_score=required_score,
            effective_score=score,
            location_status=location_status,
            reasons=normalized_reasons,
            diagnostics=diagnostics,
        )

    entry_mode, channel_required_score = _score_channel(route, score)
    return EntryPolicyV7Decision(
        decision_action=action,
        entry_mode=entry_mode,
        entry_state=(
            ENTRY_STATE_READY_RESEARCH
            if entry_mode == ENTRY_MODE_TREND_RESEARCH
            else ENTRY_STATE_READY
        ),
        required_score=channel_required_score,
        effective_score=score,
        location_status=location_status,
        reasons=(),
        diagnostics=diagnostics,
    )


def _normalize_token(value: str) -> str:
    return str(value or "").strip().upper()


def _normalize_score(value: int | float) -> tuple[int, bool]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0, False
    if not math.isfinite(numeric):
        return 0, False
    # Existing scores are integral evidence points. Preserve scores above 100
    # and clamp only the impossible negative tail.
    return max(0, int(numeric)), True


def _minimum_score_for_route(route: str) -> int:
    # All three paper-simulation routes have an executable research floor of
    # 65. Their higher thresholds determine the resulting channel, not a
    # second hidden blocker.
    del route
    return TREND_RESEARCH_MIN_SCORE


def _score_channel(route: str, score: int) -> tuple[str, int]:
    if route == ENTRY_ROUTE_H4_ALIGNED:
        if score >= STANDARD_MIN_SCORE:
            return ENTRY_MODE_STANDARD, STANDARD_MIN_SCORE
        return ENTRY_MODE_TREND_RESEARCH, TREND_RESEARCH_MIN_SCORE
    if route == ENTRY_ROUTE_H4_CLEAN_NEUTRAL:
        if score >= TREND_STARTUP_MIN_SCORE:
            return ENTRY_MODE_TREND_STARTUP, TREND_STARTUP_MIN_SCORE
        return ENTRY_MODE_TREND_RESEARCH, TREND_RESEARCH_MIN_SCORE
    return ENTRY_MODE_TREND_RESEARCH, TREND_RESEARCH_MIN_SCORE


def _location_status(request: EntryPolicyV7Input) -> str:
    return (
        LOCATION_STATUS_EXCELLENT
        if (
            request.authorized_entry_zone_exists
            and request.current_price_inside_zone
            and request.setup_valid
        )
        else LOCATION_STATUS_WAITING
    )


def _reason(
    code: str,
    category: str,
    *,
    strategy_hard_veto: bool = False,
    detail: str | None = None,
) -> EntryPolicyV7Reason:
    return EntryPolicyV7Reason(
        code=code,
        category=category,
        strategy_hard_veto=strategy_hard_veto,
        detail=detail,
    )


def _deduplicate_reasons(
    reasons: list[EntryPolicyV7Reason],
) -> tuple[EntryPolicyV7Reason, ...]:
    return tuple({reason.code: reason for reason in reasons}.values())


def _entry_state_for_reasons(
    reasons: tuple[EntryPolicyV7Reason, ...],
) -> str:
    codes = tuple(reason.code for reason in reasons)
    if any(code.startswith("PRICE_DATA_") for code in codes):
        return ENTRY_STATE_DATA_BLOCKED
    if "H4_DIRECTION_OPPOSED" in codes:
        return ENTRY_STATE_DIRECTION_BLOCKED
    if any(
        code in {
            "CANDIDATE_ACTION_NOT_ESTABLISHED",
            "CANDIDATE_ROUTE_NOT_ESTABLISHED",
        }
        for code in codes
    ):
        return ENTRY_STATE_DIRECTION_PENDING
    if any(
        code in {"SEMANTIC_STRUCTURE_UNAVAILABLE", "SETUP_INVALID"}
        for code in codes
    ):
        return ENTRY_STATE_STRUCTURE_PENDING
    if any(
        code in {
            "AUTHORIZED_ENTRY_ZONE_UNAVAILABLE",
            "CURRENT_PRICE_OUTSIDE_AUTHORIZED_ZONE",
        }
        for code in codes
    ):
        return ENTRY_STATE_ENTRY_ZONE_PENDING
    if any(
        code == "TOTAL_SCORE_INVALID"
        or code == "RESEARCH_SCORE_BELOW_MINIMUM"
        for code in codes
    ):
        return ENTRY_STATE_SCORE_PENDING
    return ENTRY_STATE_DIRECTION_PENDING
