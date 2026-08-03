from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Mapping, Sequence

from .models import Candle, PositionSide


class EvidenceFamilyV8(str, Enum):
    DIRECTION = "DIRECTION"
    STRUCTURE = "STRUCTURE"
    LOCATION = "LOCATION"
    TRIGGER = "TRIGGER"
    PARTICIPATION = "PARTICIPATION"
    PRICE_PROGRESS = "PRICE_PROGRESS"


SCORE_FAMILY_CAPS: Mapping[str, int] = {
    EvidenceFamilyV8.DIRECTION.value: 20,
    EvidenceFamilyV8.STRUCTURE.value: 30,
    EvidenceFamilyV8.LOCATION.value: 20,
    EvidenceFamilyV8.TRIGGER.value: 15,
    EvidenceFamilyV8.PARTICIPATION.value: 15,
    EvidenceFamilyV8.PRICE_PROGRESS.value: 10,
}
MAX_GROSS_SETUP_SCORE = sum(SCORE_FAMILY_CAPS.values())


PRICE_PROGRESS_UNKNOWN = "UNKNOWN"
PRICE_PROGRESS_NONE = "NO_PROGRESS"
PRICE_PROGRESS_WEAK = "WEAK_PROGRESS"
PRICE_PROGRESS_MODERATE = "MODERATE_PROGRESS"
PRICE_PROGRESS_STRONG = "STRONG_PROGRESS"

CROWD_NORMAL = "NORMAL"
LONG_CROWD_SUSPECTED = "LONG_CROWD_SUSPECTED"
LONG_CROWD_CONFIRMED = "LONG_CROWD_CONFIRMED"
SHORT_CROWD_SUSPECTED = "SHORT_CROWD_SUSPECTED"
SHORT_CROWD_CONFIRMED = "SHORT_CROWD_CONFIRMED"
CROWDING_CONFIRMED_PENALTY = -6


@dataclass(frozen=True)
class RiskPenaltyV8:
    """One score penalty in a named risk channel.

    Multiple observations from the same channel are intentionally allowed as
    input. :func:`compose_score_v8` retains only the most severe (most
    negative) observation for that channel, preventing duplicate deductions.
    """

    channel: str
    penalty: int
    reason: str = ""


@dataclass(frozen=True)
class ScoreV8Result:
    score_breakdown: Mapping[str, int]
    gross_setup_score: int
    risk_penalties: Mapping[str, int]
    total_risk_penalty: int
    setup_score: int


def compose_score_v8(
    family_scores: Mapping[str | EvidenceFamilyV8, int],
    risk_penalties: Sequence[RiskPenaltyV8] = (),
) -> ScoreV8Result:
    """Compose the six v4 evidence families into an unnormalised v8 score.

    Missing families are emitted as zero, so callers always receive a complete
    six-family breakdown. A family is bounded by its own evidence cap, but the
    final positive score is deliberately *not* capped at 100; all six families
    can produce the documented maximum of 110.

    Positive values from an unknown family are rejected. This prevents legacy
    RSI, MA/VWAP or setup-priority points from silently leaking outside the six
    v4 families.
    """

    normalized_input: dict[str, int] = {}
    for raw_family, raw_score in family_scores.items():
        family = (
            raw_family.value
            if isinstance(raw_family, EvidenceFamilyV8)
            else str(raw_family).strip().upper()
        )
        if family not in SCORE_FAMILY_CAPS:
            raise ValueError(f"unknown v8 evidence family: {raw_family!r}")
        score = _integer_score(raw_score, label=f"family {family}")
        # A mapping cannot repeat a key, but aliases such as enum/string input
        # can normalize to the same family. Keep the strongest state only.
        normalized_input[family] = max(normalized_input.get(family, 0), score)

    breakdown: dict[str, int] = {}
    for family, cap in SCORE_FAMILY_CAPS.items():
        breakdown[family] = min(cap, max(0, normalized_input.get(family, 0)))

    selected_penalties: dict[str, int] = {}
    for item in risk_penalties:
        channel = str(item.channel or "").strip().upper()
        if not channel:
            raise ValueError("risk penalty channel must not be empty")
        penalty = _integer_score(item.penalty, label=f"risk channel {channel}")
        if penalty > 0:
            raise ValueError(
                f"risk penalty must be zero or negative: {channel}={penalty}"
            )
        selected_penalties[channel] = min(
            selected_penalties.get(channel, 0),
            penalty,
        )

    gross = sum(breakdown.values())
    total_penalty = sum(selected_penalties.values())
    return ScoreV8Result(
        score_breakdown=breakdown,
        gross_setup_score=gross,
        risk_penalties=selected_penalties,
        total_risk_penalty=total_penalty,
        setup_score=gross + total_penalty,
    )


@dataclass(frozen=True)
class PriceProgressV8:
    side: PositionSide
    state: str
    score: int
    directional_displacement: float
    path: float
    efficiency: float
    magnitude: float
    quality: float
    data_valid: bool
    window_start: datetime | None = None
    window_end: datetime | None = None


def calculate_price_progress_v8(
    closed_candles: Sequence[Candle],
    atr14: float | None,
    side: PositionSide | str,
    *,
    interval: timedelta = timedelta(minutes=15),
    as_of: datetime | None = None,
) -> PriceProgressV8:
    """Calculate PRICE_PROGRESS from the latest five closed 15-minute bars.

    Candle timestamps are treated as bar-open timestamps. If ``as_of`` is
    supplied, the newest bar must have ended no later than ``as_of``. Without
    ``as_of`` the caller is responsible for supplying only already-closed bars,
    which is useful for deterministic historical replay.

    The four path segments use True Range against the preceding close, exactly
    matching the v4 formula. Any insufficient, malformed, non-contiguous or
    non-finite input returns UNKNOWN/0 instead of manufacturing a deduction.
    """

    normalized_side = _position_side(side)
    if len(closed_candles) < 5 or not _positive_finite(atr14):
        return _unknown_progress(normalized_side)
    if interval.total_seconds() <= 0:
        raise ValueError("price progress interval must be positive")

    window = tuple(closed_candles[-5:])
    if not _valid_candle_window(window, interval=interval, as_of=as_of):
        return _unknown_progress(normalized_side)

    side_sign = 1.0 if normalized_side is PositionSide.LONG else -1.0
    displacement = max(
        0.0,
        side_sign * (float(window[-1].close) - float(window[0].close)),
    )
    path = sum(
        _true_range(candle, previous_close=float(previous.close))
        for previous, candle in zip(window, window[1:])
    )
    if not _positive_finite(path):
        # A flat zero-range market is valid data, but it has no progress.
        return PriceProgressV8(
            side=normalized_side,
            state=PRICE_PROGRESS_NONE,
            score=0,
            directional_displacement=displacement,
            path=0.0,
            efficiency=0.0,
            magnitude=0.0,
            quality=0.0,
            data_valid=True,
            window_start=window[0].timestamp,
            window_end=window[-1].timestamp,
        )

    efficiency = displacement / path
    magnitude = min(displacement / float(atr14), 1.0)
    quality = efficiency * magnitude
    unrounded = 10.0 * min(quality / 0.35, 1.0)
    # The PRD says round, and conventional half-up behavior is less surprising
    # than Python's ties-to-even rule at exact x.5 boundaries.
    score = min(10, max(0, int(math.floor(unrounded + 0.5))))

    return PriceProgressV8(
        side=normalized_side,
        state=_price_progress_state(score),
        score=score,
        directional_displacement=displacement,
        path=path,
        efficiency=efficiency,
        magnitude=magnitude,
        quality=quality,
        data_valid=True,
        window_start=window[0].timestamp,
        window_end=window[-1].timestamp,
    )


@dataclass(frozen=True)
class CrowdingEvidenceBarV4:
    """Closed-bar facts used to confirm either side's crowding risk.

    Long and short fields are deliberately explicit instead of relying on a
    fuzzy implicit mirror. ``oi_building`` is direction-neutral and can be one
    (but only one) independent auxiliary fact for either candidate side.
    """

    timestamp: datetime | None = None

    long_price_failure: bool = False
    short_price_failure: bool = False

    long_account_ratio_extreme: bool = False
    long_account_ratio_worsening: bool = False
    short_account_ratio_extreme: bool = False
    short_account_ratio_worsening: bool = False

    long_funding_extreme: bool = False
    long_funding_worsening: bool = False
    short_funding_extreme: bool = False
    short_funding_worsening: bool = False

    oi_building: bool = False
    long_active_flow_reversal: bool = False
    short_active_flow_reversal: bool = False
    long_top_trader_divergence: bool = False
    short_top_trader_divergence: bool = False


@dataclass(frozen=True)
class CrowdingAssessmentV4:
    side: PositionSide
    state: str
    penalty: int
    qualifying_bars: int
    sampled_bars: int
    auxiliary_counts: tuple[int, ...]
    data_contiguous: bool

    @property
    def confirmed(self) -> bool:
        return self.state in {
            LONG_CROWD_CONFIRMED,
            SHORT_CROWD_CONFIRMED,
        }

    def as_risk_penalty(self) -> RiskPenaltyV8 | None:
        if not self.confirmed:
            return None
        return RiskPenaltyV8(
            channel="CROWDING",
            penalty=self.penalty,
            reason=self.state,
        )


def assess_crowding_v4(
    closed_bars: Sequence[CrowdingEvidenceBarV4],
    side: PositionSide | str,
    *,
    interval: timedelta = timedelta(minutes=15),
) -> CrowdingAssessmentV4:
    """Assess confirmed same-side crowding using the latest three closed bars.

    Confirmation requires, on at least two of all three latest bars:

    * a side-specific price-failure fact; and
    * at least two independent auxiliary facts.

    An extreme account ratio, incomplete supporting facts, fewer than three
    bars, or non-contiguous timestamped bars can only be SUSPECTED with zero
    penalty. The -6 deduction applies solely to the assessed candidate side.
    """

    normalized_side = _position_side(side)
    if interval.total_seconds() <= 0:
        raise ValueError("crowding interval must be positive")

    window = tuple(closed_bars[-3:])
    if not window:
        return CrowdingAssessmentV4(
            side=normalized_side,
            state=CROWD_NORMAL,
            penalty=0,
            qualifying_bars=0,
            sampled_bars=0,
            auxiliary_counts=(),
            data_contiguous=False,
        )

    contiguous = _crowding_window_contiguous(window, interval=interval)
    counts = tuple(_crowding_auxiliary_count(bar, normalized_side) for bar in window)
    qualifying = sum(
        1
        for bar, count in zip(window, counts)
        if _crowding_price_failed(bar, normalized_side) and count >= 2
    )
    has_any = any(_has_crowding_evidence(bar, normalized_side) for bar in window)
    can_confirm = len(window) == 3 and contiguous and qualifying >= 2

    if normalized_side is PositionSide.LONG:
        confirmed_state = LONG_CROWD_CONFIRMED
        suspected_state = LONG_CROWD_SUSPECTED
    else:
        confirmed_state = SHORT_CROWD_CONFIRMED
        suspected_state = SHORT_CROWD_SUSPECTED

    return CrowdingAssessmentV4(
        side=normalized_side,
        state=(
            confirmed_state
            if can_confirm
            else suspected_state
            if has_any
            else CROWD_NORMAL
        ),
        penalty=CROWDING_CONFIRMED_PENALTY if can_confirm else 0,
        qualifying_bars=qualifying,
        sampled_bars=len(window),
        auxiliary_counts=counts,
        data_contiguous=contiguous,
    )


def _integer_score(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be a finite integer")
    return int(numeric)


def _position_side(side: PositionSide | str) -> PositionSide:
    if isinstance(side, PositionSide):
        return side
    normalized = str(side or "").strip().upper()
    try:
        return PositionSide(normalized)
    except ValueError as exc:
        raise ValueError(f"unsupported position side: {side!r}") from exc


def _positive_finite(value: object) -> bool:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def _unknown_progress(side: PositionSide) -> PriceProgressV8:
    return PriceProgressV8(
        side=side,
        state=PRICE_PROGRESS_UNKNOWN,
        score=0,
        directional_displacement=0.0,
        path=0.0,
        efficiency=0.0,
        magnitude=0.0,
        quality=0.0,
        data_valid=False,
    )


def _valid_candle_window(
    candles: Sequence[Candle],
    *,
    interval: timedelta,
    as_of: datetime | None,
) -> bool:
    if len(candles) != 5:
        return False
    for candle in candles:
        values = (
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return False
        if candle.high < max(candle.open, candle.close, candle.low):
            return False
        if candle.low > min(candle.open, candle.close, candle.high):
            return False
        if candle.volume < 0:
            return False
    if any(
        current.timestamp - previous.timestamp != interval
        for previous, current in zip(candles, candles[1:])
    ):
        return False
    if as_of is not None and candles[-1].timestamp + interval > as_of:
        return False
    return True


def _true_range(candle: Candle, *, previous_close: float) -> float:
    return max(
        float(candle.high) - float(candle.low),
        abs(float(candle.high) - previous_close),
        abs(float(candle.low) - previous_close),
    )


def _price_progress_state(score: int) -> str:
    if score <= 0:
        return PRICE_PROGRESS_NONE
    if score <= 3:
        return PRICE_PROGRESS_WEAK
    if score <= 6:
        return PRICE_PROGRESS_MODERATE
    return PRICE_PROGRESS_STRONG


def _crowding_window_contiguous(
    bars: Sequence[CrowdingEvidenceBarV4],
    *,
    interval: timedelta,
) -> bool:
    timestamps = tuple(bar.timestamp for bar in bars)
    # ``None`` means the caller has already supplied an ordered closed window.
    if all(timestamp is None for timestamp in timestamps):
        return True
    if any(timestamp is None for timestamp in timestamps):
        return False
    return all(
        current - previous == interval
        for previous, current in zip(timestamps, timestamps[1:])
        if previous is not None and current is not None
    )


def _crowding_price_failed(
    bar: CrowdingEvidenceBarV4,
    side: PositionSide,
) -> bool:
    return (
        bar.long_price_failure
        if side is PositionSide.LONG
        else bar.short_price_failure
    )


def _crowding_auxiliary_count(
    bar: CrowdingEvidenceBarV4,
    side: PositionSide,
) -> int:
    if side is PositionSide.LONG:
        facts = (
            bar.long_account_ratio_extreme
            and bar.long_account_ratio_worsening,
            bar.long_funding_extreme and bar.long_funding_worsening,
            bar.oi_building,
            bar.long_active_flow_reversal,
            bar.long_top_trader_divergence,
        )
    else:
        facts = (
            bar.short_account_ratio_extreme
            and bar.short_account_ratio_worsening,
            bar.short_funding_extreme and bar.short_funding_worsening,
            bar.oi_building,
            bar.short_active_flow_reversal,
            bar.short_top_trader_divergence,
        )
    return sum(bool(fact) for fact in facts)


def _has_crowding_evidence(
    bar: CrowdingEvidenceBarV4,
    side: PositionSide,
) -> bool:
    if side is PositionSide.LONG:
        facts = (
            bar.long_price_failure,
            bar.long_account_ratio_extreme,
            bar.long_account_ratio_worsening,
            bar.long_funding_extreme,
            bar.long_funding_worsening,
            bar.oi_building,
            bar.long_active_flow_reversal,
            bar.long_top_trader_divergence,
        )
    else:
        facts = (
            bar.short_price_failure,
            bar.short_account_ratio_extreme,
            bar.short_account_ratio_worsening,
            bar.short_funding_extreme,
            bar.short_funding_worsening,
            bar.oi_building,
            bar.short_active_flow_reversal,
            bar.short_top_trader_divergence,
        )
    return any(facts)
