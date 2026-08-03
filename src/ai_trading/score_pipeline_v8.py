from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from ai_trading.models import Candle, IndicatorSnapshot, PositionSide
from ai_trading.score_v8 import (
    CrowdingEvidenceBarV4,
    PriceProgressV8,
    RiskPenaltyV8,
    assess_crowding_v4,
    calculate_price_progress_v8,
    compose_score_v8,
)


DIRECTION_SCORE_GAP = 8


@dataclass(frozen=True)
class LevelSelectionV8:
    key: str
    zone: Mapping[str, float]
    score: int
    state: str
    structural: bool

    @property
    def ready(self) -> bool:
        return self.score >= 15 and bool(self.key) and bool(self.zone)


@dataclass(frozen=True)
class SideScoreV8:
    side: PositionSide
    score_breakdown: Mapping[str, int]
    gross_setup_score: int
    risk_penalties: Mapping[str, int]
    total_risk_penalty: int
    setup_score: int
    direction_confirmation_state: str
    direction_state: str
    structure_state: str
    trigger_state: str
    participation_state: str
    price_progress: PriceProgressV8
    crowding_state: str
    selected_level: LevelSelectionV8
    structure_stop: float | None
    structure_target: float | None
    evidence_event_ids: Mapping[str, str]
    derivatives_data_complete: bool


@dataclass(frozen=True)
class DualScoreV8:
    long: SideScoreV8
    short: SideScoreV8
    selected_side: PositionSide | None
    score_gap: int


@dataclass(frozen=True)
class _StructureEvidence:
    score: int = 0
    state: str = "NONE"
    timeframe: str = ""
    event_id: str = ""
    source_event_id: str = ""
    level_key: str = ""
    zone: Mapping[str, float] | None = None
    projected_target: float | None = None
    trigger_score: int = 0
    trigger_state: str = "NONE"
    trigger_event_id: str = ""
    trigger_source_event_id: str = ""


@dataclass(frozen=True)
class _ParticipationEvidence:
    score: int
    state: str
    event_id: str
    data_complete: bool


def evaluate_dual_score_v8(
    *,
    symbol: str,
    price: float,
    timeframe_candles: Mapping[str, Sequence[Candle]],
    timeframe_indicators: Mapping[str, Sequence[IndicatorSnapshot]],
    context: Mapping[str, object],
    long_ratio_extreme: float,
    short_ratio_extreme: float,
    funding_hot_long: float,
    funding_hot_short: float,
) -> DualScoreV8:
    """Score long and short completely before selecting either direction.

    This adapter intentionally consumes structured context only.  Display
    reasons are never parsed, so changing Chinese/English copy cannot change a
    score, selected level, or entry direction.
    """

    long = _evaluate_side(
        symbol=symbol,
        side=PositionSide.LONG,
        price=price,
        timeframe_candles=timeframe_candles,
        timeframe_indicators=timeframe_indicators,
        context=context,
        long_ratio_extreme=long_ratio_extreme,
        short_ratio_extreme=short_ratio_extreme,
        funding_hot_long=funding_hot_long,
        funding_hot_short=funding_hot_short,
    )
    short = _evaluate_side(
        symbol=symbol,
        side=PositionSide.SHORT,
        price=price,
        timeframe_candles=timeframe_candles,
        timeframe_indicators=timeframe_indicators,
        context=context,
        long_ratio_extreme=long_ratio_extreme,
        short_ratio_extreme=short_ratio_extreme,
        funding_hot_long=funding_hot_long,
        funding_hot_short=funding_hot_short,
    )
    gap = abs(long.setup_score - short.setup_score)
    selected_side = _select_scored_side(long, short, gap=gap)
    return DualScoreV8(
        long=long,
        short=short,
        selected_side=selected_side,
        score_gap=gap,
    )


def _select_scored_side(
    long: SideScoreV8,
    short: SideScoreV8,
    *,
    gap: int,
) -> PositionSide | None:
    """Choose only after both sides are fully scored.

    A side whose 4-hour direction is opposed or whose direction is still
    pending is not allowed to suppress a valid opposite-side candidate merely
    because low-timeframe/location evidence gave it a similar raw score.  The
    score-gap rule is applied only when both directions are independently
    established.
    """

    established_states = {"H4_CONFIRMED", "TEMPORARY_CONFIRMED"}
    long_ready = long.direction_confirmation_state in established_states
    short_ready = short.direction_confirmation_state in established_states
    if long_ready and not short_ready:
        return PositionSide.LONG
    if short_ready and not long_ready:
        return PositionSide.SHORT
    if not long_ready and not short_ready:
        return None
    if gap < DIRECTION_SCORE_GAP:
        return None
    return (
        PositionSide.LONG
        if long.setup_score > short.setup_score
        else PositionSide.SHORT
    )


def location_for_selected_level_v8(
    *,
    price: float,
    side: PositionSide,
    key: str,
    zone: Mapping[str, object],
    structural: bool,
    atr14: float | None,
) -> LevelSelectionV8:
    """Refresh only the selected v8 location for a live mark price."""

    normalized = _normalized_zone(zone)
    if not key or normalized is None:
        return LevelSelectionV8("", {}, 0, "UNAVAILABLE", False)
    low = normalized["low"]
    high = normalized["high"]
    hit = low <= price <= high
    advantage = hit and _advantage_side_hit(price, side, low, high)
    if structural and advantage:
        score, state = 20, "STRUCTURAL_ADVANTAGE"
    elif hit:
        score, state = 15, "IN_VALID_ZONE"
    else:
        distance = min(abs(price - low), abs(price - high))
        extension_limit = max(float(atr14 or 0.0) * 2.0, price * 0.02)
        score, state = (
            (8, "DIRECTIONAL_NOT_EXTENDED")
            if distance <= extension_limit
            else (0, "EXTENDED_OR_WRONG_SIDE")
        )
    return LevelSelectionV8(
        key=key,
        zone=normalized,
        score=score,
        state=state,
        structural=structural,
    )


def _evaluate_side(
    *,
    symbol: str,
    side: PositionSide,
    price: float,
    timeframe_candles: Mapping[str, Sequence[Candle]],
    timeframe_indicators: Mapping[str, Sequence[IndicatorSnapshot]],
    context: Mapping[str, object],
    long_ratio_extreme: float,
    short_ratio_extreme: float,
    funding_hot_long: float,
    funding_hot_short: float,
) -> SideScoreV8:
    m15_candles = tuple(timeframe_candles.get("15m", ()))
    m15_indicators = tuple(timeframe_indicators.get("15m", ()))
    current_m15 = m15_indicators[-1] if m15_indicators else None
    atr14 = current_m15.atr14 if current_m15 is not None else None
    progress = calculate_price_progress_v8(
        m15_candles,
        atr14,
        side,
    )
    micro = _micro_structure_evidence(
        symbol,
        side,
        m15_candles,
        atr14,
    )
    direction_score, confirmation_state, direction_state, direction_event = (
        _direction_evidence(
            symbol,
            side,
            context,
            timeframe_candles,
            micro,
        )
    )
    structure = _strongest_structure_evidence(
        symbol,
        side,
        context,
        timeframe_candles,
        micro,
        progress,
    )
    trigger_score, trigger_state, trigger_event = _trigger_evidence(
        symbol,
        side,
        context,
        timeframe_candles,
        micro,
        structure,
        progress,
    )
    levels = _side_levels(context, side)
    if micro.zone and micro.level_key:
        levels[micro.level_key] = dict(micro.zone)
    location = _select_location(
        side=side,
        price=price,
        levels=levels,
        atr14=atr14,
        direction_score=direction_score,
    )
    participation = _participation_evidence(
        symbol,
        side,
        m15_indicators,
    )
    crowding = _crowding_evidence(
        side=side,
        candles=m15_candles,
        indicators=m15_indicators,
        long_ratio_extreme=long_ratio_extreme,
        short_ratio_extreme=short_ratio_extreme,
        funding_hot_long=funding_hot_long,
        funding_hot_short=funding_hot_short,
    )
    crowding_state = (
        "UNKNOWN"
        if not participation.data_complete and crowding.state == "NORMAL"
        else crowding.state
    )
    penalties = tuple(
        penalty
        for penalty in (crowding.as_risk_penalty(),)
        if penalty is not None
    )
    composed = compose_score_v8(
        {
            "DIRECTION": direction_score,
            "STRUCTURE": structure.score,
            "LOCATION": location.score,
            "TRIGGER": trigger_score,
            "PARTICIPATION": participation.score,
            "PRICE_PROGRESS": progress.score,
        },
        penalties,
    )
    stop = _structure_stop(side, price, location, atr14, trigger_score)
    target = _nearest_structure_target(
        side,
        price,
        context,
        structure.projected_target,
    )
    evidence_ids = {
        family: event_id
        for family, event_id, score in (
            ("DIRECTION", direction_event, direction_score),
            ("STRUCTURE", structure.event_id, structure.score),
            ("LOCATION", _level_event_id(symbol, side, location), location.score),
            ("TRIGGER", trigger_event, trigger_score),
            ("PARTICIPATION", participation.event_id, participation.score),
            (
                "PRICE_PROGRESS",
                _progress_event_id(symbol, side, progress),
                progress.score,
            ),
        )
        if score > 0 and event_id
    }
    return SideScoreV8(
        side=side,
        score_breakdown=composed.score_breakdown,
        gross_setup_score=composed.gross_setup_score,
        risk_penalties=composed.risk_penalties,
        total_risk_penalty=composed.total_risk_penalty,
        setup_score=composed.setup_score,
        direction_confirmation_state=confirmation_state,
        direction_state=direction_state,
        structure_state=structure.state,
        trigger_state=trigger_state,
        participation_state=participation.state,
        price_progress=progress,
        crowding_state=crowding_state,
        selected_level=location,
        structure_stop=stop,
        structure_target=target,
        evidence_event_ids=evidence_ids,
        derivatives_data_complete=participation.data_complete,
    )


def _direction_evidence(
    symbol: str,
    side: PositionSide,
    context: Mapping[str, object],
    timeframe_candles: Mapping[str, Sequence[Candle]],
    micro: _StructureEvidence,
) -> tuple[int, str, str, str]:
    expected = side.value
    opposite = PositionSide.SHORT.value if side is PositionSide.LONG else PositionSide.LONG.value
    expected_daily = "BULL" if side is PositionSide.LONG else "BEAR"
    opposite_daily = "BEAR" if side is PositionSide.LONG else "BULL"
    daily = str(context.get("daily_bias") or "NEUTRAL").upper()
    h4 = _mapping(context.get("h4_structure"))
    h1 = _mapping(context.get("h1_structure"))
    h1_trigger = _mapping(context.get("h1_trigger"))
    h4_direction = _normalized_structure_direction(h4)
    h1_direction = _normalized_structure_direction(h1)
    trigger_direction = str(h1_trigger.get("direction") or "NONE").upper()
    event = _event_id(
        symbol,
        side,
        "DIRECTION",
        "4h",
        _latest_timestamp(timeframe_candles.get("4h", ())),
        f"{daily}_{h4_direction}",
    )
    if h4_direction == opposite:
        return 0, "H4_OPPOSED", "H4_OPPOSED", event
    if h4_direction == expected:
        if daily == expected_daily:
            return 20, "H4_CONFIRMED", "D1_H4_ALIGNED", event
        if daily == opposite_daily:
            return 8, "H4_CONFIRMED", "H4_ALIGNED_D1_HEADWIND", event
        return 15, "H4_CONFIRMED", "H4_ALIGNED", event
    provisional = (
        h1_direction == expected
        or trigger_direction == expected
        or micro.score >= 20
    )
    if provisional:
        if daily == opposite_daily:
            return 6, "TEMPORARY_CONFIRMED", "PROVISIONAL_D1_HEADWIND", event
        return 12, "TEMPORARY_CONFIRMED", "PROVISIONAL_STARTUP", event
    if daily == expected_daily:
        return 5, "DIRECTION_PENDING", "D1_BACKGROUND_ONLY", event
    return 0, "DIRECTION_PENDING", "NO_DIRECTION", event


def _strongest_structure_evidence(
    symbol: str,
    side: PositionSide,
    context: Mapping[str, object],
    timeframe_candles: Mapping[str, Sequence[Candle]],
    micro: _StructureEvidence,
    progress: PriceProgressV8,
) -> _StructureEvidence:
    del progress
    candidates = [micro]
    for timeframe, key in (("4h", "h4_structure"), ("1h", "h1_structure")):
        structure = _mapping(context.get(key))
        score, state = _mapped_structure_score(side, structure)
        if score:
            candidates.append(
                _StructureEvidence(
                    score=score,
                    state=state,
                    timeframe=timeframe,
                    event_id=_context_structure_event_id(
                        symbol,
                        side,
                        timeframe,
                        structure,
                        state,
                    ),
                    source_event_id=_context_structure_source_event_id(
                        symbol,
                        side,
                        timeframe,
                        structure,
                        state,
                    ),
                )
            )
    h1_trigger = _mapping(context.get("h1_trigger"))
    if not any(candidate.score >= 20 for candidate in candidates):
        score, state = _trigger_as_structure_score(side, h1_trigger)
        if score:
            candidates.append(
                _StructureEvidence(
                    score=score,
                    state=state,
                    timeframe="1h",
                    event_id=_context_structure_event_id(
                        symbol,
                        side,
                        "1h",
                        h1_trigger,
                        state,
                    ),
                    source_event_id=_context_structure_source_event_id(
                        symbol,
                        side,
                        "1h",
                        h1_trigger,
                        state,
                    ),
                )
            )
    selected = max(
        candidates,
        key=lambda item: (
            item.score,
            {"4h": 3, "1h": 2, "15m": 1}.get(item.timeframe, 0),
        ),
    )
    return selected


def _trigger_evidence(
    symbol: str,
    side: PositionSide,
    context: Mapping[str, object],
    timeframe_candles: Mapping[str, Sequence[Candle]],
    micro: _StructureEvidence,
    structure: _StructureEvidence,
    progress: PriceProgressV8,
) -> tuple[int, str, str]:
    del progress
    candidates: list[tuple[int, str, str]] = []
    if (
        micro.trigger_score
        and micro.trigger_source_event_id != structure.source_event_id
    ):
        candidates.append(
            (
                micro.trigger_score,
                micro.trigger_state,
                micro.trigger_event_id,
            )
        )
    h1 = _mapping(context.get("h1_trigger"))
    h1_direction = str(h1.get("direction") or "NONE").upper()
    h1_state = str(h1.get("state") or "UNKNOWN").upper()
    expected = side.value
    if h1_direction == expected:
        base = (
            10
            if h1_state in {"RETEST", "FAKE_BREAKOUT", "FAKE_BREAKDOWN"}
            else 5
            if h1_state in {"BREAKOUT", "BREAKDOWN"}
            else 0
        )
        if base:
            event = _context_trigger_event_id(
                symbol,
                side,
                "1h",
                h1,
                h1_state,
            )
            source_event = _context_structure_source_event_id(
                symbol,
                side,
                "1h",
                h1,
                h1_state,
            )
            if (
                source_event != structure.source_event_id
                and not _same_h1_structure_trigger_event(
                    structure.state,
                    h1_state,
                )
            ):
                candidates.append((base, f"H1_{h1_state}", event))
    precision = _mapping(context.get("m15_precision"))
    expected_pullback = (
        "M15_LONG_PULLBACK"
        if side is PositionSide.LONG
        else "M15_SHORT_PULLBACK"
    )
    if str(precision.get("pullback") or "").upper() == expected_pullback:
        event = _event_id(
            symbol,
            side,
            "TRIGGER",
            "15m",
            _latest_timestamp(timeframe_candles.get("15m", ())),
            expected_pullback,
        )
        source_event = _source_event_id(
            symbol,
            side,
            "15m",
            _latest_timestamp(timeframe_candles.get("15m", ())),
        )
        if source_event != structure.source_event_id:
            candidates.append(
                (
                    10,
                    expected_pullback,
                    event,
                )
            )
    if not candidates:
        return 0, "WAIT", ""
    return max(candidates, key=lambda item: item[0])


def _participation_evidence(
    symbol: str,
    side: PositionSide,
    indicators: Sequence[IndicatorSnapshot],
) -> _ParticipationEvidence:
    recent = tuple(indicators[-3:])
    data_complete = len(recent) == 3 and all(
        item.oi_change is not None and _active_flow_ratio(item) is not None
        for item in recent
    )
    consecutive = 0
    for item in reversed(recent):
        ratio = _active_flow_ratio(item)
        oi_supports = item.oi_change is not None and item.oi_change > 0
        flow_supports = (
            ratio is not None
            and (
                ratio >= 1.05
                if side is PositionSide.LONG
                else ratio <= (1.0 / 1.05)
            )
        )
        if not (oi_supports and flow_supports):
            break
        consecutive += 1
    score = min(15, consecutive * 5)
    timestamp = recent[-1].timestamp if recent else None
    return _ParticipationEvidence(
        score=score,
        state=(
            f"CONTINUOUS_{consecutive}"
            if consecutive
            else "UNCONFIRMED"
        ),
        event_id=(
            _event_id(
                symbol,
                side,
                "PARTICIPATION",
                "15m",
                timestamp,
                f"CONTINUOUS_{consecutive}",
            )
            if score
            else ""
        ),
        data_complete=data_complete,
    )


def _crowding_evidence(
    *,
    side: PositionSide,
    candles: Sequence[Candle],
    indicators: Sequence[IndicatorSnapshot],
    long_ratio_extreme: float,
    short_ratio_extreme: float,
    funding_hot_long: float,
    funding_hot_short: float,
):
    usable = min(len(candles), len(indicators))
    if usable < 4:
        return assess_crowding_v4((), side)
    candles = tuple(candles[-usable:])
    indicators = tuple(indicators[-usable:])
    bars: list[CrowdingEvidenceBarV4] = []
    for index in range(usable - 3, usable):
        candle = candles[index]
        previous_candle = candles[index - 1]
        current = indicators[index]
        previous = indicators[index - 1]
        current_ratio = current.long_short_ratio
        previous_ratio = previous.long_short_ratio
        current_funding = current.funding_rate
        previous_funding = previous.funding_rate
        current_flow = _active_flow_ratio(current)
        previous_flow = _active_flow_ratio(previous)
        candle_range = max(candle.high - candle.low, 0.0)
        upper_rejection = (
            candle_range > 0
            and (candle.high - max(candle.open, candle.close)) / candle_range >= 0.45
        )
        lower_rejection = (
            candle_range > 0
            and (min(candle.open, candle.close) - candle.low) / candle_range >= 0.45
        )
        long_failed = (
            candle.close < previous_candle.close
            or (candle.high > previous_candle.high and candle.close <= previous_candle.high)
            or upper_rejection
        )
        short_failed = (
            candle.close > previous_candle.close
            or (candle.low < previous_candle.low and candle.close >= previous_candle.low)
            or lower_rejection
        )
        long_extreme = current_ratio is not None and current_ratio >= long_ratio_extreme
        short_extreme = current_ratio is not None and current_ratio <= short_ratio_extreme
        bars.append(
            CrowdingEvidenceBarV4(
                timestamp=candle.timestamp,
                long_price_failure=long_failed,
                short_price_failure=short_failed,
                long_account_ratio_extreme=long_extreme,
                long_account_ratio_worsening=(
                    long_extreme
                    and previous_ratio is not None
                    and current_ratio is not None
                    and current_ratio > previous_ratio
                ),
                short_account_ratio_extreme=short_extreme,
                short_account_ratio_worsening=(
                    short_extreme
                    and previous_ratio is not None
                    and current_ratio is not None
                    and current_ratio < previous_ratio
                ),
                long_funding_extreme=(
                    current_funding is not None
                    and current_funding >= funding_hot_long
                ),
                long_funding_worsening=(
                    current_funding is not None
                    and previous_funding is not None
                    and current_funding > previous_funding
                ),
                short_funding_extreme=(
                    current_funding is not None
                    and current_funding <= funding_hot_short
                ),
                short_funding_worsening=(
                    current_funding is not None
                    and previous_funding is not None
                    and current_funding < previous_funding
                ),
                oi_building=(current.oi_change or 0.0) > 0,
                long_active_flow_reversal=(
                    current_flow is not None
                    and (
                        current_flow < 1.0
                        or (
                            previous_flow is not None
                            and current_flow < previous_flow
                        )
                    )
                ),
                short_active_flow_reversal=(
                    current_flow is not None
                    and (
                        current_flow > 1.0
                        or (
                            previous_flow is not None
                            and current_flow > previous_flow
                        )
                    )
                ),
                long_top_trader_divergence=(
                    long_extreme
                    and current.top_position_long_short_ratio is not None
                    and current.top_position_long_short_ratio < 1.0
                ),
                short_top_trader_divergence=(
                    short_extreme
                    and current.top_position_long_short_ratio is not None
                    and current.top_position_long_short_ratio > 1.0
                ),
            )
        )
    return assess_crowding_v4(bars, side)


def _micro_structure_evidence(
    symbol: str,
    side: PositionSide,
    candles: Sequence[Candle],
    atr14: float | None,
) -> _StructureEvidence:
    if len(candles) < 11:
        return _StructureEvidence()
    # Keep the structural acceptance event and the executable trigger on
    # different closed bars.  The breakout is observed first, the next bar
    # establishes acceptance/retest structure, and only the newest bar may
    # become the independent trigger.
    history = tuple(candles[-23:-3])
    if len(history) < 8:
        return _StructureEvidence()
    breakout, confirmation, current = candles[-3:]
    atr = float(atr14 or current.close * 0.01)
    buffer = atr * 0.10
    range_high = max(candle.high for candle in history)
    range_low = min(candle.low for candle in history)
    if side is PositionSide.LONG:
        initial_break = breakout.close > range_high + buffer
        current_break = (
            current.close > max(range_high, confirmation.high) + buffer
        )
        accepted = initial_break and confirmation.close >= range_high
        retest_held = (
            accepted
            and confirmation.low <= range_high + atr * 0.50
            and confirmation.close >= range_high
        )
        current_held = accepted and current.close >= range_high
        renewed = (
            current_held
            and current.low <= range_high + atr * 0.50
            and current.close > confirmation.close
        )
        if accepted:
            score = 30 if retest_held else 25
            state = (
                "M15_BREAKOUT_RETEST_HELD"
                if retest_held
                else "M15_BREAKOUT_ACCEPTED"
            )
            event_timestamp = confirmation.timestamp
            trigger_score = 15 if renewed else 10 if current_held else 0
            trigger_state = (
                "M15_RENEWED_PROGRESS"
                if renewed
                else "M15_POST_STRUCTURE_HOLD"
                if current_held
                else "WAIT"
            )
            zone = {"low": range_high - atr * 0.35, "high": range_high + atr * 0.35, "price": range_high}
            target = range_high + (range_high - range_low)
        elif current_break:
            score, state, event_timestamp = 20, "M15_BREAKOUT", current.timestamp
            trigger_score, trigger_state, zone = 0, "SAME_EVENT_AS_STRUCTURE", {"low": range_high - atr * 0.35, "high": range_high + atr * 0.35, "price": range_high}
            target = range_high + (range_high - range_low)
        elif _directional_sequence(candles[-4:], side):
            score, state, event_timestamp = 12, "M15_HIGHER_SEQUENCE", current.timestamp
            trigger_score, trigger_state, zone, target = 0, "NONE", None, None
        else:
            return _StructureEvidence()
        key = "v8_breakout_retest"
    else:
        initial_break = breakout.close < range_low - buffer
        current_break = (
            current.close < min(range_low, confirmation.low) - buffer
        )
        accepted = initial_break and confirmation.close <= range_low
        retest_held = (
            accepted
            and confirmation.high >= range_low - atr * 0.50
            and confirmation.close <= range_low
        )
        current_held = accepted and current.close <= range_low
        renewed = (
            current_held
            and current.high >= range_low - atr * 0.50
            and current.close < confirmation.close
        )
        if accepted:
            score = 30 if retest_held else 25
            state = (
                "M15_BREAKDOWN_RETEST_HELD"
                if retest_held
                else "M15_BREAKDOWN_ACCEPTED"
            )
            event_timestamp = confirmation.timestamp
            trigger_score = 15 if renewed else 10 if current_held else 0
            trigger_state = (
                "M15_RENEWED_PROGRESS"
                if renewed
                else "M15_POST_STRUCTURE_HOLD"
                if current_held
                else "WAIT"
            )
            zone = {"low": range_low - atr * 0.35, "high": range_low + atr * 0.35, "price": range_low}
            target = range_low - (range_high - range_low)
        elif current_break:
            score, state, event_timestamp = 20, "M15_BREAKDOWN", current.timestamp
            trigger_score, trigger_state, zone = 0, "SAME_EVENT_AS_STRUCTURE", {"low": range_low - atr * 0.35, "high": range_low + atr * 0.35, "price": range_low}
            target = range_low - (range_high - range_low)
        elif _directional_sequence(candles[-4:], side):
            score, state, event_timestamp = 12, "M15_LOWER_SEQUENCE", current.timestamp
            trigger_score, trigger_state, zone, target = 0, "NONE", None, None
        else:
            return _StructureEvidence()
        key = "v8_breakdown_retest"
    structure_event = _event_id(
        symbol,
        side,
        "STRUCTURE",
        "15m",
        event_timestamp,
        state,
    )
    structure_source_event = _source_event_id(
        symbol,
        side,
        "15m",
        event_timestamp,
    )
    trigger_event = (
        _event_id(
            symbol,
            side,
            "TRIGGER",
            "15m",
            current.timestamp,
            trigger_state,
        )
        if trigger_score
        else ""
    )
    trigger_source_event = (
        _source_event_id(
            symbol,
            side,
            "15m",
            current.timestamp,
        )
        if trigger_score
        else ""
    )
    return _StructureEvidence(
        score=score,
        state=state,
        timeframe="15m",
        event_id=structure_event,
        source_event_id=structure_source_event,
        level_key=key,
        zone=zone,
        projected_target=target if target and target > 0 else None,
        trigger_score=trigger_score,
        trigger_state=trigger_state,
        trigger_event_id=trigger_event,
        trigger_source_event_id=trigger_source_event,
    )


def _mapped_structure_score(
    side: PositionSide,
    structure: Mapping[str, object],
) -> tuple[int, str]:
    expected = side.value
    direction = _normalized_structure_direction(structure)
    structure_type = str(structure.get("structure_type") or "UNKNOWN").upper()
    state = str(structure.get("state") or "UNKNOWN").upper()
    if direction != expected:
        return 0, "NONE"
    if structure_type in {"DESCENDING_BREAKOUT_RETEST", "ASCENDING_BREAKDOWN_RETEST"}:
        return 25, structure_type
    if state in {"BREAKOUT_UP", "BREAKDOWN_DOWN"} or structure_type in {
        "BREAKOUT_UP",
        "BREAKDOWN_DOWN",
        "DESCENDING_BREAKOUT_PENDING",
        "ASCENDING_BREAKDOWN_PENDING",
    }:
        return 20, structure_type if structure_type != "RANGE" else state
    if structure_type in {"ASCENDING_SUPPORT", "DESCENDING_RESISTANCE"}:
        return 12, structure_type
    return 0, "NONE"


def _trigger_as_structure_score(
    side: PositionSide,
    trigger: Mapping[str, object],
) -> tuple[int, str]:
    if str(trigger.get("direction") or "NONE").upper() != side.value:
        return 0, "NONE"
    state = str(trigger.get("state") or "UNKNOWN").upper()
    if state in {"RETEST", "FAKE_BREAKOUT", "FAKE_BREAKDOWN"}:
        return 25, f"H1_{state}"
    if state in {"BREAKOUT", "BREAKDOWN"}:
        return 20, f"H1_{state}"
    return 0, "NONE"


def _select_location(
    *,
    side: PositionSide,
    price: float,
    levels: Mapping[str, Mapping[str, object]],
    atr14: float | None,
    direction_score: int,
) -> LevelSelectionV8:
    structural_keys = (
        {
            "v8_breakout_retest",
            "h1_support",
            "h4_support",
            "sweep_reclaim_support",
            "breakout_retest",
            "ma_cluster_breakout",
        }
        if side is PositionSide.LONG
        else {
            "v8_breakdown_retest",
            "h1_resistance",
            "h4_resistance",
            "h4_descending_resistance",
            "sweep_reject_resistance",
            "breakdown_retest",
            "ma_cluster_breakdown",
            "distribution_range_high",
            "descending_high_trendline",
        }
    )
    candidates: list[tuple[int, float, str, Mapping[str, float], bool]] = []
    for key, raw_zone in levels.items():
        zone = _normalized_zone(raw_zone)
        if zone is None:
            continue
        low, high = zone["low"], zone["high"]
        distance = 0.0 if low <= price <= high else min(abs(price - low), abs(price - high))
        structural = key in structural_keys
        priority = 2 if structural else 1
        candidates.append((priority, distance, key, zone, structural))
    if not candidates:
        return LevelSelectionV8("", {}, 0, "UNAVAILABLE", False)
    hits = [item for item in candidates if item[1] == 0.0]
    selected = (
        max(hits, key=lambda item: (item[0], -item[1]))
        if hits
        else min(candidates, key=lambda item: (item[1], -item[0]))
    )
    _, distance, key, zone, structural = selected
    if hits:
        advantage = _advantage_side_hit(price, side, zone["low"], zone["high"])
        if structural and advantage:
            return LevelSelectionV8(key, zone, 20, "STRUCTURAL_ADVANTAGE", True)
        return LevelSelectionV8(key, zone, 15, "IN_VALID_ZONE", structural)
    extension_limit = max(float(atr14 or 0.0) * 2.0, price * 0.02)
    if direction_score > 0 and distance <= extension_limit:
        return LevelSelectionV8(key, zone, 8, "DIRECTIONAL_NOT_EXTENDED", structural)
    return LevelSelectionV8(key, zone, 0, "EXTENDED_OR_WRONG_SIDE", structural)


def _structure_stop(
    side: PositionSide,
    price: float,
    location: LevelSelectionV8,
    atr14: float | None,
    trigger_score: int,
) -> float | None:
    if (
        not location.ready
        or not location.structural
        or trigger_score < 10
    ):
        return None
    buffer = max(float(atr14 or 0.0) * 0.25, price * 0.002)
    if side is PositionSide.LONG:
        stop = location.zone["low"] - buffer
        return stop if 0 < stop < price else None
    stop = location.zone["high"] + buffer
    return stop if stop > price else None


def _nearest_structure_target(
    side: PositionSide,
    price: float,
    context: Mapping[str, object],
    projected_target: float | None,
) -> float | None:
    candidates: list[float] = []
    for key in ("h1_structure", "h4_structure", "h1_trigger"):
        mapping = _mapping(context.get(key))
        target_keys = (
            ("resistance", "resistance_zone_high")
            if side is PositionSide.LONG
            else ("support", "support_zone_low")
        )
        for target_key in target_keys:
            value = _float_or_none(mapping.get(target_key))
            if _profitable_target(side, price, value):
                candidates.append(float(value))
    for key in ("h1_ma_cluster", "h4_ma_cluster"):
        mapping = _mapping(context.get(key))
        target_key = "target_up" if side is PositionSide.LONG else "target_down"
        value = _float_or_none(mapping.get(target_key))
        if _profitable_target(side, price, value):
            candidates.append(float(value))
    if _profitable_target(side, price, projected_target):
        candidates.append(float(projected_target))
    if not candidates:
        return None
    return min(candidates) if side is PositionSide.LONG else max(candidates)


def _side_levels(
    context: Mapping[str, object],
    side: PositionSide,
) -> dict[str, Mapping[str, object]]:
    all_levels = _mapping(context.get("entry_levels"))
    side_key = "long" if side is PositionSide.LONG else "short"
    raw = _mapping(all_levels.get(side_key))
    return {
        str(key): value
        for key, value in raw.items()
        if isinstance(value, Mapping)
    }


def _normalized_structure_direction(structure: Mapping[str, object]) -> str:
    direction = str(structure.get("direction") or "NEUTRAL").upper()
    state = str(structure.get("state") or "UNKNOWN").upper()
    if direction == "NEUTRAL" and state == "BREAKOUT_UP":
        return "LONG"
    if direction == "NEUTRAL" and state == "BREAKDOWN_DOWN":
        return "SHORT"
    return direction


def _directional_sequence(candles: Sequence[Candle], side: PositionSide) -> bool:
    if len(candles) < 4:
        return False
    if side is PositionSide.LONG:
        return all(
            current.close > previous.close and current.low >= previous.low
            for previous, current in zip(candles, candles[1:])
        )
    return all(
        current.close < previous.close and current.high <= previous.high
        for previous, current in zip(candles, candles[1:])
    )


def _active_flow_ratio(indicator: IndicatorSnapshot) -> float | None:
    if indicator.taker_buy_sell_ratio is not None:
        return float(indicator.taker_buy_sell_ratio)
    if (
        indicator.taker_buy_volume is not None
        and indicator.taker_sell_volume is not None
        and indicator.taker_sell_volume > 0
    ):
        return indicator.taker_buy_volume / indicator.taker_sell_volume
    return None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _normalized_zone(value: Mapping[str, object]) -> dict[str, float] | None:
    values = [
        item
        for item in (
            _float_or_none(value.get("low")),
            _float_or_none(value.get("high")),
            _float_or_none(value.get("price")),
        )
        if item is not None
    ]
    if not values:
        return None
    return {
        "low": min(values),
        "high": max(values),
        "price": (
            float(value["price"])
            if _float_or_none(value.get("price")) is not None
            else sum(values) / len(values)
        ),
    }


def _advantage_side_hit(
    price: float,
    side: PositionSide,
    low: float,
    high: float,
) -> bool:
    if high <= low:
        return price == low
    if side is PositionSide.LONG:
        return low <= price <= low + (high - low) * 0.60
    return high - (high - low) * 0.60 <= price <= high


def _profitable_target(
    side: PositionSide,
    price: float,
    target: float | None,
) -> bool:
    if target is None:
        return False
    return target > price if side is PositionSide.LONG else target < price


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_timestamp(candles: Sequence[Candle]) -> datetime | None:
    return candles[-1].timestamp if candles else None


def _event_id(
    symbol: str,
    side: PositionSide,
    family: str,
    timeframe: str,
    timestamp: datetime | None,
    state: str,
) -> str:
    if timestamp is None:
        return ""
    return (
        f"{symbol.upper()}:{side.value}:{family}:{timeframe}:"
        f"{timestamp.isoformat()}:{state}"
    )


def _context_structure_event_id(
    symbol: str,
    side: PositionSide,
    timeframe: str,
    structure: Mapping[str, object],
    state: str,
) -> str:
    anchor = _context_structure_anchor(side, structure, state)
    return (
        f"{symbol.upper()}:{side.value}:STRUCTURE:{timeframe}:"
        f"{anchor}"
    )


def _context_trigger_event_id(
    symbol: str,
    side: PositionSide,
    timeframe: str,
    trigger: Mapping[str, object],
    state: str,
) -> str:
    anchor = _context_structure_anchor(side, trigger, state)
    return (
        f"{symbol.upper()}:{side.value}:TRIGGER:{timeframe}:"
        f"{anchor}"
    )


def _context_structure_source_event_id(
    symbol: str,
    side: PositionSide,
    timeframe: str,
    structure: Mapping[str, object],
    state: str,
) -> str:
    anchor = _context_structure_anchor(side, structure, state)
    return (
        f"{symbol.upper()}:{side.value}:SOURCE:{timeframe}:"
        f"{anchor}"
    )


def _context_structure_anchor(
    side: PositionSide,
    structure: Mapping[str, object],
    state: str,
) -> str:
    # Upstream context does not yet expose a dedicated event timestamp.  Use
    # it when available; otherwise fingerprint only structural state/anchors.
    # This remains stable while the same structure persists across new bars,
    # and changes when a support/resistance structure actually advances.
    explicit = next(
        (
            str(structure.get(key)).strip()
            for key in ("event_timestamp", "confirmed_at", "anchor_time")
            if str(structure.get(key) or "").strip()
        ),
        "",
    )
    if explicit:
        return f"{state}:{explicit}"
    parts = [
        str(structure.get("direction") or "NEUTRAL").upper(),
        str(structure.get("state") or state or "UNKNOWN").upper(),
        str(structure.get("structure_type") or "UNKNOWN").upper(),
    ]
    anchor_keys = (
        ("support", "support_zone_low", "support_zone_high")
        if side is PositionSide.LONG
        else (
            "resistance",
            "resistance_zone_low",
            "resistance_zone_high",
        )
    )
    for key in anchor_keys:
        value = _float_or_none(structure.get(key))
        if value is not None:
            parts.append(f"{key}={value:.10g}")
    return ":".join(parts)


def _same_h1_structure_trigger_event(
    structure_state: str,
    trigger_state: str,
) -> bool:
    structure = str(structure_state or "").upper()
    trigger = str(trigger_state or "").upper()
    if trigger == "BREAKOUT":
        return structure in {"BREAKOUT_UP", "H1_BREAKOUT"}
    if trigger == "BREAKDOWN":
        return structure in {"BREAKDOWN_DOWN", "H1_BREAKDOWN"}
    if trigger == "RETEST":
        return structure in {
            "H1_RETEST",
            "DESCENDING_BREAKOUT_RETEST",
            "ASCENDING_BREAKDOWN_RETEST",
        }
    if trigger == "FAKE_BREAKOUT":
        return structure == "H1_FAKE_BREAKOUT"
    if trigger == "FAKE_BREAKDOWN":
        return structure == "H1_FAKE_BREAKDOWN"
    return False


def _source_event_id(
    symbol: str,
    side: PositionSide,
    timeframe: str,
    timestamp: datetime | None,
) -> str:
    if timestamp is None:
        return ""
    return (
        f"{symbol.upper()}:{side.value}:SOURCE:{timeframe}:"
        f"{timestamp.isoformat()}"
    )


def _level_event_id(
    symbol: str,
    side: PositionSide,
    location: LevelSelectionV8,
) -> str:
    if not location.key:
        return ""
    low = _float_or_none(location.zone.get("low"))
    high = _float_or_none(location.zone.get("high"))
    anchor = (
        f"{low:.10g}:{high:.10g}"
        if low is not None and high is not None
        else "UNKNOWN"
    )
    return (
        f"{symbol.upper()}:{side.value}:LOCATION:{location.key}:{anchor}"
    )


def _progress_event_id(
    symbol: str,
    side: PositionSide,
    progress: PriceProgressV8,
) -> str:
    if progress.window_end is None:
        return ""
    return (
        f"{symbol.upper()}:{side.value}:PRICE_PROGRESS:15m:"
        f"{progress.window_end.isoformat()}"
    )
