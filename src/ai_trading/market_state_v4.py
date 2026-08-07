from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
import hashlib
import json
import math
from statistics import median
from typing import Sequence

from ai_trading.models import Candle


MARKET_STATE_VERSION = 4
CHANNEL_GEOMETRY_VERSION = 2


class MarketStateV4(str, Enum):
    CHANNEL_UP = "CHANNEL_UP"
    CHANNEL_DOWN = "CHANNEL_DOWN"
    STRONG_UP = "STRONG_UP"
    STRONG_DOWN = "STRONG_DOWN"
    RANGE = "RANGE"


class H4BackgroundV4(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    CLEAN_NEUTRAL = "CLEAN_NEUTRAL"
    CONFLICT_NEUTRAL = "CONFLICT_NEUTRAL"


class H4StructureKindV4(str, Enum):
    ASCENDING_SUPPORT = "ASCENDING_SUPPORT"
    DESCENDING_RESISTANCE = "DESCENDING_RESISTANCE"
    DESCENDING_BREAKOUT_RETEST = "DESCENDING_BREAKOUT_RETEST"
    ASCENDING_BREAKDOWN_RETEST = "ASCENDING_BREAKDOWN_RETEST"
    CLOSED_BREAKOUT = "CLOSED_BREAKOUT"
    CLOSED_BREAKDOWN = "CLOSED_BREAKDOWN"
    HH_HL = "HH_HL"
    LH_LL = "LH_LL"


class DirectionV4(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class DataStatusV4(str, Enum):
    COMPLETE = "COMPLETE"
    INSUFFICIENT = "INSUFFICIENT"
    DISCONTINUOUS = "DISCONTINUOUS"
    INVALID = "INVALID"


class ChannelShapeV4(str, Enum):
    PAIRED_GEOMETRIC_CHANNEL = "PAIRED_GEOMETRIC_CHANNEL"
    PRIMARY_RAIL_TREND = "PRIMARY_RAIL_TREND"


class RangePositionV4(str, Enum):
    LOWER_EDGE = "LOWER_EDGE"
    UPPER_EDGE = "UPPER_EDGE"
    MIDDLE = "MIDDLE"
    OVERLAP = "OVERLAP"
    OUTSIDE = "OUTSIDE"
    NONE = "NONE"


class SetupKindV4(str, Enum):
    TREND = "TREND"
    RANGE_EDGE = "RANGE_EDGE"
    EARLY_RESEARCH = "EARLY_RESEARCH"
    H4_BULLISH_SUPPORT = "H4_BULLISH_SUPPORT"
    H4_BULLISH_BREAKOUT_RETEST = "H4_BULLISH_BREAKOUT_RETEST"
    H4_BEARISH_RESISTANCE = "H4_BEARISH_RESISTANCE"
    H4_BEARISH_BREAKDOWN_RETEST = "H4_BEARISH_BREAKDOWN_RETEST"


@dataclass(frozen=True)
class MarketStateV4Config:
    atr_period: int = 14
    pivot_left: int = 2
    pivot_right: int = 2
    touch_atr_multiple: float = 0.15
    invalidation_atr_multiple: float = 0.40
    ema_fast_period: int = 20
    ema_slow_period: int = 60
    ema_enter_quantile: float = 0.55
    ema_exit_quantile: float = 0.35
    ema_min_strength_samples: int = 12

    def __post_init__(self) -> None:
        if self.atr_period < 2:
            raise ValueError("atr_period must be at least 2")
        if self.pivot_left < 1 or self.pivot_right < 1:
            raise ValueError("pivot confirmation spans must be positive")
        if self.touch_atr_multiple <= 0:
            raise ValueError("touch_atr_multiple must be positive")
        if self.invalidation_atr_multiple <= 0:
            raise ValueError("invalidation_atr_multiple must be positive")
        if self.touch_atr_multiple >= self.invalidation_atr_multiple:
            raise ValueError(
                "touch_atr_multiple must be below invalidation_atr_multiple"
            )
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("ema_fast_period must be shorter than ema_slow_period")
        if not (
            0 <= self.ema_exit_quantile <= self.ema_enter_quantile <= 1
        ):
            raise ValueError(
                "EMA quantiles must satisfy 0 <= exit <= enter <= 1"
            )
        if self.ema_min_strength_samples < 1:
            raise ValueError("ema_min_strength_samples must be positive")


DEFAULT_CONFIG = MarketStateV4Config()


@dataclass(frozen=True)
class SwingPointV4:
    event_id: str
    index: int
    timestamp: datetime
    kind: str
    price: float
    atr: float


@dataclass(frozen=True)
class RailV4:
    structure_id: str
    direction: DirectionV4
    side: str
    anchor_ids: tuple[str, ...]
    first_anchor_index: int
    last_anchor_index: int
    confirmation_index: int
    confirmation_time: datetime
    last_validation_index: int
    validation_count: int
    span_bars: int
    slope_per_bar: float
    intercept: float
    touch_buffer: float
    invalidation_buffer: float
    residual: float
    current_boundary: float
    entry_side_valid: bool

    def boundary_at(self, index: int) -> float:
        return self.intercept + self.slope_per_bar * index


@dataclass(frozen=True)
class FrozenRangeV4:
    structure_id: str
    lower: float
    upper: float
    support_anchor_ids: tuple[str, ...]
    resistance_anchor_ids: tuple[str, ...]
    confirmation_index: int
    confirmation_time: datetime
    first_anchor_time: datetime
    last_validation_index: int
    validation_count: int
    span_bars: int
    touch_buffer: float
    invalidation_buffer: float
    residual: float
    position: RangePositionV4


@dataclass(frozen=True)
class MarketStateSnapshotV4:
    state: MarketStateV4 | None
    data_status: DataStatusV4
    as_of: datetime
    data_cutoff: datetime | None
    market_state_version: int = MARKET_STATE_VERSION
    channel_geometry_version: int = CHANNEL_GEOMETRY_VERSION
    direction: DirectionV4 | None = None
    channel_shape: ChannelShapeV4 | None = None
    channel_id: str | None = None
    range_id: str | None = None
    range_tradeable: bool | None = None
    range_position: RangePositionV4 = RangePositionV4.NONE
    primary_rail: RailV4 | None = None
    paired_rail: RailV4 | None = None
    frozen_range: FrozenRangeV4 | None = None
    entry_permission: bool = False
    direction_conflict: bool = False
    structure_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class H4StructureFactV4:
    direction: DirectionV4
    kind: H4StructureKindV4 | str
    structure_id: str
    confirmed_at: datetime
    invalidated: bool = False
    invalidated_at: datetime | None = None
    last_validated_at: datetime | None = None


@dataclass(frozen=True)
class H4BackgroundSnapshotV4:
    background: H4BackgroundV4 | None
    data_status: DataStatusV4
    as_of: datetime
    data_cutoff: datetime | None
    market_state_version: int = MARKET_STATE_VERSION
    channel_geometry_version: int = CHANNEL_GEOMETRY_VERSION
    bullish_structure_ids: tuple[str, ...] = ()
    bearish_structure_ids: tuple[str, ...] = ()
    source: str | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateSetupV4:
    direction: DirectionV4
    kind: SetupKindV4 = SetupKindV4.TREND
    setup_id: str = ""
    valid: bool = True
    entry_zone_exists: bool = True
    entry_zone_hit: bool = False
    structure_confirmed: bool = True
    invalidation_anchor_exists: bool = True
    mirror_structure_opposed: bool = False


@dataclass(frozen=True)
class DirectionRouteDecisionV4:
    allowed: bool
    direction: DirectionV4 | None
    provisional: bool = False
    hard_veto: bool = False
    reason_code: str | None = None


@dataclass(frozen=True)
class RoutedCandidateV4:
    candidate: CandidateSetupV4
    decision: DirectionRouteDecisionV4


@dataclass(frozen=True)
class DirectionRoutingResultV4:
    routes: tuple[RoutedCandidateV4, ...]
    allowed_directions: tuple[DirectionV4, ...]
    direction_conflict: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EarlyResearchDecisionV4:
    candidate: CandidateSetupV4 | None
    data_status: DataStatusV4
    as_of: datetime
    data_cutoff: datetime | None
    market_state_version: int = MARKET_STATE_VERSION
    channel_geometry_version: int = CHANNEL_GEOMETRY_VERSION
    structure_id: str | None = None
    structure_mode: str | None = None
    invalidation_anchor_id: str | None = None
    invalidation_price: float | None = None
    invalidation_buffer: float | None = None
    entry_zone: tuple[float, float] | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class _PreparedCandles:
    candles: tuple[Candle, ...]
    atr: tuple[float | None, ...]
    status: DataStatusV4
    data_cutoff: datetime | None
    reason: str | None = None


@dataclass(frozen=True)
class _StructureSignal:
    direction: DirectionV4
    structure_id: str
    anchor_ids: tuple[str, ...]
    first_index: int
    last_index: int


@dataclass(frozen=True)
class _PointCluster:
    points: tuple[SwingPointV4, ...]
    boundary: float
    residual: float


@dataclass(frozen=True)
class _EarlyStructureFact:
    structure_id: str
    mode: str
    confirmation_index: int
    zone_anchor: SwingPointV4
    invalidation_anchor: SwingPointV4


@dataclass(frozen=True)
class _ActiveH4Fact:
    direction: DirectionV4
    structure_id: str
    confirmed_at: datetime
    last_validated_at: datetime
    span_bars: int = 0


def classify_market_state_v4(
    candles: Sequence[Candle],
    *,
    as_of: datetime,
    previous_snapshot: MarketStateSnapshotV4 | None = None,
    config: MarketStateV4Config = DEFAULT_CONFIG,
) -> MarketStateSnapshotV4:
    """Classify one immutable 1H snapshot from closed candles only.

    Candle timestamps are interpreted as interval-open timestamps. Candles
    whose one-hour interval has not closed at ``as_of`` are ignored.
    Realtime and replay callers pass the immediately preceding snapshot so a
    retired frozen range cannot be rebuilt from anchors preceding its breach.
    ``None`` is reserved for bootstrap; a historical bootstrap should replay
    closed candles sequentially and then persist the returned snapshot.
    """

    observed_at = _aware(as_of)
    prepared = _prepare_candles(
        candles,
        as_of=observed_at,
        timeframe=timedelta(hours=1),
        config=config,
    )
    if prepared.status is not DataStatusV4.COMPLETE:
        return MarketStateSnapshotV4(
            state=None,
            data_status=prepared.status,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            reasons=((prepared.reason,) if prepared.reason else ()),
        )

    pivots = _confirmed_pivots(prepared, "1h", config)
    (
        carried_primary,
        retired_primary_direction,
        minimum_primary_anchor_index,
    ) = _carry_forward_primary_rail(
        prepared,
        previous_snapshot,
        timeframe_name="1h",
        config=config,
    )
    bullish_rail = (
        carried_primary
        if carried_primary is not None
        and carried_primary.direction is DirectionV4.LONG
        else _select_primary_rail(
            prepared,
            pivots,
            DirectionV4.LONG,
            "1h",
            config,
            minimum_anchor_index=(
                minimum_primary_anchor_index
                if retired_primary_direction is DirectionV4.LONG
                else 0
            ),
        )
    )
    bearish_rail = (
        carried_primary
        if carried_primary is not None
        and carried_primary.direction is DirectionV4.SHORT
        else _select_primary_rail(
            prepared,
            pivots,
            DirectionV4.SHORT,
            "1h",
            config,
            minimum_anchor_index=(
                minimum_primary_anchor_index
                if retired_primary_direction is DirectionV4.SHORT
                else 0
            ),
        )
    )
    carried_range, minimum_range_anchor_index = _carry_forward_range(
        prepared,
        previous_snapshot,
        timeframe_name="1h",
    )
    frozen_range = carried_range
    if frozen_range is None:
        frozen_range = _select_frozen_range(
            prepared,
            pivots,
            "1h",
            config,
            minimum_anchor_index=minimum_range_anchor_index,
        )
    paired_up = (
        _select_paired_rail(
            prepared,
            pivots,
            bullish_rail,
            "1h",
            config,
        )
        if bullish_rail is not None
        else None
    )
    paired_down = (
        _select_paired_rail(
            prepared,
            pivots,
            bearish_rail,
            "1h",
            config,
        )
        if bearish_rail is not None
        else None
    )

    latest_index = len(prepared.candles) - 1
    latest_close = prepared.candles[-1].close
    active_rails = tuple(
        rail for rail in (bullish_rail, bearish_rail) if rail is not None
    )
    widest_rail_span = max(
        (rail.span_bars for rail in active_rails),
        default=-1,
    )
    if (
        frozen_range is not None
        and frozen_range.span_bars > widest_rail_span
    ):
        return _range_snapshot(
            prepared,
            observed_at,
            frozen_range,
            reason="HIGHER_SPAN_FROZEN_RANGE",
        )

    if bullish_rail is not None and bearish_rail is not None:
        return MarketStateSnapshotV4(
            state=MarketStateV4.RANGE,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            range_tradeable=False,
            range_position=RangePositionV4.NONE,
            entry_permission=False,
            direction_conflict=True,
            structure_ids=tuple(
                sorted(
                    (
                        bullish_rail.structure_id,
                        bearish_rail.structure_id,
                    )
                )
            ),
            reasons=("OPPOSING_PRIMARY_RAILS",),
        )

    if bullish_rail is not None and _paired_contains(
        bullish_rail,
        paired_up,
        latest_index,
        latest_close,
    ):
        return _channel_snapshot(
            prepared,
            observed_at,
            bullish_rail,
            paired_up,
        )
    if bearish_rail is not None and _paired_contains(
        bearish_rail,
        paired_down,
        latest_index,
        latest_close,
    ):
        return _channel_snapshot(
            prepared,
            observed_at,
            bearish_rail,
            paired_down,
        )

    bullish_structure = _directional_structure(
        prepared,
        pivots,
        DirectionV4.LONG,
        "1h",
        config,
    )
    bearish_structure = _directional_structure(
        prepared,
        pivots,
        DirectionV4.SHORT,
        "1h",
        config,
    )
    range_contains = frozen_range is not None
    if (
        bullish_structure is not None
        and bearish_structure is not None
    ):
        return MarketStateSnapshotV4(
            state=MarketStateV4.RANGE,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            range_tradeable=False,
            direction_conflict=True,
            structure_ids=tuple(
                sorted(
                    (
                        bullish_structure.structure_id,
                        bearish_structure.structure_id,
                    )
                )
            ),
            reasons=("OPPOSING_DIRECTIONAL_STRUCTURES",),
        )
    if (
        bullish_structure is not None
        and bearish_rail is None
        and not range_contains
        and paired_up is None
    ):
        return _strong_snapshot(
            prepared,
            observed_at,
            bullish_structure,
        )
    if (
        bearish_structure is not None
        and bullish_rail is None
        and not range_contains
        and paired_down is None
    ):
        return _strong_snapshot(
            prepared,
            observed_at,
            bearish_structure,
        )

    if bullish_rail is not None:
        return _channel_snapshot(
            prepared,
            observed_at,
            bullish_rail,
            None,
        )
    if bearish_rail is not None:
        return _channel_snapshot(
            prepared,
            observed_at,
            bearish_rail,
            None,
        )
    if frozen_range is not None:
        return _range_snapshot(
            prepared,
            observed_at,
            frozen_range,
            reason="FROZEN_RANGE",
        )
    return MarketStateSnapshotV4(
        state=MarketStateV4.RANGE,
        data_status=DataStatusV4.COMPLETE,
        as_of=observed_at,
        data_cutoff=prepared.data_cutoff,
        range_tradeable=False,
        range_position=RangePositionV4.NONE,
        entry_permission=False,
        reasons=("NO_RELIABLE_BOUNDARY",),
    )


def classify_h4_background_v4(
    candles: Sequence[Candle],
    *,
    as_of: datetime,
    structure_facts: Sequence[H4StructureFactV4] = (),
    previous_background: (
        H4BackgroundSnapshotV4 | H4BackgroundV4
    ) = H4BackgroundV4.CLEAN_NEUTRAL,
    config: MarketStateV4Config = DEFAULT_CONFIG,
) -> H4BackgroundSnapshotV4:
    """Return the three-way 4H background plus explicit conflict-neutral.

    The 35-percent EMA retention threshold is available only when
    ``previous_background`` is a complete snapshot whose source is
    ``EMA_FALLBACK``.  A bare direction enum cannot prove that provenance and
    is therefore evaluated with the 55-percent new-entry threshold.
    """

    observed_at = _aware(as_of)
    prepared = _prepare_candles(
        candles,
        as_of=observed_at,
        timeframe=timedelta(hours=4),
        config=config,
    )
    if prepared.status is not DataStatusV4.COMPLETE:
        return H4BackgroundSnapshotV4(
            background=None,
            data_status=prepared.status,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            reasons=((prepared.reason,) if prepared.reason else ()),
        )

    pivots = _confirmed_pivots(prepared, "4h", config)
    bullish_rail = _select_primary_rail(
        prepared,
        pivots,
        DirectionV4.LONG,
        "4h",
        config,
    )
    bearish_rail = _select_primary_rail(
        prepared,
        pivots,
        DirectionV4.SHORT,
        "4h",
        config,
    )
    frozen_range = _select_frozen_range(
        prepared,
        pivots,
        "4h",
        config,
    )
    bullish_structure = _directional_structure(
        prepared,
        pivots,
        DirectionV4.LONG,
        "4h",
        config,
    )
    bearish_structure = _directional_structure(
        prepared,
        pivots,
        DirectionV4.SHORT,
        "4h",
        config,
    )

    detected_bullish: list[tuple[str, int]] = []
    detected_bearish: list[tuple[str, int]] = []
    if bullish_rail is not None:
        detected_bullish.append(
            (bullish_rail.structure_id, bullish_rail.span_bars)
        )
    if bearish_rail is not None:
        detected_bearish.append(
            (bearish_rail.structure_id, bearish_rail.span_bars)
        )
    if bullish_structure is not None:
        detected_bullish.append(
            (
                bullish_structure.structure_id,
                bullish_structure.last_index - bullish_structure.first_index,
            )
        )
    if bearish_structure is not None:
        detected_bearish.append(
            (
                bearish_structure.structure_id,
                bearish_structure.last_index - bearish_structure.first_index,
            )
        )

    closed_fact_cutoff = prepared.data_cutoff or observed_at
    active_external = _active_h4_facts(
        structure_facts,
        cutoff=closed_fact_cutoff,
        frozen_range=frozen_range,
    )
    external_bullish = {
        fact.structure_id
        for fact in active_external
        if fact.direction is DirectionV4.LONG
    }
    external_bearish = {
        fact.structure_id
        for fact in active_external
        if fact.direction is DirectionV4.SHORT
    }

    detected_spans = [
        span for _, span in (*detected_bullish, *detected_bearish)
    ]
    higher_range_dominates = (
        frozen_range is not None
        and bool(detected_spans)
        and frozen_range.span_bars > max(detected_spans)
    )
    bullish_ids = set(external_bullish)
    bearish_ids = set(external_bearish)
    if not higher_range_dominates:
        bullish_ids.update(item_id for item_id, _ in detected_bullish)
        bearish_ids.update(item_id for item_id, _ in detected_bearish)

    if bullish_ids and bearish_ids:
        return H4BackgroundSnapshotV4(
            background=H4BackgroundV4.CONFLICT_NEUTRAL,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            bullish_structure_ids=tuple(sorted(bullish_ids)),
            bearish_structure_ids=tuple(sorted(bearish_ids)),
            source="PRICE_STRUCTURE_CONFLICT",
            reasons=("H4_STRUCTURE_CONFLICT",),
        )
    if bullish_ids:
        return H4BackgroundSnapshotV4(
            background=H4BackgroundV4.BULLISH,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            bullish_structure_ids=tuple(sorted(bullish_ids)),
            source="PRICE_STRUCTURE",
        )
    if bearish_ids:
        return H4BackgroundSnapshotV4(
            background=H4BackgroundV4.BEARISH,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            bearish_structure_ids=tuple(sorted(bearish_ids)),
            source="PRICE_STRUCTURE",
        )

    if frozen_range is not None and (
        higher_range_dominates or not detected_spans
    ):
        return H4BackgroundSnapshotV4(
            background=H4BackgroundV4.CLEAN_NEUTRAL,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            source="FROZEN_RANGE",
            reasons=("H4_RELIABLE_RANGE",),
        )

    ema_direction = _ema_fallback_direction(
        prepared,
        _previous_ema_background(previous_background),
        config,
    )
    if ema_direction is DirectionV4.LONG:
        return H4BackgroundSnapshotV4(
            background=H4BackgroundV4.BULLISH,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            source="EMA_FALLBACK",
        )
    if ema_direction is DirectionV4.SHORT:
        return H4BackgroundSnapshotV4(
            background=H4BackgroundV4.BEARISH,
            data_status=DataStatusV4.COMPLETE,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            source="EMA_FALLBACK",
        )
    return H4BackgroundSnapshotV4(
        background=H4BackgroundV4.CLEAN_NEUTRAL,
        data_status=DataStatusV4.COMPLETE,
        as_of=observed_at,
        data_cutoff=prepared.data_cutoff,
        source="NO_RELIABLE_DIRECTION",
    )


def route_candidate_direction_v4(
    h4: H4BackgroundSnapshotV4 | H4BackgroundV4,
    h1: MarketStateSnapshotV4,
    candidate: CandidateSetupV4,
) -> DirectionRouteDecisionV4:
    """Apply the single v7.5 authority table exactly once."""

    h4_data_complete = (
        not isinstance(h4, H4BackgroundSnapshotV4)
        or h4.data_status is DataStatusV4.COMPLETE
    )
    background = h4.background if isinstance(h4, H4BackgroundSnapshotV4) else h4
    if (
        not h4_data_complete
        or h1.data_status is not DataStatusV4.COMPLETE
        or background is None
    ):
        return _blocked_route("MARKET_STATE_DATA_BLOCKED")
    if not candidate.valid:
        return _blocked_route("SETUP_INVALID")
    if not candidate.setup_id:
        return _blocked_route("SETUP_ID_MISSING")
    if not candidate.entry_zone_exists:
        return _blocked_route("AUTHORIZED_ENTRY_ZONE_MISSING")
    if background is H4BackgroundV4.CONFLICT_NEUTRAL:
        return _blocked_route("H4_STRUCTURE_CONFLICT")

    if _h4_opposes(background, candidate.direction):
        return DirectionRouteDecisionV4(
            allowed=False,
            direction=None,
            hard_veto=True,
            reason_code="H4_DIRECTION_OPPOSED",
        )

    if candidate.kind is SetupKindV4.EARLY_RESEARCH:
        return _route_early_research(background, h1, candidate)

    state_direction = h1.direction
    h4_specific = _h4_authorizes_specific(background, candidate)
    if h1.state is MarketStateV4.RANGE:
        if h1.range_tradeable:
            if _correct_range_edge(h1.range_position, candidate):
                return _allowed_route(candidate.direction)
            if h4_specific and candidate.entry_zone_hit:
                return _allowed_route(candidate.direction)
            return _blocked_route("RANGE_EDGE_DIRECTION_NOT_AUTHORIZED")
        if h4_specific:
            return _allowed_route(candidate.direction)
        return _blocked_route("NO_AUTHORIZED_DIRECTION")

    if (
        candidate.kind is SetupKindV4.TREND
        and state_direction is candidate.direction
    ):
        return _allowed_route(candidate.direction)
    if h4_specific and background in {
        H4BackgroundV4.BULLISH,
        H4BackgroundV4.BEARISH,
    }:
        return _allowed_route(candidate.direction)
    return _blocked_route("MARKET_STATE_DIRECTION_NOT_AUTHORIZED")


def route_candidate_directions_v4(
    h4: H4BackgroundSnapshotV4 | H4BackgroundV4,
    h1: MarketStateSnapshotV4,
    candidates: Sequence[CandidateSetupV4],
) -> DirectionRoutingResultV4:
    """Route a complete same-snapshot candidate set without loop-order bias.

    This is the authoritative multi-candidate entry point. In particular, two
    opposite provisional 15m candidates under CLEAN_NEUTRAL cannot each win by
    being visited first; both are retained as diagnostics and neither side is
    authorized.
    """

    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.setup_id,
                item.kind.value,
                item.direction.value,
                item.valid,
                item.entry_zone_exists,
                item.entry_zone_hit,
                item.structure_confirmed,
                item.invalidation_anchor_exists,
                item.mirror_structure_opposed,
            ),
        )
    )
    routed = tuple(
        RoutedCandidateV4(
            candidate=candidate,
            decision=route_candidate_direction_v4(h4, h1, candidate),
        )
        for candidate in ordered
    )
    provisional_directions = {
        route.decision.direction
        for route in routed
        if route.decision.allowed
        and route.decision.provisional
        and route.decision.direction is not None
    }
    if len(provisional_directions) > 1:
        conflict_code = "OPPOSING_EARLY_RESEARCH_CANDIDATES"
        conflicted_routes = tuple(
            RoutedCandidateV4(
                candidate=route.candidate,
                decision=(
                    _blocked_route(conflict_code)
                    if route.decision.provisional
                    else route.decision
                ),
            )
            for route in routed
        )
        directions = tuple(
            sorted(
                {
                    route.decision.direction
                    for route in conflicted_routes
                    if route.decision.allowed
                    and route.decision.direction is not None
                },
                key=lambda item: item.value,
            )
        )
        return DirectionRoutingResultV4(
            routes=conflicted_routes,
            allowed_directions=directions,
            direction_conflict=True,
            reasons=_route_reasons(conflicted_routes, extra=(conflict_code,)),
        )
    directions = tuple(
        sorted(
            {
                route.decision.direction
                for route in routed
                if route.decision.allowed
                and route.decision.direction is not None
            },
            key=lambda item: item.value,
        )
    )
    return DirectionRoutingResultV4(
        routes=routed,
        allowed_directions=directions,
        reasons=_route_reasons(routed),
    )


def evaluate_early_research_v4(
    h4: H4BackgroundSnapshotV4 | H4BackgroundV4,
    h1: MarketStateSnapshotV4,
    m15_candles: Sequence[Candle],
    *,
    direction: DirectionV4,
    as_of: datetime,
    config: MarketStateV4Config = DEFAULT_CONFIG,
) -> EarlyResearchDecisionV4:
    """Build a provisional RANGE(false) candidate from closed 15m facts.

    Score, current-price zone hit and execution-plan results are deliberately
    absent from this API. They are downstream gates and must not participate
    in provisional direction creation.
    """

    observed_at = _aware(as_of)
    prepared = _prepare_candles(
        m15_candles,
        as_of=observed_at,
        timeframe=timedelta(minutes=15),
        config=config,
    )
    if prepared.status is not DataStatusV4.COMPLETE:
        return EarlyResearchDecisionV4(
            candidate=None,
            data_status=prepared.status,
            as_of=observed_at,
            data_cutoff=prepared.data_cutoff,
            reason_code=prepared.reason,
        )
    h4_data_complete = (
        not isinstance(h4, H4BackgroundSnapshotV4)
        or h4.data_status is DataStatusV4.COMPLETE
    )
    background = h4.background if isinstance(h4, H4BackgroundSnapshotV4) else h4
    if not h4_data_complete or background is None:
        return _early_blocked(
            prepared,
            observed_at,
            "MARKET_STATE_DATA_BLOCKED",
        )
    if background is H4BackgroundV4.CONFLICT_NEUTRAL:
        return _early_blocked(
            prepared,
            observed_at,
            "H4_STRUCTURE_CONFLICT",
        )
    if _h4_opposes(background, direction):
        return _early_blocked(
            prepared,
            observed_at,
            "H4_DIRECTION_OPPOSED",
        )
    if (
        h1.data_status is not DataStatusV4.COMPLETE
        or h1.state is not MarketStateV4.RANGE
    ):
        return _early_blocked(
            prepared,
            observed_at,
            "EARLY_RESEARCH_REQUIRES_UNDIRECTED_H1",
        )
    if h1.range_tradeable:
        return _early_blocked(
            prepared,
            observed_at,
            "RANGE_EDGE_SETUP_REQUIRED",
        )
    if h1.direction_conflict:
        return _early_blocked(
            prepared,
            observed_at,
            "EARLY_RESEARCH_MIRROR_STRUCTURE_OPPOSED",
        )

    pivots = _confirmed_pivots(prepared, "15m", config)
    signal = _directional_structure(
        prepared,
        pivots,
        direction,
        "15m",
        config,
    )
    mirror = _directional_structure(
        prepared,
        pivots,
        (
            DirectionV4.SHORT
            if direction is DirectionV4.LONG
            else DirectionV4.LONG
        ),
        "15m",
        config,
    )
    if mirror is not None:
        return _early_blocked(
            prepared,
            observed_at,
            "EARLY_RESEARCH_MIRROR_STRUCTURE_OPPOSED",
        )
    if signal is None:
        return _early_blocked(
            prepared,
            observed_at,
            "EARLY_RESEARCH_STRUCTURE_UNCONFIRMED",
        )
    early_fact = _early_structure_fact(
        prepared,
        pivots,
        direction,
        "15m",
        config,
    )
    if early_fact is None:
        return _early_blocked(
            prepared,
            observed_at,
            "EARLY_RESEARCH_STRUCTURE_UNCONFIRMED",
        )
    invalidation_anchor = early_fact.invalidation_anchor
    zone_anchor = early_fact.zone_anchor
    zone_buffer = config.touch_atr_multiple * zone_anchor.atr
    invalidation_buffer = (
        config.invalidation_atr_multiple * invalidation_anchor.atr
    )
    invalidation_price = (
        invalidation_anchor.price - invalidation_buffer
        if direction is DirectionV4.LONG
        else invalidation_anchor.price + invalidation_buffer
    )
    if direction is DirectionV4.LONG:
        entry_zone = (
            zone_anchor.price,
            zone_anchor.price + zone_buffer,
        )
    else:
        entry_zone = (
            zone_anchor.price - zone_buffer,
            zone_anchor.price,
        )
    setup_id = _stable_id(
        "m15_early_setup",
        {
            "market_state_version": MARKET_STATE_VERSION,
            "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
            "direction": direction.value,
            "structure_id": early_fact.structure_id,
            "mode": early_fact.mode,
            "invalidation_anchor_id": invalidation_anchor.event_id,
            "zone_anchor_id": zone_anchor.event_id,
        },
    )
    candidate = CandidateSetupV4(
        direction=direction,
        kind=SetupKindV4.EARLY_RESEARCH,
        setup_id=setup_id,
        valid=True,
        entry_zone_exists=True,
        entry_zone_hit=False,
        structure_confirmed=True,
        invalidation_anchor_exists=True,
        mirror_structure_opposed=False,
    )
    return EarlyResearchDecisionV4(
        candidate=candidate,
        data_status=DataStatusV4.COMPLETE,
        as_of=observed_at,
        data_cutoff=prepared.data_cutoff,
        structure_id=early_fact.structure_id,
        structure_mode=early_fact.mode,
        invalidation_anchor_id=invalidation_anchor.event_id,
        invalidation_price=invalidation_price,
        invalidation_buffer=invalidation_buffer,
        entry_zone=entry_zone,
    )


def mirror_candles_v4(
    candles: Sequence[Candle],
    *,
    axis: float,
) -> tuple[Candle, ...]:
    """Return the exact price-sign mirror used by symmetry tests."""

    return tuple(
        Candle(
            timestamp=candle.timestamp,
            open=2 * axis - candle.open,
            high=2 * axis - candle.low,
            low=2 * axis - candle.high,
            close=2 * axis - candle.close,
            volume=candle.volume,
        )
        for candle in candles
    )


def _range_snapshot(
    prepared: _PreparedCandles,
    observed_at: datetime,
    frozen_range: FrozenRangeV4,
    *,
    reason: str,
) -> MarketStateSnapshotV4:
    return MarketStateSnapshotV4(
        state=MarketStateV4.RANGE,
        data_status=DataStatusV4.COMPLETE,
        as_of=observed_at,
        data_cutoff=prepared.data_cutoff,
        range_id=frozen_range.structure_id,
        range_tradeable=True,
        range_position=frozen_range.position,
        frozen_range=frozen_range,
        entry_permission=frozen_range.position in {
            RangePositionV4.LOWER_EDGE,
            RangePositionV4.UPPER_EDGE,
        },
        structure_ids=(frozen_range.structure_id,),
        reasons=(reason,),
    )


def _channel_snapshot(
    prepared: _PreparedCandles,
    observed_at: datetime,
    primary: RailV4,
    paired: RailV4 | None,
) -> MarketStateSnapshotV4:
    upward = primary.direction is DirectionV4.LONG
    shape = (
        ChannelShapeV4.PAIRED_GEOMETRIC_CHANNEL
        if paired is not None
        else ChannelShapeV4.PRIMARY_RAIL_TREND
    )
    confirmation_time = max(
        primary.confirmation_time,
        (
            paired.confirmation_time
            if paired is not None
            else primary.confirmation_time
        ),
    )
    channel_anchor_events = [
        (primary.first_anchor_index, primary.anchor_ids[0]),
        (primary.last_anchor_index, primary.anchor_ids[-1]),
    ]
    if paired is not None:
        channel_anchor_events.extend(
            (
                (paired.first_anchor_index, paired.anchor_ids[0]),
                (paired.last_anchor_index, paired.anchor_ids[-1]),
            )
        )
    ordered_anchor_ids = [
        event_id
        for _, event_id in sorted(channel_anchor_events)
    ]
    channel_id = _stable_id(
        "channel",
        {
            "market_state_version": MARKET_STATE_VERSION,
            "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
            "direction": primary.direction.value,
            "anchor_ids": ordered_anchor_ids,
            "primary_id": primary.structure_id,
            "paired_id": paired.structure_id if paired else None,
            "confirmation_time": _iso(confirmation_time),
        },
    )
    structure_ids = [primary.structure_id]
    if paired is not None:
        structure_ids.append(paired.structure_id)
    current_close = prepared.candles[-1].close
    if primary.direction is DirectionV4.LONG:
        current_inside_primary_zone = (
            primary.current_boundary
            <= current_close
            <= primary.current_boundary + primary.touch_buffer
        )
    else:
        current_inside_primary_zone = (
            primary.current_boundary - primary.touch_buffer
            <= current_close
            <= primary.current_boundary
        )
    return MarketStateSnapshotV4(
        state=(MarketStateV4.CHANNEL_UP if upward else MarketStateV4.CHANNEL_DOWN),
        data_status=DataStatusV4.COMPLETE,
        as_of=observed_at,
        data_cutoff=prepared.data_cutoff,
        direction=primary.direction,
        channel_shape=shape,
        channel_id=channel_id,
        primary_rail=primary,
        paired_rail=paired,
        entry_permission=(
            primary.entry_side_valid and current_inside_primary_zone
        ),
        structure_ids=tuple(sorted(structure_ids)),
    )


def _strong_snapshot(
    prepared: _PreparedCandles,
    observed_at: datetime,
    structure: _StructureSignal,
) -> MarketStateSnapshotV4:
    upward = structure.direction is DirectionV4.LONG
    return MarketStateSnapshotV4(
        state=(MarketStateV4.STRONG_UP if upward else MarketStateV4.STRONG_DOWN),
        data_status=DataStatusV4.COMPLETE,
        as_of=observed_at,
        data_cutoff=prepared.data_cutoff,
        direction=structure.direction,
        # A strong state authorizes tactical setup generation, but the 1H
        # classifier alone cannot prove that price is inside an independent
        # 15m EMA or other setup zone.
        entry_permission=False,
        structure_ids=(structure.structure_id,),
    )


def _prepare_candles(
    candles: Sequence[Candle],
    *,
    as_of: datetime,
    timeframe: timedelta,
    config: MarketStateV4Config,
) -> _PreparedCandles:
    by_time: dict[datetime, Candle] = {}
    for raw in candles:
        timestamp = _aware(raw.timestamp)
        normalized = Candle(
            timestamp=timestamp,
            open=float(raw.open),
            high=float(raw.high),
            low=float(raw.low),
            close=float(raw.close),
            volume=float(raw.volume),
        )
        if timestamp + timeframe > as_of:
            continue
        if not _valid_candle(normalized):
            return _PreparedCandles(
                candles=(),
                atr=(),
                status=DataStatusV4.INVALID,
                data_cutoff=None,
                reason="INVALID_CANDLE",
            )
        existing = by_time.get(timestamp)
        if existing is not None and existing != normalized:
            return _PreparedCandles(
                candles=(),
                atr=(),
                status=DataStatusV4.INVALID,
                data_cutoff=None,
                reason="CONFLICTING_DUPLICATE_CANDLE",
            )
        by_time[timestamp] = normalized

    ordered = tuple(by_time[key] for key in sorted(by_time))
    cutoff = ordered[-1].timestamp + timeframe if ordered else None
    minimum = max(
        config.atr_period + config.pivot_left + config.pivot_right + 1,
        8,
    )
    if len(ordered) < minimum:
        return _PreparedCandles(
            candles=ordered,
            atr=(),
            status=DataStatusV4.INSUFFICIENT,
            data_cutoff=cutoff,
            reason="INSUFFICIENT_CLOSED_CANDLES",
        )
    expected_seconds = timeframe.total_seconds()
    for previous, current in zip(ordered, ordered[1:]):
        delta = (current.timestamp - previous.timestamp).total_seconds()
        if not math.isclose(delta, expected_seconds, abs_tol=1e-6):
            return _PreparedCandles(
                candles=ordered,
                atr=(),
                status=DataStatusV4.DISCONTINUOUS,
                data_cutoff=cutoff,
                reason="DISCONTINUOUS_CLOSED_CANDLES",
            )
    if cutoff is not None and as_of - cutoff >= timeframe:
        return _PreparedCandles(
            candles=ordered,
            atr=(),
            status=DataStatusV4.DISCONTINUOUS,
            data_cutoff=cutoff,
            reason="STALE_CLOSED_CANDLES",
        )
    return _PreparedCandles(
        candles=ordered,
        atr=_atr_series(ordered, config.atr_period),
        status=DataStatusV4.COMPLETE,
        data_cutoff=cutoff,
    )


def _valid_candle(candle: Candle) -> bool:
    values = (
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
    )
    return (
        all(math.isfinite(value) for value in values)
        and candle.low > 0
        and candle.volume >= 0
        and candle.high >= max(candle.open, candle.close)
        and candle.low <= min(candle.open, candle.close)
    )


def _atr_series(
    candles: Sequence[Candle],
    period: int,
) -> tuple[float | None, ...]:
    true_ranges: list[float] = []
    for index, candle in enumerate(candles):
        if index == 0:
            true_range = candle.high - candle.low
        else:
            previous_close = candles[index - 1].close
            true_range = max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        true_ranges.append(true_range)
    return tuple(
        value if value is not None and value > 0 else None
        for value in _ema_series(true_ranges, period)
    )


def _confirmed_pivots(
    prepared: _PreparedCandles,
    timeframe_name: str,
    config: MarketStateV4Config,
) -> tuple[SwingPointV4, ...]:
    candles = prepared.candles
    pivots: list[SwingPointV4] = []
    start = max(config.pivot_left, config.atr_period - 1)
    stop = len(candles) - config.pivot_right
    for index in range(start, stop):
        atr = prepared.atr[index]
        if atr is None:
            continue
        window = candles[
            index - config.pivot_left : index + config.pivot_right + 1
        ]
        highs = [item.high for item in window]
        lows = [item.low for item in window]
        candle = candles[index]
        if candle.high == max(highs) and highs.count(candle.high) == 1:
            pivots.append(
                _swing_point(
                    candle,
                    index,
                    "HIGH",
                    candle.high,
                    atr,
                    timeframe_name,
                )
            )
        if candle.low == min(lows) and lows.count(candle.low) == 1:
            pivots.append(
                _swing_point(
                    candle,
                    index,
                    "LOW",
                    candle.low,
                    atr,
                    timeframe_name,
                )
            )
    return tuple(
        sorted(
            pivots,
            key=lambda item: (item.index, item.kind, item.event_id),
        )
    )


def _swing_point(
    candle: Candle,
    index: int,
    kind: str,
    price: float,
    atr: float,
    timeframe_name: str,
) -> SwingPointV4:
    event_id = _stable_id(
        "anchor",
        {
            "market_state_version": MARKET_STATE_VERSION,
            "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
            "timeframe": timeframe_name,
            "kind": kind,
            "timestamp": _iso(candle.timestamp),
            "price": _number(price),
            "atr": _number(atr),
        },
    )
    return SwingPointV4(
        event_id=event_id,
        index=index,
        timestamp=candle.timestamp,
        kind=kind,
        price=price,
        atr=atr,
    )


def _carry_forward_primary_rail(
    prepared: _PreparedCandles,
    previous_snapshot: MarketStateSnapshotV4 | None,
    *,
    timeframe_name: str,
    config: MarketStateV4Config,
) -> tuple[RailV4 | None, DirectionV4 | None, int]:
    if (
        previous_snapshot is None
        or previous_snapshot.data_status is not DataStatusV4.COMPLETE
        or previous_snapshot.market_state_version != MARKET_STATE_VERSION
        or previous_snapshot.channel_geometry_version
        != CHANNEL_GEOMETRY_VERSION
        or previous_snapshot.primary_rail is None
        or previous_snapshot.data_cutoff is None
    ):
        return None, None, 0
    prior = previous_snapshot.primary_rail
    prior_cutoff = _aware(previous_snapshot.data_cutoff)
    if (
        prepared.data_cutoff is None
        or prior_cutoff > prepared.data_cutoff
    ):
        return None, None, 0

    validation_count = prior.validation_count
    residual_total = prior.residual * validation_count
    last_validation_index = prior.last_validation_index
    for index, candle in enumerate(prepared.candles):
        if _closed_at(prepared, index, timeframe_name) <= prior_cutoff:
            continue
        boundary = prior.boundary_at(index)
        if _rail_invalidated(
            candle.close,
            boundary,
            prior.invalidation_buffer,
            prior.direction,
        ):
            return None, prior.direction, index + 1
        atr = prepared.atr[index]
        if atr is not None and _rail_touch(
            candle,
            boundary,
            atr,
            prior.direction,
            config,
        ):
            validation_count += 1
            residual_total += _touch_residual(
                candle,
                boundary,
                prior.direction,
            )
            last_validation_index = index

    current_index = len(prepared.candles) - 1
    current_boundary = prior.boundary_at(current_index)
    current_close = prepared.candles[-1].close
    entry_side_valid = _price_on_entry_side(
        current_close,
        current_boundary,
        prior.direction,
    )
    return (
        replace(
            prior,
            last_validation_index=last_validation_index,
            validation_count=validation_count,
            span_bars=current_index - prior.first_anchor_index,
            residual=residual_total / validation_count,
            current_boundary=current_boundary,
            entry_side_valid=entry_side_valid,
        ),
        None,
        0,
    )


def _select_primary_rail(
    prepared: _PreparedCandles,
    pivots: Sequence[SwingPointV4],
    direction: DirectionV4,
    timeframe_name: str,
    config: MarketStateV4Config,
    *,
    minimum_anchor_index: int = 0,
) -> RailV4 | None:
    kind = "LOW" if direction is DirectionV4.LONG else "HIGH"
    points = [
        point
        for point in pivots
        if point.kind == kind
        and point.index >= minimum_anchor_index
    ]
    candidates: list[RailV4] = []
    for first_offset, first in enumerate(points):
        for second in points[first_offset + 1 :]:
            signed_change = (
                second.price - first.price
                if direction is DirectionV4.LONG
                else first.price - second.price
            )
            # The anchor event freezes ATR14 from the second reaction candle.
            anchor_noise = config.touch_atr_multiple * second.atr
            if signed_change < anchor_noise:
                continue
            index_span = second.index - first.index
            if index_span <= 0:
                continue
            slope = (second.price - first.price) / index_span
            candidate = _validated_primary_rail(
                prepared,
                direction,
                first,
                second,
                slope,
                timeframe_name,
                config,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda rail: (
            -rail.validation_count,
            -rail.span_bars,
            -rail.last_validation_index,
            rail.residual,
            rail.structure_id,
        ),
    )


def _validated_primary_rail(
    prepared: _PreparedCandles,
    direction: DirectionV4,
    first: SwingPointV4,
    second: SwingPointV4,
    slope: float,
    timeframe_name: str,
    config: MarketStateV4Config,
) -> RailV4 | None:
    validation_index: int | None = None
    validation_indices: list[int] = []
    residuals: list[float] = []
    for index in range(second.index + 1, len(prepared.candles)):
        candle = prepared.candles[index]
        atr = prepared.atr[index]
        if atr is None:
            continue
        boundary = first.price + slope * (index - first.index)
        if validation_index is None:
            if not _rail_touch(candle, boundary, atr, direction, config):
                continue
            validation_index = index
            validation_indices.append(index)
            residuals.append(_touch_residual(candle, boundary, direction))
            continue
        if _rail_touch(candle, boundary, atr, direction, config):
            validation_indices.append(index)
            residuals.append(_touch_residual(candle, boundary, direction))
    if validation_index is None:
        return None

    # A pivot is not available until its right-hand confirmation candle has
    # closed.  A validation may occur sooner, but the rail becomes observable
    # only when both facts are closed and available.
    confirmation_index = max(
        validation_index,
        second.index + config.pivot_right,
    )
    confirmation_atr = prepared.atr[confirmation_index]
    if confirmation_atr is None:
        return None
    invalidation_buffer = (
        config.invalidation_atr_multiple * confirmation_atr
    )
    touch_buffer = config.touch_atr_multiple * confirmation_atr
    for index in range(confirmation_index, len(prepared.candles)):
        boundary = first.price + slope * (index - first.index)
        if _rail_invalidated(
            prepared.candles[index].close,
            boundary,
            invalidation_buffer,
            direction,
        ):
            return None

    current_index = len(prepared.candles) - 1
    current_boundary = first.price + slope * (current_index - first.index)
    current_close = prepared.candles[-1].close
    entry_side_valid = (
        current_close >= current_boundary
        if direction is DirectionV4.LONG
        else current_close <= current_boundary
    )
    confirmation_time = _closed_at(
        prepared,
        confirmation_index,
        timeframe_name,
    )
    structure_id = _stable_id(
        "rail",
        {
            "market_state_version": MARKET_STATE_VERSION,
            "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
            "timeframe": timeframe_name,
            "direction": direction.value,
            "side": "PRIMARY",
            "anchor_ids": [first.event_id, second.event_id],
            "confirmation_time": _iso(confirmation_time),
        },
    )
    return RailV4(
        structure_id=structure_id,
        direction=direction,
        side="PRIMARY",
        anchor_ids=(first.event_id, second.event_id),
        first_anchor_index=first.index,
        last_anchor_index=second.index,
        confirmation_index=confirmation_index,
        confirmation_time=confirmation_time,
        last_validation_index=validation_indices[-1],
        validation_count=len(validation_indices),
        span_bars=current_index - first.index,
        slope_per_bar=slope,
        intercept=first.price - slope * first.index,
        touch_buffer=touch_buffer,
        invalidation_buffer=invalidation_buffer,
        residual=(sum(residuals) / len(residuals)),
        current_boundary=current_boundary,
        entry_side_valid=entry_side_valid,
    )


def _rail_touch(
    candle: Candle,
    boundary: float,
    atr: float,
    direction: DirectionV4,
    config: MarketStateV4Config,
) -> bool:
    buffer = config.touch_atr_multiple * atr
    if direction is DirectionV4.LONG:
        return (
            boundary - buffer <= candle.low <= boundary + buffer
            and candle.close >= boundary
        )
    return (
        boundary - buffer <= candle.high <= boundary + buffer
        and candle.close <= boundary
    )


def _touch_residual(
    candle: Candle,
    boundary: float,
    direction: DirectionV4,
) -> float:
    extreme = candle.low if direction is DirectionV4.LONG else candle.high
    return abs(extreme - boundary)


def _rail_invalidated(
    close: float,
    boundary: float,
    buffer: float,
    direction: DirectionV4,
) -> bool:
    if direction is DirectionV4.LONG:
        return close < boundary - buffer
    return close > boundary + buffer


def _select_paired_rail(
    prepared: _PreparedCandles,
    pivots: Sequence[SwingPointV4],
    primary: RailV4,
    timeframe_name: str,
    config: MarketStateV4Config,
) -> RailV4 | None:
    opposite_kind = "HIGH" if primary.direction is DirectionV4.LONG else "LOW"
    points = [
        point
        for point in pivots
        if point.kind == opposite_kind
        and point.index >= primary.first_anchor_index
    ]
    candidates: list[RailV4] = []
    for first_offset, first in enumerate(points):
        for second in points[first_offset + 1 :]:
            span = second.index - first.index
            if span <= 0:
                continue
            slope = (second.price - first.price) / span
            intercept = first.price - slope * first.index
            confirmation_index = second.index + config.pivot_right
            confirmation_atr = prepared.atr[confirmation_index]
            if confirmation_atr is None:
                continue
            buffer = config.invalidation_atr_multiple * confirmation_atr
            touch_buffer = config.touch_atr_multiple * confirmation_atr
            invalidated = False
            reactions = [first, second]
            residuals = [0.0, 0.0]
            for point in points:
                if point.index <= second.index:
                    continue
                boundary = intercept + slope * point.index
                if abs(point.price - boundary) <= (
                    config.invalidation_atr_multiple * point.atr
                ):
                    reactions.append(point)
                    residuals.append(abs(point.price - boundary))
            for index in range(
                confirmation_index,
                len(prepared.candles),
            ):
                boundary = intercept + slope * index
                close = prepared.candles[index].close
                if primary.direction is DirectionV4.LONG:
                    if close > boundary + buffer:
                        invalidated = True
                        break
                elif close < boundary - buffer:
                    invalidated = True
                    break
            if invalidated:
                continue
            current_index = len(prepared.candles) - 1
            current_boundary = intercept + slope * current_index
            primary_boundary = primary.boundary_at(current_index)
            geometry_start = max(
                primary.confirmation_index,
                confirmation_index,
            )
            start_boundary = intercept + slope * geometry_start
            primary_start_boundary = primary.boundary_at(geometry_start)
            if primary.direction is DirectionV4.LONG:
                if (
                    start_boundary <= primary_start_boundary
                    or current_boundary <= primary_boundary
                ):
                    continue
                side = "UPPER"
            else:
                if (
                    start_boundary >= primary_start_boundary
                    or current_boundary >= primary_boundary
                ):
                    continue
                side = "LOWER"
            anchor_ids = (first.event_id, second.event_id)
            confirmation_time = _closed_at(
                prepared,
                confirmation_index,
                timeframe_name,
            )
            structure_id = _stable_id(
                "paired_rail",
                {
                    "market_state_version": MARKET_STATE_VERSION,
                    "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
                    "timeframe": timeframe_name,
                    "direction": primary.direction.value,
                    "side": side,
                    "anchor_ids": anchor_ids,
                    "confirmation_time": _iso(confirmation_time),
                },
            )
            candidates.append(
                RailV4(
                    structure_id=structure_id,
                    direction=primary.direction,
                    side=side,
                    anchor_ids=anchor_ids,
                    first_anchor_index=first.index,
                    last_anchor_index=second.index,
                    confirmation_index=confirmation_index,
                    confirmation_time=confirmation_time,
                    last_validation_index=max(
                        point.index for point in reactions
                    ),
                    validation_count=len(reactions),
                    span_bars=current_index - first.index,
                    slope_per_bar=slope,
                    intercept=intercept,
                    touch_buffer=touch_buffer,
                    invalidation_buffer=buffer,
                    residual=sum(residuals) / len(residuals),
                    current_boundary=current_boundary,
                    entry_side_valid=True,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda rail: (
            -rail.validation_count,
            -rail.span_bars,
            -rail.last_validation_index,
            rail.residual,
            rail.structure_id,
        ),
    )


def _paired_contains(
    primary: RailV4,
    paired: RailV4 | None,
    index: int,
    close: float,
) -> bool:
    if paired is None:
        return False
    primary_boundary = primary.boundary_at(index)
    paired_boundary = paired.boundary_at(index)
    if primary.direction is DirectionV4.LONG:
        return (
            close >= primary_boundary - primary.invalidation_buffer
            and close <= paired_boundary + paired.invalidation_buffer
        )
    return (
        close <= primary_boundary + primary.invalidation_buffer
        and close >= paired_boundary - paired.invalidation_buffer
    )


def _carry_forward_range(
    prepared: _PreparedCandles,
    previous_snapshot: MarketStateSnapshotV4 | None,
    *,
    timeframe_name: str,
) -> tuple[FrozenRangeV4 | None, int]:
    if (
        previous_snapshot is None
        or previous_snapshot.data_status is not DataStatusV4.COMPLETE
        or previous_snapshot.market_state_version != MARKET_STATE_VERSION
        or previous_snapshot.channel_geometry_version
        != CHANNEL_GEOMETRY_VERSION
        or previous_snapshot.frozen_range is None
    ):
        return None, 0
    prior = previous_snapshot.frozen_range
    prior_cutoff = previous_snapshot.data_cutoff
    if prior_cutoff is None:
        return None, 0
    prior_cutoff = _aware(prior_cutoff)
    if (
        prepared.data_cutoff is None
        or prior_cutoff > prepared.data_cutoff
    ):
        return None, 0

    for index, candle in enumerate(prepared.candles):
        if _closed_at(prepared, index, timeframe_name) <= prior_cutoff:
            continue
        if (
            candle.close < prior.lower - prior.invalidation_buffer
            or candle.close > prior.upper + prior.invalidation_buffer
        ):
            return None, index + 1

    latest_close = prepared.candles[-1].close
    first_anchor_index = next(
        (
            index
            for index in range(len(prepared.candles))
            if _closed_at(prepared, index, timeframe_name)
            >= prior.first_anchor_time
        ),
        0,
    )
    return (
        replace(
            prior,
            span_bars=(
                len(prepared.candles) - 1 - first_anchor_index
            ),
            position=_range_position(
                latest_close,
                lower=prior.lower,
                upper=prior.upper,
                touch_buffer=prior.touch_buffer,
            ),
        ),
        0,
    )


def _range_position(
    close: float,
    *,
    lower: float,
    upper: float,
    touch_buffer: float,
) -> RangePositionV4:
    if close < lower or close > upper:
        return RangePositionV4.OUTSIDE
    lower_edge = close <= lower + touch_buffer
    upper_edge = close >= upper - touch_buffer
    if lower_edge and upper_edge:
        return RangePositionV4.OVERLAP
    if lower_edge:
        return RangePositionV4.LOWER_EDGE
    if upper_edge:
        return RangePositionV4.UPPER_EDGE
    return RangePositionV4.MIDDLE


def _select_frozen_range(
    prepared: _PreparedCandles,
    pivots: Sequence[SwingPointV4],
    timeframe_name: str,
    config: MarketStateV4Config,
    *,
    minimum_anchor_index: int = 0,
) -> FrozenRangeV4 | None:
    supports = _horizontal_clusters(
        [
            point
            for point in pivots
            if point.kind == "LOW"
            and point.index >= minimum_anchor_index
        ],
        config,
    )
    resistances = _horizontal_clusters(
        [
            point
            for point in pivots
            if point.kind == "HIGH"
            and point.index >= minimum_anchor_index
        ],
        config,
    )
    candidates: list[FrozenRangeV4] = []
    for support in supports:
        for resistance in resistances:
            if support.boundary >= resistance.boundary:
                continue
            all_points = tuple(
                sorted(
                    support.points + resistance.points,
                    key=lambda item: (item.index, item.kind, item.event_id),
                )
            )
            confirmation_index = max(
                support.points[1].index + config.pivot_right,
                resistance.points[1].index + config.pivot_right,
            )
            atr = prepared.atr[confirmation_index]
            if atr is None:
                continue
            buffer = config.invalidation_atr_multiple * atr
            touch_buffer = config.touch_atr_multiple * atr
            invalidated = False
            structure_epoch_start = min(
                point.index for point in all_points
            )
            for index in range(
                structure_epoch_start,
                len(prepared.candles),
            ):
                close = prepared.candles[index].close
                if (
                    close < support.boundary - buffer
                    or close > resistance.boundary + buffer
                ):
                    invalidated = True
                    break
            if invalidated:
                continue
            latest_close = prepared.candles[-1].close
            if (
                latest_close < support.boundary - buffer
                or latest_close > resistance.boundary + buffer
            ):
                continue
            position = _range_position(
                latest_close,
                lower=support.boundary,
                upper=resistance.boundary,
                touch_buffer=touch_buffer,
            )
            support_ids = tuple(
                point.event_id
                for point in sorted(
                    support.points,
                    key=lambda item: (item.index, item.event_id),
                )
            )
            resistance_ids = tuple(
                point.event_id
                for point in sorted(
                    resistance.points,
                    key=lambda item: (item.index, item.event_id),
                )
            )
            support_reactions = [
                point
                for point in pivots
                if point.kind == "LOW"
                and point.index >= minimum_anchor_index
                and point.index >= support.points[0].index
                and abs(point.price - support.boundary)
                <= config.invalidation_atr_multiple * point.atr
            ]
            resistance_reactions = [
                point
                for point in pivots
                if point.kind == "HIGH"
                and point.index >= minimum_anchor_index
                and point.index >= resistance.points[0].index
                and abs(point.price - resistance.boundary)
                <= config.invalidation_atr_multiple * point.atr
            ]
            validation_points = (
                support_reactions + resistance_reactions
            )
            confirmation_time = _closed_at(
                prepared,
                confirmation_index,
                timeframe_name,
            )
            structure_id = _stable_id(
                "range",
                {
                    "market_state_version": MARKET_STATE_VERSION,
                    "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
                    "timeframe": timeframe_name,
                    "direction": "NEUTRAL",
                    "anchor_ids": [
                        point.event_id for point in all_points
                    ],
                    "support_anchor_ids": support_ids,
                    "resistance_anchor_ids": resistance_ids,
                    "confirmation_time": _iso(confirmation_time),
                },
            )
            candidates.append(
                FrozenRangeV4(
                    structure_id=structure_id,
                    lower=support.boundary,
                    upper=resistance.boundary,
                    support_anchor_ids=support_ids,
                    resistance_anchor_ids=resistance_ids,
                    confirmation_index=confirmation_index,
                    confirmation_time=confirmation_time,
                    first_anchor_time=_closed_at(
                        prepared,
                        min(point.index for point in all_points),
                        timeframe_name,
                    ),
                    last_validation_index=max(
                        point.index for point in validation_points
                    ),
                    validation_count=len(validation_points),
                    span_bars=(
                        len(prepared.candles)
                        - 1
                        - min(point.index for point in all_points)
                    ),
                    touch_buffer=touch_buffer,
                    invalidation_buffer=buffer,
                    residual=support.residual + resistance.residual,
                    position=position,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -item.validation_count,
            -item.span_bars,
            -item.last_validation_index,
            item.residual,
            item.structure_id,
        ),
    )


def _horizontal_clusters(
    points: Sequence[SwingPointV4],
    config: MarketStateV4Config,
) -> tuple[_PointCluster, ...]:
    clusters: dict[tuple[str, ...], _PointCluster] = {}
    for first_offset, first in enumerate(points):
        for second in points[first_offset + 1 :]:
            selected = [first, second]
            boundary = float(median(point.price for point in selected))
            if any(
                abs(point.price - boundary)
                > config.invalidation_atr_multiple * point.atr
                for point in selected
            ):
                continue
            ids = tuple(point.event_id for point in selected)
            residual = sum(
                abs(point.price - boundary) for point in selected
            ) / len(selected)
            clusters[ids] = _PointCluster(
                points=tuple(selected),
                boundary=boundary,
                residual=residual,
            )
    return tuple(
        sorted(
            clusters.values(),
            key=lambda item: (
                -len(item.points),
                -(item.points[-1].index - item.points[0].index),
                item.residual,
                tuple(point.event_id for point in item.points),
            ),
        )
    )


def _directional_structure(
    prepared: _PreparedCandles,
    pivots: Sequence[SwingPointV4],
    direction: DirectionV4,
    timeframe_name: str,
    config: MarketStateV4Config,
) -> _StructureSignal | None:
    highs = [point for point in pivots if point.kind == "HIGH"]
    lows = [point for point in pivots if point.kind == "LOW"]
    if not highs or not lows:
        return None

    current_close = prepared.candles[-1].close
    if direction is DirectionV4.LONG:
        invalidation_anchor = lows[-1]
        buffer = config.invalidation_atr_multiple * invalidation_anchor.atr
        if current_close < invalidation_anchor.price - buffer:
            return None
        breakout_index = _unreclaimed_breakout(
            prepared,
            highs[-1],
            direction,
            config,
        )
        mature = _mature_directional_swings(
            highs,
            lows,
            direction,
        )
        if breakout_index is None and not mature:
            return None
        anchors = (
            (highs[-2], lows[-2], highs[-1], lows[-1])
            if mature
            else (highs[-1], invalidation_anchor)
        )
    else:
        invalidation_anchor = highs[-1]
        buffer = config.invalidation_atr_multiple * invalidation_anchor.atr
        if current_close > invalidation_anchor.price + buffer:
            return None
        breakout_index = _unreclaimed_breakout(
            prepared,
            lows[-1],
            direction,
            config,
        )
        mature = _mature_directional_swings(
            highs,
            lows,
            direction,
        )
        if breakout_index is None and not mature:
            return None
        anchors = (
            (highs[-2], lows[-2], highs[-1], lows[-1])
            if mature
            else (lows[-1], invalidation_anchor)
        )

    ordered_anchors = tuple(
        sorted(
            {point.event_id: point for point in anchors}.values(),
            key=lambda item: (item.index, item.kind, item.event_id),
        )
    )
    anchor_ready_index = min(
        len(prepared.candles) - 1,
        max(point.index for point in ordered_anchors)
        + config.pivot_right,
    )
    confirmation_index = max(
        anchor_ready_index,
        breakout_index if breakout_index is not None else anchor_ready_index,
    )
    structure_id = _stable_id(
        "directional_structure",
        {
            "market_state_version": MARKET_STATE_VERSION,
            "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
            "timeframe": timeframe_name,
            "direction": direction.value,
            "anchor_ids": [point.event_id for point in ordered_anchors],
            "confirmation_time": _iso(
                _closed_at(
                    prepared,
                    confirmation_index,
                    timeframe_name,
                )
            ),
            "mode": "MATURE" if breakout_index is None else "BREAKOUT",
        },
    )
    return _StructureSignal(
        direction=direction,
        structure_id=structure_id,
        anchor_ids=tuple(point.event_id for point in ordered_anchors),
        first_index=min(point.index for point in ordered_anchors),
        last_index=confirmation_index,
    )


def _mature_directional_swings(
    highs: Sequence[SwingPointV4],
    lows: Sequence[SwingPointV4],
    direction: DirectionV4,
) -> bool:
    if len(highs) < 2 or len(lows) < 2:
        return False
    if direction is DirectionV4.LONG:
        return (
            highs[-1].price > highs[-2].price
            and lows[-1].price > lows[-2].price
        )
    return (
        highs[-1].price < highs[-2].price
        and lows[-1].price < lows[-2].price
    )


def _early_structure_fact(
    prepared: _PreparedCandles,
    pivots: Sequence[SwingPointV4],
    direction: DirectionV4,
    timeframe_name: str,
    config: MarketStateV4Config,
) -> _EarlyStructureFact | None:
    highs = [point for point in pivots if point.kind == "HIGH"]
    lows = [point for point in pivots if point.kind == "LOW"]
    if not highs or not lows:
        return None

    level_points = highs if direction is DirectionV4.LONG else lows
    invalidation_points = (
        lows if direction is DirectionV4.LONG else highs
    )
    candidates: list[_EarlyStructureFact] = []
    for level in level_points:
        breakout_index = _unreclaimed_breakout(
            prepared,
            level,
            direction,
            config,
        )
        if breakout_index is None:
            continue
        for index in range(breakout_index + 1, len(prepared.candles)):
            atr = prepared.atr[index]
            if atr is None:
                continue
            candle = prepared.candles[index]
            touch_buffer = config.touch_atr_multiple * atr
            extreme = (
                candle.low
                if direction is DirectionV4.LONG
                else candle.high
            )
            recovered = (
                candle.close >= level.price
                if direction is DirectionV4.LONG
                else candle.close <= level.price
            )
            if (
                abs(extreme - level.price) > touch_buffer
                or not recovered
            ):
                continue
            eligible_anchors = [
                point
                for point in invalidation_points
                if point.index <= index
            ]
            if not eligible_anchors:
                continue
            invalidation_anchor = eligible_anchors[-1]
            if not _invalidation_anchor_active(
                prepared.candles[-1].close,
                invalidation_anchor,
                direction,
                config,
            ) or not _price_on_entry_side(
                prepared.candles[-1].close,
                invalidation_anchor.price,
                direction,
            ):
                continue
            if not _price_on_entry_side(
                prepared.candles[-1].close,
                level.price,
                direction,
            ):
                continue
            confirmation_index = max(
                index,
                level.index + config.pivot_right,
                invalidation_anchor.index + config.pivot_right,
            )
            candidates.append(
                _make_early_structure_fact(
                    prepared,
                    mode="BREAKOUT_RETEST",
                    direction=direction,
                    anchors=(level, invalidation_anchor),
                    confirmation_index=confirmation_index,
                    zone_anchor=level,
                    invalidation_anchor=invalidation_anchor,
                    timeframe_name=timeframe_name,
                )
            )
            break

    if _mature_directional_swings(highs, lows, direction):
        swing_anchors = (highs[-2], lows[-2], highs[-1], lows[-1])
        invalidation_anchor = (
            lows[-1] if direction is DirectionV4.LONG else highs[-1]
        )
        if _invalidation_anchor_active(
            prepared.candles[-1].close,
            invalidation_anchor,
            direction,
            config,
        ) and _price_on_entry_side(
            prepared.candles[-1].close,
            invalidation_anchor.price,
            direction,
        ):
            confirmation_index = min(
                len(prepared.candles) - 1,
                max(point.index for point in swing_anchors)
                + config.pivot_right,
            )
            candidates.append(
                _make_early_structure_fact(
                    prepared,
                    mode="SWING_RECLAIM",
                    direction=direction,
                    anchors=swing_anchors,
                    confirmation_index=confirmation_index,
                    zone_anchor=invalidation_anchor,
                    invalidation_anchor=invalidation_anchor,
                    timeframe_name=timeframe_name,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -item.confirmation_index,
            item.mode,
            item.structure_id,
        ),
    )


def _make_early_structure_fact(
    prepared: _PreparedCandles,
    *,
    mode: str,
    direction: DirectionV4,
    anchors: Sequence[SwingPointV4],
    confirmation_index: int,
    zone_anchor: SwingPointV4,
    invalidation_anchor: SwingPointV4,
    timeframe_name: str,
) -> _EarlyStructureFact:
    ordered = tuple(
        sorted(
            {point.event_id: point for point in anchors}.values(),
            key=lambda item: (item.index, item.kind, item.event_id),
        )
    )
    confirmation_time = _closed_at(
        prepared,
        confirmation_index,
        timeframe_name,
    )
    structure_id = _stable_id(
        "early_structure",
        {
            "market_state_version": MARKET_STATE_VERSION,
            "channel_geometry_version": CHANNEL_GEOMETRY_VERSION,
            "timeframe": timeframe_name,
            "direction": direction.value,
            "mode": mode,
            "anchor_ids": [point.event_id for point in ordered],
            "confirmation_time": _iso(confirmation_time),
        },
    )
    return _EarlyStructureFact(
        structure_id=structure_id,
        mode=mode,
        confirmation_index=confirmation_index,
        zone_anchor=zone_anchor,
        invalidation_anchor=invalidation_anchor,
    )


def _invalidation_anchor_active(
    current_close: float,
    anchor: SwingPointV4,
    direction: DirectionV4,
    config: MarketStateV4Config,
) -> bool:
    buffer = config.invalidation_atr_multiple * anchor.atr
    if direction is DirectionV4.LONG:
        return current_close >= anchor.price - buffer
    return current_close <= anchor.price + buffer


def _price_on_entry_side(
    price: float,
    boundary: float,
    direction: DirectionV4,
) -> bool:
    if direction is DirectionV4.LONG:
        return price >= boundary
    return price <= boundary


def _active_h4_facts(
    facts: Sequence[H4StructureFactV4],
    *,
    cutoff: datetime,
    frozen_range: FrozenRangeV4 | None,
) -> tuple[_ActiveH4Fact, ...]:
    active: list[_ActiveH4Fact] = []
    observed_structures: list[tuple[DirectionV4, datetime]] = []
    for fact in sorted(
        facts,
        key=lambda item: (
            _aware(item.confirmed_at),
            item.structure_id,
        ),
    ):
        expected_direction = _h4_kind_direction(fact.kind)
        if expected_direction is not fact.direction:
            continue
        confirmed_at = _aware(fact.confirmed_at)
        if confirmed_at > cutoff:
            continue
        observed_structures.append((fact.direction, confirmed_at))
        invalidated_at = (
            _aware(fact.invalidated_at)
            if fact.invalidated_at is not None
            else None
        )
        # A timeless invalidation flag cannot be replayed safely.  It is
        # conservatively inactive until the producer supplies its event time.
        if fact.invalidated and invalidated_at is None:
            continue
        if invalidated_at is not None and invalidated_at <= cutoff:
            continue
        last_validated_at = confirmed_at
        if fact.last_validated_at is not None:
            candidate_time = _aware(fact.last_validated_at)
            if confirmed_at <= candidate_time <= cutoff:
                last_validated_at = candidate_time
        if (
            frozen_range is not None
            and frozen_range.first_anchor_time > last_validated_at
        ):
            continue
        active.append(
            _ActiveH4Fact(
                direction=fact.direction,
                structure_id=fact.structure_id,
                confirmed_at=confirmed_at,
                last_validated_at=last_validated_at,
            )
        )

    surviving = [
        fact
        for fact in active
        if not any(
            mirror_direction is not fact.direction
            and mirror_confirmed_at > fact.last_validated_at
            for mirror_direction, mirror_confirmed_at in observed_structures
        )
    ]
    return tuple(
        sorted(
            surviving,
            key=lambda item: (
                item.confirmed_at,
                item.direction.value,
                item.structure_id,
            ),
        )
    )


def _h4_kind_direction(
    kind: H4StructureKindV4 | str,
) -> DirectionV4 | None:
    raw_kind = kind.value if isinstance(kind, H4StructureKindV4) else kind
    try:
        normalized = H4StructureKindV4(raw_kind)
    except ValueError:
        return None
    if normalized in {
        H4StructureKindV4.ASCENDING_SUPPORT,
        H4StructureKindV4.DESCENDING_BREAKOUT_RETEST,
        H4StructureKindV4.CLOSED_BREAKOUT,
        H4StructureKindV4.HH_HL,
    }:
        return DirectionV4.LONG
    return DirectionV4.SHORT


def _unreclaimed_breakout(
    prepared: _PreparedCandles,
    level: SwingPointV4,
    direction: DirectionV4,
    config: MarketStateV4Config,
) -> int | None:
    breakout_index: int | None = None
    buffer = config.invalidation_atr_multiple * level.atr
    for index in range(level.index + 1, len(prepared.candles)):
        close = prepared.candles[index].close
        if (
            direction is DirectionV4.LONG
            and close > level.price + buffer
        ):
            breakout_index = index
            break
        if (
            direction is DirectionV4.SHORT
            and close < level.price - buffer
        ):
            breakout_index = index
            break
    if breakout_index is None:
        return None
    for candle in prepared.candles[breakout_index:]:
        if direction is DirectionV4.LONG and candle.close < level.price - buffer:
            return None
        if direction is DirectionV4.SHORT and candle.close > level.price + buffer:
            return None
    return breakout_index


def _previous_ema_background(
    previous: H4BackgroundSnapshotV4 | H4BackgroundV4,
) -> H4BackgroundV4:
    if (
        isinstance(previous, H4BackgroundSnapshotV4)
        and previous.data_status is DataStatusV4.COMPLETE
        and previous.source == "EMA_FALLBACK"
        and previous.background in {
            H4BackgroundV4.BULLISH,
            H4BackgroundV4.BEARISH,
        }
    ):
        return previous.background
    return H4BackgroundV4.CLEAN_NEUTRAL


def _ema_fallback_direction(
    prepared: _PreparedCandles,
    previous_background: H4BackgroundV4,
    config: MarketStateV4Config,
) -> DirectionV4 | None:
    closes = [candle.close for candle in prepared.candles]
    fast = _ema_series(closes, config.ema_fast_period)
    slow = _ema_series(closes, config.ema_slow_period)
    strengths: list[tuple[int, float]] = []
    for index, (fast_value, slow_value, atr) in enumerate(
        zip(fast, slow, prepared.atr)
    ):
        if fast_value is None or slow_value is None or atr is None or atr <= 0:
            continue
        strengths.append((index, abs(fast_value - slow_value) / atr))
    if len(strengths) < config.ema_min_strength_samples + 2:
        return None
    latest_indices = {len(closes) - 2, len(closes) - 1}
    if not latest_indices.issubset({index for index, _ in strengths}):
        return None
    baseline = [
        value
        for index, value in strengths
        if index < len(closes) - 2
    ]
    if len(baseline) < config.ema_min_strength_samples:
        return None

    relation_long = all(
        fast[index] is not None
        and slow[index] is not None
        and fast[index] > slow[index]
        for index in (len(closes) - 2, len(closes) - 1)
    )
    relation_short = all(
        fast[index] is not None
        and slow[index] is not None
        and fast[index] < slow[index]
        for index in (len(closes) - 2, len(closes) - 1)
    )
    current_strengths = [
        abs(float(fast[index]) - float(slow[index]))
        / float(prepared.atr[index])
        for index in (len(closes) - 2, len(closes) - 1)
    ]
    if relation_long:
        quantile = (
            config.ema_exit_quantile
            if previous_background is H4BackgroundV4.BULLISH
            else config.ema_enter_quantile
        )
        threshold = _quantile(baseline, quantile)
        if all(value >= threshold for value in current_strengths):
            return DirectionV4.LONG
    if relation_short:
        quantile = (
            config.ema_exit_quantile
            if previous_background is H4BackgroundV4.BEARISH
            else config.ema_enter_quantile
        )
        threshold = _quantile(baseline, quantile)
        if all(value >= threshold for value in current_strengths):
            return DirectionV4.SHORT
    return None


def _ema_series(
    values: Sequence[float],
    period: int,
) -> tuple[float | None, ...]:
    output: list[float | None] = [None] * len(values)
    if len(values) < period:
        return tuple(output)
    seed = sum(values[:period]) / period
    output[period - 1] = seed
    multiplier = 2 / (period + 1)
    current = seed
    for index in range(period, len(values)):
        current = (values[index] - current) * multiplier + current
        output[index] = current
    return tuple(output)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * min(1.0, max(0.0, probability))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _route_early_research(
    background: H4BackgroundV4,
    h1: MarketStateSnapshotV4,
    candidate: CandidateSetupV4,
) -> DirectionRouteDecisionV4:
    if background not in {
        H4BackgroundV4.BULLISH,
        H4BackgroundV4.BEARISH,
        H4BackgroundV4.CLEAN_NEUTRAL,
    }:
        return _blocked_route("EARLY_RESEARCH_H4_NOT_AUTHORIZED")
    if h1.state is not MarketStateV4.RANGE:
        return _blocked_route("EARLY_RESEARCH_REQUIRES_UNDIRECTED_H1")
    if h1.direction_conflict or candidate.mirror_structure_opposed:
        return _blocked_route("EARLY_RESEARCH_MIRROR_STRUCTURE_OPPOSED")
    if not candidate.structure_confirmed:
        return _blocked_route("EARLY_RESEARCH_STRUCTURE_UNCONFIRMED")
    if not candidate.invalidation_anchor_exists:
        return _blocked_route("EARLY_RESEARCH_INVALIDATION_ANCHOR_MISSING")
    if h1.range_tradeable:
        return _blocked_route("RANGE_EDGE_SETUP_REQUIRED")
    return DirectionRouteDecisionV4(
        allowed=True,
        direction=candidate.direction,
        provisional=True,
    )


def _early_blocked(
    prepared: _PreparedCandles,
    observed_at: datetime,
    reason: str,
) -> EarlyResearchDecisionV4:
    return EarlyResearchDecisionV4(
        candidate=None,
        data_status=prepared.status,
        as_of=observed_at,
        data_cutoff=prepared.data_cutoff,
        reason_code=reason,
    )


def _h4_opposes(
    background: H4BackgroundV4,
    direction: DirectionV4,
) -> bool:
    return (
        background is H4BackgroundV4.BULLISH
        and direction is DirectionV4.SHORT
    ) or (
        background is H4BackgroundV4.BEARISH
        and direction is DirectionV4.LONG
    )


def _is_matching_h4_setup(candidate: CandidateSetupV4) -> bool:
    if candidate.direction is DirectionV4.LONG:
        return candidate.kind in {
            SetupKindV4.H4_BULLISH_SUPPORT,
            SetupKindV4.H4_BULLISH_BREAKOUT_RETEST,
        }
    return candidate.kind in {
        SetupKindV4.H4_BEARISH_RESISTANCE,
        SetupKindV4.H4_BEARISH_BREAKDOWN_RETEST,
    }


def _h4_authorizes_specific(
    background: H4BackgroundV4,
    candidate: CandidateSetupV4,
) -> bool:
    if not _is_matching_h4_setup(candidate):
        return False
    return (
        candidate.direction is DirectionV4.LONG
        and background is H4BackgroundV4.BULLISH
    ) or (
        candidate.direction is DirectionV4.SHORT
        and background is H4BackgroundV4.BEARISH
    )


def _correct_range_edge(
    position: RangePositionV4,
    candidate: CandidateSetupV4,
) -> bool:
    if candidate.kind is not SetupKindV4.RANGE_EDGE:
        return False
    return (
        candidate.direction is DirectionV4.LONG
        and position is RangePositionV4.LOWER_EDGE
    ) or (
        candidate.direction is DirectionV4.SHORT
        and position is RangePositionV4.UPPER_EDGE
    )


def _allowed_route(
    direction: DirectionV4,
) -> DirectionRouteDecisionV4:
    return DirectionRouteDecisionV4(
        allowed=True,
        direction=direction,
    )


def _blocked_route(reason: str) -> DirectionRouteDecisionV4:
    return DirectionRouteDecisionV4(
        allowed=False,
        direction=None,
        reason_code=reason,
    )


def _route_reasons(
    routes: Sequence[RoutedCandidateV4],
    *,
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *extra,
                *(
                    route.decision.reason_code
                    for route in routes
                    if route.decision.reason_code is not None
                ),
            }
        )
    )


def _stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:20]}"


def _number(value: float) -> str:
    return float(value).hex()


def _iso(value: datetime) -> str:
    return _aware(value).isoformat(timespec="microseconds")


def _closed_at(
    prepared: _PreparedCandles,
    index: int,
    timeframe_name: str,
) -> datetime:
    durations = {
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
    }
    try:
        duration = durations[timeframe_name]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe: {timeframe_name}") from exc
    return prepared.candles[index].timestamp + duration


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
