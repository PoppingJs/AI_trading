from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.market_state_v4 import (
    CandidateSetupV4,
    ChannelShapeV4,
    DataStatusV4,
    DirectionV4,
    H4BackgroundSnapshotV4,
    H4BackgroundV4,
    H4StructureFactV4,
    MarketStateV4Config,
    MarketStateSnapshotV4,
    MarketStateV4,
    RangePositionV4,
    SetupKindV4,
    classify_h4_background_v4,
    classify_market_state_v4,
    evaluate_early_research_v4,
    mirror_candles_v4,
    route_candidate_direction_v4,
    route_candidate_directions_v4,
)
from ai_trading.models import Candle


START = datetime(2026, 1, 1, tzinfo=UTC)


def test_primary_rail_uses_two_anchors_plus_later_closed_validation() -> None:
    candles = _primary_up_candles()
    result = _classify(candles)

    assert result.state is MarketStateV4.CHANNEL_UP
    assert result.channel_shape is ChannelShapeV4.PRIMARY_RAIL_TREND
    assert result.primary_rail is not None
    assert result.primary_rail.validation_count >= 1
    assert result.paired_rail is None
    assert not result.entry_permission
    assert result.market_state_version == 4
    assert result.channel_geometry_version == 2


def test_missing_paired_rail_does_not_cancel_direction() -> None:
    result = _classify(_primary_up_candles())

    assert result.direction is DirectionV4.LONG
    assert result.channel_id
    assert result.range_tradeable is None


def test_reliable_opposite_reactions_create_paired_channel() -> None:
    result = _classify(_paired_up_candles())

    assert result.state is MarketStateV4.CHANNEL_UP
    assert result.channel_shape is ChannelShapeV4.PAIRED_GEOMETRIC_CHANNEL
    assert result.paired_rail is not None
    assert len(result.paired_rail.anchor_ids) >= 2


def test_market_state_is_price_mirror_symmetric() -> None:
    upward = _primary_up_candles()
    downward = mirror_candles_v4(upward, axis=120.0)

    up_result = _classify(upward)
    down_result = _classify(downward)

    assert up_result.state is MarketStateV4.CHANNEL_UP
    assert down_result.state is MarketStateV4.CHANNEL_DOWN
    assert down_result.direction is DirectionV4.SHORT
    assert down_result.channel_shape is up_result.channel_shape


def test_reliable_range_uses_two_reactions_on_each_side() -> None:
    result = _classify(_range_candles())

    assert result.state is MarketStateV4.RANGE
    assert result.range_tradeable is True
    assert result.range_id
    assert result.frozen_range is not None
    assert len(result.frozen_range.support_anchor_ids) >= 2
    assert len(result.frozen_range.resistance_anchor_ids) >= 2
    assert result.range_position is RangePositionV4.LOWER_EDGE


def test_opposing_primary_rails_never_choose_loop_order_side() -> None:
    candles = _conflicting_rails_candles()

    forward = _classify(candles)
    reversed_input = classify_market_state_v4(
        list(reversed(candles)),
        as_of=_as_of(candles),
    )

    assert forward.state is MarketStateV4.RANGE
    assert forward.range_tradeable is False
    assert forward.direction_conflict
    assert len(forward.structure_ids) == 2
    assert reversed_input == forward


def test_closed_structure_without_channel_is_strong_one_way() -> None:
    upward = _strong_up_candles()
    downward = mirror_candles_v4(upward, axis=120.0)

    up_result = _classify(upward)
    down_result = _classify(downward)

    assert up_result.state is MarketStateV4.STRONG_UP
    assert down_result.state is MarketStateV4.STRONG_DOWN
    assert up_result.channel_shape is None
    assert down_result.channel_shape is None


def test_complete_unstructured_data_is_range_false() -> None:
    candles = _flat_candles(30)
    result = _classify(candles)

    assert result.state is MarketStateV4.RANGE
    assert result.range_tradeable is False
    assert not result.direction_conflict


def test_unclosed_candle_cannot_change_state_or_stable_id() -> None:
    candles = _primary_up_candles()
    as_of = _as_of(candles)
    baseline = classify_market_state_v4(candles, as_of=as_of)
    open_candle = Candle(
        timestamp=as_of,
        open=100.0,
        high=160.0,
        low=40.0,
        close=50.0,
        volume=10_000.0,
    )

    with_open = classify_market_state_v4(
        [*candles, open_candle],
        as_of=as_of + timedelta(minutes=59),
    )

    assert with_open.state is baseline.state
    assert with_open.channel_id == baseline.channel_id
    assert with_open.structure_ids == baseline.structure_ids


def test_closed_harmless_extension_keeps_frozen_channel_id() -> None:
    candles = _primary_up_candles()
    baseline = _classify(candles)
    extended = [
        *candles,
        _candle(33, close=107.6, high=109.6, low=107.1),
    ]

    later = _classify(extended)

    assert later.state is MarketStateV4.CHANNEL_UP
    assert later.channel_id == baseline.channel_id
    assert later.primary_rail is not None
    assert baseline.primary_rail is not None
    assert later.primary_rail.structure_id == baseline.primary_rail.structure_id


def test_previous_active_primary_rail_is_carried_without_reanchoring() -> None:
    candles = _primary_up_candles()
    previous = _classify(candles[:30])
    assert previous.primary_rail is not None

    result = classify_market_state_v4(
        candles,
        as_of=_as_of(candles),
        previous_snapshot=previous,
    )

    assert result.state is MarketStateV4.CHANNEL_UP
    assert result.primary_rail is not None
    assert result.primary_rail.structure_id == (
        previous.primary_rail.structure_id
    )
    assert result.channel_id == previous.channel_id


def test_primary_rail_wrong_side_uses_frozen_three_zone_rule() -> None:
    candles = _primary_up_candles()
    baseline = _classify(candles)
    assert baseline.primary_rail is not None
    rail = baseline.primary_rail
    index = len(candles) - 1
    boundary = rail.boundary_at(index)

    retained = list(candles)
    retained_close = boundary - rail.invalidation_buffer / 2
    retained[index] = _candle(
        index,
        close=retained_close,
        high=boundary + 0.2,
        low=retained_close - 0.2,
    )
    retained_result = _classify(retained)

    invalidated = list(candles)
    invalidated_close = boundary - rail.invalidation_buffer - 0.01
    invalidated[index] = _candle(
        index,
        close=invalidated_close,
        high=boundary + 0.2,
        low=invalidated_close - 0.2,
    )
    invalidated_result = _classify(invalidated)

    assert retained_result.state is MarketStateV4.CHANNEL_UP
    assert retained_result.channel_id == baseline.channel_id
    assert not retained_result.entry_permission
    assert (
        invalidated_result.state is not MarketStateV4.CHANNEL_UP
        or invalidated_result.channel_id != baseline.channel_id
    )


def test_channel_entry_requires_current_price_inside_frozen_touch_zone() -> None:
    candles = _primary_up_candles()
    baseline = _classify(candles)
    assert baseline.primary_rail is not None
    rail = baseline.primary_rail
    index = len(candles) - 1
    close = rail.current_boundary + rail.touch_buffer / 2
    near_rail = list(candles)
    near_rail[index] = _candle(
        index,
        close=close,
        high=close + 0.2,
        low=rail.current_boundary - rail.touch_buffer / 2,
    )

    result = _classify(near_rail)

    assert result.state is MarketStateV4.CHANNEL_UP
    assert result.entry_permission


def test_range_wrong_side_uses_frozen_three_zone_rule() -> None:
    candles = _range_candles()
    baseline = _classify(candles)
    assert baseline.frozen_range is not None
    frozen = baseline.frozen_range
    index = len(candles) - 1
    retained_close = frozen.lower - frozen.invalidation_buffer / 2
    retained = list(candles)
    retained[index] = _candle(
        index,
        close=retained_close,
        high=frozen.lower + 0.2,
        low=retained_close - 0.2,
    )

    result = _classify(retained)

    assert result.state is MarketStateV4.RANGE
    assert result.range_tradeable is True
    assert result.range_id == baseline.range_id
    assert result.range_position is RangePositionV4.OUTSIDE
    assert not result.entry_permission


def test_discontinuous_data_is_not_mislabeled_as_range() -> None:
    candles = _flat_candles(30)
    discontinuous = [*candles[:18], *candles[19:]]

    result = classify_market_state_v4(
        discontinuous,
        as_of=_as_of(candles),
    )

    assert result.state is None
    assert result.data_status is DataStatusV4.DISCONTINUOUS


def test_stale_tail_is_not_mislabeled_as_range() -> None:
    candles = _flat_candles(30)

    result = classify_market_state_v4(
        candles,
        as_of=_as_of(candles) + timedelta(hours=1),
    )

    assert result.state is None
    assert result.data_status is DataStatusV4.DISCONTINUOUS
    assert result.reasons == ("STALE_CLOSED_CANDLES",)


def test_h4_reliable_structure_overrides_ema_and_conflicts_are_explicit() -> None:
    candles = _ema_up_candles()
    as_of = _as_of(candles, hours=4)
    bearish = H4StructureFactV4(
        direction=DirectionV4.SHORT,
        kind="DESCENDING_RESISTANCE",
        structure_id="h4_short",
        confirmed_at=as_of - timedelta(hours=4),
    )
    bullish = H4StructureFactV4(
        direction=DirectionV4.LONG,
        kind="ASCENDING_SUPPORT",
        structure_id="h4_long",
        confirmed_at=as_of - timedelta(hours=8),
        last_validated_at=as_of,
    )

    price_first = classify_h4_background_v4(
        candles,
        as_of=as_of,
        structure_facts=(bearish,),
    )
    conflict = classify_h4_background_v4(
        candles,
        as_of=as_of,
        structure_facts=(bearish, bullish),
    )

    assert price_first.background is H4BackgroundV4.BEARISH
    assert price_first.source == "PRICE_STRUCTURE"
    assert conflict.background is H4BackgroundV4.CONFLICT_NEUTRAL
    assert conflict.bullish_structure_ids == ("h4_long",)
    assert conflict.bearish_structure_ids == ("h4_short",)


def test_h4_ema_fallback_requires_two_closed_confirmations() -> None:
    candles = _ema_up_candles()
    result = classify_h4_background_v4(
        candles,
        as_of=_as_of(candles, hours=4),
    )

    assert result.background is H4BackgroundV4.BULLISH
    assert result.source == "EMA_FALLBACK"

    unclosed_crash = Candle(
        timestamp=_as_of(candles, hours=4),
        open=130.0,
        high=131.0,
        low=80.0,
        close=81.0,
        volume=1000.0,
    )
    unchanged = classify_h4_background_v4(
        [*candles, unclosed_crash],
        as_of=_as_of(candles, hours=4) + timedelta(hours=3),
    )
    assert unchanged.background is H4BackgroundV4.BULLISH

    mirrored = classify_h4_background_v4(
        mirror_candles_v4(candles, axis=140.0),
        as_of=_as_of(candles, hours=4),
    )
    assert mirrored.background is H4BackgroundV4.BEARISH
    assert mirrored.source == "EMA_FALLBACK"


def test_h4_fact_after_closed_data_cutoff_is_ignored() -> None:
    candles = _ema_up_candles()
    as_of = _as_of(candles, hours=4)
    future_bearish = H4StructureFactV4(
        direction=DirectionV4.SHORT,
        kind="DESCENDING_RESISTANCE",
        structure_id="future_h4_short",
        confirmed_at=as_of + timedelta(minutes=1),
    )

    result = classify_h4_background_v4(
        candles,
        as_of=as_of + timedelta(hours=3),
        structure_facts=(future_bearish,),
    )

    assert result.background is H4BackgroundV4.BULLISH
    assert result.source == "EMA_FALLBACK"
    assert result.bearish_structure_ids == ()


def test_h4_external_facts_are_typed_and_replay_invalidation_time() -> None:
    candles = _flat_h4_candles(90)
    as_of = _as_of(candles, hours=4)
    future_invalidation = H4StructureFactV4(
        direction=DirectionV4.LONG,
        kind="ASCENDING_SUPPORT",
        structure_id="active_until_future",
        confirmed_at=as_of - timedelta(hours=8),
        invalidated=True,
        invalidated_at=as_of + timedelta(hours=4),
    )
    arbitrary_fact = H4StructureFactV4(
        direction=DirectionV4.SHORT,
        kind="OI_DISTRIBUTION_LABEL",
        structure_id="not_price_structure",
        confirmed_at=as_of - timedelta(hours=4),
    )

    active = classify_h4_background_v4(
        candles,
        as_of=as_of,
        structure_facts=(future_invalidation, arbitrary_fact),
    )
    invalidated = classify_h4_background_v4(
        candles,
        as_of=as_of,
        structure_facts=(
            H4StructureFactV4(
                direction=DirectionV4.LONG,
                kind="ASCENDING_SUPPORT",
                structure_id="already_invalid",
                confirmed_at=as_of - timedelta(hours=8),
                invalidated=True,
                invalidated_at=as_of - timedelta(hours=4),
            ),
            arbitrary_fact,
        ),
    )

    assert active.background is H4BackgroundV4.BULLISH
    assert active.bullish_structure_ids == ("active_until_future",)
    assert active.bearish_structure_ids == ()
    assert invalidated.background is H4BackgroundV4.CLEAN_NEUTRAL


def test_newer_h4_mirror_supersedes_unrevalidated_old_fact() -> None:
    candles = _flat_h4_candles(90)
    as_of = _as_of(candles, hours=4)
    old_long = H4StructureFactV4(
        direction=DirectionV4.LONG,
        kind="ASCENDING_SUPPORT",
        structure_id="old_long",
        confirmed_at=as_of - timedelta(hours=12),
    )
    newer_short = H4StructureFactV4(
        direction=DirectionV4.SHORT,
        kind="DESCENDING_RESISTANCE",
        structure_id="newer_short",
        confirmed_at=as_of - timedelta(hours=4),
    )

    result = classify_h4_background_v4(
        candles,
        as_of=as_of,
        structure_facts=(newer_short, old_long),
    )

    assert result.background is H4BackgroundV4.BEARISH
    assert result.bullish_structure_ids == ()
    assert result.bearish_structure_ids == ("newer_short",)


def test_newer_h4_range_retires_older_external_direction() -> None:
    candles = _with_timeframe(_range_candles(), minutes=240)
    as_of = max(item.timestamp for item in candles) + timedelta(hours=4)
    old_long = H4StructureFactV4(
        direction=DirectionV4.LONG,
        kind="ASCENDING_SUPPORT",
        structure_id="old_long",
        confirmed_at=START + timedelta(hours=4),
    )

    result = classify_h4_background_v4(
        candles,
        as_of=as_of,
        structure_facts=(old_long,),
    )

    assert result.background is H4BackgroundV4.CLEAN_NEUTRAL
    assert result.source == "FROZEN_RANGE"
    assert result.bullish_structure_ids == ()


def test_only_prior_ema_fallback_receives_35_percent_retention() -> None:
    candles = _ema_hysteresis_candles()
    as_of = _as_of(candles, hours=4)
    config = MarketStateV4Config(
        ema_enter_quantile=0.90,
        ema_exit_quantile=0.10,
    )
    previous_ema = H4BackgroundSnapshotV4(
        background=H4BackgroundV4.BULLISH,
        data_status=DataStatusV4.COMPLETE,
        as_of=as_of - timedelta(hours=4),
        data_cutoff=as_of - timedelta(hours=4),
        source="EMA_FALLBACK",
    )
    previous_structure = H4BackgroundSnapshotV4(
        background=H4BackgroundV4.BULLISH,
        data_status=DataStatusV4.COMPLETE,
        as_of=as_of - timedelta(hours=4),
        data_cutoff=as_of - timedelta(hours=4),
        source="PRICE_STRUCTURE",
    )

    retained = classify_h4_background_v4(
        candles,
        as_of=as_of,
        previous_background=previous_ema,
        config=config,
    )
    reentered = classify_h4_background_v4(
        candles,
        as_of=as_of,
        previous_background=previous_structure,
        config=config,
    )

    assert retained.background is H4BackgroundV4.BULLISH
    assert retained.source == "EMA_FALLBACK"
    assert reentered.background is H4BackgroundV4.CLEAN_NEUTRAL


@pytest.mark.parametrize(
    ("background", "direction", "allowed", "hard_veto"),
    [
        (H4BackgroundV4.BULLISH, DirectionV4.LONG, True, False),
        (H4BackgroundV4.BULLISH, DirectionV4.SHORT, False, True),
        (H4BackgroundV4.BEARISH, DirectionV4.SHORT, True, False),
        (H4BackgroundV4.BEARISH, DirectionV4.LONG, False, True),
    ],
)
def test_h4_opposition_is_applied_once_in_authority_router(
    background: H4BackgroundV4,
    direction: DirectionV4,
    allowed: bool,
    hard_veto: bool,
) -> None:
    h1 = _snapshot(
        MarketStateV4.CHANNEL_UP
        if direction is DirectionV4.LONG
        else MarketStateV4.CHANNEL_DOWN
    )
    candidate = CandidateSetupV4(
        setup_id="candidate",
        direction=direction,
    )

    decision = route_candidate_direction_v4(background, h1, candidate)

    assert decision.allowed is allowed
    assert decision.hard_veto is hard_veto
    assert decision.reason_code == (
        None if allowed else "H4_DIRECTION_OPPOSED"
    )


def test_h4_setup_and_correct_box_edge_are_both_reachable() -> None:
    lower_edge = _snapshot(
        MarketStateV4.RANGE,
        range_tradeable=True,
        position=RangePositionV4.LOWER_EDGE,
    )
    ordinary = CandidateSetupV4(
        setup_id="range_long",
        direction=DirectionV4.LONG,
        kind=SetupKindV4.RANGE_EDGE,
    )
    h4_setup = CandidateSetupV4(
        setup_id="h4_long",
        direction=DirectionV4.LONG,
        kind=SetupKindV4.H4_BULLISH_SUPPORT,
        entry_zone_hit=True,
    )

    assert route_candidate_direction_v4(
        H4BackgroundV4.BULLISH,
        lower_edge,
        ordinary,
    ).allowed
    assert route_candidate_direction_v4(
        H4BackgroundV4.BULLISH,
        lower_edge,
        h4_setup,
    ).allowed


def test_clean_neutral_cannot_route_an_unreflected_h4_specific_setup() -> None:
    h4_setup = CandidateSetupV4(
        setup_id="h4_long",
        direction=DirectionV4.LONG,
        kind=SetupKindV4.H4_BULLISH_SUPPORT,
        entry_zone_hit=True,
    )

    decision = route_candidate_direction_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        _snapshot(MarketStateV4.RANGE, range_tradeable=False),
        h4_setup,
    )

    assert not decision.allowed
    assert decision.reason_code == "NO_AUTHORIZED_DIRECTION"


def test_range_setup_kind_is_not_reused_as_a_trend_setup() -> None:
    candidate = CandidateSetupV4(
        setup_id="wrong_role",
        direction=DirectionV4.LONG,
        kind=SetupKindV4.RANGE_EDGE,
    )

    decision = route_candidate_direction_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        _snapshot(MarketStateV4.CHANNEL_UP),
        candidate,
    )

    assert not decision.allowed
    assert decision.reason_code == "MARKET_STATE_DIRECTION_NOT_AUTHORIZED"


def test_range_false_early_research_is_reachable_before_score_or_plan() -> None:
    range_false = _snapshot(
        MarketStateV4.RANGE,
        range_tradeable=False,
    )
    early = CandidateSetupV4(
        setup_id="m15_long",
        direction=DirectionV4.LONG,
        kind=SetupKindV4.EARLY_RESEARCH,
        entry_zone_hit=False,
    )

    aligned = route_candidate_direction_v4(
        H4BackgroundV4.BULLISH,
        range_false,
        early,
    )
    neutral = route_candidate_direction_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        range_false,
        early,
    )

    assert aligned.allowed and aligned.provisional
    assert neutral.allowed and neutral.provisional


def test_early_research_builder_uses_only_closed_m15_structure() -> None:
    m15 = _with_timeframe(_strong_up_candles(), minutes=15)
    h1 = _snapshot(MarketStateV4.RANGE, range_tradeable=False)
    as_of = max(item.timestamp for item in m15) + timedelta(minutes=15)

    baseline = evaluate_early_research_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        h1,
        m15,
        direction=DirectionV4.LONG,
        as_of=as_of,
    )
    unclosed = Candle(
        timestamp=as_of,
        open=110.0,
        high=111.0,
        low=70.0,
        close=71.0,
        volume=1000.0,
    )
    with_unclosed = evaluate_early_research_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        h1,
        [*m15, unclosed],
        direction=DirectionV4.LONG,
        as_of=as_of + timedelta(minutes=14),
    )

    assert baseline.candidate is not None
    assert baseline.candidate.kind is SetupKindV4.EARLY_RESEARCH
    assert baseline.entry_zone is not None
    assert baseline.invalidation_anchor_id
    assert with_unclosed.structure_id == baseline.structure_id
    assert with_unclosed.candidate == baseline.candidate


def test_early_research_rejects_bare_breakout_and_accepts_retest() -> None:
    h1 = _snapshot(MarketStateV4.RANGE, range_tradeable=False)
    bare = _bare_breakout_m15_candles(with_retest=False)
    retested = _bare_breakout_m15_candles(with_retest=True)

    rejected = evaluate_early_research_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        h1,
        bare,
        direction=DirectionV4.LONG,
        as_of=max(item.timestamp for item in bare)
        + timedelta(minutes=15),
    )
    accepted = evaluate_early_research_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        h1,
        retested,
        direction=DirectionV4.LONG,
        as_of=max(item.timestamp for item in retested)
        + timedelta(minutes=15),
    )

    assert rejected.candidate is None
    assert rejected.reason_code == "EARLY_RESEARCH_STRUCTURE_UNCONFIRMED"
    assert accepted.candidate is not None
    assert accepted.structure_id
    assert accepted.structure_mode == "BREAKOUT_RETEST"
    assert accepted.invalidation_price is not None
    assert accepted.invalidation_buffer is not None
    assert accepted.entry_zone is not None
    assert accepted.entry_zone[0] >= 104.0


def test_early_research_retest_is_price_mirror_symmetric() -> None:
    long_candles = _bare_breakout_m15_candles(with_retest=True)
    short_candles = mirror_candles_v4(long_candles, axis=120.0)
    h1 = _snapshot(MarketStateV4.RANGE, range_tradeable=False)

    long_result = evaluate_early_research_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        h1,
        long_candles,
        direction=DirectionV4.LONG,
        as_of=max(item.timestamp for item in long_candles)
        + timedelta(minutes=15),
    )
    short_result = evaluate_early_research_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        h1,
        short_candles,
        direction=DirectionV4.SHORT,
        as_of=max(item.timestamp for item in short_candles)
        + timedelta(minutes=15),
    )

    assert long_result.candidate is not None
    assert short_result.candidate is not None
    assert short_result.structure_mode == long_result.structure_mode
    assert long_result.entry_zone is not None
    assert short_result.entry_zone is not None
    assert short_result.entry_zone == pytest.approx(
        (
            240.0 - long_result.entry_zone[1],
            240.0 - long_result.entry_zone[0],
        )
    )
    assert long_result.invalidation_price is not None
    assert short_result.invalidation_price == pytest.approx(
        240.0 - long_result.invalidation_price
    )


def test_opposite_early_research_candidates_do_not_select_iteration_side() -> None:
    range_false = _snapshot(
        MarketStateV4.RANGE,
        range_tradeable=False,
    )
    candidates = (
        CandidateSetupV4(
            setup_id="long",
            direction=DirectionV4.LONG,
            kind=SetupKindV4.EARLY_RESEARCH,
        ),
        CandidateSetupV4(
            setup_id="short",
            direction=DirectionV4.SHORT,
            kind=SetupKindV4.EARLY_RESEARCH,
        ),
    )

    forward = route_candidate_directions_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        range_false,
        candidates,
    )
    reverse = route_candidate_directions_v4(
        H4BackgroundV4.CLEAN_NEUTRAL,
        range_false,
        tuple(reversed(candidates)),
    )

    assert forward == reverse
    assert forward.direction_conflict
    assert forward.allowed_directions == ()
    assert all(not item.decision.allowed for item in forward.routes)


def test_batch_router_records_h4_opposition_once() -> None:
    candidates = tuple(
        CandidateSetupV4(
            setup_id=f"short_{index}",
            direction=DirectionV4.SHORT,
        )
        for index in range(2)
    )

    result = route_candidate_directions_v4(
        H4BackgroundV4.BULLISH,
        _snapshot(MarketStateV4.CHANNEL_DOWN),
        candidates,
    )

    assert result.allowed_directions == ()
    assert result.reasons == ("H4_DIRECTION_OPPOSED",)
    assert all(item.decision.hard_veto for item in result.routes)


def test_router_and_early_builder_block_incomplete_h4_snapshot() -> None:
    incomplete_h4 = H4BackgroundSnapshotV4(
        background=H4BackgroundV4.BULLISH,
        data_status=DataStatusV4.DISCONTINUOUS,
        as_of=START,
        data_cutoff=START,
    )
    h1 = _snapshot(MarketStateV4.CHANNEL_UP)
    route = route_candidate_direction_v4(
        incomplete_h4,
        h1,
        CandidateSetupV4(
            setup_id="long",
            direction=DirectionV4.LONG,
        ),
    )
    m15 = _with_timeframe(_strong_up_candles(), minutes=15)
    early = evaluate_early_research_v4(
        incomplete_h4,
        _snapshot(MarketStateV4.RANGE, range_tradeable=False),
        m15,
        direction=DirectionV4.LONG,
        as_of=max(item.timestamp for item in m15)
        + timedelta(minutes=15),
    )

    assert not route.allowed
    assert route.reason_code == "MARKET_STATE_DATA_BLOCKED"
    assert early.candidate is None
    assert early.reason_code == "MARKET_STATE_DATA_BLOCKED"


def test_conflict_neutral_blocks_without_masquerading_as_h4_opposition() -> None:
    candidate = CandidateSetupV4(
        setup_id="long",
        direction=DirectionV4.LONG,
    )

    decision = route_candidate_direction_v4(
        H4BackgroundV4.CONFLICT_NEUTRAL,
        _snapshot(MarketStateV4.CHANNEL_UP),
        candidate,
    )

    assert not decision.allowed
    assert not decision.hard_veto
    assert decision.reason_code == "H4_STRUCTURE_CONFLICT"


def test_paired_channel_has_no_legacy_four_atr_width_gate() -> None:
    candles = _primary_up_candles()
    for index, high in ((17, 130.0), (23, 131.2), (29, 132.4)):
        candle = candles[index]
        candles[index] = _candle(
            index,
            close=candle.close,
            high=high,
            low=candle.low,
        )

    result = _classify(candles)

    assert result.state is MarketStateV4.CHANNEL_UP
    assert result.channel_shape is ChannelShapeV4.PAIRED_GEOMETRIC_CHANNEL
    assert result.primary_rail is not None
    assert result.paired_rail is not None
    paired_atr = (
        result.paired_rail.invalidation_buffer / 0.40
    )
    width = (
        result.paired_rail.current_boundary
        - result.primary_rail.current_boundary
    )
    assert width > 4 * paired_atr


def test_structure_confirmation_time_is_a_closed_candle_boundary() -> None:
    result = _classify(_paired_up_candles())

    assert result.data_cutoff is not None
    assert result.primary_rail is not None
    assert result.paired_rail is not None
    assert result.primary_rail.confirmation_time <= result.data_cutoff
    assert result.paired_rail.confirmation_time <= result.data_cutoff
    assert result.primary_rail.confirmation_time.minute == 0
    assert result.paired_rail.confirmation_time.minute == 0


def test_strong_structure_wrong_side_is_retained_but_cannot_enter() -> None:
    candles = _strong_up_candles()
    index = len(candles) - 1
    candles[index] = _candle(
        index,
        close=99.5,
        high=99.8,
        low=99.3,
    )

    result = _classify(candles)

    assert result.state is MarketStateV4.STRONG_UP
    assert not result.entry_permission


def test_range_does_not_reform_from_pre_break_and_post_break_anchors() -> None:
    result = _classify(_post_break_range_candles())

    assert result.frozen_range is None
    assert result.range_tradeable is not True


def test_previous_range_breach_starts_a_new_anchor_epoch() -> None:
    candles = _expanded_post_break_range_candles()
    before_break = candles[:28]
    previous = _classify(before_break)
    assert previous.range_tradeable is True
    assert previous.frozen_range is not None

    result = classify_market_state_v4(
        candles,
        as_of=_as_of(candles),
        previous_snapshot=previous,
    )

    assert result.frozen_range is None
    assert result.range_tradeable is not True


def test_previous_active_range_keeps_frozen_id_and_boundaries() -> None:
    candles = _range_candles()
    previous = _classify(candles[:28])
    assert previous.frozen_range is not None

    result = classify_market_state_v4(
        candles,
        as_of=_as_of(candles),
        previous_snapshot=previous,
    )

    assert result.range_tradeable is True
    assert result.range_id == previous.range_id
    assert result.frozen_range is not None
    assert result.frozen_range.lower == previous.frozen_range.lower
    assert result.frozen_range.upper == previous.frozen_range.upper


def _classify(candles: list[Candle]) -> MarketStateSnapshotV4:
    return classify_market_state_v4(candles, as_of=_as_of(candles))


def _as_of(candles: list[Candle] | tuple[Candle, ...], *, hours: int = 1) -> datetime:
    return max(candle.timestamp for candle in candles) + timedelta(hours=hours)


def _candle(
    index: int,
    *,
    close: float,
    high: float | None = None,
    low: float | None = None,
    hours: int = 1,
) -> Candle:
    high_value = high if high is not None else close + 1.0
    low_value = low if low is not None else close - 1.0
    return Candle(
        timestamp=START + timedelta(hours=hours * index),
        open=close,
        high=high_value,
        low=low_value,
        close=close,
        volume=1000.0,
    )


def _flat_candles(count: int) -> list[Candle]:
    return [_candle(index, close=100.0) for index in range(count)]


def _primary_up_candles() -> list[Candle]:
    candles = [
        _candle(
            index,
            close=101.0 + index * 0.20,
            high=103.0 + index * 0.20,
            low=100.5 + index * 0.20,
        )
        for index in range(33)
    ]
    for index, low in ((15, 100.0), (21, 101.2), (27, 102.4)):
        candle = candles[index]
        candles[index] = _candle(
            index,
            close=candle.close,
            high=candle.high,
            low=low,
        )
    return candles


def _paired_up_candles() -> list[Candle]:
    candles = _primary_up_candles()
    for index, high in ((17, 110.0), (23, 111.2), (29, 112.4)):
        candle = candles[index]
        candles[index] = _candle(
            index,
            close=candle.close,
            high=high,
            low=candle.low,
        )
    return candles


def _range_candles() -> list[Candle]:
    candles = [
        _candle(index, close=105.0, high=106.0, low=104.0)
        for index in range(34)
    ]
    overrides = {
        15: (103.0, 104.0, 100.0),
        18: (107.0, 110.0, 106.0),
        21: (103.0, 104.0, 100.0),
        24: (107.0, 110.0, 106.0),
        31: (100.4, 102.0, 99.9),
        32: (100.5, 101.5, 100.0),
        33: (100.4, 101.4, 100.1),
    }
    for index, (close, high, low) in overrides.items():
        candles[index] = _candle(index, close=close, high=high, low=low)
    return candles


def _conflicting_rails_candles() -> list[Candle]:
    candles = [
        _candle(index, close=105.0, high=106.0, low=104.0)
        for index in range(34)
    ]
    for index, low in ((15, 95.0), (21, 96.2), (27, 97.4)):
        candles[index] = _candle(index, close=105.0, high=106.0, low=low)
    for index, high in ((16, 115.0), (22, 113.8), (28, 112.6)):
        candles[index] = _candle(index, close=105.0, high=high, low=104.0)
    return candles


def _strong_up_candles() -> list[Candle]:
    candles = _flat_candles(29)
    path = {
        14: (101.0, 102.0, 100.0),
        15: (100.0, 101.0, 98.0),
        16: (101.0, 102.0, 100.5),
        17: (102.0, 103.0, 101.0),
        18: (103.0, 104.0, 102.0),
        19: (102.0, 103.0, 101.0),
        20: (101.5, 102.5, 100.5),
        21: (102.0, 103.0, 100.0),
        22: (104.0, 105.0, 103.0),
        23: (106.0, 107.0, 105.0),
        24: (107.0, 108.0, 106.0),
        25: (106.5, 107.0, 105.5),
        26: (107.0, 107.5, 106.0),
        27: (109.0, 109.5, 108.0),
        28: (110.0, 110.5, 109.0),
    }
    for index, (close, high, low) in path.items():
        candles[index] = _candle(index, close=close, high=high, low=low)
    return candles


def _ema_up_candles() -> list[Candle]:
    return [
        _candle(
            index,
            close=100.0 + index * 0.25,
            high=101.0 + index * 0.25,
            low=99.0 + index * 0.25,
            hours=4,
        )
        for index in range(90)
    ]


def _flat_h4_candles(count: int) -> list[Candle]:
    return [
        _candle(index, close=100.0, hours=4)
        for index in range(count)
    ]


def _ema_hysteresis_candles() -> list[Candle]:
    candles: list[Candle] = []
    close = 100.0
    index = 0
    for count, slope in ((60, 0.05), (10, 0.50), (20, 0.10)):
        for _ in range(count):
            close += slope
            candles.append(
                _candle(
                    index,
                    close=close,
                    high=close + 1.0,
                    low=close - 1.0,
                    hours=4,
                )
            )
            index += 1
    return candles


def _bare_breakout_m15_candles(*, with_retest: bool) -> list[Candle]:
    hourly = _flat_candles(32)
    overrides = {
        15: (100.0, 101.0, 98.0),
        18: (102.0, 104.0, 101.0),
    }
    for index in range(21, len(hourly)):
        close = 105.0 + (index - 21) * 0.30
        overrides[index] = (close, close + 0.50, close - 0.50)
    if with_retest:
        overrides[24] = (105.0, 105.7, 104.0)
    for index, (close, high, low) in overrides.items():
        hourly[index] = _candle(
            index,
            close=close,
            high=high,
            low=low,
        )
    return _with_timeframe(hourly, minutes=15)


def _post_break_range_candles() -> list[Candle]:
    candles = [
        _candle(index, close=105.0, high=106.0, low=104.0)
        for index in range(42)
    ]
    overrides = {
        15: (103.0, 104.0, 100.0),
        18: (107.0, 110.0, 106.0),
        21: (103.0, 104.0, 100.0),
        24: (107.0, 110.0, 106.0),
        27: (95.0, 96.0, 94.0),
        32: (103.0, 104.0, 100.0),
        36: (107.0, 110.0, 106.0),
    }
    for index, (close, high, low) in overrides.items():
        candles[index] = _candle(
            index,
            close=close,
            high=high,
            low=low,
        )
    return candles


def _expanded_post_break_range_candles() -> list[Candle]:
    candles = [
        _candle(index, close=105.0, high=106.0, low=104.0)
        for index in range(44)
    ]
    overrides = {
        15: (103.0, 104.0, 100.0),
        18: (107.0, 110.0, 106.0),
        21: (103.0, 104.0, 100.0),
        24: (107.0, 110.0, 106.0),
        28: (115.0, 116.0, 114.0),
        34: (117.0, 120.0, 116.0),
        40: (117.0, 120.0, 116.0),
    }
    for index, (close, high, low) in overrides.items():
        candles[index] = _candle(
            index,
            close=close,
            high=high,
            low=low,
        )
    return candles


def _snapshot(
    state: MarketStateV4,
    *,
    range_tradeable: bool | None = None,
    position: RangePositionV4 = RangePositionV4.NONE,
) -> MarketStateSnapshotV4:
    direction = None
    if state in {MarketStateV4.CHANNEL_UP, MarketStateV4.STRONG_UP}:
        direction = DirectionV4.LONG
    elif state in {MarketStateV4.CHANNEL_DOWN, MarketStateV4.STRONG_DOWN}:
        direction = DirectionV4.SHORT
    return MarketStateSnapshotV4(
        state=state,
        data_status=DataStatusV4.COMPLETE,
        as_of=START,
        data_cutoff=START,
        direction=direction,
        range_tradeable=range_tradeable,
        range_position=position,
    )


def _with_timeframe(
    candles: list[Candle],
    *,
    minutes: int,
) -> list[Candle]:
    return [
        Candle(
            timestamp=START + timedelta(minutes=minutes * index),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )
        for index, candle in enumerate(candles)
    ]
