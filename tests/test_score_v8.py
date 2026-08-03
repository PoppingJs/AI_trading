from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ai_trading.models import Candle, PositionSide
from ai_trading.score_v8 import (
    LONG_CROWD_CONFIRMED,
    LONG_CROWD_SUSPECTED,
    MAX_GROSS_SETUP_SCORE,
    PRICE_PROGRESS_MODERATE,
    PRICE_PROGRESS_NONE,
    PRICE_PROGRESS_STRONG,
    PRICE_PROGRESS_UNKNOWN,
    PRICE_PROGRESS_WEAK,
    SCORE_FAMILY_CAPS,
    SHORT_CROWD_CONFIRMED,
    SHORT_CROWD_SUSPECTED,
    CrowdingEvidenceBarV4,
    EvidenceFamilyV8,
    RiskPenaltyV8,
    assess_crowding_v4,
    calculate_price_progress_v8,
    compose_score_v8,
)


START = datetime(2026, 7, 30, 10, 0, tzinfo=timezone.utc)


def _candles(closes: list[float]) -> list[Candle]:
    result: list[Candle] = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index else close
        result.append(
            Candle(
                timestamp=START + timedelta(minutes=15 * index),
                open=open_price,
                high=max(open_price, close) + 0.1,
                low=min(open_price, close) - 0.1,
                close=close,
                volume=100.0,
            )
        )
    return result


def _complete_long_crowd_bar(index: int) -> CrowdingEvidenceBarV4:
    return CrowdingEvidenceBarV4(
        timestamp=START + timedelta(minutes=15 * index),
        long_price_failure=True,
        long_account_ratio_extreme=True,
        long_account_ratio_worsening=True,
        oi_building=True,
    )


def _mirror_bar(bar: CrowdingEvidenceBarV4) -> CrowdingEvidenceBarV4:
    return CrowdingEvidenceBarV4(
        timestamp=bar.timestamp,
        long_price_failure=bar.short_price_failure,
        short_price_failure=bar.long_price_failure,
        long_account_ratio_extreme=bar.short_account_ratio_extreme,
        long_account_ratio_worsening=bar.short_account_ratio_worsening,
        short_account_ratio_extreme=bar.long_account_ratio_extreme,
        short_account_ratio_worsening=bar.long_account_ratio_worsening,
        long_funding_extreme=bar.short_funding_extreme,
        long_funding_worsening=bar.short_funding_worsening,
        short_funding_extreme=bar.long_funding_extreme,
        short_funding_worsening=bar.long_funding_worsening,
        oi_building=bar.oi_building,
        long_active_flow_reversal=bar.short_active_flow_reversal,
        short_active_flow_reversal=bar.long_active_flow_reversal,
        long_top_trader_divergence=bar.short_top_trader_divergence,
        short_top_trader_divergence=bar.long_top_trader_divergence,
    )


def test_family_caps_total_110_without_legacy_100_clamp() -> None:
    result = compose_score_v8(SCORE_FAMILY_CAPS)

    assert MAX_GROSS_SETUP_SCORE == 110
    assert result.gross_setup_score == 110
    assert result.setup_score == 110
    assert result.score_breakdown == SCORE_FAMILY_CAPS


def test_family_values_are_capped_individually_and_missing_families_are_zero() -> None:
    result = compose_score_v8(
        {
            EvidenceFamilyV8.DIRECTION: 200,
            "structure": 25,
        }
    )

    assert result.score_breakdown == {
        "DIRECTION": 20,
        "STRUCTURE": 25,
        "LOCATION": 0,
        "TRIGGER": 0,
        "PARTICIPATION": 0,
        "PRICE_PROGRESS": 0,
    }
    assert result.gross_setup_score == 45


def test_unknown_positive_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown v8 evidence family"):
        compose_score_v8({"RSI_HEALTHY": 10})


def test_same_risk_channel_only_uses_most_severe_penalty() -> None:
    result = compose_score_v8(
        SCORE_FAMILY_CAPS,
        risk_penalties=(
            RiskPenaltyV8("crowding", -2),
            RiskPenaltyV8("CROWDING", -6),
            RiskPenaltyV8("execution", -3),
        ),
    )

    assert result.gross_setup_score == 110
    assert result.risk_penalties == {"CROWDING": -6, "EXECUTION": -3}
    assert result.total_risk_penalty == -9
    assert result.setup_score == 101


def test_positive_risk_penalty_cannot_manufacture_setup_points() -> None:
    with pytest.raises(ValueError, match="zero or negative"):
        compose_score_v8({}, (RiskPenaltyV8("crowding", 6),))


def test_clean_directional_progress_scores_ten_and_opposite_side_zero() -> None:
    candles = _candles([100.0, 101.0, 102.0, 103.0, 104.0])

    long_result = calculate_price_progress_v8(candles, 4.0, "LONG")
    short_result = calculate_price_progress_v8(candles, 4.0, "SHORT")

    assert long_result.data_valid is True
    assert long_result.directional_displacement == pytest.approx(4.0)
    assert long_result.path == pytest.approx(4.8)
    assert long_result.efficiency == pytest.approx(4.0 / 4.8)
    assert long_result.magnitude == pytest.approx(1.0)
    assert long_result.score == 10
    assert long_result.state == PRICE_PROGRESS_STRONG
    assert short_result.score == 0
    assert short_result.state == PRICE_PROGRESS_NONE


def test_price_progress_long_short_formula_is_exactly_mirrored() -> None:
    long_candles = _candles([100.0, 101.0, 102.0, 103.0, 104.0])
    short_candles = _candles([100.0, 99.0, 98.0, 97.0, 96.0])

    long_result = calculate_price_progress_v8(long_candles, 4.0, "LONG")
    short_result = calculate_price_progress_v8(short_candles, 4.0, "SHORT")

    assert short_result.directional_displacement == pytest.approx(
        long_result.directional_displacement
    )
    assert short_result.path == pytest.approx(long_result.path)
    assert short_result.efficiency == pytest.approx(long_result.efficiency)
    assert short_result.magnitude == pytest.approx(long_result.magnitude)
    assert short_result.quality == pytest.approx(long_result.quality)
    assert short_result.score == long_result.score == 10


def test_atr_magnitude_prevents_tiny_monotonic_drift_from_scoring_high() -> None:
    result = calculate_price_progress_v8(
        _candles([100.0, 100.1, 100.2, 100.3, 100.4]),
        atr14=2.0,
        side="LONG",
    )

    assert result.efficiency == pytest.approx(1.0 / 3.0)
    assert result.magnitude == pytest.approx(0.2)
    assert result.score == 2
    assert result.state == PRICE_PROGRESS_WEAK


def test_choppy_path_lowers_efficiency_and_progress_score() -> None:
    result = calculate_price_progress_v8(
        _candles([100.0, 102.0, 99.0, 103.0, 100.5]),
        atr14=2.0,
        side="LONG",
    )

    assert result.directional_displacement == pytest.approx(0.5)
    assert result.efficiency < 0.05
    assert result.score == 0
    assert result.state == PRICE_PROGRESS_NONE


def test_progress_state_bands_cover_medium_score() -> None:
    # displacement=1.6, path=2.4, magnitude=.4, quality=.2667 -> score 8
    strong = calculate_price_progress_v8(
        _candles([100.0, 100.4, 100.8, 101.2, 101.6]),
        atr14=4.0,
        side="LONG",
    )
    # Same path shape with a larger ATR lowers only the magnitude -> score 4.
    moderate = calculate_price_progress_v8(
        _candles([100.0, 100.4, 100.8, 101.2, 101.6]),
        atr14=8.0,
        side="LONG",
    )

    assert strong.state == PRICE_PROGRESS_STRONG
    assert strong.score == 8
    assert moderate.state == PRICE_PROGRESS_MODERATE
    assert moderate.score == 4


@pytest.mark.parametrize(
    "candles, atr14",
    [
        (_candles([100.0, 101.0, 102.0, 103.0]), 2.0),
        (_candles([100.0, 101.0, 102.0, 103.0, 104.0]), None),
        (_candles([100.0, 101.0, 102.0, 103.0, 104.0]), 0.0),
    ],
)
def test_insufficient_or_invalid_progress_input_is_unknown(
    candles: list[Candle],
    atr14: float | None,
) -> None:
    result = calculate_price_progress_v8(candles, atr14, "LONG")

    assert result.state == PRICE_PROGRESS_UNKNOWN
    assert result.score == 0
    assert result.data_valid is False


def test_gap_or_unclosed_last_candle_is_unknown() -> None:
    candles = _candles([100.0, 101.0, 102.0, 103.0, 104.0])
    candles[3] = replace(
        candles[3],
        timestamp=candles[3].timestamp + timedelta(minutes=1),
    )

    gap_result = calculate_price_progress_v8(candles, 4.0, "LONG")
    unclosed_result = calculate_price_progress_v8(
        _candles([100.0, 101.0, 102.0, 103.0, 104.0]),
        4.0,
        "LONG",
        as_of=START + timedelta(minutes=74),
    )

    assert gap_result.state == PRICE_PROGRESS_UNKNOWN
    assert unclosed_result.state == PRICE_PROGRESS_UNKNOWN


def test_extreme_account_ratio_alone_is_suspected_with_zero_penalty() -> None:
    bars = [
        CrowdingEvidenceBarV4(long_account_ratio_extreme=True)
        for _ in range(3)
    ]

    result = assess_crowding_v4(bars, "LONG")

    assert result.state == LONG_CROWD_SUSPECTED
    assert result.penalty == 0
    assert result.qualifying_bars == 0


def test_incomplete_or_price_less_crowding_evidence_does_not_deduct() -> None:
    only_one_auxiliary = [
        CrowdingEvidenceBarV4(
            long_price_failure=True,
            oi_building=True,
        )
        for _ in range(3)
    ]
    no_price_failure = [
        CrowdingEvidenceBarV4(
            long_account_ratio_extreme=True,
            long_account_ratio_worsening=True,
            oi_building=True,
        )
        for _ in range(3)
    ]

    assert assess_crowding_v4(only_one_auxiliary, "LONG").penalty == 0
    assert assess_crowding_v4(no_price_failure, "LONG").penalty == 0


def test_two_of_latest_three_complete_bars_confirm_long_crowding() -> None:
    bars = [
        _complete_long_crowd_bar(0),
        _complete_long_crowd_bar(1),
        CrowdingEvidenceBarV4(
            timestamp=START + timedelta(minutes=30),
            long_account_ratio_extreme=True,
        ),
    ]

    result = assess_crowding_v4(bars, PositionSide.LONG)

    assert result.state == LONG_CROWD_CONFIRMED
    assert result.penalty == -6
    assert result.confirmed is True
    assert result.qualifying_bars == 2
    assert result.as_risk_penalty() == RiskPenaltyV8(
        channel="CROWDING",
        penalty=-6,
        reason=LONG_CROWD_CONFIRMED,
    )


def test_confirmed_long_crowding_penalty_never_leaks_to_short_candidate() -> None:
    bars = [_complete_long_crowd_bar(index) for index in range(3)]

    long_result = assess_crowding_v4(bars, "LONG")
    short_result = assess_crowding_v4(bars, "SHORT")

    assert long_result.state == LONG_CROWD_CONFIRMED
    assert long_result.penalty == -6
    # Direction-neutral OI accumulation can still be shown as an incomplete
    # observation for the other side, but it must never inherit the deduction.
    assert short_result.state == SHORT_CROWD_SUSPECTED
    assert short_result.penalty == 0


def test_short_crowding_is_an_exact_explicit_mirror_of_long_crowding() -> None:
    long_bars = [_complete_long_crowd_bar(index) for index in range(3)]
    short_bars = [_mirror_bar(bar) for bar in long_bars]

    long_result = assess_crowding_v4(long_bars, "LONG")
    short_result = assess_crowding_v4(short_bars, "SHORT")

    assert long_result.state == LONG_CROWD_CONFIRMED
    assert short_result.state == SHORT_CROWD_CONFIRMED
    assert long_result.penalty == short_result.penalty == -6
    assert long_result.qualifying_bars == short_result.qualifying_bars == 3
    assert long_result.auxiliary_counts == short_result.auxiliary_counts


def test_extreme_short_ratio_alone_is_mirrored_suspected_zero_penalty() -> None:
    bars = [
        CrowdingEvidenceBarV4(short_account_ratio_extreme=True)
        for _ in range(3)
    ]

    result = assess_crowding_v4(bars, "SHORT")

    assert result.state == SHORT_CROWD_SUSPECTED
    assert result.penalty == 0


def test_trend_continuation_without_price_failure_cannot_confirm_crowding() -> None:
    # OI and active flow continue with the trend. Even an extreme and worsening
    # account ratio is not a reversal signal without price failure.
    bars = [
        CrowdingEvidenceBarV4(
            long_account_ratio_extreme=True,
            long_account_ratio_worsening=True,
            oi_building=True,
            long_top_trader_divergence=True,
            long_price_failure=False,
        )
        for _ in range(3)
    ]

    result = assess_crowding_v4(bars, "LONG")

    assert result.state == LONG_CROWD_SUSPECTED
    assert result.penalty == 0


def test_fewer_than_three_or_non_contiguous_bars_cannot_confirm() -> None:
    two_bars = [_complete_long_crowd_bar(index) for index in range(2)]
    gapped_bars = [_complete_long_crowd_bar(index) for index in range(3)]
    gapped_bars[2] = replace(
        gapped_bars[2],
        timestamp=gapped_bars[2].timestamp + timedelta(minutes=1),
    )

    short_window = assess_crowding_v4(two_bars, "LONG")
    gapped = assess_crowding_v4(gapped_bars, "LONG")

    assert short_window.state == LONG_CROWD_SUSPECTED
    assert short_window.penalty == 0
    assert gapped.state == LONG_CROWD_SUSPECTED
    assert gapped.penalty == 0
    assert gapped.data_contiguous is False
