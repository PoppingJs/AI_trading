from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ai_trading.models import Candle, IndicatorSnapshot, PositionSide
from ai_trading.score_pipeline_v8 import evaluate_dual_score_v8


START = datetime(2026, 7, 30, 4, 0, tzinfo=timezone.utc)


def _candles(*, falling: bool = False) -> list[Candle]:
    closes = [100.0] * 22
    closes.extend(
        [99.0, 98.9, 98.7]
        if falling
        else [101.0, 101.1, 101.3]
    )
    rows: list[Candle] = []
    for index, close in enumerate(closes):
        previous = closes[index - 1] if index else close
        rows.append(
            Candle(
                timestamp=START + timedelta(minutes=15 * index),
                open=previous,
                high=max(previous, close) + 0.2,
                low=min(previous, close) - 0.2,
                close=close,
                volume=100.0,
            )
        )
    return rows


def _flat_candles() -> list[Candle]:
    return [
        Candle(
            timestamp=START + timedelta(minutes=15 * index),
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
            volume=100.0,
        )
        for index in range(25)
    ]


def _indicators(
    candles: list[Candle],
    *,
    flow_ratio: float,
    account_ratio: float = 1.0,
) -> list[IndicatorSnapshot]:
    return [
        IndicatorSnapshot(
            timestamp=candle.timestamp,
            close=candle.close,
            ema20=100.0,
            ema50=100.0,
            ema200=100.0,
            ma100=100.0,
            boll_mid=100.0,
            boll_upper=102.0,
            boll_lower=98.0,
            rsi14=50.0,
            atr14=4.0,
            volume_sma20=100.0,
            volume_ratio=1.0,
            ema50_slope=0.0,
            vwap=100.0,
            open_interest=1_000_000.0 + index * 10_000.0,
            oi_change=0.01,
            long_short_ratio=account_ratio,
            funding_rate=0.0,
            taker_buy_sell_ratio=flow_ratio,
            top_position_long_short_ratio=1.0,
        )
        for index, candle in enumerate(candles)
    ]


def _higher_timeframe_rows() -> dict[str, list[Candle]]:
    return {
        "1h": [
            Candle(
                timestamp=START + timedelta(hours=index),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=100.0,
            )
            for index in range(8)
        ],
        "4h": [
            Candle(
                timestamp=START + timedelta(hours=4 * index),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=100.0,
            )
            for index in range(8)
        ],
    }


def _context(*, h4_direction: str = "LONG") -> dict[str, object]:
    return {
        "daily_bias": "BULL" if h4_direction == "LONG" else "BEAR",
        "h4_structure": {
            "direction": h4_direction,
            "state": "BREAKOUT_UP" if h4_direction == "LONG" else "BREAKDOWN_DOWN",
            "structure_type": "ASCENDING_SUPPORT" if h4_direction == "LONG" else "DESCENDING_RESISTANCE",
            "support": 99.5,
            "support_zone_low": 99.2,
            "support_zone_high": 99.8,
            "resistance": 102.0,
            "resistance_zone_low": 101.8,
            "resistance_zone_high": 102.2,
        },
        "h1_structure": {
            "direction": h4_direction,
            "state": "BREAKOUT_UP" if h4_direction == "LONG" else "BREAKDOWN_DOWN",
            "structure_type": "ASCENDING_SUPPORT" if h4_direction == "LONG" else "DESCENDING_RESISTANCE",
            "support": 99.5,
            "resistance": 102.0,
        },
        "h1_trigger": {
            "direction": h4_direction,
            "state": "RETEST",
            "support": 99.5,
            "resistance": 102.0,
        },
        "m15_precision": {
            "pullback": "M15_LONG_PULLBACK" if h4_direction == "LONG" else "M15_SHORT_PULLBACK",
        },
        "entry_levels": {
            "long": {
                "h1_support": {"low": 99.0, "high": 101.5, "price": 100.0},
                "h1_ema20": {"low": 99.0, "high": 101.5, "price": 100.0},
                "h1_boll_mid": {"low": 99.0, "high": 101.5, "price": 100.0},
            },
            "short": {
                "h1_resistance": {"low": 98.0, "high": 101.5, "price": 100.0},
                "h1_ema20": {"low": 98.0, "high": 101.5, "price": 100.0},
            },
        },
    }


def _evaluate(
    *,
    candles: list[Candle],
    context: dict[str, object],
    flow_ratio: float,
    account_ratio: float = 1.0,
    price: float | None = None,
):
    higher = _higher_timeframe_rows()
    return evaluate_dual_score_v8(
        symbol="TESTUSDT",
        price=candles[-1].close if price is None else price,
        timeframe_candles={"15m": candles, **higher},
        timeframe_indicators={
            "15m": _indicators(
                candles,
                flow_ratio=flow_ratio,
                account_ratio=account_ratio,
            )
        },
        context=context,
        long_ratio_extreme=2.0,
        short_ratio_extreme=0.5,
        funding_hot_long=0.001,
        funding_hot_short=-0.001,
    )


def test_complete_long_pipeline_can_exceed_100_without_duplicate_location_points() -> None:
    result = _evaluate(
        candles=_candles(),
        context=_context(),
        flow_ratio=1.2,
        account_ratio=4.0,
        price=100.3,
    )

    assert result.selected_side is PositionSide.LONG
    assert result.long.setup_score > 100
    assert result.long.score_breakdown["LOCATION"] == 20
    assert result.long.score_breakdown["PARTICIPATION"] == 15
    # A lone extreme account ratio remains diagnostic and cannot deduct.
    assert result.long.risk_penalties == {}


def test_structure_and_trigger_use_different_closed_candle_events() -> None:
    result = _evaluate(
        candles=_candles(),
        context=_context(),
        flow_ratio=1.2,
        price=100.3,
    ).long

    assert result.score_breakdown["STRUCTURE"] == 30
    assert result.score_breakdown["TRIGGER"] == 15
    assert result.evidence_event_ids["STRUCTURE"] != result.evidence_event_ids["TRIGGER"]


def test_h4_opposed_side_cannot_suppress_the_only_established_direction() -> None:
    # Falling microstructure can make the short raw score stronger, but H4 is
    # explicitly long. Selection must retain the established long candidate;
    # its missing structure/trigger will then be diagnosed by entry policy.
    result = _evaluate(
        candles=_candles(falling=True),
        context=_context(h4_direction="LONG"),
        flow_ratio=0.8,
    )

    assert result.short.direction_confirmation_state == "H4_OPPOSED"
    assert result.long.direction_confirmation_state == "H4_CONFIRMED"
    assert result.selected_side is PositionSide.LONG


def test_short_pipeline_is_directionally_mirrored() -> None:
    result = _evaluate(
        candles=_candles(falling=True),
        context=_context(h4_direction="SHORT"),
        flow_ratio=0.8,
        account_ratio=0.2,
        price=99.7,
    )

    assert result.selected_side is PositionSide.SHORT
    assert result.short.score_breakdown["DIRECTION"] == 20
    assert result.short.score_breakdown["PARTICIPATION"] == 15
    assert result.short.risk_penalties == {}


def test_same_h1_closed_event_cannot_score_as_structure_and_trigger() -> None:
    candles = _candles()
    context = _context()
    context["h1_structure"] = {
        "direction": "NEUTRAL",
        "state": "UNKNOWN",
        "structure_type": "RANGE",
    }
    context["h4_structure"] = {
        "direction": "LONG",
        "state": "BOX_UPPER_HALF",
        "structure_type": "RANGE",
        "support": 99.5,
        "resistance": 102.0,
    }
    context["h1_trigger"] = {
        "direction": "LONG",
        "state": "RETEST",
        "support": 99.5,
        "resistance": 102.0,
    }
    context["m15_precision"] = {}
    # Remove the M15 breakout so the H1 trigger is the selected structure.
    flat = [
        Candle(
            timestamp=row.timestamp,
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
            volume=row.volume,
        )
        for row in candles
    ]

    result = _evaluate(
        candles=flat,
        context=context,
        flow_ratio=1.2,
        price=100.0,
    ).long

    assert result.score_breakdown["STRUCTURE"] == 25
    assert result.score_breakdown["TRIGGER"] == 0


def test_persistent_higher_timeframe_structure_keeps_a_stable_event_id() -> None:
    candles = _flat_candles()
    first = _evaluate(
        candles=candles,
        context=_context(),
        flow_ratio=1.2,
        price=100.3,
    ).long
    shifted = _higher_timeframe_rows()
    shifted["4h"].append(
        Candle(
            timestamp=shifted["4h"][-1].timestamp + timedelta(hours=4),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=100.0,
        )
    )
    second = evaluate_dual_score_v8(
        symbol="TESTUSDT",
        price=100.3,
        timeframe_candles={"15m": candles, **shifted},
        timeframe_indicators={
            "15m": _indicators(candles, flow_ratio=1.2)
        },
        context=_context(),
        long_ratio_extreme=2.0,
        short_ratio_extreme=0.5,
        funding_hot_long=0.001,
        funding_hot_short=-0.001,
    ).long

    assert first.evidence_event_ids["STRUCTURE"] == second.evidence_event_ids["STRUCTURE"]


def test_indicator_only_location_cannot_manufacture_a_structure_stop() -> None:
    candles = _flat_candles()
    context = _context()
    context["entry_levels"] = {
        "long": {
            "h1_ema20": {
                "low": 99.0,
                "high": 101.5,
                "price": 100.0,
            }
        },
        "short": {},
    }

    result = _evaluate(
        candles=candles,
        context=context,
        flow_ratio=1.2,
        price=100.3,
    ).long

    assert result.selected_level.structural is False
    assert result.score_breakdown["LOCATION"] == 15
    assert result.structure_stop is None
