from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from ai_trading.config import AppSettings
from ai_trading.entry_policy_v4 import (
    ENTRY_MODE_TREND_RESEARCH,
    ENTRY_MODE_TREND_STARTUP,
)
from ai_trading.models import (
    Candle,
    IndicatorSnapshot,
    Position,
    PositionSide,
    SignalAction,
)
from ai_trading.paper import (
    ENTRY_QUALITY_B,
    ENTRY_QUALITY_S,
    PaperTradingEngine,
    _activate_plan3_runner_stop,
    _apply_entry_policy_v4_fields,
    _candles_current_for_scoring,
    _clear_transient_auto_entry_blocks,
    _entry_context_from_signal,
    _entry_quality_grade,
    _exit_timeframe_from_position,
    _latest_closed_slot,
    _promote_plan3_stop_after_structure,
    _record_auto_entry_block,
    _side_entry_timing,
    _timeframe_seconds,
    _v3_runner_can_continue,
)


NOW = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)


def _position(
    *,
    side: PositionSide = PositionSide.LONG,
    first_tp_done: bool = False,
) -> Position:
    return Position(
        symbol="TESTUSDT",
        side=side,
        entry_price=100.0,
        quantity=1.0,
        opened_at=NOW,
        stop_price=90.0 if side is PositionSide.LONG else 110.0,
        take_profit_1=108.0 if side is PositionSide.LONG else 92.0,
        take_profit_2=120.0 if side is PositionSide.LONG else 80.0,
        first_tp_done=first_tp_done,
        metadata={
            "margin_usdt": 100.0,
            "initial_stop_distance": 10.0,
            "plan_version": 3,
            "entry_context": {
                "long_evidence_event_ids": {"STRUCTURE": "ENTRY-LONG"},
                "short_evidence_event_ids": {"STRUCTURE": "ENTRY-SHORT"},
            },
        },
    )


def _indicator(
    close: float = 110.0,
    *,
    timestamp: datetime = NOW,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=timestamp,
        close=close,
        ema20=105.0,
        ema50=103.0,
        ema200=100.0,
        ma100=101.0,
        boll_mid=104.0,
        boll_upper=112.0,
        boll_lower=96.0,
        rsi14=60.0,
        atr14=1.0,
        volume_sma20=100.0,
        volume_ratio=1.0,
        ema50_slope=0.1,
    )


def _plan4_position(
    *,
    stop_timeframe: str = "1h",
    disaster_stop: float = 80.0,
) -> Position:
    position = _position()
    position.metadata.update(
        {
            "plan_version": 4,
            "structure_stop_price": 90.0,
            "soft_stop_price": 90.0,
            "hard_stop_price": disaster_stop,
            "disaster_stop_price": disaster_stop,
            "stop_execution_mode": "HTF_CLOSE",
            "entry_context": {
                "entry_source": "AUTO_STRATEGY",
                "score_model_version": 8,
                "stop_timeframe": stop_timeframe,
                "stop_execution_mode": "HTF_CLOSE",
            },
        }
    )
    return position


def _closed_candle_series(
    timeframe: str,
    *,
    stale_periods: int = 0,
) -> list[Candle]:
    seconds = _timeframe_seconds(timeframe)
    latest_open = _latest_closed_slot(
        timeframe,
        NOW,
    ) - timedelta(seconds=seconds * (1 + stale_periods))
    return [
        Candle(
            timestamp=latest_open - timedelta(seconds=seconds * (49 - index)),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000.0,
        )
        for index in range(50)
    ]


def _v8_ready_signal(
    *,
    score: int,
    direction_score: int,
    progress_score: int,
) -> dict[str, object]:
    breakdown = {
        "DIRECTION": direction_score,
        "STRUCTURE": 25,
        "LOCATION": 15,
        "TRIGGER": 10,
        "PARTICIPATION": 0,
        "PRICE_PROGRESS": progress_score,
    }
    return {
        "prd_version": "5.0",
        "score_model_version": 8,
        "entry_pipeline_version": 5,
        "timestamp": NOW.isoformat(),
        "action": SignalAction.ENTRY_LONG.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "decision_action": SignalAction.ENTRY_LONG.value,
        "direction": PositionSide.LONG.value,
        "direction_confirmation_state": "TEMPORARY_CONFIRMED",
        "score": score,
        "setup_score": score,
        "score_before_entry_quality": score,
        "gross_setup_score": score,
        "score_breakdown": breakdown,
        "score_evidence_families": breakdown,
        "risk_penalties": {},
        "total_risk_penalty": 0,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.8,
        "m15_atr14": 1.0,
        "selected_level_key": "h1_ema20",
        "selected_level_zone": {
            "low": 99.0,
            "high": 101.0,
            "price": 100.0,
        },
        "selected_level_structural": True,
        "selected_level_eligible": True,
        "v8_structure_stop": 98.0,
        "v8_structure_target": 110.0,
        "entry_levels": {
            "long": {
                "h1_ema20": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
        "h1_structure": {
            "direction": "LONG",
            "state": "BREAKOUT_UP",
            "support": 99.0,
            "support_zone_low": 99.0,
            "resistance": 110.0,
            "resistance_zone_high": 110.0,
        },
        "h4_structure": {
            "direction": "NEUTRAL",
            "state": "RANGE",
            "structure_type": "RANGE",
        },
        "data_confidence": {
            "price_data_contiguous": True,
            "price_data_fresh": True,
            "derivatives_data_complete": True,
        },
        "policy_blocks": (),
        "vetoes": (),
        "auto_entry_blocks": (),
        "reasons": (),
        "market_context": {
            "liquidity_state": "NORMAL",
            "system_risk_state": "NORMAL",
        },
    }


class _FlatPaperMarket:
    async def klines(
        self,
        symbol: str,
        interval: str = "15m",
        *,
        limit: int = 500,
        start_time_ms=None,
        end_time_ms=None,
    ) -> list[Candle]:
        del symbol, start_time_ms, end_time_ms
        seconds = _timeframe_seconds(interval)
        latest_open = _latest_closed_slot(
            interval,
            NOW,
        ) - timedelta(seconds=seconds)
        return [
            Candle(
                timestamp=latest_open
                - timedelta(seconds=seconds * (limit - index - 1)),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000.0,
            )
            for index in range(limit)
        ]


def test_startup_and_research_modes_reach_execution_grade_without_hidden_75_gate() -> None:
    assert _entry_quality_grade(score=70, entry_mode=ENTRY_MODE_TREND_STARTUP) == ENTRY_QUALITY_B
    assert _entry_quality_grade(score=67, entry_mode=ENTRY_MODE_TREND_RESEARCH) == ENTRY_QUALITY_B
    assert _entry_quality_grade(score=110) == ENTRY_QUALITY_S


@pytest.mark.parametrize(
    (
        "score",
        "direction_score",
        "progress_score",
        "expected_mode",
    ),
    [
        (65, 6, 9, ENTRY_MODE_TREND_RESEARCH),
        (70, 12, 8, ENTRY_MODE_TREND_STARTUP),
    ],
)
def test_v4_research_and_startup_scores_execute_real_open_orders(
    score: int,
    direction_score: int,
    progress_score: int,
    expected_mode: str,
) -> None:
    settings = AppSettings()
    settings.risk.max_open_positions = 1
    engine = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatPaperMarket(),
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.8)
    engine.latest_signals["TESTUSDT"] = _v8_ready_signal(
        score=score,
        direction_score=direction_score,
        progress_score=progress_score,
    )

    asyncio.run(engine._auto_trade_once())

    assert "TESTUSDT" in engine.account.positions
    position = engine.account.positions["TESTUSDT"]
    entry_context = position.metadata["entry_context"]
    assert entry_context["entry_mode"] == expected_mode
    assert entry_context["setup_score"] == score
    assert entry_context["plan_version"] == 4
    assert any(
        fill.action == "OPEN" and fill.symbol == "TESTUSDT"
        for fill in engine.account.fills
    )


@pytest.mark.parametrize(
    (
        "score",
        "direction_score",
        "structure_score",
        "participation_score",
        "expected_quality",
        "expected_max_units",
    ),
    [
        (75, 12, 25, 5, "B", 1),
        (80, 12, 25, 10, "A", 1),
        (95, 20, 27, 15, "S", 2),
    ],
)
def test_paper_auto_entry_uses_one_initial_unit_for_every_quality(
    score: int,
    direction_score: int,
    structure_score: int,
    participation_score: int,
    expected_quality: str,
    expected_max_units: int,
) -> None:
    settings = AppSettings()
    settings.execution.paper_trading = True
    settings.risk.paper_fixed_unit_sizing = True
    settings.risk.max_open_positions = 1
    engine = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatPaperMarket(),
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.8)
    signal = _v8_ready_signal(
        score=score,
        direction_score=direction_score,
        progress_score=8,
    )
    score_breakdown = dict(signal["score_breakdown"])
    score_breakdown["STRUCTURE"] = structure_score
    score_breakdown["PARTICIPATION"] = participation_score
    assert sum(score_breakdown.values()) == score
    signal["score_breakdown"] = score_breakdown
    signal["score_evidence_families"] = dict(score_breakdown)
    engine.latest_signals["TESTUSDT"] = signal

    asyncio.run(engine._auto_trade_once())

    position = engine.account.positions["TESTUSDT"]
    context = position.metadata["entry_context"]
    assert position.metadata["margin_usdt"] == pytest.approx(190.0)
    assert engine.account.fills[-1].margin_usdt == pytest.approx(190.0)
    assert context["entry_quality"] == expected_quality
    assert context["capital_units"] == pytest.approx(1.0)
    assert context["max_capital_units"] == expected_max_units
    assert context["position_sizing_mode"] == "FIXED_CAPITAL_UNIT"
    assert context["entry_source"] == "AUTO_STRATEGY"
    assert context["risk_gate_status"] == "PAPER_FIXED_UNIT"
    assert context["allocated_margin_usdt"] == pytest.approx(190.0)
    assert context["risk_sized_margin_usdt"] < 190.0
    assert context["planned_risk_usdt"] > context["risk_budget_usdt"]
    assert context["open_risk_after_usdt"] == pytest.approx(
        context["open_risk_before_usdt"]
        + context["planned_risk_usdt"]
    )
    assert position.metadata["initial_risk_usdt"] == pytest.approx(
        context["planned_risk_usdt"]
    )


@pytest.mark.parametrize(
    ("paper_trading", "fixed_unit_sizing"),
    [(True, False), (False, True)],
)
def test_fixed_unit_mode_falls_back_to_risk_sizing_when_disabled(
    paper_trading: bool,
    fixed_unit_sizing: bool,
) -> None:
    settings = AppSettings()
    settings.execution.paper_trading = paper_trading
    settings.risk.paper_fixed_unit_sizing = fixed_unit_sizing
    settings.risk.max_open_positions = 1
    engine = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatPaperMarket(),
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.8)
    engine.latest_signals["TESTUSDT"] = _v8_ready_signal(
        score=95,
        direction_score=12,
        progress_score=8,
    )

    asyncio.run(engine._auto_trade_once())

    position = engine.account.positions["TESTUSDT"]
    context = position.metadata["entry_context"]
    assert position.metadata["margin_usdt"] < 190.0
    assert context["position_sizing_mode"] == "RISK_BASED"
    assert context["risk_gate_status"] == "ALLOWED"


def test_fixed_unit_mode_keeps_account_loss_gate_as_a_hard_block() -> None:
    settings = AppSettings()
    settings.execution.paper_trading = True
    settings.risk.paper_fixed_unit_sizing = True
    settings.risk.daily_loss_limit = 0.01
    engine = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatPaperMarket(),
        clock=lambda: NOW,
    )
    engine._account_risk_snapshot(engine.status(), now=NOW)
    engine.account.wallet_balance = 900.0
    engine._remember_mark_price("TESTUSDT", 100.8)
    signal = _v8_ready_signal(
        score=75,
        direction_score=12,
        progress_score=8,
    )
    score_breakdown = dict(signal["score_breakdown"])
    score_breakdown["PARTICIPATION"] = 5
    signal["score_breakdown"] = score_breakdown
    signal["score_evidence_families"] = dict(score_breakdown)
    engine.latest_signals["TESTUSDT"] = signal

    asyncio.run(engine._auto_trade_once())

    assert "TESTUSDT" not in engine.account.positions
    published = engine.latest_signals["TESTUSDT"]
    assert published["risk_gate_status"] == "BLOCKED"
    assert published["risk_gate_code"] == "DAILY_LOSS_LIMIT"
    assert any(
        "daily loss limit reached" in str(reason)
        for reason in published["auto_entry_blocks"]
    )


def test_v8_stale_closed_candle_updates_policy_and_blocks_entry() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatPaperMarket(),
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.8)
    engine._oi_ratio_updated_at["TESTUSDT"] = NOW
    engine._derivatives_updated_at["TESTUSDT"] = NOW
    engine._timeframe_candles["TESTUSDT"] = {
        "15m": _closed_candle_series("15m", stale_periods=2),
        "1h": _closed_candle_series("1h"),
        "4h": _closed_candle_series("4h"),
    }
    signal = _v8_ready_signal(
        score=70,
        direction_score=12,
        progress_score=8,
    )

    blocks = engine._data_freshness_blocks("TESTUSDT", signal)

    assert not _candles_current_for_scoring(
        engine._timeframe_candles["TESTUSDT"]["15m"],
        "15m",
        NOW,
    )
    assert "15m K-line context is stale" in blocks
    assert signal["data_confidence"]["price_data_fresh"] is False
    assert "PRICE_DATA_STALE" in signal["policy_blocks"]
    assert signal["decision_action"] == "WATCH"


def test_pipeline4_runtime_blocks_do_not_feed_back_into_strategy_vetoes() -> None:
    signal: dict[str, object] = {
        "entry_pipeline_version": 4,
        "vetoes": (),
    }

    _record_auto_entry_block(signal, "ENTRY_TRIGGER_UNAVAILABLE")

    assert signal["vetoes"] == ()
    assert signal["auto_entry_blocks"] == ("ENTRY_TRIGGER_UNAVAILABLE",)
    _clear_transient_auto_entry_blocks(signal)
    assert signal["auto_entry_blocks"] == ()


def test_v8_strategy_signal_defers_low_net_r_to_final_execution() -> None:
    signal = _v8_ready_signal(
        score=75,
        direction_score=12,
        progress_score=8,
    )
    signal.update(
        {
            "entry_timing": "GOOD",
            "preview_structure_stop": 98.0,
            "nearest_structure_target": 102.0,
            "net_plan_r": 0.7595,
            "entry_quality_status": "BELOW_MIN_NET_R",
        }
    )

    _apply_entry_policy_v4_fields(signal)

    assert signal["decision_action"] == SignalAction.ENTRY_LONG.value
    assert signal["action"] == SignalAction.ENTRY_LONG.value
    assert signal["policy_blocks"] == ()
    assert signal["entry_quality_status"] == "BELOW_MIN_NET_R"


def test_entry_context_persists_v8_decision_and_structured_evidence() -> None:
    context = _entry_context_from_signal(
        {
            "prd_version": "4.0",
            "score_model_version": 8,
            "entry_pipeline_version": 4,
            "entry_mode": ENTRY_MODE_TREND_RESEARCH,
            "gross_setup_score": 104,
            "setup_score": 98,
            "risk_penalties": {"CROWDING": -6},
            "evidence_event_ids": {"STRUCTURE": "S1"},
            "long_evidence_event_ids": {"STRUCTURE": "LS1"},
            "selected_level_key": "h1_support",
            "selected_level_zone": {"low": 99.0, "high": 100.0},
            "data_confidence": {"derivatives_data_complete": False},
        }
    )

    assert context["prd_version"] == "4.0"
    assert context["entry_mode"] == ENTRY_MODE_TREND_RESEARCH
    assert context["gross_setup_score"] == 104
    assert context["risk_penalties"] == {"CROWDING": -6}
    assert context["long_evidence_event_ids"]["STRUCTURE"] == "LS1"


def test_plan3_tp1_reduces_position_without_immediate_breakeven_stop() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1_000.0, symbols=["TESTUSDT"])
    position = _position()
    engine.account.positions[position.symbol] = position

    engine._manage_open_position_v3(
        position,
        108.5,
        {
            "timestamp": NOW.isoformat(),
            "long_evidence_event_ids": {"STRUCTURE": "ENTRY-LONG"},
            "long_score_breakdown": {"STRUCTURE": 30},
        },
        None,
        strong_trend=True,
    )

    assert position.first_tp_done is True
    assert position.remaining_fraction == pytest.approx(
        1.0 - AppSettings().risk.first_take_profit_fraction
    )
    assert position.stop_price == 90.0
    assert position.metadata["position_stage"] == "TP1_DONE"
    assert position.metadata["tp1_structure_event_id"] == "ENTRY-LONG"


def test_plan3_time_stop_respects_explicit_zero_side_progress() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1_000.0, symbols=["TESTUSDT"])
    position = _position()
    position.bars_held = AppSettings().risk.time_stop_bars
    engine.account.positions[position.symbol] = position

    engine._manage_open_position_v3(
        position,
        99.0,
        {
            "long_price_progress_state": "NO_PROGRESS",
            "long_price_progress_score": 0,
            # Generic progress belongs to a newly selected short candidate and
            # must not overwrite the held long's explicit zero.
            "price_progress_state": "STRONG_PROGRESS",
            "price_progress_score": 10,
        },
        None,
        strong_trend=False,
    )

    assert position.symbol not in engine.account.positions
    assert "no 0.5R progress" in engine.account.fills[-1].reason


def test_plan3_stop_promotes_only_after_a_new_structure_event() -> None:
    position = _position(first_tp_done=True)
    base_signal = {
        "timestamp": NOW.isoformat(),
        "long_score_breakdown": {"STRUCTURE": 30},
        "h1_structure": {
            "direction": "LONG",
            "state": "BREAKOUT_UP",
            "support": 106.0,
        },
    }

    # A legacy/loaded plan-3 position may not have the H1 snapshot yet.  The
    # first observation establishes a baseline and must not tighten the stop.
    unchanged = _promote_plan3_stop_after_structure(
        position,
        112.0,
        base_signal,
        _indicator(112.0),
    )
    repeated = _promote_plan3_stop_after_structure(
        position,
        112.0,
        base_signal,
        _indicator(112.0),
    )
    promoted = _promote_plan3_stop_after_structure(
        position,
        112.0,
        {
            **base_signal,
            "h1_structure": {
                "direction": "LONG",
                "state": "BREAKOUT_UP",
                "support": 107.0,
            },
        },
        _indicator(112.0),
    )

    assert unchanged is False
    assert repeated is False
    assert promoted is True
    assert position.stop_price == pytest.approx(106.75)
    assert position.metadata["plan3_stop_structure_event_id"] == (
        "TESTUSDT:LONG:H1_STRUCTURE:h1_structure:BREAKOUT_UP:107"
    )


def test_plan3_runner_rejects_h4_opposition_even_when_low_timeframe_is_strong() -> None:
    position = _position(first_tp_done=True)
    signal = {
        "h4_structure": {"direction": "SHORT", "state": "BREAKDOWN_DOWN"},
        "short_price_progress_score": 0,
        "long_score_breakdown": {"STRUCTURE": 30},
        "long_evidence_event_ids": {"STRUCTURE": "NEW-LONG"},
    }

    assert _v3_runner_can_continue(position, signal, strong_trend=True) is False


def test_plan3_runner_fails_closed_when_h4_structure_is_unknown() -> None:
    position = _position(first_tp_done=True)
    signal = {
        "h4_structure": {"direction": "NEUTRAL", "state": "UNKNOWN"},
        "short_price_progress_score": 0,
        "long_score_breakdown": {"STRUCTURE": 30},
        "long_evidence_event_ids": {"STRUCTURE": "NEW-LONG"},
    }

    assert _v3_runner_can_continue(position, signal, strong_trend=True) is False


def test_plan3_stop_ignores_new_non_h1_structure_event() -> None:
    position = _position(first_tp_done=True)
    h1_signal = {
        "long_score_breakdown": {"STRUCTURE": 30},
        "h1_structure": {
            "direction": "LONG",
            "state": "BREAKOUT_UP",
            "support": 106.0,
        },
    }
    baseline = _promote_plan3_stop_after_structure(
        position,
        112.0,
        h1_signal,
        _indicator(112.0),
    )
    promoted = _promote_plan3_stop_after_structure(
        position,
        112.0,
        {
            **h1_signal,
            "long_evidence_event_ids": {
                "STRUCTURE": "TESTUSDT:LONG:STRUCTURE:15m:NEW-EVENT"
            },
        },
        _indicator(112.0),
    )

    assert baseline is False
    assert promoted is False
    assert position.stop_price == 90.0


def test_confirmed_crowding_tightens_runner_but_does_not_exit_it() -> None:
    normal = _position(first_tp_done=True)
    crowded = _position(first_tp_done=True)

    assert _activate_plan3_runner_stop(
        normal,
        130.0,
        {"crowding_state": "NORMAL"},
        None,
        lock_r=0.5,
    )
    assert _activate_plan3_runner_stop(
        crowded,
        130.0,
        {"crowding_state": "LONG_CROWD_CONFIRMED"},
        None,
        lock_r=0.5,
    )

    assert normal.stop_price == 105.0
    assert crowded.stop_price == 107.5
    assert crowded.metadata["plan3_crowding_tightened"] is True


def test_plan4_structure_stop_waits_for_new_h1_close() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
    )
    position = _plan4_position(stop_timeframe="1h")
    engine.account.positions[position.symbol] = position

    # The live mark (or an intrabar wick) may cross the ordinary structure
    # price, but plan 4 must wait for a newly closed H1 body.
    engine._manage_open_position_v3(
        position,
        89.0,
        {},
        _indicator(100.0, timestamp=NOW),
        strong_trend=False,
    )

    assert position.symbol in engine.account.positions

    engine._manage_open_position_v3(
        position,
        89.0,
        {},
        _indicator(89.0, timestamp=NOW + timedelta(hours=1)),
        strong_trend=False,
    )

    assert position.symbol not in engine.account.positions
    assert "structure close confirmed" in engine.account.fills[-1].reason


def test_plan4_h4_position_ignores_h1_failure_until_h4_close() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
    )
    position = _plan4_position(stop_timeframe="4h")
    engine.account.positions[position.symbol] = position
    engine.latest_prices[position.symbol] = 89.0
    engine.latest_timeframe_indicators[position.symbol] = {
        "1h": [_indicator(89.0, timestamp=NOW + timedelta(hours=1))],
        "4h": [_indicator(100.0, timestamp=NOW)],
    }
    engine.latest_signals[position.symbol] = {
        "h1_trigger": {"direction": "SHORT", "state": "BREAKDOWN"},
        "h4_structure": {"direction": "LONG", "state": "RANGE"},
    }

    engine._manage_open_positions()

    assert position.symbol in engine.account.positions

    engine.latest_timeframe_indicators[position.symbol]["4h"] = [
        _indicator(89.0, timestamp=NOW + timedelta(hours=4))
    ]
    engine.latest_signals[position.symbol]["h4_structure"] = {
        "direction": "SHORT",
        "state": "BREAKDOWN_DOWN",
    }
    engine._manage_open_positions()

    assert position.symbol not in engine.account.positions
    assert "structure close confirmed" in engine.account.fills[-1].reason


def test_plan4_m15_tactical_stop_confirmation_is_promoted_to_h1() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
    )
    position = _plan4_position(stop_timeframe="15m")
    engine.account.positions[position.symbol] = position
    engine.latest_prices[position.symbol] = 89.0
    engine.latest_timeframe_indicators[position.symbol] = {
        "15m": [_indicator(100.0, timestamp=NOW)],
        "1h": [_indicator(100.0, timestamp=NOW)],
    }

    assert _exit_timeframe_from_position(position) == "1h"
    engine._manage_open_positions()
    assert position.symbol in engine.account.positions

    engine.latest_timeframe_indicators[position.symbol]["15m"] = [
        _indicator(89.0, timestamp=NOW + timedelta(minutes=15))
    ]
    engine._manage_open_positions()
    assert position.symbol in engine.account.positions

    engine.latest_timeframe_indicators[position.symbol]["1h"] = [
        _indicator(89.0, timestamp=NOW + timedelta(hours=1))
    ]
    engine._manage_open_positions()

    assert position.symbol not in engine.account.positions
    assert "structure close confirmed" in engine.account.fills[-1].reason


def test_plan4_disaster_stop_executes_on_mark_without_atr_reason() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
    )
    position = _plan4_position(disaster_stop=80.0)
    engine.account.positions[position.symbol] = position

    engine._manage_open_position_v3(
        position,
        79.0,
        {},
        _indicator(100.0),
        strong_trend=False,
    )

    assert position.symbol not in engine.account.positions
    assert "灾难保护触发" in engine.account.fills[-1].reason
    assert "ATR" not in engine.account.fills[-1].reason


def test_plan3_loaded_position_keeps_legacy_mark_stop() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
    )
    position = _position()
    engine.account.positions[position.symbol] = position

    engine._manage_open_position_v3(
        position,
        89.0,
        {},
        _indicator(100.0),
        strong_trend=False,
    )

    assert position.symbol not in engine.account.positions


def test_qualified_suggested_zone_hit_still_means_good_entry_timing() -> None:
    signal = _v8_ready_signal(
        score=75,
        direction_score=12,
        progress_score=13,
    )

    timing, reason = _side_entry_timing(
        PositionSide.LONG,
        100.8,
        signal,
    )

    assert timing == "GOOD"
    assert reason == "已到优势入场区"


def test_unclosed_live_m15_does_not_replace_v5_authorized_zone() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
    )
    confirmed = [
        Candle(
            timestamp=NOW - timedelta(minutes=15 * (60 - index)),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000.0,
        )
        for index in range(60)
    ]
    engine._timeframe_candles["TESTUSDT"] = {"15m": confirmed}
    engine._live_m15_candles["TESTUSDT"] = Candle(
        timestamp=NOW,
        open=100.0,
        high=103.0,
        low=97.0,
        close=102.0,
        volume=0.0,
    )
    signal = _v8_ready_signal(
        score=75,
        direction_score=12,
        progress_score=13,
    )
    original_precision = {
        "trend": "UP",
        "pullback": "WAIT",
    }
    signal["m15_precision"] = dict(original_precision)
    original_levels = dict(signal["entry_levels"])

    engine._apply_live_m15_overlay("TESTUSDT", signal)

    assert signal["m15_precision"] == original_precision
    assert signal["entry_levels"] == original_levels
    assert "live_m15_precision" in signal


def test_stale_v8_entry_zone_policy_waits_for_v5_rescore() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatPaperMarket(),
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.8)
    signal = _v8_ready_signal(
        score=75,
        direction_score=12,
        progress_score=13,
    )
    signal["entry_pipeline_version"] = 4
    engine.latest_signals["TESTUSDT"] = signal

    asyncio.run(engine._auto_trade_once())

    assert "TESTUSDT" not in engine.account.positions
    assert any(
        "entry-zone policy is stale" in str(reason)
        for reason in engine.latest_signals["TESTUSDT"].get(
            "auto_entry_blocks",
            (),
        )
    )
