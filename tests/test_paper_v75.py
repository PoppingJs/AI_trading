from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.config import AppSettings
from ai_trading.execution_plan_v7 import (
    PROTECTION_ACTIVE,
    PROTECTION_ELIGIBLE,
)
from ai_trading.models import (
    PLAN_TARGET_MODE_BOUNDED_TARGETS,
    PLAN_TARGET_MODE_OPEN_SPACE,
    Candle,
    IndicatorSnapshot,
    PositionSide,
    SignalAction,
)
from ai_trading.paper import (
    PaperTradingEngine,
    _advance_open_space_position_plan,
    _apply_entry_policy_v4_fields,
    _auto_entry_decision_id,
    _auto_entry_status_signal,
    _clear_execution_stage_signal_fields,
    _clear_transient_auto_entry_blocks,
    _entry_policy_v7_reason_text,
    _record_auto_entry_block,
    _refresh_entry_quality_for_live_price,
    _timeframe_seconds,
    _update_entry_position_fields,
    _visible_auto_entry_block,
)


NOW = datetime(2026, 8, 7, 10, 0, tzinfo=UTC)


class _FlatMarket:
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
        return [
            Candle(
                timestamp=NOW
                - timedelta(seconds=seconds * (limit - index)),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000.0,
            )
            for index in range(limit)
        ]


def _ready_long_signal() -> dict[str, object]:
    breakdown = {
        "DIRECTION": 12,
        "STRUCTURE": 25,
        "LOCATION": 15,
        "TRIGGER": 10,
        "PARTICIPATION": 5,
        "PRICE_PROGRESS": 8,
    }
    signal: dict[str, object] = {
        "prd_version": "7.5",
        "score_model_version": 8,
        "entry_pipeline_version": 7,
        "timestamp": NOW.isoformat(),
        "action": SignalAction.ENTRY_LONG.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "decision_action": SignalAction.ENTRY_LONG.value,
        "direction": PositionSide.LONG.value,
        "direction_confirmation_state": "TEMPORARY_CONFIRMED",
        "score": 75,
        "setup_score": 75,
        "score_before_entry_quality": 75,
        "gross_setup_score": 75,
        "score_breakdown": breakdown,
        "score_evidence_families": breakdown,
        "risk_penalties": {},
        "total_risk_penalty": 0,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
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
    _apply_entry_policy_v4_fields(signal)
    return signal


def _indicator(timestamp: datetime, close: float) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=timestamp,
        close=close,
        ema20=close - 1.0,
        ema50=close - 2.0,
        ema200=close - 3.0,
        ma100=close - 2.5,
        boll_mid=close - 1.0,
        boll_upper=close + 2.0,
        boll_lower=close - 4.0,
        rsi14=60.0,
        atr14=1.0,
        volume_sma20=1_000.0,
        volume_ratio=1.0,
        ema50_slope=0.1,
    )


def test_decision_audit_persists_and_same_decision_is_never_retried(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "paper-v75.json"
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        state_path=state_path,
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.0)
    signal = _ready_long_signal()
    _clear_execution_stage_signal_fields(signal)
    _clear_transient_auto_entry_blocks(signal)
    _refresh_entry_quality_for_live_price(signal)
    _update_entry_position_fields(signal)
    decision_id = _auto_entry_decision_id("TESTUSDT", signal)
    signal.update(
        {
            "decision_id": decision_id,
            "eligible_at_submit": True,
            "auto_open_attempted": True,
            "auto_open_succeeded": False,
            "open_attempt_count": 1,
            "decision_reason_ledger": (),
        }
    )
    engine.latest_signals["TESTUSDT"] = dict(signal)
    engine.account.latest_signals["TESTUSDT"] = dict(signal)

    assert engine._append_decision_audit_from_signal(
        "TESTUSDT", signal
    )
    assert not engine._append_decision_audit_from_signal(
        "TESTUSDT", signal
    )
    engine._save_state_unlocked()

    restored = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        state_path=state_path,
        clock=lambda: NOW,
    )
    attempt_events = [
        item
        for item in restored.account.decision_audit_log
        if item.get("decision_id") == decision_id
        and item.get("code") == "AUTO_OPEN_ATTEMPT_FAILED"
    ]
    assert len(attempt_events) == 1
    assert decision_id in restored._attempted_auto_decision_ids

    submissions = 0

    async def unexpected_submission(*args, **kwargs):
        del args, kwargs
        nonlocal submissions
        submissions += 1
        raise AssertionError("the same decision was submitted twice")

    monkeypatch.setattr(restored, "open_position", unexpected_submission)
    asyncio.run(restored._auto_trade_once())

    assert submissions == 0
    assert any(
        "当前决策已提交过开仓尝试" in str(reason)
        for reason in restored.latest_signals["TESTUSDT"].get(
            "auto_entry_blocks", ()
        )
    )
    assert len(
        [
            item
            for item in restored.account.decision_audit_log
            if item.get("decision_id") == decision_id
            and str(item.get("code") or "").startswith(
                "AUTO_OPEN_ATTEMPT_"
            )
        ]
    ) == 1


@pytest.mark.parametrize(
    ("zone", "targets", "expected_error"),
    [
        (
            {"low": 90.0, "high": 95.0},
            (105.0, 110.0),
            "FINAL_PRICE_OUTSIDE_ENTRY_ZONE",
        ),
        (
            {"low": 99.0, "high": 101.0},
            (100.5, 101.0),
            "below minimum 1.30R",
        ),
    ],
)
def test_rejected_final_execution_has_zero_account_side_effects(
    zone: dict[str, float],
    targets: tuple[float, float],
    expected_error: str,
) -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.0)
    before = (
        engine.account.wallet_balance,
        engine.account.fees_paid,
        dict(engine.account.positions),
        list(engine.account.fills),
    )

    with pytest.raises(ValueError, match=expected_error):
        asyncio.run(
            engine.open_position(
                "TESTUSDT",
                "LONG",
                margin_usdt=100.0,
                leverage=5,
                stop_loss=90.0,
                take_profit_1=targets[0],
                take_profit_2=targets[1],
                entry_context={
                    "plan_target_mode": (
                        PLAN_TARGET_MODE_BOUNDED_TARGETS
                    ),
                    "execution_plan_formula_version": 2,
                    "execution_entry_zone": zone,
                    "execution_target_weights": [0.5, 0.5],
                    "disaster_stop_price": 80.0,
                    "tick_size": 0.01,
                    "entry_fee_rate": 0.0004,
                    "execution_slippage_rate": 0.0002,
                },
            )
        )

    assert engine.account.wallet_balance == before[0]
    assert engine.account.fees_paid == before[1]
    assert engine.account.positions == before[2]
    assert engine.account.fills == before[3]


def test_open_space_survives_restart_and_waits_for_new_closed_h1_swing(
    tmp_path,
) -> None:
    state_path = tmp_path / "open-space.json"
    settings = AppSettings()
    engine = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        state_path=state_path,
        clock=lambda: NOW,
    )
    engine._remember_mark_price("TESTUSDT", 100.0)
    asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100.0,
            leverage=5,
            stop_loss=90.0,
            entry_context={
                "entry_source": "AUTO_STRATEGY",
                "score_model_version": 8,
                "plan_version": 4,
                "plan_target_mode": PLAN_TARGET_MODE_OPEN_SPACE,
                "execution_plan_formula_version": 2,
                "execution_entry_zone": {"low": 99.0, "high": 101.0},
                "disaster_stop_price": 80.0,
                "tick_size": 0.01,
                "entry_fee_rate": 0.0004,
                "execution_slippage_rate": 0.0002,
                "execution_market_state": "STRONG_UP",
                "reliable_historical_target_absent": True,
                "h1_structure": {
                    "direction": "LONG",
                    "state": "BREAKOUT_UP",
                    "support": 95.0,
                    "support_zone_low": 95.0,
                },
            },
        )
    )

    restored = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        state_path=state_path,
        clock=lambda: NOW,
    )
    position = restored.account.positions["TESTUSDT"]
    context = position.metadata["entry_context"]
    reference = float(context["open_space_reference_price"])
    breakeven = float(context["fee_adjusted_breakeven_price"])

    assert position.plan_target_mode == PLAN_TARGET_MODE_OPEN_SPACE
    assert position.take_profit_1 is None
    assert position.take_profit_2 is None
    assert context["open_space_protection_state"] == "INACTIVE"

    threshold_structure = {
        "h1_structure": {
            "direction": "LONG",
            "state": "BREAKOUT_UP",
            "support": breakeven + 1.0,
            "support_zone_low": breakeven + 1.0,
        }
    }
    assert _advance_open_space_position_plan(
        position,
        threshold_structure,
        _indicator(NOW + timedelta(hours=1), reference + 0.5),
    )
    assert context["open_space_protection_state"] == PROTECTION_ELIGIBLE
    assert "protected_structure_invalidation_price" not in context

    assert _advance_open_space_position_plan(
        position,
        threshold_structure,
        _indicator(NOW + timedelta(hours=2), reference + 1.0),
    )
    assert context["open_space_protection_state"] == PROTECTION_ELIGIBLE
    assert "protected_structure_invalidation_price" not in context

    later_structure = {
        "h1_structure": {
            "direction": "LONG",
            "state": "BREAKOUT_UP",
            "support": breakeven + 2.0,
            "support_zone_low": breakeven + 2.0,
        }
    }
    assert _advance_open_space_position_plan(
        position,
        later_structure,
        _indicator(NOW + timedelta(hours=3), reference + 1.5),
    )
    assert context["open_space_protection_state"] == PROTECTION_ACTIVE
    assert context["protected_structure_invalidation_price"] > breakeven

    restored._save_state_unlocked()
    second_restart = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        state_path=state_path,
        clock=lambda: NOW,
    )
    persisted = second_restart.account.positions["TESTUSDT"]
    assert persisted.plan_target_mode == PLAN_TARGET_MODE_OPEN_SPACE
    assert persisted.take_profit_1 is None
    assert persisted.take_profit_2 is None
    assert (
        persisted.metadata["entry_context"][
            "open_space_protection_state"
        ]
        == PROTECTION_ACTIVE
    )


def test_raw_h4_opposition_is_the_strategy_hard_veto() -> None:
    signal = _ready_long_signal()
    signal["h4_direction"] = "SHORT"

    _apply_entry_policy_v4_fields(signal)

    assert signal["candidate_action"] == SignalAction.ENTRY_LONG.value
    assert signal["decision_action"] == SignalAction.WATCH.value
    assert signal["action"] == SignalAction.WATCH.value
    assert signal["policy_block_codes"] == ("H4_DIRECTION_OPPOSED",)
    assert signal["policy_blocks"] == (
        "方向、结构与市场环境：已闭合4小时方向与候选方向明确相反",
    )
    assert signal["vetoes"] == ()


def test_signal_stage_clears_every_previous_exact_exit_plan_value() -> None:
    signal = _ready_long_signal()
    signal.update(
        {
            "preview_structure_stop": 97.0,
            "preview_structure_stop_basis": "stale-previous-scan",
            "planned_take_profit_1": 110.0,
            "planned_take_profit_2": 120.0,
            "net_plan_r": 2.5,
            "plan_target_mode": PLAN_TARGET_MODE_BOUNDED_TARGETS,
            "final_fill_price": 100.1,
            "trade_plan": {
                "invalidation_price": 97.0,
                "target_1": 110.0,
                "target_2": 120.0,
                "net_plan_r": 2.5,
            },
        }
    )

    _clear_execution_stage_signal_fields(signal)
    _apply_entry_policy_v4_fields(signal)

    for key in (
        "preview_structure_stop",
        "preview_structure_stop_basis",
        "planned_take_profit_1",
        "planned_take_profit_2",
        "net_plan_r",
        "plan_target_mode",
        "final_fill_price",
    ):
        assert signal.get(key) in {None, ""}
    trade_plan = signal["trade_plan"]
    assert trade_plan["invalidation_price"] is None
    assert trade_plan["target_1"] is None
    assert trade_plan["target_2"] is None
    assert trade_plan["net_plan_r"] is None


def test_paper_fixed_unit_daily_loss_and_drawdown_are_advisory_only() -> None:
    settings = AppSettings()
    settings.execution.paper_trading = True
    settings.risk.paper_fixed_unit_sizing = True
    settings.risk.daily_loss_limit = 0.01
    settings.risk.max_drawdown_circuit_breaker = 0.05
    engine = PaperTradingEngine(
        settings,
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        clock=lambda: NOW,
    )
    engine.running = True
    engine.auto_trade = True
    engine.account.risk_day_key = "2026-08-07"
    engine.account.risk_day_start_equity = 1_000.0
    engine.account.daily_loss_locked = True
    engine.account.risk_peak_equity = 1_000.0
    engine.account.drawdown_locked = True

    gate = engine._new_entry_gate_status(900.0, now=NOW)

    assert gate["allowed"] is True
    assert gate["blocked_codes"] == []
    assert set(gate["advisory_codes"]) == {
        "DAILY_LOSS_LIMIT",
        "MAX_DRAWDOWN",
    }


@pytest.mark.parametrize(
    ("code", "category_prefix"),
    [
        (
            "CANDIDATE_ACTION_NOT_ESTABLISHED",
            "方向、结构与市场环境：",
        ),
        (
            "CANDIDATE_ROUTE_NOT_ESTABLISHED",
            "方向、结构与市场环境：",
        ),
        ("PRICE_DATA_DISCONTINUOUS", "标的资格与行情数据："),
        ("PRICE_DATA_STALE", "标的资格与行情数据："),
        ("H4_DIRECTION_OPPOSED", "方向、结构与市场环境："),
        (
            "SEMANTIC_STRUCTURE_UNAVAILABLE",
            "方向、结构与市场环境：",
        ),
        ("SETUP_INVALID", "评分与入场："),
        ("AUTHORIZED_ENTRY_ZONE_UNAVAILABLE", "评分与入场："),
        ("CURRENT_PRICE_OUTSIDE_AUTHORIZED_ZONE", "评分与入场："),
        ("TOTAL_SCORE_INVALID", "评分与入场："),
        ("RESEARCH_SCORE_BELOW_MINIMUM", "评分与入场："),
    ],
)
def test_v7_policy_projection_is_complete_classified_and_chinese(
    code: str,
    category_prefix: str,
) -> None:
    text = _entry_policy_v7_reason_text(
        code,
        decision=SimpleNamespace(
            effective_score=64,
            required_score=65,
        ),
    )

    assert text.startswith(category_prefix)
    assert "其他风控条件未满足" not in text
    assert "未识别的决策原因" not in text
    assert code not in text
    assert re.search(r"[\u3400-\u9fff]", text)
    assert not re.search(r"[A-Z]{3,}(?:_[A-Z0-9]+)+", text)


def test_policy_projection_keeps_machine_codes_separate_without_ledger_duplication() -> None:
    signal = _ready_long_signal()
    signal["h4_direction"] = "SHORT"

    _apply_entry_policy_v4_fields(signal)

    assert signal["policy_block_codes"] == ("H4_DIRECTION_OPPOSED",)
    visible_reason = signal["policy_blocks"][0]
    original_ledger = signal["decision_reason_ledger"]

    _record_auto_entry_block(signal, visible_reason)

    assert signal["auto_entry_blocks"] == (visible_reason,)
    assert signal["decision_reason_ledger"] == original_ledger


def test_status_preserves_visible_execution_and_account_denials_with_exact_codes() -> None:
    signal: dict[str, object] = {
        "entry_pipeline_version": 7,
        "action": SignalAction.ENTRY_LONG.value,
        "score": 75,
        "policy_blocks": (),
        "policy_block_codes": (),
        "vetoes": (),
        "auto_entry_blocks": (),
        "decision_reason_ledger": (),
    }
    _record_auto_entry_block(
        signal,
        "交易计划：目标价格方向与候选交易方向不一致",
        reason_code="TARGET_DIRECTION_INVALID",
        reason_category="TRADE_PLAN",
        actual_value=0.92,
        threshold_value=1.30,
    )
    _record_auto_entry_block(
        signal,
        "weekly loss limit reached",
        reason_code="WEEKLY_LOSS_LIMIT",
        reason_category="ACCOUNT_ORDER",
        actual_value=-0.06,
        threshold_value=-0.05,
    )

    projected = _auto_entry_status_signal(
        "TESTUSDT",
        signal,
        auto_trade=True,
        has_position=False,
    )

    assert projected["auto_entry_blocks"] == (
        "交易计划：目标价格方向与候选交易方向不一致",
        "资金、仓位与交易限制：已达到明确启用的本周亏损上限",
    )
    ledger = {
        str(item["code"]): item
        for item in projected["decision_reason_ledger"]
    }
    assert ledger["TARGET_DIRECTION_INVALID"] == {
        "code": "TARGET_DIRECTION_INVALID",
        "category": "TRADE_PLAN",
        "detail": "交易计划：目标价格方向与候选交易方向不一致",
        "stage": "EXECUTION",
        "actual_value": 0.92,
        "threshold_value": 1.30,
    }
    assert ledger["WEEKLY_LOSS_LIMIT"] == {
        "code": "WEEKLY_LOSS_LIMIT",
        "category": "ACCOUNT_ORDER",
        "detail": "资金、仓位与交易限制：已达到明确启用的本周亏损上限",
        "stage": "EXECUTION",
        "actual_value": -0.06,
        "threshold_value": -0.05,
    }


def test_active_reason_transitions_to_resolved_when_ledger_context_remains() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        clock=lambda: NOW,
    )
    detail = "交易计划：目标方向无效"
    signal: dict[str, object] = {
        "decision_id": "AUTO-ACTIVE-RESOLVED",
        "timestamp": NOW.isoformat(),
        "auto_entry_blocks": (detail,),
        "policy_blocks": (),
        "policy_block_codes": (),
        "vetoes": (),
        "decision_reason_ledger": (
            {
                "code": "TARGET_DIRECTION_INVALID",
                "category": "TRADE_PLAN",
                "detail": detail,
                "stage": "EXECUTION",
            },
        ),
    }

    assert engine._append_decision_audit_from_signal("TESTUSDT", signal)
    signal["auto_entry_blocks"] = ()
    assert engine._append_decision_audit_from_signal("TESTUSDT", signal)
    assert not engine._append_decision_audit_from_signal(
        "TESTUSDT",
        signal,
    )

    events = [
        item
        for item in engine.account.decision_audit_log
        if item.get("code") == "TARGET_DIRECTION_INVALID"
    ]
    assert [item["status"] for item in events] == ["ACTIVE", "RESOLVED"]
    assert events[0]["is_blocking"] is True
    assert events[1]["is_blocking"] is False
    assert events[1]["previous_event_id"] == events[0]["event_id"]


def test_failed_attempt_is_informational_while_concrete_failure_is_active() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1_000.0,
        symbols=["TESTUSDT"],
        market_data=_FlatMarket(),
        clock=lambda: NOW,
    )
    detail = "系统与下单：自动开仓执行失败，账户状态未发生副作用"
    signal: dict[str, object] = {
        "decision_id": "AUTO-FAILED-WITH-CAUSE",
        "timestamp": NOW.isoformat(),
        "auto_open_attempted": True,
        "auto_open_succeeded": False,
        "auto_entry_blocks": (detail,),
        "policy_blocks": (),
        "policy_block_codes": (),
        "vetoes": (),
        "decision_reason_ledger": (
            {
                "code": "ORDER_SUBMISSION_FAILED",
                "category": "SYSTEM_EXECUTION",
                "detail": detail,
                "stage": "EXECUTION",
            },
        ),
    }

    assert engine._append_decision_audit_from_signal("TESTUSDT", signal)

    events = {
        str(item["code"]): item
        for item in engine.account.decision_audit_log
    }
    assert events["ORDER_SUBMISSION_FAILED"]["status"] == "ACTIVE"
    assert events["ORDER_SUBMISSION_FAILED"]["is_blocking"] is True
    assert events["AUTO_OPEN_ATTEMPT_FAILED"]["status"] == "OBSERVED"
    assert events["AUTO_OPEN_ATTEMPT_FAILED"]["is_blocking"] is False


def test_unknown_runtime_denial_is_chinese_and_keeps_stable_system_code() -> None:
    signal: dict[str, object] = {
        "entry_pipeline_version": 7,
        "auto_entry_blocks": (),
        "decision_reason_ledger": (),
    }

    _record_auto_entry_block(signal, "opaque broker adapter fault")

    assert signal["auto_entry_blocks"] == (
        "系统与下单：自动开仓前置审核返回未分类异常，已写入决策流水",
    )
    visible = signal["auto_entry_blocks"][0]
    assert re.search(r"[\u3400-\u9fff]", visible)
    assert not re.search(r"[A-Za-z]{3,}", visible)
    assert signal["decision_reason_ledger"] == (
        {
            "code": "SYSTEM_OR_EXECUTION_BLOCKED",
            "category": "SYSTEM_EXECUTION",
            "detail": visible,
            "stage": "EXECUTION",
        },
    )


@pytest.mark.parametrize(
    "raw_reason",
    [
        "BTC 4h extreme volatility; pause new altcoin entries",
        "ZECUSDT derivatives data is incomplete",
        "交易计划：执行计划未通过：OPEN_SPACE_STRONG_TREND_REQUIRED。",
        "交易计划：开放空间方向不一致：候选方向为LONG，行情状态为STRONG_DOWN。",
        "auto entry execution failed: 账户锁异常",
    ],
)
def test_known_runtime_projection_survives_legacy_veto_text_contract(
    raw_reason: str,
) -> None:
    visible = _visible_auto_entry_block(raw_reason)

    assert re.search(r"[\u3400-\u9fff]", visible)
    assert not re.search(r"[A-Za-z]{3,}", visible)
    assert "其他风控条件未满足" not in visible
    assert "未分类" not in visible
