from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.market_context import (
    LiquidityHistory,
    LiquidityObservation,
    MarketContextTracker,
    StateMemory,
    crowding_candidate,
    direction_candidate,
    liquidity_candidate,
    system_risk_candidate,
)
from ai_trading.models import Candle, IndicatorSnapshot


def test_state_memory_requires_distinct_observations_for_normal_transition() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    memory = StateMemory()

    assert memory.update("TREND_LONG", start) == "UNKNOWN"
    assert memory.confirm_count == 1
    assert memory.update("TREND_LONG", start) == "UNKNOWN"
    assert memory.confirm_count == 1
    assert (
        memory.update("TREND_LONG", start + timedelta(hours=4))
        == "TREND_LONG"
    )


def test_risk_state_enters_immediately_and_recovers_after_two_checks() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    memory = StateMemory(current="NORMAL")

    assert (
        memory.update(
            "STRESS",
            start,
            immediate_states=frozenset({"STRESS", "UNKNOWN"}),
        )
        == "STRESS"
    )
    assert (
        memory.update(
            "NORMAL",
            start + timedelta(minutes=1),
            immediate_states=frozenset({"STRESS", "UNKNOWN"}),
        )
        == "STRESS"
    )
    assert (
        memory.update(
            "NORMAL",
            start + timedelta(minutes=2),
            immediate_states=frozenset({"STRESS", "UNKNOWN"}),
        )
        == "NORMAL"
    )


def test_direction_uses_relative_strength_and_separate_h4_direction() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    indicators = [
        _indicator(
            start + timedelta(hours=4 * index),
            ema_gap=0.10 + index * 0.03,
        )
        for index in range(21)
    ]
    indicators.append(
        _indicator(
            start + timedelta(hours=4 * 21),
            ema_gap=2.0,
        )
    )
    context = {
        "h4_structure": {
            "direction": "LONG",
            "state": "BREAKOUT_UP",
        }
    }

    assert (
        direction_candidate(
            indicators,
            context,
            current_state="UNKNOWN",
        )
        == "TREND_LONG"
    )


def test_crowding_is_independent_from_direction() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    indicators = [
        _indicator(
            start + timedelta(hours=index),
            long_short_ratio=1.0 + index * 0.001,
            funding_rate=0.0001,
        )
        for index in range(21)
    ]
    indicators.append(
        _indicator(
            start + timedelta(hours=21),
            long_short_ratio=2.0,
            funding_rate=0.0001,
        )
    )

    assert (
        crowding_candidate(indicators, current_state="NORMAL")
        == "LONG_CROWDED"
    )


def test_liquidity_uses_history_and_blocks_spread_shock() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    history = LiquidityHistory(
        volumes=[100_000_000.0] * 10,
        spreads_bps=[2.0] * 10,
        last_observed_at=start - timedelta(minutes=1),
    )
    normal = LiquidityObservation(
        symbol="TESTUSDT",
        quote_volume=100_000_000.0,
        best_bid=99.99,
        best_ask=100.01,
        timestamp=start,
    )
    thin = LiquidityObservation(
        symbol="TESTUSDT",
        quote_volume=100_000_000.0,
        best_bid=99.8,
        best_ask=100.2,
        timestamp=start + timedelta(minutes=1),
    )

    assert (
        liquidity_candidate(
            normal,
            history,
            current_state="NORMAL",
            observed_at=start,
        )
        == "NORMAL"
    )
    assert (
        liquidity_candidate(
            thin,
            history,
            current_state="NORMAL",
            observed_at=start + timedelta(minutes=1),
        )
        == "THIN"
    )


def test_liquidity_rejects_non_finite_market_data() -> None:
    observation = LiquidityObservation(
        symbol="TESTUSDT",
        quote_volume=float("nan"),
        best_bid=99.99,
        best_ask=100.01,
        timestamp=datetime(2026, 7, 1, tzinfo=UTC),
    )

    assert observation.spread_bps is None


def test_stale_liquidity_snapshot_immediately_changes_state_to_unknown() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    tracker = MarketContextTracker()
    tracker.states["TESTUSDT"] = {
        "liquidity_state": StateMemory(
            current="NORMAL",
            last_observed_at=start,
        )
    }
    tracker.liquidity_histories["TESTUSDT"] = LiquidityHistory(
        volumes=[100_000_000.0] * 10,
        spreads_bps=[2.0] * 10,
        last_observed_at=start,
    )
    stale = LiquidityObservation(
        symbol="TESTUSDT",
        quote_volume=100_000_000.0,
        best_bid=99.99,
        best_ask=100.01,
        timestamp=start,
    )

    context = tracker.update_symbol(
        "TESTUSDT",
        h4_indicators=[],
        crowding_indicators=[],
        h4_context=None,
        liquidity=stale,
        observed_at=start + timedelta(minutes=3),
    )

    assert context["liquidity_state"] == "UNKNOWN"


def test_system_risk_detects_atr_relative_btc_shock() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + timedelta(hours=4 * index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        )
        for index in range(22)
    ]
    candles[-1] = Candle(
        timestamp=candles[-1].timestamp,
        open=100.0,
        high=108.0,
        low=92.0,
        close=94.0,
        volume=4000.0,
    )

    assert (
        system_risk_candidate(
            candles,
            {},
            current_state="NORMAL",
        )
        == "STRESS"
    )


def test_system_risk_uses_exit_threshold_during_pool_recovery() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    btc_candles = _range_candles(start, latest_range=2.0)
    pool_candles = {
        f"ALT{index}USDT": _range_candles(start, latest_range=5.0)
        for index in range(5)
    }

    assert (
        system_risk_candidate(
            btc_candles,
            pool_candles,
            current_state="STRESS",
        )
        == "STRESS"
    )


def test_system_risk_is_unknown_when_pool_sample_is_insufficient() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    assert (
        system_risk_candidate(
            _range_candles(start, latest_range=2.0),
            {},
            current_state="NORMAL",
        )
        == "UNKNOWN"
    )


def test_tracker_state_round_trips_for_restart_hysteresis() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    tracker = MarketContextTracker()
    memory = tracker.states.setdefault("TESTUSDT", {}).setdefault(
        "direction_state",
        StateMemory(),
    )
    memory.update("TREND_LONG", start)

    restored = MarketContextTracker.from_payload(tracker.to_payload())
    restored_memory = restored.states["TESTUSDT"]["direction_state"]

    assert restored_memory.current == "UNKNOWN"
    assert restored_memory.pending == "TREND_LONG"
    assert restored_memory.confirm_count == 1
    assert restored_memory.last_observed_at == start


def _range_candles(
    start: datetime,
    *,
    latest_range: float,
) -> list[Candle]:
    candles = [
        Candle(
            timestamp=start + timedelta(hours=4 * index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        )
        for index in range(22)
    ]
    half_range = latest_range / 2
    candles[-1] = Candle(
        timestamp=candles[-1].timestamp,
        open=100.0,
        high=100.0 + half_range,
        low=100.0 - half_range,
        close=100.0,
        volume=1000.0,
    )
    return candles


def _indicator(
    timestamp: datetime,
    *,
    ema_gap: float = 1.0,
    long_short_ratio: float | None = 1.0,
    funding_rate: float | None = 0.0001,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=timestamp,
        close=100.0 + ema_gap,
        ema20=100.0 + ema_gap,
        ema50=100.0,
        ema200=95.0,
        ma100=97.0,
        boll_mid=100.0,
        boll_upper=102.0,
        boll_lower=98.0,
        rsi14=60.0,
        atr14=1.0,
        volume_sma20=1000.0,
        volume_ratio=1.0,
        ema50_slope=0.1,
        vwap=100.0,
        kc_mid=100.0,
        kc_upper=102.0,
        kc_lower=98.0,
        long_short_ratio=long_short_ratio,
        funding_rate=funding_rate,
    )
