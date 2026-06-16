from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from ai_trading.api import create_app
from ai_trading.config import AppSettings
from ai_trading.models import Candle, IndicatorSnapshot, PositionSide
from ai_trading.paper import (
    PaperTradingEngine,
    _adaptive_exits,
    _apply_multi_timeframe_context,
    _auto_signal_allowed,
    _confirmed_structure_exit_reason,
    _daily_bias_margin_factor,
    _daily_pnl_payload,
    _leverage_for_signal,
    _margin_for_signal,
    _ma_cluster_signal_adjustment,
    _pnl_history_payload,
    _protect_confirmed_breakout_position,
    _precision_stop_allowed,
    _refine_stop_with_ma_cluster,
    _refine_stop_with_precision,
    _refine_take_profit_with_ma_cluster,
    _pyramid_allowed,
    _rotation_candidate_allowed,
    _risk_exit_reason,
    _stop_exit_reason,
)


class FakeMarketData:
    def __init__(self) -> None:
        self.price = 100.0
        self.btc_4h_extreme = False

    async def klines(self, symbol: str, interval: str = "15m", *, limit: int = 500, start_time_ms=None, end_time_ms=None):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        high = self.price + 1
        low = self.price - 1
        if symbol == "BTCUSDT" and interval == "4h" and self.btc_4h_extreme:
            high = self.price * 1.10
            low = self.price * 0.90
        return [
            Candle(
                timestamp=start + timedelta(minutes=15 * idx),
                open=self.price,
                high=high,
                low=low,
                close=self.price,
                volume=1000,
            )
            for idx in range(limit)
        ]

    async def top_usdt_perpetuals(self, limit: int = 30):
        class Symbol:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT"] + [f"TEST{idx}USDT" for idx in range(limit)]
        return [Symbol(symbol) for symbol in symbols]


def indicator_snapshot(
    *,
    close: float = 100.0,
    atr: float = 1.0,
    volume_ratio: float = 1.5,
    ema20: float = 100.0,
    oi_change: float | None = None,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        close=close,
        ema20=ema20,
        ema50=99.0,
        ema200=98.0,
        ma100=98.0,
        boll_mid=100.0,
        boll_upper=102.0,
        boll_lower=98.0,
        rsi14=55.0,
        atr14=atr,
        volume_sma20=1000.0,
        volume_ratio=volume_ratio,
        ema50_slope=0.1,
        oi_change=oi_change,
    )


def test_paper_engine_manual_long_close_profit() -> None:
    market = FakeMarketData()
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=market)

    import asyncio

    asyncio.run(engine.open_position("BTCUSDT", "LONG", margin_usdt=100, leverage=5))
    open_fill = engine.status()["fills"][0]
    assert open_fill["closed_at"] is None
    assert open_fill["leverage"] == 5
    assert open_fill["margin_usdt"] == 100
    assert open_fill["stop_price"] > 0
    assert open_fill["take_profit_1"] > 0
    assert open_fill["take_profit_2"] > open_fill["take_profit_1"]

    market.price = 110.0
    engine.latest_prices["BTCUSDT"] = 110.0
    status = engine.status()
    assert status["unrealized_pnl"] == 50.0

    trade = asyncio.run(engine.close_position("BTCUSDT"))
    close_fill = engine.status()["fills"][-1]
    assert trade.pnl > 0
    assert close_fill["closed_at"] is not None
    assert close_fill["return_pct"] > 0
    assert not engine.account.positions
    assert engine.status()["equity"] > 1000


def test_paper_api_manual_order_with_fake_engine(tmp_path: Path) -> None:
    app = create_app(state_path=tmp_path / "paper_state.json")
    app.state.paper_engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    client = TestClient(app)

    opened = client.post(
        "/api/paper/order/open",
        json={"symbol": "ETHUSDT", "side": "SHORT", "margin_usdt": 50, "leverage": 5},
    )
    assert opened.status_code == 200
    assert opened.json()["positions"][0]["symbol"] == "ETHUSDT"

    closed = client.post("/api/paper/order/close", json={"symbol": "ETHUSDT"})
    assert closed.status_code == 200
    assert closed.json()["positions"] == []


def test_paper_engine_persists_positions_and_fills(tmp_path: Path) -> None:
    state_path = tmp_path / "paper_state.json"
    market = FakeMarketData()
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=market, state_path=state_path)

    asyncio.run(engine.open_position("BTCUSDT", "LONG", margin_usdt=100, leverage=5))
    assert state_path.exists()

    restored = PaperTradingEngine(AppSettings(), starting_balance=500, market_data=FakeMarketData(), state_path=state_path)
    status = restored.status()
    assert status["starting_balance"] == 1000
    assert status["positions"][0]["symbol"] == "BTCUSDT"
    assert status["fills"][0]["action"] == "OPEN"

    restored.latest_prices["BTCUSDT"] = 110.0
    asyncio.run(restored.close_position("BTCUSDT", reason="test close"))
    restored_again = PaperTradingEngine(AppSettings(), starting_balance=500, market_data=FakeMarketData(), state_path=state_path)
    assert restored_again.status()["positions"] == []
    assert restored_again.status()["fills"][-1]["action"] == "CLOSE"


def test_auto_leverage_scales_with_signal_score() -> None:
    assert _leverage_for_signal(95, 10, "ONE_WAY_UP") == 10
    assert _leverage_for_signal(85, 10, "TREND_LONG") == 7
    assert _leverage_for_signal(85, 10, "CHOP") == 5
    assert _leverage_for_signal(78, 10, "TREND_LONG") == 5
    assert _leverage_for_signal(75, 10) == 5
    assert _leverage_for_signal(95, 7, "ONE_WAY_DOWN") == 7


def test_adaptive_exits_expand_profit_targets_by_trend_state() -> None:
    chop_stop, _, chop_tp = _adaptive_exits(PositionSide.LONG, 100.0, 5, "CHOP", None)
    trend_stop, _, trend_tp = _adaptive_exits(PositionSide.LONG, 100.0, 5, "TREND_LONG", None)
    one_way_stop, _, one_way_tp = _adaptive_exits(PositionSide.LONG, 100.0, 5, "ONE_WAY_UP", None)

    assert chop_stop > trend_stop > one_way_stop
    assert chop_tp < trend_tp < one_way_tp


def test_adaptive_exits_do_not_pull_atr_stop_closer_than_percent_floor() -> None:
    indicator = IndicatorSnapshot(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        close=100.0,
        ema20=100.0,
        ema50=99.0,
        ema200=98.0,
        ma100=98.0,
        boll_mid=100.0,
        boll_upper=102.0,
        boll_lower=98.0,
        rsi14=55.0,
        atr14=0.3,
        volume_sma20=1000.0,
        volume_ratio=1.0,
        ema50_slope=0.1,
    )

    long_stop, _, _ = _adaptive_exits(PositionSide.LONG, 100.0, 5, "TREND_LONG", indicator)
    short_stop, _, _ = _adaptive_exits(PositionSide.SHORT, 100.0, 5, "TREND_SHORT", indicator)

    assert long_stop <= 97.6
    assert short_stop >= 102.4


def test_adaptive_exits_widen_for_high_volatility() -> None:
    normal = indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.2)
    high_vol = indicator_snapshot(close=100.0, atr=3.0, volume_ratio=1.2)

    normal_stop, _, normal_tp = _adaptive_exits(PositionSide.LONG, 100.0, 5, "TREND_LONG", normal)
    high_stop, _, high_tp = _adaptive_exits(PositionSide.LONG, 100.0, 5, "TREND_LONG", high_vol)

    assert high_stop < normal_stop
    assert high_tp > normal_tp


def test_pyramid_requires_profit_strong_trend_and_pullback_to_ema20() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 97.0
    position.metadata["initial_stop_distance"] = 3.0
    signal = {
        "score": 88,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "h4_oi": {"state": "DELEVERAGE_HOLD_LONG"},
    }
    indicator = indicator_snapshot(close=103.0, atr=1.0, volume_ratio=1.1, ema20=102.5)

    assert _pyramid_allowed(position, 103.0, signal, indicator)
    assert not _pyramid_allowed(position, 101.0, signal, indicator)
    assert not _pyramid_allowed(position, 103.0, {**signal, "risk_state": "LONG_CROWD"}, indicator)
    assert not _pyramid_allowed(position, 106.0, signal, indicator)
    assert not _pyramid_allowed(position, 103.0, {"score": 88, "trend_state": "ONE_WAY_UP", "risk_state": "NORMAL"}, indicator)


def test_pyramid_allows_resistance_grind_breakout_confirmation() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 97.0
    position.metadata["initial_stop_distance"] = 3.0
    signal = {
        "score": 90,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "h1_trigger": {"direction": "LONG", "state": "BREAKOUT"},
        "h4_oi": {"state": "REBUILD_BREAKOUT_LONG"},
        "reasons": ("market structure: resistance grind broke upward, shorts may be squeezed",),
    }

    assert _pyramid_allowed(position, 103.0, signal, indicator_snapshot(close=103.0, atr=1.0, volume_ratio=1.2, ema20=102.3))


def test_confirmed_breakout_moves_long_stop_to_structure_protection() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 96.0
    signal = {
        "trend_state": "ONE_WAY_UP",
        "h4_structure": {"state": "BREAKOUT_UP", "resistance": 103.0},
        "h1_trigger": {"direction": "LONG", "state": "BREAKOUT"},
        "reasons": ("market structure: resistance grind broke upward, shorts may be squeezed",),
    }

    _protect_confirmed_breakout_position(position, signal, indicator_snapshot(close=105.0, atr=1.0, ema20=104.0))

    assert position.stop_price >= 102.4
    assert position.metadata["breakout_protected"] is True


def test_confirmed_structure_exit_uses_1h_4h_body_break_not_m15_noise() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.quantity = 5.0

    wick_only_signal = {
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
    }
    failed_signal = {
        "h4_structure": {"state": "BREAKDOWN_DOWN"},
        "h1_trigger": {"direction": "SHORT", "state": "BREAKDOWN"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
    }

    assert _confirmed_structure_exit_reason(position, 99.0, wick_only_signal) is None
    assert _confirmed_structure_exit_reason(position, 97.0, failed_signal) == "stop loss: 1h/4h body closed below support or EMA/BOLL zone"
    assert _confirmed_structure_exit_reason(position, 102.0, failed_signal) == "take profit: 1h/4h body closed below support or EMA/BOLL zone"


def test_precision_stop_only_allowed_for_strong_m15_tactical_pullback() -> None:
    precision = {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 97.2}

    assert _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "NORMAL", 90, precision)
    assert not _precision_stop_allowed(PositionSide.LONG, "TREND_LONG", "NORMAL", 90, precision)
    assert not _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "LONG_CROWD", 90, precision)
    assert not _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "NORMAL", 80, precision)


def test_stop_exit_reason_explicitly_marks_stop_or_take_profit() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 100.4

    assert _stop_exit_reason(position, {}) == "take profit: floating profit trailing stop"

    position.stop_price = 96.0
    assert _stop_exit_reason(position, {"action": "ENTRY_SHORT", "trend_state": "ONE_WAY_DOWN"}) == "stop loss: signal structure failed"
    assert _stop_exit_reason(position, {"action": "WATCH", "trend_state": "TREND_LONG", "score": 80}) == "stop loss: ATR volatility hard stop"


def test_risk_exit_reason_is_specific_to_one_way_position_risk() -> None:
    assert _risk_exit_reason(PositionSide.LONG, "ONE_WAY_UP", "LONG_CROWD") == "risk exit: LONG_CROWD"
    assert _risk_exit_reason(PositionSide.SHORT, "ONE_WAY_DOWN", "SHORT_CROWD") == "risk exit: SHORT_CROWD"
    assert _risk_exit_reason(PositionSide.LONG, "ONE_WAY_UP", "OI_ABNORMAL") == "risk exit: OI_ABNORMAL"
    assert _risk_exit_reason(PositionSide.SHORT, "ONE_WAY_DOWN", "FUNDING_HOT") == "risk exit: FUNDING_HOT"
    assert _risk_exit_reason(PositionSide.LONG, "TREND_LONG", "LONG_CROWD") is None


def test_auto_top30_universe_refreshes_symbols() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())

    import asyncio

    assert engine.symbols == ["AUTO_TOP30"]
    asyncio.run(engine.refresh_universe_if_needed())

    assert len(engine.symbols) == 30
    assert "BTCUSDT" not in engine.symbols
    assert "ETHUSDT" not in engine.symbols
    assert "SOLUSDT" not in engine.symbols
    assert engine.symbols[0] == "TEST0USDT"


def test_auto_signal_score_tiers_and_margins() -> None:
    assert _auto_signal_allowed({"score": 85, "risk_state": "LONG_CROWD"})
    assert _auto_signal_allowed({"score": 78, "risk_state": "FUNDING_HOT"})
    assert _auto_signal_allowed({"score": 76, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({"score": 76, "risk_state": "LONG_CROWD"})
    assert not _auto_signal_allowed({"score": 74, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({"score": 90, "risk_state": "NORMAL", "vetoes": ("1h trigger opposes long entry",)})

    assert _margin_for_signal(90, 1000) == 280
    assert _margin_for_signal(80, 1000) == 230
    assert _margin_for_signal(76, 1000) == 180
    assert _margin_for_signal(90, 1000, 950, 5) == 190
    assert _margin_for_signal(76, 1000, 950, 5) == 180


def test_multi_timeframe_context_adjusts_score_veto_and_margin() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 80,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BEAR",
        "h4_structure": {"state": "BREAKOUT_UP"},
        "h1_trigger": {"direction": "SHORT", "state": "FAKE_BREAKOUT"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["score"] == 64
    assert "1h trigger opposes long entry" in adjusted["vetoes"]
    assert "high area without pullback confirmation; wait for 1h/4h pullback before long" in adjusted["vetoes"]
    assert _daily_bias_margin_factor("LONG", adjusted) == 0.5


def test_multi_timeframe_healthy_pullback_adds_score() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 80,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["score"] == 103
    assert "1h BOLL/EMA pullback held with clean risk" in adjusted["reasons"]
    assert not adjusted["vetoes"]


def test_high_area_long_waits_for_pullback_confirmation() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 90,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "high area without pullback confirmation; wait for 1h/4h pullback before long" in adjusted["vetoes"]
    assert not _auto_signal_allowed(adjusted)


def test_high_area_long_allows_retest_confirmation() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 90,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "LONG", "state": "RETEST"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "high area without pullback confirmation; wait for 1h/4h pullback before long" not in adjusted["vetoes"]
    assert _auto_signal_allowed(adjusted)


def test_one_way_high_area_allows_15m_boll_ema9_pullback() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 90,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "m15_precision": {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 97.2},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "high area without pullback confirmation; wait for 1h/4h pullback before long" not in adjusted["vetoes"]
    assert "one-way uptrend 15m BOLL/EMA9 pullback confirmed; allow tactical long" in adjusted["reasons"]
    assert adjusted["m15_precision"]["long_stop_anchor"] == 97.2
    assert _auto_signal_allowed(adjusted)


def test_ma_cluster_breakout_scores_and_dense_waits() -> None:
    score, reasons, vetoes = _ma_cluster_signal_adjustment(
        PositionSide.LONG,
        {"state": "DENSE", "price": 1.30},
        {"state": "BREAKOUT_UP", "price": 1.31},
    )

    assert score == 12
    assert "MA cluster breakout up: price=1.31" in reasons
    assert not vetoes

    score, reasons, vetoes = _ma_cluster_signal_adjustment(
        PositionSide.LONG,
        {"state": "DENSE", "price": 1.30},
        {"state": "WAIT", "price": 1.29},
    )

    assert score == 0
    assert not reasons
    assert "MA cluster dense; wait for breakout or MA20 retest: price=1.29" in vetoes


def test_ma_cluster_refines_stop_and_take_profit() -> None:
    context = {
        "h4_structure": {"resistance": 112.0},
        "h1_ma_cluster": {
            "state": "BREAKOUT_UP",
            "price": 100.0,
            "lower": 99.5,
            "upper": 100.5,
            "ema20": 100.1,
            "target_up": 110.0,
        },
    }

    stop = _refine_stop_with_ma_cluster(PositionSide.LONG, 97.0, context)
    tp1, tp2 = _refine_take_profit_with_ma_cluster(PositionSide.LONG, 102.0, stop, 116.0, 122.0, context)

    assert stop < 99.5
    assert tp1 == 110.0
    assert tp2 == 122.0


def test_multi_timeframe_high_pullback_with_distribution_risk_vetoes_long() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 95,
        "risk_state": "OI_ABNORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "LONG", "state": "HIGH_PULLBACK"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "high pullback with OI/funding/crowd risk; avoid long entry" in adjusted["vetoes"]
    assert not _auto_signal_allowed(adjusted)


def test_oi_deleverage_hold_long_caps_leverage_and_margin() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 95,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h4_oi": {"state": "DELEVERAGE_HOLD_LONG"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["leverage_cap"] == 5
    assert adjusted["margin_factor"] == 0.5
    assert _daily_bias_margin_factor("LONG", adjusted) == 0.5
    assert min(_leverage_for_signal(adjusted["score"], 10, "ONE_WAY_UP"), adjusted["leverage_cap"]) == 5


def test_oi_deleverage_with_long_short_ratio_rising_allows_tiny_long_only() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 95,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h4_oi": {"state": "DELEVERAGE_CROWD_HOLD_LONG"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["leverage_cap"] == 3
    assert adjusted["margin_factor"] == 0.3
    assert "4h OI deleveraged while long/short ratio rose; 1h support held, only tiny long allowed" in adjusted["reasons"]


def test_oi_deleverage_breakdown_vetoes_long_and_waits_for_short_retest() -> None:
    long_signal = {
        "action": "ENTRY_LONG",
        "score": 95,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    short_signal = {
        "action": "ENTRY_SHORT",
        "score": 80,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BEAR",
        "h4_structure": {"state": "BREAKDOWN_DOWN"},
        "h4_oi": {"state": "DELEVERAGE_CROWD_BREAKDOWN"},
        "h1_trigger": {"direction": "SHORT", "state": "BREAKDOWN"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "summary": "MTF: test",
    }

    adjusted_long = _apply_multi_timeframe_context(long_signal, context)
    adjusted_short = _apply_multi_timeframe_context(short_signal, context)

    assert "4h OI deleverage with price breakdown; avoid long entry" in adjusted_long["vetoes"]
    assert not _auto_signal_allowed(adjusted_long)
    assert "4h OI deleverage breakdown; wait for resistance retest or upper-wick rejection before short" in adjusted_short["vetoes"]
    assert not _auto_signal_allowed(adjusted_short)


def test_oi_deleverage_breakdown_improves_short_after_failed_bounce() -> None:
    short_signal = {
        "action": "ENTRY_SHORT",
        "score": 80,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BEAR",
        "h4_structure": {"state": "BREAKDOWN_DOWN"},
        "h4_oi": {"state": "DELEVERAGE_CROWD_BREAKDOWN"},
        "h1_trigger": {"direction": "SHORT", "state": "RETEST"},
        "h1_pullback": {"direction": "SHORT", "state": "HEALTHY_PULLBACK"},
        "summary": "MTF: test",
    }

    adjusted_short = _apply_multi_timeframe_context(short_signal, context)

    assert "4h OI deleverage breakdown with failed bounce; short candidate improved" in adjusted_short["reasons"]
    assert adjusted_short["score"] > short_signal["score"]


def test_fifteen_minute_precision_refines_stop_outside_anchor() -> None:
    assert _refine_stop_with_precision(PositionSide.LONG, 98.0, {"long_stop_anchor": 97.2}) == 97.2
    assert _refine_stop_with_precision(PositionSide.SHORT, 102.0, {"short_stop_anchor": 103.1}) == 103.1


def test_rotation_candidate_requires_one_way_volatility_and_clean_risk() -> None:
    good = {"action": "ENTRY_LONG", "score": 92, "trend_state": "ONE_WAY_UP", "risk_state": "NORMAL"}

    assert _rotation_candidate_allowed(good, indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5))
    assert _rotation_candidate_allowed({**good, "score": 89}, indicator_snapshot())
    assert not _rotation_candidate_allowed({**good, "trend_state": "TREND_LONG"}, indicator_snapshot())
    assert not _rotation_candidate_allowed({**good, "risk_state": "LONG_CROWD"}, indicator_snapshot())
    assert not _rotation_candidate_allowed(good, indicator_snapshot(close=100.0, atr=0.5, volume_ratio=1.5))
    assert not _rotation_candidate_allowed(good, indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.0))


def test_auto_trade_caps_positions_and_prefers_highest_scores() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_prices = {f"TEST{idx}USDT": 100.0 for idx in range(6)}
    for idx, score in enumerate([75, 92, 81, 88, 79, 95]):
        engine.latest_signals[f"TEST{idx}USDT"] = {
            "action": "ENTRY_LONG",
            "score": score,
            "trend_state": "TREND_LONG",
        }

    import asyncio

    asyncio.run(engine._auto_trade_once())

    assert len(engine.account.positions) == 5
    assert "TEST0USDT" not in engine.account.positions
    assert set(engine.account.positions) == {"TEST1USDT", "TEST2USDT", "TEST3USDT", "TEST4USDT", "TEST5USDT"}
    used_margin = sum(float(position.metadata["margin_usdt"]) for position in engine.account.positions.values())
    assert 760 <= used_margin <= 790
    assert engine.status()["available_balance"] >= 200
    assert engine.account.positions["TEST5USDT"].metadata["margin_usdt"] >= 180


def test_auto_trade_pauses_altcoin_entries_when_btc_4h_is_extreme() -> None:
    market = FakeMarketData()
    market.btc_4h_extreme = True
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=market)
    engine.latest_prices = {"TESTUSDT": 100.0}
    engine.latest_signals["TESTUSDT"] = {"action": "ENTRY_LONG", "score": 95, "trend_state": "ONE_WAY_UP"}

    import asyncio

    asyncio.run(engine._auto_trade_once())

    assert not engine.account.positions


def test_auto_trade_rotates_weak_position_for_much_stronger_candidate() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_prices = {f"TEST{idx}USDT": 100.0 for idx in range(6)}

    import asyncio

    for idx in range(5):
        asyncio.run(engine.open_position(f"TEST{idx}USDT", "LONG", margin_usdt=100, leverage=5))
        engine.account.positions[f"TEST{idx}USDT"].opened_at = datetime.now(UTC) - timedelta(hours=1)
        engine.latest_signals[f"TEST{idx}USDT"] = {
            "score": 75 + idx,
            "action": "ENTRY_SHORT" if idx == 0 else "WATCH",
            "trend_state": "ONE_WAY_DOWN" if idx == 0 else "TREND_LONG",
            "risk_state": "NORMAL",
        }
        engine.latest_indicators[f"TEST{idx}USDT"] = [indicator_snapshot(close=100.0, atr=0.6, volume_ratio=1.0, oi_change=0.0)]
    engine.latest_signals["TEST5USDT"] = {
        "action": "ENTRY_LONG",
        "score": 95,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
    }
    engine.latest_indicators["TEST5USDT"] = [indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5)]

    asyncio.run(engine._auto_trade_once())

    assert "TEST0USDT" not in engine.account.positions
    assert "TEST5USDT" in engine.account.positions
    assert len(engine.account.positions) == 5
    assert engine.account.fills[-2].reason == "rotation exit: symbol=TEST5USDT score=95 current_score=75"


def test_auto_trade_does_not_rotate_for_ordinary_high_score_candidate() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_prices = {f"TEST{idx}USDT": 100.0 for idx in range(6)}

    import asyncio

    for idx in range(5):
        asyncio.run(engine.open_position(f"TEST{idx}USDT", "LONG", margin_usdt=100, leverage=5))
        engine.account.positions[f"TEST{idx}USDT"].opened_at = datetime.now(UTC) - timedelta(hours=1)
        engine.latest_signals[f"TEST{idx}USDT"] = {"score": 75 + idx}
    engine.latest_signals["TEST5USDT"] = {
        "action": "ENTRY_LONG",
        "score": 95,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
    }
    engine.latest_indicators["TEST5USDT"] = [indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5)]

    asyncio.run(engine._auto_trade_once())

    assert "TEST5USDT" not in engine.account.positions
    assert set(engine.account.positions) == {f"TEST{idx}USDT" for idx in range(5)}


def test_daily_pnl_uses_8am_total_pnl_snapshot_delta() -> None:
    baselines: dict[str, float] = {}

    intraday = _daily_pnl_payload(baselines, -2.24, now=datetime(2026, 6, 11, 3, 0, tzinfo=UTC))
    assert intraday["today"] == "2026-06-11"
    assert intraday["days"][0]["date"] == "2026-06-11"
    assert intraday["days"][0]["net_pnl"] == 0.0

    later = _daily_pnl_payload(baselines, 12.5, now=datetime(2026, 6, 11, 12, 0, tzinfo=UTC))
    assert later["days"][0]["net_pnl"] == 0.0

    next_day = _daily_pnl_payload(baselines, 18.0, now=datetime(2026, 6, 12, 1, 0, tzinfo=UTC))
    assert next_day["today"] == "2026-06-12"
    assert next_day["days"][0]["date"] == "2026-06-11"
    assert round(float(next_day["days"][0]["net_pnl"]), 2) == 20.24
    assert next_day["days"][1]["date"] == "2026-06-12"
    assert next_day["days"][1]["net_pnl"] == 0.0


def test_pnl_history_keeps_first_value_per_15_minute_bucket() -> None:
    samples: dict[str, float] = {}

    first = _pnl_history_payload(samples, -1.2, now=datetime(2026, 6, 11, 1, 2, tzinfo=UTC))
    second = _pnl_history_payload(samples, 8.8, now=datetime(2026, 6, 11, 1, 14, tzinfo=UTC))
    third = _pnl_history_payload(samples, 3.5, now=datetime(2026, 6, 11, 1, 15, tzinfo=UTC))

    assert first == [{"timestamp": "2026-06-11T01:00:00+00:00", "total_pnl": -1.2}]
    assert second == first
    assert third == [
        {"timestamp": "2026-06-11T01:00:00+00:00", "total_pnl": -1.2},
        {"timestamp": "2026-06-11T01:15:00+00:00", "total_pnl": 3.5},
    ]


def test_background_pnl_sample_records_without_status_request() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, symbols=["BTCUSDT"], market_data=FakeMarketData())
    engine.account.wallet_balance = 1037.01

    engine._record_pnl_history_sample(now=datetime(2026, 6, 12, 1, 17, tzinfo=UTC))

    assert round(engine.account.pnl_history["2026-06-12T01:15:00+00:00"], 2) == 37.01


def test_background_daily_pnl_baseline_records_without_status_request() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, symbols=["BTCUSDT"], market_data=FakeMarketData())
    engine.account.wallet_balance = 990.0
    engine._record_account_snapshots(now=datetime(2026, 6, 11, 0, 1, tzinfo=UTC))

    engine.account.wallet_balance = 1035.0
    engine._record_account_snapshots(now=datetime(2026, 6, 12, 0, 1, tzinfo=UTC))

    daily = _daily_pnl_payload(engine.account.daily_pnl_baselines, 35.0, now=datetime(2026, 6, 12, 1, 0, tzinfo=UTC))
    assert daily["days"][0]["date"] == "2026-06-11"
    assert round(float(daily["days"][0]["net_pnl"]), 2) == 45.0
