from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from ai_trading.api import create_app
from ai_trading.config import AppSettings
from ai_trading.models import (
    Candle,
    DerivativesSnapshot,
    IndicatorSnapshot,
    Position,
    PositionSide,
    SignalAction,
)
from ai_trading.risk import TradePlan
from ai_trading.paper import (
    AUTO_ENTRY_MIN_SCORE,
    DISTRIBUTION_STAGE_DESCENDING,
    DISTRIBUTION_STAGE_MARKDOWN,
    DISTRIBUTION_STAGE_RANGE,
    MARKET_PRICE_STALE_SECONDS,
    SETUP_H1_PULLBACK_LONG,
    SETUP_H4_DESCENDING_RESISTANCE_SHORT,
    SETUP_H4_PULLBACK_SHORT,
    SETUP_OI_VALLEY_REVERSAL_LONG,
    STATE_SAVE_SECONDS,
    PaperStateError,
    PaperTradingEngine,
    _clear_transient_auto_entry_blocks,
    _closed_candles,
    _adaptive_exits,
    _apply_multi_timeframe_context,
    _auto_entry_prerequisite_blocks,
    _auto_signal_allowed,
    _candidate_prefilter_allowed,
    _confirmed_structure_exit_reason,
    _direction_validation_exit_reason,
    _daily_pnl_payload,
    _entry_quality_grade,
    _distribution_short_stage,
    _distribution_short_structure,
    _distribution_projection_supported,
    _entry_reward_r,
    _ema20_ema60_band,
    _entry_signal_timeframe,
    _entry_timeframe_for_signal,
    _entry_stop_error,
    _exit_plan_error,
    _four_hour_oi_valley,
    _four_hour_oi_valley_long_setup,
    _four_hour_structure,
    _fixed_margin_for_quality,
    _high_distribution_handoff,
    _high_distribution_handoff_exit_reason,
    _leverage_for_signal,
    _leverage_for_entry_quality,
    _large_timeframe_wick_rejections,
    _merge_candles,
    _ma_cluster_signal_adjustment,
    _one_hour_ema_reliability,
    _paper_account_from_payload,
    _oi_valley_short_exit_reason,
    _oi_valley_long_invalidation_reason,
    _pnl_history_payload,
    _profit_drawdown_exit_reason,
    _required_entry_timeframes,
    _preferred_exit_indicator,
    _protect_confirmed_breakout_position,
    _precision_stop_allowed,
    _position_profit_management_enabled,
    _position_reached_initial_r,
    _refine_stop_with_ma_cluster,
    _refine_stop_with_precision,
    _refine_stop_with_distribution_stage,
    _refine_stop_with_entry_zone,
    _refine_stop_with_setup_structure,
    _refine_stop_with_retest_structure,
    _refine_take_profit_with_ma_cluster,
    _pyramid_allowed,
    _rotation_candidate_allowed,
    _risk_exit_reason,
    _scored_entry_levels,
    _signal_entry_timing,
    _soft_stop_close_exit_reason,
    _stop_exit_reason,
    _strong_trend_invalidated,
    _structure_take_profit_reason,
    _take_profits_for_final_stop,
    _tighten_short_support_stop,
    _update_position_excursions,
    _update_position_validation,
    _update_entry_position_fields,
)


def test_periodic_state_checkpoint_is_throttled_to_five_minutes() -> None:
    assert STATE_SAVE_SECONDS == 300.0


def test_legacy_paper_state_loads_with_safe_risk_defaults() -> None:
    account = _paper_account_from_payload(
        {
            "starting_balance": 1000.0,
            "wallet_balance": 950.0,
            "positions": {},
            "fills": [],
        }
    )

    assert account.risk_peak_equity == 0.0
    assert not account.daily_loss_locked
    assert not account.weekly_loss_locked
    assert not account.drawdown_locked
    assert account.consecutive_losses == 0
    assert account.cooldown_until is None


def test_account_risk_snapshot_latches_daily_loss_and_drawdown() -> None:
    settings = AppSettings()
    settings.risk.daily_loss_limit = 0.02
    settings.risk.max_drawdown_circuit_breaker = 0.20
    engine = PaperTradingEngine(
        settings,
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    now = datetime(2026, 7, 12, 4, 0, tzinfo=UTC)
    engine._account_risk_snapshot(engine.status(), now=now)
    engine.account.wallet_balance = 790.0

    snapshot = engine._account_risk_snapshot(engine.status(), now=now)

    assert snapshot.daily_loss_locked
    assert snapshot.drawdown_locked


def test_disabled_paper_loss_breakers_ignore_and_clear_persisted_locks() -> None:
    settings = AppSettings()
    assert settings.risk.max_drawdown_circuit_breaker == 0.0
    engine = PaperTradingEngine(
        settings,
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine.running = True
    engine.auto_trade = True
    engine.account.wallet_balance = 700.0
    engine.account.daily_loss_locked = True
    engine.account.weekly_loss_locked = True
    engine.account.risk_peak_equity = 1000.0
    engine.account.drawdown_locked = True
    engine.account.consecutive_losses = 3
    engine.account.cooldown_until = datetime.now(UTC) + timedelta(hours=6)

    status = engine.status()

    assert status["new_entries_allowed"]
    assert not status["risk"]["daily_loss_locked"]
    assert not status["risk"]["weekly_loss_locked"]
    assert not status["risk"]["drawdown_enabled"]
    assert not status["risk"]["drawdown_locked"]
    snapshot = engine._account_risk_snapshot(status)
    assert not snapshot.daily_loss_locked
    assert not snapshot.weekly_loss_locked
    assert not snapshot.drawdown_locked
    assert snapshot.consecutive_losses == 0
    assert snapshot.cooldown_until is None
    assert not engine.account.daily_loss_locked
    assert not engine.account.weekly_loss_locked
    assert not engine.account.drawdown_locked


def test_status_marks_new_entries_blocked_by_active_weekly_lock() -> None:
    settings = AppSettings()
    settings.risk.weekly_loss_limit = 0.15
    engine = PaperTradingEngine(
        settings,
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine.running = True
    engine.auto_trade = True
    engine._account_risk_snapshot(engine.status())
    engine.account.weekly_loss_locked = True

    status = engine.status()

    assert not status["new_entries_allowed"]
    assert "WEEKLY_LOSS_LIMIT" in status["new_entry_block_codes"]


def test_manual_risk_entry_rejects_margin_above_single_symbol_limit() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )

    with pytest.raises(ValueError, match="risk-gated maximum"):
        asyncio.run(
            engine.open_position_with_risk(
                "BTCUSDT",
                "LONG",
                margin_usdt=200,
                leverage=5,
            )
        )


def test_three_losing_position_lifecycles_start_cooldown() -> None:
    settings = AppSettings()
    settings.risk.max_consecutive_losses = 3
    settings.risk.cooldown_hours = 6
    engine = PaperTradingEngine(
        settings,
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    for index in range(3):
        symbol = f"LOSS{index}USDT"
        position = asyncio.run(
            engine.open_position(
                symbol,
                "LONG",
                margin_usdt=50,
                leverage=5,
                stop_loss=98.0,
            )
        )
        engine._close_position_unlocked(position, 98.0, "stop loss")

    assert engine.account.consecutive_losses == 3
    assert engine.account.cooldown_until is not None
    decision = engine.portfolio_risk.evaluate(
        TradePlan(
            symbol="NEXTUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
            leverage=5,
        ),
        engine._account_risk_snapshot(engine.status()),
    )
    assert not decision.allowed
    assert decision.blocked_code == "LOSS_COOLDOWN"


def test_replay_clock_and_execution_price_are_injectable() -> None:
    replay_time = datetime(2025, 5, 1, 8, 0, tzinfo=UTC)
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
        clock=lambda: replay_time,
        fill_price_resolver=lambda price, side, entering: (
            price * 1.001
            if side == PositionSide.LONG and entering
            else price * 0.999
            if side == PositionSide.LONG
            else price
        ),
    )
    position = asyncio.run(
        engine.open_position(
            "CLOCKUSDT",
            "LONG",
            margin_usdt=50,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
        )
    )

    assert position.opened_at == replay_time
    assert position.entry_price == pytest.approx(100.1)
    trade = engine._close_position_unlocked(position, 100.0, "replay close")
    assert trade.closed_at == replay_time
    assert trade.exit_price == pytest.approx(99.9)


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
                self.quote_volume = 100_000_000
                self.price_change_percent = 5.0
                self.last_price = 100.0
                self.high_price = 105.0
                self.low_price = 95.0
                self.open_price = 99.0

        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT"] + [f"TEST{idx}USDT" for idx in range(limit)]
        return [Symbol(symbol) for symbol in symbols]


def test_single_symbol_incomplete_derivatives_is_local_and_recovers() -> None:
    class PartialDerivativesMarket(FakeMarketData):
        def __init__(self) -> None:
            super().__init__()
            self.fail_zec = True

        async def derivatives_bundle(
            self,
            symbol,
            interval,
            candle_times,
            *,
            include_funding=True,
        ):
            timestamps = list(candle_times)
            if symbol == "ZECUSDT" and self.fail_zec:
                raise RuntimeError(
                    "ZECUSDT derivatives data is incomplete"
                )
            return [
                DerivativesSnapshot(
                    timestamp=timestamp,
                    open_interest=1_000_000.0,
                    long_short_ratio=1.0,
                    funding_rate=0.0 if include_funding else None,
                )
                for timestamp in timestamps
            ]

    market = PartialDerivativesMarket()
    engine = PaperTradingEngine(
        AppSettings(),
        symbols=["ZECUSDT", "REUSDT"],
        market_data=market,
    )

    asyncio.run(engine.refresh_once())

    assert engine.last_error is None
    assert set(engine.status()["symbol_data_warnings"]) == {"ZECUSDT"}
    assert "REUSDT" not in engine._symbol_data_warnings
    engine.running = True
    zec_signal = engine.status()["latest_signals"]["ZECUSDT"]
    assert zec_signal["data_warning"].startswith("衍生品数据不完整")
    assert any(
        str(reason).startswith("衍生品数据不完整")
        for reason in zec_signal["vetoes"]
    )

    market.fail_zec = False
    asyncio.run(engine.refresh_once())

    assert engine.last_error is None
    assert engine.status()["symbol_data_warnings"] == {}


def indicator_snapshot(
    *,
    close: float = 100.0,
    atr: float = 1.0,
    volume_ratio: float = 1.5,
    ema20: float = 100.0,
    oi_change: float | None = None,
    open_interest: float | None = None,
    long_short_ratio: float | None = None,
    timestamp: datetime | None = None,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=timestamp or datetime(2026, 1, 1, tzinfo=UTC),
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
        open_interest=open_interest,
        long_short_ratio=long_short_ratio,
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
    assert close_fill["entry_position"] == "手动开仓≈100"
    assert not engine.account.positions
    assert engine.status()["equity"] > 1000


def test_open_partial_and_close_share_one_trade_cycle_id() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
        )
    )

    engine._close_position_fraction_unlocked(
        position,
        102.0,
        0.35,
        "take profit: target 1 reached",
    )
    engine._close_position_unlocked(
        position,
        103.0,
        "take profit: target 2 reached",
    )

    cycle_ids = {fill.trade_cycle_id for fill in engine.account.fills}
    assert cycle_ids == {position.metadata["trade_cycle_id"]}
    assert [fill.action for fill in engine.account.fills] == [
        "OPEN",
        "PARTIAL_CLOSE",
        "CLOSE",
    ]
    assert engine.account.fills[-1].exit_category == "TAKE_PROFIT"


def test_take_profit_1_partially_closes_and_moves_stop_above_break_even() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
        )
    )
    engine.latest_prices["TESTUSDT"] = 102.0
    engine.latest_signals["TESTUSDT"] = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "vetoes": (),
    }

    engine._manage_open_positions()

    position = engine.account.positions["TESTUSDT"]
    assert position.first_tp_done
    assert position.remaining_fraction == pytest.approx(0.65)
    assert position.metadata["margin_usdt"] == pytest.approx(65.0)
    assert position.stop_price > position.entry_price


def test_planned_point_eight_r_partial_profit_does_not_wait_for_one_r() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    position = asyncio.run(
        engine.open_position(
            "TEST08RUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=95.0,
            take_profit_1=104.0,
            take_profit_2=106.0,
        )
    )
    engine.latest_prices[position.symbol] = 104.0
    engine.latest_signals[position.symbol] = {
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
    }

    engine._manage_open_positions()

    assert position.first_tp_done
    assert position.remaining_fraction < 1.0
    assert position.stop_price > position.entry_price
    assert engine.account.fills[-1].action == "PARTIAL_CLOSE"
    assert engine.account.fills[-1].reason == "take profit: target 1 reached"
    assert engine._reentry_block_reason(position.symbol) is None


@pytest.mark.parametrize(
    "exit_reason",
    (
        "stop loss: test",
        "take profit: profit drawdown after long crowd risk",
        "take profit: target 2 reached",
        "rotation: stronger setup",
    ),
)
def test_full_exit_requires_new_entry_timeframe_candle_before_reentry(
    exit_reason: str,
) -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    candle_time = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    engine._timeframe_candles["TESTUSDT"] = {
        "1h": [
            Candle(
                timestamp=candle_time,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000.0,
            )
        ]
    }
    asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
            entry_context={"stop_timeframe": "1h"},
        )
    )
    position = engine.account.positions["TESTUSDT"]

    engine._close_position_unlocked(position, 98.0, exit_reason)

    assert (
        engine._reentry_block_reason("TESTUSDT")
        == "waiting for new 1h closed candle after full exit"
    )
    engine._timeframe_candles["TESTUSDT"]["1h"].append(
        Candle(
            timestamp=candle_time + timedelta(hours=1),
            open=98.0,
            high=99.0,
            low=97.0,
            close=98.5,
            volume=1000.0,
        )
    )
    assert engine._reentry_block_reason("TESTUSDT") is None


def test_auto_reentry_requires_a_new_confirmed_structure_not_only_a_new_candle() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    candle_time = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    engine._timeframe_candles["TESTUSDT"] = {
        "1h": [
            Candle(
                timestamp=candle_time,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000.0,
            )
        ]
    }
    position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
            entry_context={
                "stop_timeframe": "1h",
                "h1_trigger": {"direction": "LONG", "state": "RETEST"},
            },
        )
    )
    engine.latest_signals["TESTUSDT"] = {
        "action": SignalAction.WATCH.value,
        "h4_structure": {"state": "BREAKDOWN_DOWN", "support": 98.0},
        "h1_trigger": {"direction": "SHORT", "state": "BREAKDOWN"},
    }
    engine._close_position_unlocked(position, 98.0, "stop loss: structure failed")
    engine._timeframe_candles["TESTUSDT"]["1h"].append(
        Candle(
            timestamp=candle_time + timedelta(hours=1),
            open=98.0,
            high=99.0,
            low=97.0,
            close=98.5,
            volume=1000.0,
        )
    )

    assert (
        engine._reentry_block_reason("TESTUSDT")
        == "waiting for a new confirmed structure after full exit"
    )

    engine.latest_signals["TESTUSDT"] = {
        "action": SignalAction.ENTRY_LONG.value,
        "h4_structure": {"state": "BOX_LOWER_HALF", "support": 97.0},
        "h1_trigger": {"direction": "LONG", "state": "RETEST"},
        "entry_levels": {
            "long": {
                "h1_support": {"low": 97.0, "high": 98.5, "price": 98.0},
            }
        },
    }

    assert engine._reentry_block_reason("TESTUSDT") is None


def test_paper_status_pnl_components_reconcile_to_total() -> None:
    market = FakeMarketData()
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=market)

    asyncio.run(engine.open_position("BTCUSDT", "LONG", margin_usdt=100, leverage=5))
    market.price = 110.0
    engine.latest_prices["BTCUSDT"] = 110.0
    asyncio.run(engine.close_position("BTCUSDT"))
    asyncio.run(engine.open_position("ETHUSDT", "SHORT", margin_usdt=50, leverage=5))
    engine.latest_prices["ETHUSDT"] = 105.0

    status = engine.status()
    components = status["realized_pnl"] + status["unrealized_pnl"] - status["fees_paid"]
    assert components == pytest.approx(status["total_pnl"])


def test_open_position_strategy_signal_only_shows_already_open_veto() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    asyncio.run(engine.open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    engine.latest_signals["TESTUSDT"] = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 70,
        "entry_timing": "WAIT",
        "entry_timing_reason": "high area without pullback confirmation",
        "vetoes": (
            "high area without pullback confirmation; wait for 1h/4h pullback before long",
            "final score 70 below auto-entry minimum 80",
            "current entry position is not excellent",
        ),
    }
    engine.running = True

    signal = engine.status()["latest_signals"]["TESTUSDT"]

    assert signal["vetoes"] == ("symbol already has an open position",)


def test_closed_fill_preserves_actual_auto_entry_position() -> None:
    market = FakeMarketData()
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=market)
    entry_context = {
        "entry_levels": {
            "long": {
                "h1_support": {"low": 99.0, "high": 101.0, "price": 100.0},
                "h1_boll_mid": {"low": 99.5, "high": 100.5, "price": 100.0},
            }
        }
    }

    import asyncio

    position = asyncio.run(
        engine.open_position(
            "BTCUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            reason="auto strategy score=96; state=TREND_LONG",
            entry_context=entry_context,
        )
    )
    assert position.metadata["entry_position"] == (
        "1H支撑回踩≈99-101；实际开仓≈100"
    )

    market.price = 105.0
    engine.latest_prices["BTCUSDT"] = 105.0
    asyncio.run(engine.close_position("BTCUSDT", reason="take profit: test"))
    close_fill = engine.status()["fills"][-1]

    assert close_fill["entry_position"] == position.metadata["entry_position"]


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
    assert status["fills"][0]["trade_cycle_id"]
    assert status["fills"][0]["validation_state"] == "UNVALIDATED"

    restored.latest_prices["BTCUSDT"] = 110.0
    asyncio.run(restored.close_position("BTCUSDT", reason="test close"))
    restored_again = PaperTradingEngine(AppSettings(), starting_balance=500, market_data=FakeMarketData(), state_path=state_path)
    assert restored_again.status()["positions"] == []
    assert restored_again.status()["fills"][-1]["action"] == "CLOSE"
    assert (
        restored_again.status()["fills"][0]["trade_cycle_id"]
        == restored_again.status()["fills"][-1]["trade_cycle_id"]
    )


def test_paper_engine_persists_last_mark_price_for_open_positions(tmp_path: Path) -> None:
    state_path = tmp_path / "paper_state.json"
    market = FakeMarketData()
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=market, state_path=state_path)

    asyncio.run(engine.open_position("BTCUSDT", "LONG", margin_usdt=100, leverage=5))
    market.price = 110.0
    assert asyncio.run(engine.refresh_open_position_prices()) == 1

    restored = PaperTradingEngine(AppSettings(), starting_balance=500, market_data=FakeMarketData(), state_path=state_path)
    status = restored.status()
    assert status["positions"][0]["mark_price"] == 110.0
    assert status["unrealized_pnl"] == 50.0


def test_stop_disables_entries_without_stopping_background_worker() -> None:
    async def scenario() -> None:
        engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
        refreshed = asyncio.Event()

        async def fake_refresh() -> None:
            refreshed.set()

        engine.refresh_once = fake_refresh  # type: ignore[method-assign]
        await engine.start(auto_trade=True, poll_seconds=5)
        await asyncio.wait_for(refreshed.wait(), timeout=1)
        worker = engine._task

        await engine.stop()

        assert engine.running is True
        assert engine.auto_trade is False
        assert worker is not None and not worker.done()

        await engine.shutdown()
        assert engine.running is False
        assert engine._task is None

    asyncio.run(scenario())


def test_auto_universe_configuration_keeps_resolved_pool() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    asyncio.run(engine.refresh_universe_if_needed())
    resolved = list(engine.symbols)

    engine.configure_symbols(["AUTO_TOP30"])

    assert engine.symbols == resolved
    assert engine._auto_universe is True


def test_live_entry_timing_rechecks_when_price_reaches_zone() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_signals["TESTUSDT"] = {
        "action": "ENTRY_LONG",
        "score": 90,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 110.0,
            "entry_levels": {
                "long": {
                    "h1_support": {"low": 99.0, "high": 101.0, "price": 100.0},
                }
            },
            "h1_structure": {"resistance": 105.0},
        }
    engine._remember_mark_price("TESTUSDT", 110.0)
    engine._refresh_live_entry_timing()
    assert engine.latest_signals["TESTUSDT"]["entry_timing"] == "WAIT"

    engine._remember_mark_price("TESTUSDT", 100.0)
    engine._refresh_live_entry_timing()

    assert engine.latest_signals["TESTUSDT"]["entry_timing"] == "GOOD"


def test_transient_entry_blocks_are_removed_but_strategy_veto_remains() -> None:
    signal = {
        "vetoes": (
            "latest price is stale for more than 15 seconds",
            "current funding rate data is stale for more than 15 minutes",
            "symbol excluded from automatic universe",
            "等待 1H支撑回踩区",
            "已进入建议区，但未处于优势侧",
            "暂无有效建议入场区",
            "funding rate too hot for long entry",
        ),
        "entry_timing": "BLOCK",
        "entry_timing_reason": "latest price is stale for more than 15 seconds",
    }

    _clear_transient_auto_entry_blocks(signal)

    assert signal["vetoes"] == ("funding rate too hot for long entry",)
    assert signal["entry_timing"] == "BLOCK"
    assert signal["entry_timing_reason"] == "latest price is stale for more than 15 seconds"


def test_data_freshness_only_requires_the_actual_entry_timeframe() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "price": 100.0,
        "signal_timeframe": "4h",
        "entry_levels": {
            "long": {
                "h4_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
    }
    now = datetime.now(UTC)
    engine._warmup_complete = True
    engine._price_updated_at["TESTUSDT"] = now
    engine._oi_ratio_updated_at["TESTUSDT"] = now

    assert _required_entry_timeframes(signal) == ("4h",)
    blocks = engine._data_freshness_blocks("TESTUSDT", signal)
    assert "4h K-line context is missing or discontinuous" in blocks
    assert "1h K-line context is missing or discontinuous" not in blocks


def test_excluded_symbol_is_not_added_as_an_auto_entry_veto() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine.latest_signals["BTCUSDT"] = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
    }

    vetoes = engine.status()["latest_signals"]["BTCUSDT"]["vetoes"]

    assert "symbol excluded from automatic universe" not in vetoes


def test_stale_price_is_recorded_as_auto_entry_veto_and_clears_after_refresh() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    symbol = "TESTUSDT"
    now = datetime.now(UTC)
    engine.running = True
    engine._warmup_complete = True
    engine._symbol_cache_valid = lambda _symbol: True
    engine.latest_prices[symbol] = 100.0
    engine._price_updated_at[symbol] = now - timedelta(
        seconds=MARKET_PRICE_STALE_SECONDS + 1
    )
    engine._oi_ratio_updated_at[symbol] = now
    engine.latest_signals[symbol] = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {"low": 99.0, "high": 101.0, "price": 100.0},
            }
        },
        "h1_structure": {"resistance": 105.0},
    }

    asyncio.run(engine._auto_trade_once())

    stale_reason = "latest price is stale for more than 15 seconds"
    assert symbol not in engine.account.positions
    assert stale_reason in engine.latest_signals[symbol]["vetoes"]
    assert stale_reason in engine.status()["latest_signals"][symbol]["vetoes"]

    engine._remember_mark_price(symbol, 100.0)
    asyncio.run(engine._auto_trade_once())

    assert symbol in engine.account.positions
    assert stale_reason not in engine.latest_signals[symbol]["vetoes"]


def test_status_surfaces_stale_derivatives_vetoes_when_auto_trade_is_off() -> None:
    class FundingAwareMarket(FakeMarketData):
        async def current_funding_rates(self, symbols):
            return {symbol: 0.0 for symbol in symbols}

    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FundingAwareMarket(),
    )
    now = datetime.now(UTC)
    engine.running = True
    engine.auto_trade = False
    engine._warmup_complete = True
    engine._symbol_cache_valid = lambda _symbol: True
    engine.latest_signals["TESTUSDT"] = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "vetoes": (),
        "reasons": (),
        "entry_timing": "GOOD",
    }
    engine._price_updated_at["TESTUSDT"] = now
    engine._oi_ratio_updated_at["TESTUSDT"] = now - timedelta(seconds=181)
    engine._funding_updated_at["TESTUSDT"] = now - timedelta(minutes=16)

    vetoes = engine.status()["latest_signals"]["TESTUSDT"]["vetoes"]

    assert "OI/long-short ratio data is stale for more than 180 seconds" in vetoes
    assert "current funding rate data is stale for more than 15 minutes" not in vetoes


def test_status_clears_persisted_stale_price_veto_after_price_refresh() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    now = datetime.now(UTC)
    engine.running = True
    engine._warmup_complete = True
    engine._symbol_cache_valid = lambda _symbol: True
    engine.latest_signals["TESTUSDT"] = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "vetoes": ("latest price is stale for more than 15 seconds",),
    }
    engine._price_updated_at["TESTUSDT"] = now
    engine._oi_ratio_updated_at["TESTUSDT"] = now

    vetoes = engine.status()["latest_signals"]["TESTUSDT"]["vetoes"]

    assert "latest price is stale for more than 15 seconds" not in vetoes


def test_closed_candle_cache_ignores_open_bar_and_merges_incrementally() -> None:
    now = datetime(2026, 1, 1, 1, 15, 6, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + minute,
            volume=1000.0,
        )
        for minute in (30, 45)
    ]
    candles.append(
        Candle(
            timestamp=datetime(2026, 1, 1, 1, 15, tzinfo=UTC),
            open=102.0,
            high=103.0,
            low=101.0,
            close=102.5,
            volume=500.0,
        )
    )

    closed = _closed_candles(candles, "15m", now)
    merged = _merge_candles(closed[:1], closed[1:], max_length=240)

    assert [candle.timestamp.minute for candle in closed] == [30, 45]
    assert merged == closed


@pytest.mark.parametrize(
    ("timeframe", "duration"),
    (("1h", timedelta(hours=1)), ("4h", timedelta(hours=4))),
)
def test_directional_timeframes_ignore_the_open_bar(
    timeframe: str,
    duration: timedelta,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    now = start + duration * 2 + timedelta(seconds=6)
    candles = [
        Candle(
            timestamp=start + duration * index,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + index,
            volume=1000.0,
        )
        for index in range(3)
    ]

    closed = _closed_candles(candles, timeframe, now)

    assert [candle.timestamp for candle in closed] == [
        start,
        start + duration,
    ]


def test_disabled_entries_still_manage_and_close_existing_position() -> None:
    async def scenario() -> None:
        engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
        await engine.open_position(
            "BTCUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=99.0,
            take_profit_1=110.0,
            take_profit_2=120.0,
        )

        async def fake_refresh() -> None:
            engine._remember_mark_price("BTCUSDT", 98.0)

        engine.refresh_once = fake_refresh  # type: ignore[method-assign]
        await engine.start(auto_trade=False, poll_seconds=5)

        for _ in range(20):
            if not engine.account.positions:
                break
            await asyncio.sleep(0.01)

        status = engine.status()
        assert engine.auto_trade is False
        assert engine.running is True
        assert status["positions"] == []
        assert status["fills"][-1]["action"] == "CLOSE"
        assert status["realized_pnl"] < 0
        assert status["fees_paid"] > 0

        await engine.shutdown()

    asyncio.run(scenario())


def test_position_safety_continues_while_market_scan_is_blocked() -> None:
    async def scenario() -> None:
        engine = PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=FakeMarketData(),
        )
        await engine.open_position(
            "BTCUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=99.0,
            take_profit_1=110.0,
            take_profit_2=120.0,
        )
        scan_started = asyncio.Event()

        async def blocked_refresh() -> None:
            scan_started.set()
            await asyncio.Event().wait()

        engine.refresh_once = blocked_refresh  # type: ignore[method-assign]
        await engine.start(auto_trade=False)
        await asyncio.wait_for(scan_started.wait(), timeout=1)

        engine._remember_mark_price("BTCUSDT", 98.0)
        for _ in range(50):
            if not engine.account.positions:
                break
            await asyncio.sleep(0.01)

        assert engine.status()["positions"] == []
        assert engine.status()["fills"][-1]["action"] == "CLOSE"
        await engine.shutdown()

    asyncio.run(scenario())


def test_rest_price_fallback_continues_while_market_scan_is_blocked() -> None:
    class MarkPriceMarket(FakeMarketData):
        async def mark_prices(self, symbols):
            return {symbol: 101.0 for symbol in symbols}

    async def scenario() -> None:
        engine = PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=MarkPriceMarket(),
        )
        engine.latest_signals["TESTUSDT"] = {
            "action": "ENTRY_LONG",
            "score": 90,
        }
        scan_started = asyncio.Event()

        async def blocked_refresh() -> None:
            scan_started.set()
            await asyncio.Event().wait()

        engine.refresh_once = blocked_refresh  # type: ignore[method-assign]
        await engine.start(auto_trade=False)
        await asyncio.wait_for(scan_started.wait(), timeout=1)
        for _ in range(50):
            if engine.latest_prices.get("TESTUSDT") == 101.0:
                break
            await asyncio.sleep(0.01)

        assert engine.latest_prices["TESTUSDT"] == 101.0
        assert engine.health_status()["price_fallback_alive"] is True
        await engine.shutdown()

    asyncio.run(scenario())


def test_rest_fallback_refreshes_only_stale_symbols() -> None:
    class MarkPriceMarket(FakeMarketData):
        def __init__(self) -> None:
            super().__init__()
            self.requested: list[str] = []

        async def mark_prices(self, symbols):
            self.requested = list(symbols)
            return {symbol: 101.0 for symbol in self.requested}

    async def scenario() -> None:
        market = MarkPriceMarket()
        engine = PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=market,
        )
        await engine.open_position(
            "FRESHUSDT",
            "LONG",
            margin_usdt=50,
            leverage=5,
        )
        await engine.open_position(
            "STALEUSDT",
            "LONG",
            margin_usdt=50,
            leverage=5,
        )
        now = datetime.now(UTC)
        engine._price_updated_at["FRESHUSDT"] = now
        engine._price_updated_at["STALEUSDT"] = now - timedelta(
            seconds=MARKET_PRICE_STALE_SECONDS + 1
        )

        await engine._refresh_rest_price_fallback(now)

        assert market.requested == ["STALEUSDT"]
        assert engine.latest_prices["STALEUSDT"] == 101.0

    asyncio.run(scenario())


def test_rest_fallback_refreshes_price_before_stale_veto_threshold() -> None:
    class MarkPriceMarket(FakeMarketData):
        def __init__(self) -> None:
            super().__init__()
            self.requested: list[str] = []

        async def mark_prices(self, symbols):
            self.requested = list(symbols)
            return {symbol: 101.0 for symbol in self.requested}

    async def scenario() -> None:
        market = MarkPriceMarket()
        engine = PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=market,
        )
        engine.latest_signals["TESTUSDT"] = {
            "action": "ENTRY_LONG",
            "score": 90,
        }
        now = datetime.now(UTC)
        engine._price_updated_at["TESTUSDT"] = now - timedelta(seconds=11)

        await engine._refresh_rest_price_fallback(now)

        assert market.requested == ["TESTUSDT"]
        assert engine.latest_prices["TESTUSDT"] == 101.0
        assert "latest price is stale for more than 15 seconds" not in engine._data_freshness_blocks("TESTUSDT")

    asyncio.run(scenario())


def test_historical_warmup_does_not_overwrite_live_mark_price() -> None:
    async def scenario() -> None:
        engine = PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=FakeMarketData(),
        )
        engine._remember_mark_price("TESTUSDT", 111.0)

        assert await engine._refresh_symbol("TESTUSDT") is True
        assert engine.latest_prices["TESTUSDT"] == 111.0

    asyncio.run(scenario())


def test_corrupt_state_recovers_from_backup(tmp_path: Path) -> None:
    state_path = tmp_path / "paper_state.json"
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
        state_path=state_path,
    )
    asyncio.run(
        engine.open_position(
            "BTCUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
        )
    )
    engine._save_state_unlocked()
    state_path.write_text("{broken", encoding="utf-8")

    restored = PaperTradingEngine(
        AppSettings(),
        starting_balance=500,
        market_data=FakeMarketData(),
        state_path=state_path,
    )

    assert restored.status()["starting_balance"] == 1000
    assert restored.status()["positions"][0]["symbol"] == "BTCUSDT"


def test_corrupt_state_without_backup_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "paper_state.json"
    state_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(PaperStateError, match="状态损坏"):
        PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=FakeMarketData(),
            state_path=state_path,
        )


def test_state_file_allows_only_one_running_engine(tmp_path: Path) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "paper_state.json"
        first = PaperTradingEngine(
            AppSettings(),
            market_data=FakeMarketData(),
            state_path=state_path,
        )
        second = PaperTradingEngine(
            AppSettings(),
            market_data=FakeMarketData(),
            state_path=state_path,
        )
        await first.start(auto_trade=False)
        try:
            with pytest.raises(PaperStateError, match="另一个交易进程"):
                await second.start(auto_trade=False)
        finally:
            await first.shutdown()

    asyncio.run(scenario())


def test_app_starts_position_management_with_new_entries_disabled(tmp_path: Path) -> None:
    app = create_app(state_path=tmp_path / "paper_state.json")
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    app.state.paper_engine = engine

    with TestClient(app) as client:
        initial = client.get("/api/paper/status").json()
        assert initial["running"] is True
        assert initial["auto_trade"] is False

        enabled = client.post("/api/paper/start", json={"auto_trade": True}).json()
        assert enabled["running"] is True
        assert enabled["auto_trade"] is True

        disabled = client.post("/api/paper/stop").json()
        assert disabled["running"] is True
        assert disabled["auto_trade"] is False

    assert engine.running is False


def test_auto_leverage_scales_with_signal_score() -> None:
    assert _leverage_for_signal(95, 10, "ONE_WAY_UP") == 10
    assert _leverage_for_signal(85, 10, "TREND_LONG") == 7
    assert _leverage_for_signal(85, 10, "CHOP") == 5
    assert _leverage_for_signal(78, 10, "TREND_LONG") == 5
    assert _leverage_for_signal(75, 10) == 5
    assert _leverage_for_signal(95, 7, "ONE_WAY_DOWN") == 7
    assert _leverage_for_signal(110, 10, "ONE_WAY_UP", trend_stage="LATE") == 5


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
        "score": 100,
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
    assert not _pyramid_allowed(position, 103.0, {"score": 99, "trend_state": "ONE_WAY_UP", "risk_state": "NORMAL"}, indicator)


def test_pyramid_allows_resistance_grind_breakout_confirmation() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 97.0
    position.metadata["initial_stop_distance"] = 3.0
    signal = {
        "score": 100,
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

    _protect_confirmed_breakout_position(position, 105.0, signal, indicator_snapshot(close=105.0, atr=1.0, ema20=104.0))

    assert position.stop_price >= 102.4
    assert position.metadata["breakout_protected"] is True


def test_breakout_protection_does_not_tighten_before_one_r_validation() -> None:
    position = asyncio.run(
        PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=FakeMarketData(),
        ).open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=99.0,
        )
    )
    position.entry_price = 100.0
    position.stop_price = 99.0
    position.metadata["initial_stop_distance"] = 1.0
    signal = {
        "h4_structure": {"state": "BREAKOUT_UP", "resistance": 100.5},
        "h1_trigger": {"direction": "LONG", "state": "BREAKOUT"},
    }

    _protect_confirmed_breakout_position(
        position,
        100.7,
        signal,
        indicator_snapshot(close=100.7, atr=0.5, ema20=100.4),
    )

    assert position.stop_price == 99.0
    assert not position.metadata.get("breakout_protected")


def test_soft_structure_stop_waits_for_new_candle_close_while_hard_stop_stays_fixed() -> None:
    opened = datetime(2026, 7, 20, 1, tzinfo=UTC)
    position = Position(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
        opened_at=opened,
        stop_price=98.0,
        take_profit_1=102.0,
        take_profit_2=104.0,
        metadata={
            "soft_stop_price": 99.0,
            "hard_stop_price": 98.0,
            "initial_stop_distance": 2.0,
        },
    )

    assert _soft_stop_close_exit_reason(
        position,
        indicator_snapshot(
            close=100.0,
            timestamp=opened + timedelta(hours=1),
        ),
    ) is None
    assert _soft_stop_close_exit_reason(
        position,
        indicator_snapshot(
            close=98.8,
            timestamp=opened + timedelta(hours=2),
        ),
    ) == "stop loss: structure close confirmed beyond the primary setup"
    assert position.stop_price == 98.0


def test_unvalidated_direction_exits_on_opposite_structure_but_validated_trade_does_not() -> None:
    position = Position(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
        opened_at=datetime(2026, 7, 20, 1, tzinfo=UTC),
        stop_price=98.0,
        take_profit_1=102.0,
        take_profit_2=104.0,
        metadata={
            "validation_state": "UNVALIDATED",
            "initial_stop_distance": 2.0,
            "max_favorable_distance": 0.2,
        },
    )
    opposite = {
        "action": SignalAction.ENTRY_SHORT.value,
        "trend_state": "ONE_WAY_DOWN",
        "h4_structure": {"state": "BREAKDOWN_DOWN"},
        "h1_trigger": {"direction": "SHORT", "state": "BREAKDOWN"},
    }

    reason = _direction_validation_exit_reason(position, 99.0, opposite)
    assert reason is not None and "direction unvalidated" in reason

    position.metadata["max_favorable_distance"] = 0.7
    _update_position_validation(position)

    assert position.metadata["validation_state"] == "VALIDATED"
    assert _direction_validation_exit_reason(position, 99.0, opposite) is None


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


def test_confirmed_structure_exit_does_not_close_short_on_healthy_bounce_only() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("BCHUSDT", "SHORT", margin_usdt=100, leverage=10))
    position.entry_price = 190.3
    position.stop_price = 196.6
    signal = {
        "h4_structure": {"state": "BOX_LOWER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
    }

    assert _confirmed_structure_exit_reason(position, 192.6, signal) is None


def test_four_hour_setup_ignores_one_hour_failure_until_four_hour_breaks() -> None:
    position = asyncio.run(
        PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=FakeMarketData(),
        ).open_position(
            "BNBUSDT",
            "SHORT",
            margin_usdt=100,
            leverage=5,
            entry_context={
                "setup_type": SETUP_H4_PULLBACK_SHORT,
                "stop_timeframe": "4h",
            },
        )
    )
    position.entry_price = 573.85
    position.stop_price = 587.05
    h1_only_failure = {
        "h4_structure": {"state": "BOX_LOWER_HALF"},
        "h1_trigger": {"direction": "LONG", "state": "BREAKOUT"},
    }
    h4_failure = {
        **h1_only_failure,
        "h4_structure": {"state": "BREAKOUT_UP"},
    }

    assert _confirmed_structure_exit_reason(
        position,
        575.5,
        h1_only_failure,
    ) is None
    assert _confirmed_structure_exit_reason(
        position,
        588.0,
        h4_failure,
    ) == "stop loss: 4h body closed above resistance or EMA/BOLL zone"


def test_four_hour_setup_is_not_closed_by_ema50_weakness_alone() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    h4_position = asyncio.run(
        engine.open_position(
            "TEST4HUSDT",
            "SHORT",
            margin_usdt=100,
            leverage=5,
            entry_context={
                "setup_type": SETUP_H4_PULLBACK_SHORT,
                "stop_timeframe": "4h",
            },
        )
    )
    h1_position = asyncio.run(
        engine.open_position(
            "TEST1HUSDT",
            "SHORT",
            margin_usdt=100,
            leverage=5,
            entry_context={"stop_timeframe": "1h"},
        )
    )
    invalidated = indicator_snapshot(
        close=103.0,
        atr=1.0,
    )

    assert not _strong_trend_invalidated(h4_position, invalidated)
    assert _strong_trend_invalidated(h1_position, invalidated)


def test_breakout_protection_waits_for_real_profit_before_tightening_stop() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("BCHUSDT", "SHORT", margin_usdt=100, leverage=10))
    position.entry_price = 190.3
    position.stop_price = 196.6
    position.metadata["initial_stop_distance"] = 6.3
    signal = {
        "h4_structure": {"state": "BREAKDOWN_DOWN", "support": 180.0},
        "h1_trigger": {"direction": "SHORT", "state": "BREAKDOWN"},
    }

    _protect_confirmed_breakout_position(position, 192.6, signal, indicator_snapshot(close=192.6, atr=1.0, ema20=193.0))

    assert position.stop_price == 196.6


def test_precision_stop_only_allowed_for_strong_m15_tactical_pullback() -> None:
    precision = {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 97.2, "trend": "UP"}
    squeeze = "SHORT_SQUEEZE_MARKUP"

    assert _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "NORMAL", 90, precision, squeeze)
    assert _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "SHORT_CROWD", 90, precision, squeeze)
    assert not _precision_stop_allowed(PositionSide.SHORT, "ONE_WAY_DOWN", "NORMAL", 90, precision, squeeze)
    assert not _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "NORMAL", 90, precision)
    assert not _precision_stop_allowed(PositionSide.LONG, "TREND_LONG", "NORMAL", 90, precision, squeeze)
    assert not _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "LONG_CROWD", 90, precision, squeeze)
    assert not _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "NORMAL", 80, precision, squeeze)
    assert not _precision_stop_allowed(PositionSide.LONG, "ONE_WAY_UP", "NORMAL", 90, {**precision, "trend": "CHOP"}, squeeze)


def test_stop_exit_reason_explicitly_marks_stop_or_take_profit() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 100.4

    assert _stop_exit_reason(position, {}, 100.4, 0.0004) == "take profit: protected stop after profit lock"
    assert _stop_exit_reason(position, {}, 99.8, 0.0004) == "stop loss: protected stop slipped below entry"

    position.stop_price = 96.0
    assert _stop_exit_reason(position, {"action": "ENTRY_SHORT", "trend_state": "ONE_WAY_DOWN"}, 96.0, 0.0004) == "stop loss: signal direction or structure failed"
    assert _stop_exit_reason(position, {"action": "WATCH", "trend_state": "TREND_LONG", "score": 80}, 96.0, 0.0004) == "stop loss: ATR volatility hard stop"
    position.metadata["entry_context"] = {"stop_basis": "15m_precision_structure"}
    assert _stop_exit_reason(position, {"action": "WATCH", "trend_state": "TREND_LONG", "score": 80}, 96.0, 0.0004) == "stop loss: 15m entry structure stop"


def test_profit_drawdown_exit_protects_winning_position_with_risk_signal() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 96.0
    position.metadata["initial_stop_distance"] = 4.0
    _update_position_excursions(position, 108.0)
    _update_position_excursions(position, 104.5)

    reason = _profit_drawdown_exit_reason(
        position,
        104.5,
        {"risk_state": "OI_ABNORMAL", "trend_state": "ONE_WAY_UP"},
        indicator_snapshot(close=104.5, atr=1.0),
    )

    assert reason == "take profit: profit drawdown after OI abnormal risk"


def test_structure_take_profit_near_4h_level() -> None:
    position = asyncio.run(PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData()).open_position("TESTUSDT", "LONG", margin_usdt=100, leverage=5))
    position.entry_price = 100.0
    position.stop_price = 96.0
    position.metadata["initial_stop_distance"] = 4.0

    reason = _structure_take_profit_reason(
        position,
        104.9,
        {"h4_structure": {"resistance": 105.0}, "risk_state": "NORMAL"},
        indicator_snapshot(close=104.9, atr=0.5),
    )

    assert reason == "take profit: near 4h resistance with profit protection"


def test_risk_exit_reason_is_specific_to_one_way_position_risk() -> None:
    assert _risk_exit_reason(PositionSide.LONG, "ONE_WAY_UP", "LONG_CROWD") == "risk exit: LONG_CROWD"
    assert _risk_exit_reason(PositionSide.SHORT, "ONE_WAY_DOWN", "SHORT_CROWD") == "risk exit: SHORT_CROWD"
    assert _risk_exit_reason(PositionSide.LONG, "ONE_WAY_UP", "OI_ABNORMAL") == "risk exit: OI_ABNORMAL"
    assert _risk_exit_reason(PositionSide.SHORT, "ONE_WAY_DOWN", "FUNDING_HOT") == "risk exit: FUNDING_HOT"
    assert _risk_exit_reason(PositionSide.LONG, "TREND_LONG", "LONG_CROWD") is None


def test_risk_exit_management_waits_until_position_reaches_one_r() -> None:
    position = Position(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        stop_price=95.0,
        take_profit_1=105.0,
        take_profit_2=110.0,
        metadata={"initial_stop_distance": 5.0},
    )

    assert not _position_reached_initial_r(position, 99.0)
    assert not _position_reached_initial_r(position, 104.9)
    assert _position_reached_initial_r(position, 105.0)

    position.metadata["max_favorable_distance"] = 5.1
    assert _position_reached_initial_r(position, 101.0)
    assert _position_profit_management_enabled(position, 101.0)
    assert not _position_profit_management_enabled(position, 99.0)


def test_auto_top50_universe_refreshes_symbols() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_signals["STALEUSDT"] = {"score": 99}
    engine.account.latest_signals["STALEUSDT"] = {"score": 99}

    import asyncio

    assert engine.symbols == ["AUTO_TOP50"]
    asyncio.run(engine.refresh_universe_if_needed())

    assert len(engine.symbols) == 50
    assert len(engine._candidate_symbols) == 30
    assert "BTCUSDT" not in engine.symbols
    assert "ETHUSDT" not in engine.symbols
    assert "SOLUSDT" not in engine.symbols
    assert engine.symbols[0] == "TEST0USDT"
    assert "STALEUSDT" not in engine.latest_signals
    assert "STALEUSDT" not in engine.account.latest_signals


def test_auto_top50_checks_the_top_80_by_turnover() -> None:
    class CaptureLimitMarket(FakeMarketData):
        requested_limit: int | None = None

        async def top_usdt_perpetuals(self, limit: int = 30):
            self.requested_limit = limit
            return await super().top_usdt_perpetuals(limit=limit)

    market = CaptureLimitMarket()
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=market,
    )

    asyncio.run(engine.refresh_universe_if_needed())

    assert market.requested_limit == 80


def test_candidate_prefilter_only_enforces_the_liquidity_floor() -> None:
    class Candidate:
        quote_volume = 100_000_000
        price_change_percent = 120.0
        last_price = 50.0
        high_price = 120.0
        low_price = 40.0
        open_price = 100.0

    assert _candidate_prefilter_allowed(Candidate())
    Candidate.quote_volume = 49_999_999
    assert not _candidate_prefilter_allowed(Candidate())


def test_auto_universe_does_not_backfill_ineligible_symbols() -> None:
    class PartiallyEligibleMarket(FakeMarketData):
        async def top_usdt_perpetuals(self, limit: int = 30):
            class Symbol:
                def __init__(self, symbol: str, quote_volume: float) -> None:
                    self.symbol = symbol
                    self.quote_volume = quote_volume
                    self.price_change_percent = 5.0
                    self.last_price = 100.0
                    self.high_price = 105.0
                    self.low_price = 95.0
                    self.open_price = 99.0

            qualified = [
                Symbol(f"GOOD{idx}USDT", 100_000_000)
                for idx in range(18)
            ]
            rejected = [
                Symbol(f"THIN{idx}USDT", 10_000_000)
                for idx in range(30)
            ]
            return qualified + rejected

    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=PartiallyEligibleMarket(),
    )

    asyncio.run(engine.refresh_universe_if_needed())

    assert engine.symbols == [f"GOOD{idx}USDT" for idx in range(18)]
    assert not any(symbol.startswith("THIN") for symbol in engine.symbols)


def test_periodic_universe_swap_keeps_eligible_old_pool_when_new_warmup_fails() -> None:
    class RotatingMarket(FakeMarketData):
        async def top_usdt_perpetuals(self, limit: int = 30):
            class Symbol:
                def __init__(self, symbol: str) -> None:
                    self.symbol = symbol
                    self.quote_volume = 100_000_000

            return [
                *(Symbol(f"NEW{idx}USDT") for idx in range(30)),
                *(Symbol(f"OLD{idx}USDT") for idx in range(30)),
            ][:limit]

    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=RotatingMarket(),
    )
    engine.symbols = [f"OLD{idx}USDT" for idx in range(30)]
    engine.latest_signals = {
        symbol: {"action": "NO_TRADE", "score": 0}
        for symbol in engine.symbols
    }

    async def warmup_fails(symbol: str) -> bool:
        return False

    async def no_stream_restart() -> None:
        return None

    engine._refresh_symbol = warmup_fails  # type: ignore[method-assign]
    engine._restart_price_stream_if_needed = no_stream_restart  # type: ignore[method-assign]

    asyncio.run(
        engine._refresh_universe_and_new_symbols(
            datetime.now(UTC),
        )
    )

    assert engine.symbols == [f"OLD{idx}USDT" for idx in range(30)]


def test_auto_main_pool_uses_score_65_regardless_of_signal_action() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine._universe_symbols = [
        "ENTRYUSDT",
        "WATCHUSDT",
        "LOWUSDT",
        "BLOCKEDUSDT",
    ]
    engine.latest_signals = {
        "ENTRYUSDT": {"action": "ENTRY_LONG", "score": 65},
        "WATCHUSDT": {"action": "WATCH", "score": 65},
        "LOWUSDT": {"action": "WATCH", "score": 64},
        "BLOCKEDUSDT": {"action": "NO_TRADE", "score": 99},
    }

    engine._rebalance_auto_signal_pools()

    assert engine.symbols == ["BLOCKEDUSDT", "ENTRYUSDT", "WATCHUSDT"]
    assert engine._candidate_symbols == ["LOWUSDT"]

    engine.latest_signals["LOWUSDT"] = {
        "action": "ENTRY_SHORT",
        "score": 70,
    }
    engine._rebalance_auto_signal_pools()

    assert engine.symbols == [
        "BLOCKEDUSDT",
        "LOWUSDT",
        "ENTRYUSDT",
        "WATCHUSDT",
    ]
    assert engine._candidate_symbols == []


@pytest.mark.parametrize(
    ("score", "expected_action"),
    [
        (64, SignalAction.NO_TRADE.value),
        (65, SignalAction.WATCH.value),
        (79, SignalAction.WATCH.value),
        (80, SignalAction.ENTRY_LONG.value),
        (84, SignalAction.ENTRY_LONG.value),
        (85, SignalAction.ENTRY_LONG.value),
    ],
)
def test_final_score_band_normalizes_signal_action(
    score: int,
    expected_action: str,
) -> None:
    signal = _apply_multi_timeframe_context(
        {
            "action": SignalAction.WATCH.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": score,
            "reasons": (),
            "vetoes": (),
            "risk_state": "NORMAL",
            "trend_state": "TREND_LONG",
        },
        {},
    )

    assert signal["score"] == score
    assert signal["action"] == expected_action


def test_multi_timeframe_score_deduplicates_evidence_across_families() -> None:
    signal = {
        "action": SignalAction.WATCH.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "score": 60,
        "score_evidence_families": {
            "MA_POSITION": 20,
            "DERIVATIVES": 15,
        },
        "reasons": (),
        "vetoes": (),
        "risk_state": "NORMAL",
        "trend_state": "TREND_LONG",
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BREAKOUT_UP"},
        "h4_ma_cluster": {"state": "BREAKOUT_UP"},
        "h1_ma_cluster": {"state": "RETEST_UP"},
        "h1_trigger": {"direction": "LONG", "state": "RETEST"},
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "h4_oi": {"state": "REBUILD_BREAKOUT_LONG"},
        "summary": "MTF: dedup test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["score"] == 79
    assert adjusted["score_evidence_families"] == {
        "DERIVATIVES": 15,
        "DIRECTION": 9,
        "MA_POSITION": 20,
        "TRIGGER": 10,
    }
    assert adjusted["legacy_score"] == 76
    assert adjusted["score_delta_vs_legacy"] == 3


def test_direction_score_uses_h4_as_primary_with_limited_daily_alignment() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.WATCH.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 71,
            "reasons": (),
            "vetoes": (),
            "risk_state": "NORMAL",
            "trend_state": "TREND_LONG",
        },
        {
            "daily_bias": "BULL",
            "h4_structure": {
                "state": "BOX_LOWER_HALF",
                "direction": "LONG",
                "structure_type": "ASCENDING_SUPPORT",
            },
            "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        },
    )

    assert adjusted["score"] == 80
    assert adjusted["action"] == SignalAction.ENTRY_LONG.value
    assert adjusted["score_evidence_families"]["DIRECTION"] == 9
    assert adjusted["legacy_score"] == 77
    assert adjusted["legacy_action"] == SignalAction.WATCH.value
    assert adjusted["score_delta_vs_legacy"] == 3
    assert adjusted["direction_score_breakdown"] == {
        "candidate_direction": "LONG",
        "daily_bias": "BULL",
        "h4_direction": "LONG",
        "h4_base_score": 6,
        "daily_background_score": 0,
        "multi_timeframe_alignment_bonus": 3,
        "direction_positive_score": 9,
        "direction_penalty": 0,
        "direction_cap": 9,
        "daily_aligned": True,
        "h4_aligned": True,
    }


def test_direction_alignment_is_symmetric_for_short_entries() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.WATCH.value,
            "candidate_action": SignalAction.ENTRY_SHORT.value,
            "score": 71,
            "reasons": (),
            "vetoes": (),
            "risk_state": "NORMAL",
            "trend_state": "TREND_SHORT",
        },
        {
            "daily_bias": "BEAR",
            "h4_structure": {
                "state": "BOX_UPPER_HALF",
                "direction": "SHORT",
                "structure_type": "RANGE",
            },
            "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        },
    )

    assert adjusted["score"] == 80
    assert adjusted["action"] == SignalAction.ENTRY_SHORT.value
    assert adjusted["score_evidence_families"]["DIRECTION"] == 9
    assert adjusted["legacy_score"] == 77
    assert adjusted["legacy_action"] == SignalAction.WATCH.value
    assert adjusted["score_delta_vs_legacy"] == 3


def test_direction_family_replaces_stale_oversized_cached_score() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.WATCH.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 90,
            "score_evidence_families": {"DIRECTION": 20},
            "reasons": (),
            "vetoes": (),
            "risk_state": "NORMAL",
            "trend_state": "TREND_LONG",
        },
        {
            "daily_bias": "BULL",
            "h4_structure": {
                "state": "ASCENDING_SUPPORT",
                "direction": "LONG",
                "structure_type": "TREND",
            },
            "h1_trigger": {"direction": "LONG", "state": "WAIT"},
        },
    )

    assert adjusted["score"] == 79
    assert adjusted["score_evidence_families"]["DIRECTION"] == 9
    assert adjusted["direction_score_breakdown"]["direction_cap"] == 9


def test_daily_direction_alone_is_background_not_a_full_h4_score() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.WATCH.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 70,
            "reasons": (),
            "vetoes": (),
            "risk_state": "NORMAL",
            "trend_state": "TREND_LONG",
        },
        {
            "daily_bias": "BULL",
            "h4_structure": {
                "state": "BOX_LOWER_HALF",
                "direction": "NEUTRAL",
                "structure_type": "RANGE",
            },
            "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        },
    )

    assert adjusted["score"] == 73
    assert adjusted["score_evidence_families"]["DIRECTION"] == 3
    assert adjusted["direction_score_breakdown"]["h4_base_score"] == 0
    assert adjusted["direction_score_breakdown"]["daily_background_score"] == 3
    assert adjusted["direction_score_breakdown"]["multi_timeframe_alignment_bonus"] == 0


def test_unconfirmed_h1_direction_does_not_add_trigger_score() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.WATCH.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 70,
            "reasons": (),
            "vetoes": (),
            "risk_state": "NORMAL",
            "trend_state": "TREND_LONG",
        },
        {
            "daily_bias": "NEUTRAL",
            "h4_structure": {
                "state": "BOX_LOWER_HALF",
                "direction": "LONG",
                "structure_type": "ASCENDING_SUPPORT",
            },
            "h1_trigger": {"direction": "LONG", "state": "WAIT"},
        },
    )

    assert adjusted["score"] == 76
    assert "TRIGGER" not in adjusted["score_evidence_families"]
    assert adjusted["entry_trigger"] == ""


def test_ma_cluster_uses_strongest_location_evidence_only() -> None:
    score, reasons, vetoes = _ma_cluster_signal_adjustment(
        PositionSide.LONG,
        {"state": "BREAKOUT_UP", "price": 1.30},
        {"state": "RETEST_UP", "price": 1.31},
    )

    assert score == 14
    assert "MA cluster breakout up: price=1.31" in reasons
    assert "MA cluster retest held near MA20: price=1.31" in reasons
    assert not vetoes


def test_auto_main_pool_never_exceeds_50_symbols() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine._universe_symbols = [
        f"TEST{index}USDT"
        for index in range(80)
    ]
    engine.latest_signals = {
        symbol: {
            "action": "WATCH",
            "score": 65 + index,
        }
        for index, symbol in enumerate(engine._universe_symbols)
    }

    engine._rebalance_auto_signal_pools()

    assert len(engine.symbols) == 50
    assert len(engine._candidate_symbols) == 30
    assert set(engine.symbols).isdisjoint(engine._candidate_symbols)


def test_auto_main_pool_rebalances_on_15m_close_not_only_1h_close() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine._universe_symbols = ["PROMOTEUSDT"]
    engine.symbols = []
    engine._candidate_symbols = ["PROMOTEUSDT"]
    engine.latest_signals = {
        "PROMOTEUSDT": {
            "action": SignalAction.WATCH.value,
            "score": 70,
        }
    }
    engine._symbols_with_latest_closed_timeframe = (  # type: ignore[method-assign]
        lambda timeframe, now=None: {"PROMOTEUSDT"}
        if timeframe == "15m"
        else set()
    )
    now = datetime(2026, 7, 3, 2, 15, tzinfo=UTC)

    engine._rebalance_auto_signal_pools_if_due(("1h",), now)
    assert engine.symbols == []

    engine._rebalance_auto_signal_pools_if_due(("15m",), now)
    assert engine.symbols == ["PROMOTEUSDT"]
    assert engine._candidate_symbols == []


def test_auto_status_only_exposes_main_pool_signals() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine._universe_symbols = ["MAINUSDT", "CANDIDATEUSDT"]
    engine.symbols = ["MAINUSDT"]
    engine._candidate_symbols = ["CANDIDATEUSDT"]
    engine.latest_signals = {
        "MAINUSDT": {
            "action": SignalAction.WATCH.value,
            "score": 70,
        },
        "CANDIDATEUSDT": {
            "action": SignalAction.NO_TRADE.value,
            "score": 64,
        },
    }

    status = engine.status()

    assert list(status["latest_signals"]) == ["MAINUSDT"]


def test_pool_rotation_keeps_stale_symbols_in_their_existing_pool() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine._universe_symbols = [
        "STALEMAINUSDT",
        "FRESHCANDIDATEUSDT",
        "STALECANDIDATEUSDT",
    ]
    engine.symbols = ["STALEMAINUSDT"]
    engine._candidate_symbols = [
        "FRESHCANDIDATEUSDT",
        "STALECANDIDATEUSDT",
    ]
    engine.latest_signals = {
        "STALEMAINUSDT": {
            "action": "NO_TRADE",
            "score": 0,
        },
        "FRESHCANDIDATEUSDT": {
            "action": "WATCH",
            "score": 70,
        },
        "STALECANDIDATEUSDT": {
            "action": "ENTRY_LONG",
            "score": 99,
        },
    }

    engine._rebalance_auto_signal_pools(
        fresh_symbols={"FRESHCANDIDATEUSDT"},
    )

    assert engine.symbols == [
        "STALEMAINUSDT",
        "FRESHCANDIDATEUSDT",
    ]
    assert engine._candidate_symbols == [
        "STALECANDIDATEUSDT",
    ]

    engine._rebalance_auto_signal_pools(
        fresh_symbols=set(engine._universe_symbols),
    )

    assert "STALEMAINUSDT" not in engine.symbols
    assert engine.symbols == [
        "STALECANDIDATEUSDT",
        "FRESHCANDIDATEUSDT",
    ]


def test_position_is_pinned_first_and_waits_for_post_close_rescore() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine._universe_symbols = [
        "HOLDUSDT",
        "WATCHUSDT",
    ]
    engine.symbols = ["WATCHUSDT"]
    engine._candidate_symbols = ["HOLDUSDT"]
    position = asyncio.run(
        engine.open_position(
            "HOLDUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
        )
    )
    engine.latest_prices["HOLDUSDT"] = 100.0
    engine.latest_signals = {
        "HOLDUSDT": {
            "action": "NO_TRADE",
            "score": 0,
            "reasons": (),
            "vetoes": (),
        },
        "WATCHUSDT": {
            "action": "WATCH",
            "score": 70,
            "reasons": (),
            "vetoes": (),
        },
    }

    engine._rebalance_auto_signal_pools(
        fresh_symbols=set(engine._universe_symbols),
    )

    assert engine.symbols[0] == "HOLDUSDT"
    assert list(engine.status()["latest_signals"])[0] == "HOLDUSDT"

    engine._close_position_unlocked(position, 100.0, "manual close")
    engine._rebalance_auto_signal_pools(
        fresh_symbols={"WATCHUSDT"},
    )

    assert "HOLDUSDT" in engine.symbols
    assert "HOLDUSDT" in engine._post_close_pool_review

    engine._rebalance_auto_signal_pools(
        fresh_symbols={"HOLDUSDT", "WATCHUSDT"},
    )

    assert "HOLDUSDT" not in engine.symbols
    assert "HOLDUSDT" not in engine._post_close_pool_review
    assert "HOLDUSDT" in engine._candidate_symbols


def test_candidate_pool_signal_cannot_open_until_promoted_to_main_pool() -> None:
    settings = AppSettings()
    settings.risk.max_open_positions = 2
    engine = PaperTradingEngine(
        settings,
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine._universe_symbols = ["MAINUSDT", "CANDIDATEUSDT"]
    engine.symbols = ["MAINUSDT"]
    engine._candidate_symbols = ["CANDIDATEUSDT"]
    engine.latest_prices = {
        "MAINUSDT": 100.0,
        "CANDIDATEUSDT": 100.0,
    }
    signal = {
        "action": "ENTRY_LONG",
        "score": 95,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                },
            },
        },
        "h1_structure": {
            "resistance_zone_low": 105.0,
            "resistance": 106.0,
        },
    }
    engine.latest_signals = {
        "MAINUSDT": dict(signal),
        "CANDIDATEUSDT": dict(signal),
    }

    asyncio.run(engine._auto_trade_once())

    assert set(engine.account.positions) == {"MAINUSDT"}


def test_auto_signal_score_tiers_and_margins() -> None:
    base_signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
        "h1_structure": {"resistance": 105.0},
    }
    assert not _auto_signal_allowed({
        **base_signal,
        "score": 85,
        "risk_state": "LONG_CROWD",
        "vetoes": ("long side overcrowded",),
    })
    assert _auto_signal_allowed({**base_signal, "score": 85, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({**base_signal, "score": 78, "risk_state": "FUNDING_HOT"})
    assert _auto_signal_allowed({**base_signal, "score": 81, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({**base_signal, "score": 79, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({**base_signal, "score": 74, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({**base_signal, "score": 90, "risk_state": "NORMAL", "vetoes": ("1h trigger opposes long entry",)})
    assert not _auto_signal_allowed({"score": 90, "risk_state": "NORMAL"})


def test_version_two_entry_pipeline_requires_an_explicit_trigger() -> None:
    signal = {
        "entry_pipeline_version": 2,
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {"low": 99.0, "high": 101.0, "price": 100.0},
            }
        },
        "vetoes": (),
    }

    _update_entry_position_fields(signal)

    assert signal["entry_timing"] == "GOOD"
    assert signal["entry_state"] == "TRIGGER_PENDING"
    assert "entry trigger is not confirmed" in _auto_entry_prerequisite_blocks(signal)


def test_watch_signal_below_entry_threshold_is_score_pending_not_direction_pending() -> None:
    signal = {
        "entry_pipeline_version": 2,
        "action": SignalAction.WATCH.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "score": 76,
        "h4_direction": "LONG",
        "h1_trigger": {
            "direction": "LONG",
            "state": "RETEST",
        },
        "vetoes": (),
    }

    _update_entry_position_fields(signal)

    assert signal["entry_state"] == "SCORE_PENDING"
    assert signal["entry_trigger"] == "1H_RETEST"
    assert signal["entry_timing_reason"] == (
        "entry position blocked: final score 76 below "
        "auto-entry minimum 80"
    )
    assert _auto_entry_prerequisite_blocks(signal) == (
        "final score 76 below auto-entry minimum 80",
    )


def test_watch_signal_with_hard_veto_is_not_reported_as_missing_direction() -> None:
    signal = {
        "entry_pipeline_version": 2,
        "action": SignalAction.WATCH.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "h4_direction": "SHORT",
        "vetoes": ("4h direction is short",),
    }

    _update_entry_position_fields(signal)

    assert signal["entry_state"] == "VETOED"
    assert "entry has an active veto" in _auto_entry_prerequisite_blocks(signal)


def test_watch_signal_with_h4_direction_reports_position_pending() -> None:
    signal = {
        "entry_pipeline_version": 2,
        "action": SignalAction.WATCH.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "h4_direction": "LONG",
        "price": 105.0,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                },
            },
        },
        "vetoes": (),
    }

    _update_entry_position_fields(signal)

    assert signal["entry_state"] == "POSITION_PENDING"
    assert signal["entry_timing"] == "WAIT"
    assert "direction" not in signal["entry_timing_reason"]
    assert _auto_entry_prerequisite_blocks(signal) == (
        "current price has not reached the selected primary setup",
    )


def test_watch_signal_with_h4_direction_reports_trigger_pending() -> None:
    signal = {
        "entry_pipeline_version": 2,
        "action": SignalAction.WATCH.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "h4_direction": "LONG",
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                },
            },
        },
        "h1_trigger": {
            "direction": "LONG",
            "state": "WAIT",
        },
        "vetoes": (),
    }

    _update_entry_position_fields(signal)

    assert signal["entry_state"] == "TRIGGER_PENDING"
    assert signal["entry_timing"] == "GOOD"
    assert _auto_entry_prerequisite_blocks(signal) == (
        "entry trigger is not confirmed",
    )


def test_legacy_watch_signal_without_candidate_direction_remains_compatible() -> None:
    signal = {
        "entry_pipeline_version": 2,
        "action": SignalAction.WATCH.value,
        "score": 90,
        "vetoes": (),
    }

    _update_entry_position_fields(signal)

    assert signal["entry_state"] == "DIRECTION_PENDING"
    assert _auto_entry_prerequisite_blocks(signal) == (
        "directional entry signal not established",
        "higher-timeframe direction is not established",
    )


def test_entry_quality_uses_only_a_and_s_score_bands() -> None:
    assert _entry_quality_grade(
        score=100,
        reward_r=1.6,
        stop_pct=0.025,
        setup_type=SETUP_H1_PULLBACK_LONG,
    ) == "S"
    assert _entry_quality_grade(
        score=96,
        reward_r=1.6,
        stop_pct=0.03,
        setup_type=SETUP_H1_PULLBACK_LONG,
    ) == "A"
    assert _entry_quality_grade(
        score=90,
        reward_r=1.25,
        stop_pct=0.04,
        setup_type=SETUP_H1_PULLBACK_LONG,
    ) == "A"
    assert _entry_quality_grade(
        score=79,
        reward_r=1.25,
        stop_pct=0.08,
        setup_type=SETUP_H1_PULLBACK_LONG,
    ) is None

    assert _leverage_for_entry_quality("S", stop_pct=0.03, leverage_max=10) == 10
    assert _leverage_for_entry_quality("A", stop_pct=0.04, leverage_max=10) == 10
    assert _leverage_for_entry_quality("A", stop_pct=0.05, leverage_max=10) == 7
    assert _leverage_for_entry_quality(None, stop_pct=0.06, leverage_max=10) == 0


def test_s_quality_uses_only_the_one_remaining_capital_unit() -> None:
    assert _fixed_margin_for_quality(
        "A",
        equity=1200.0,
        used_margin=0.0,
        available_balance=1200.0,
    ) == pytest.approx(228.0)
    assert _fixed_margin_for_quality(
        "S",
        equity=1200.0,
        used_margin=0.0,
        available_balance=1200.0,
    ) == pytest.approx(456.0)
    assert _fixed_margin_for_quality(
        "S",
        equity=1000.0,
        used_margin=760.0,
        available_balance=240.0,
    ) == pytest.approx(190.0)

def test_auto_entry_prerequisites_explain_score_direction_and_timing_blocks() -> None:
    wait_signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 79,
        "risk_state": "NORMAL",
        "price": 106.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }

    blocks = _auto_entry_prerequisite_blocks(wait_signal)

    assert "final score 79 below auto-entry minimum 80" in blocks
    assert "等待 1H支撑回踩区" in blocks
    assert _auto_entry_prerequisite_blocks(
        wait_signal,
        include_entry_position=False,
    ) == ("final score 79 below auto-entry minimum 80",)
    assert _auto_entry_prerequisite_blocks({"action": SignalAction.WATCH.value, "score": 90}) == (
        "directional entry signal not established",
    )


def test_status_exposes_strategy_switch_without_mutating_live_signal() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_signals["TESTUSDT"] = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
        "h1_structure": {"resistance": 105.0},
    }

    status_signal = engine.status()["latest_signals"]["TESTUSDT"]

    assert "auto strategy disabled; new entries are paused" in status_signal["vetoes"]
    assert not engine.latest_signals["TESTUSDT"].get("vetoes")


def test_entry_stop_validation_requires_stop_on_loss_side() -> None:
    assert _entry_stop_error(PositionSide.LONG, 100.0, 99.0) is None
    assert _entry_stop_error(PositionSide.SHORT, 100.0, 101.0) is None
    assert _entry_stop_error(PositionSide.LONG, 100.0, 100.0) == "long stop must be below entry price"
    assert _entry_stop_error(PositionSide.SHORT, 100.0, 99.0) == "short stop must be above entry price"


def test_auto_signal_requires_real_entry_zone_for_ordinary_short() -> None:
    mid_zone_short = {
        "action": "ENTRY_SHORT",
        "score": 108,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "price": 254.0,
        "entry_levels": {
            "short": {
                "h1_resistance": {"low": 260.0, "high": 262.0, "price": 261.0},
                "h1_ema20_ema60": {"low": 260.0, "high": 262.0, "price": 261.0},
            }
        },
    }
    timing, reason = _signal_entry_timing(mid_zone_short)

    assert timing == "WAIT"
    assert reason == "等待 1H压力反抽区"
    assert not _auto_signal_allowed(mid_zone_short)

    resistance_retest = {**mid_zone_short, "price": 261.0}
    timing, reason = _signal_entry_timing(resistance_retest)

    assert timing == "GOOD"
    assert reason == "已到优势入场区"
    assert _auto_signal_allowed(resistance_retest)


def test_entry_timing_turns_good_when_price_reaches_suggested_zone() -> None:
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 96,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 106.0,
        "entry_levels": {
            "long": {
                "h1_support": {"low": 99.0, "high": 101.0, "price": 100.0},
                "h1_boll_mid": {"low": 99.5, "high": 100.5, "price": 100.0},
            }
        },
    }

    timing, reason = _signal_entry_timing(signal)
    assert timing == "WAIT"
    assert reason == "等待 1H支撑回踩区"
    assert not _auto_signal_allowed(signal)

    signal["price"] = 100.0
    timing, reason = _signal_entry_timing(signal)
    assert timing == "GOOD"
    assert reason == "已到优势入场区"
    assert _auto_signal_allowed(signal)


def test_entry_position_requires_advantage_side_inside_scored_zone() -> None:
    signal = {
        "timestamp": "2026-07-01T00:00:00+00:00",
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.7,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
    }

    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["price"] = 98.8
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["price"] = 99.8
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"


def test_four_hour_short_entry_requires_upper_half_of_retest_zone() -> None:
    signal = {
        "timestamp": "2026-07-10T00:00:00+00:00",
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 95,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "entry_timeframe_override": "4h",
        "h1_trigger": {"direction": "SHORT", "state": "FAKE_BREAKOUT"},
        "price": 7.75032,
        "entry_levels": {
            "short": {
                "h4_ema20_ema60": {
                    "low": 7.70310,
                    "high": 7.80759,
                    "price": 7.75535,
                }
            }
        },
    }

    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["price"] = 7.78000
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"


def test_four_hour_indicator_override_does_not_create_direction_in_chop() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 95,
        "trend_state": "CHOP",
        "risk_state": "NORMAL",
        "entry_timeframe_override": "4h",
        "price": 7.79,
        "entry_levels": {
            "short": {
                "h4_ema20_ema60": {
                    "low": 7.70,
                    "high": 7.80,
                    "price": 7.75,
                }
            }
        },
    }

    _update_entry_position_fields(signal)

    assert signal["entry_timing"] == "WAIT"


def test_chop_h4_pullback_keeps_boundary_not_mid_indicator_zones() -> None:
    selected = _scored_entry_levels(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "trend_state": "CHOP",
            "setup_type": SETUP_H4_PULLBACK_SHORT,
            "reasons": ("4h structure supports downside",),
        },
        {
            "short": {
                "h4_resistance": {"low": 103.0, "high": 104.0},
                "h4_boll_mid": {"low": 99.5, "high": 100.5},
                "h4_ema20_ema60": {"low": 99.0, "high": 101.0},
            }
        },
    )

    assert selected == {
        "short": {
            "h4_resistance": {"low": 103.0, "high": 104.0},
        }
    }


def test_suggested_entry_text_uses_the_same_filtered_entry_levels() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 90,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "price": 98.0,
        "entry_levels": {
            "short": {
                "h1_resistance": {"low": 101.0, "high": 102.0, "price": 101.5},
                "breakdown_retest": {"low": 99.0, "high": 100.0, "price": 99.5},
            }
        },
    }

    _update_entry_position_fields(signal)

    assert signal["entry_timing"] == "WAIT"
    assert signal["entry_timing_reason"] == "等待 1H压力反抽区"
    assert signal["suggested_entry_text"] == (
        "1H压力反抽区≈101-102；前支撑跌破后反抽确认区≈99-100"
    )


def test_signal_without_filtered_entry_levels_has_no_suggested_entry_zone() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 90,
        "price": 98.0,
        "entry_levels": {"short": {}},
    }

    _update_entry_position_fields(signal)

    assert signal["entry_timing"] == "BLOCK"
    assert signal["entry_timing_reason"] == "暂无有效建议入场区"
    assert signal["suggested_entry_text"] == "暂无有效建议入场区"


def test_entry_position_rejects_aave_like_lower_single_indicator_zone() -> None:
    signal = {
        "timestamp": "2026-06-28T07:00:00+00:00",
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 96,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "reasons": ("1h BOLL/EMA pullback rejected with clean risk",),
        "price": 90.93,
        "entry_levels": {
            "short": {
                "h1_boll_mid": {
                    "low": 93.8,
                    "high": 94.8,
                    "price": 94.3,
                },
                "h1_ema20_ema60": {
                    "low": 93.7,
                    "high": 94.7,
                    "price": 94.2,
                },
                "h4_ema20_ema60": {
                    "low": 90.4,
                    "high": 91.3,
                    "price": 90.85,
                },
                "oi_distribution": {
                    "low": 90.15,
                    "high": 91.71,
                    "price": 90.93,
                },
            }
        },
    }

    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["price"] = 94.5
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"


def test_correlated_indicator_zones_need_scored_pullback_confirmation() -> None:
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 96,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "reasons": ("4h structure supports upside",),
        "entry_levels": {
            "long": {
                "h1_boll_mid": {
                    "low": 99.5,
                    "high": 100.5,
                    "price": 100.0,
                },
                "h1_ema20_ema60": {
                    "low": 99.6,
                    "high": 100.4,
                    "price": 100.0,
                },
            }
        },
    }

    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["reasons"] = (
        "1h BOLL/EMA pullback held with clean risk",
    )
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"


def test_four_hour_override_requires_price_rejection_confirmation() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 96,
        "trend_state": "ONE_WAY_DOWN",
        "risk_state": "NORMAL",
        "entry_timeframe_override": "4h",
        "price": 100.0,
        "reasons": ("4h structure supports downside",),
        "entry_levels": {
            "short": {
                "h4_ema20_ema60": {
                    "low": 99.5,
                    "high": 100.5,
                    "price": 100.0,
                },
            }
        },
    }

    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["h1_trigger"] = {
        "direction": "SHORT",
        "state": "FAKE_BREAKOUT",
    }
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"


def test_setup_type_selects_compatible_scored_entry_zone_without_reason_text() -> None:
    levels = {
        "long": {
            "h1_ema20_ema60": {
                "low": 99.0,
                "high": 101.0,
                "price": 100.0,
            },
        },
    }

    selected = _scored_entry_levels(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "setup_type": SETUP_H1_PULLBACK_LONG,
            "reasons": (),
        },
        levels,
    )

    assert selected == {"long": {"h1_ema20_ema60": levels["long"]["h1_ema20_ema60"]}}


def test_multi_timeframe_context_publishes_setup_type_for_four_hour_short_override() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "candidate_action": SignalAction.ENTRY_SHORT.value,
        "score": 90,
        "trend_state": "ONE_WAY_DOWN",
        "risk_state": "NORMAL",
        "price": 100.0,
        "reasons": ("base short direction",),
    }
    context = {
        "summary": "context",
        "daily_bias": "NEUTRAL",
        "h1_ema_reliability": {"short_state": "UNRELIABLE"},
        "entry_levels": {
            "short": {
                "h4_ema20_ema60": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                },
            },
        },
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["setup_type"] == SETUP_H4_PULLBACK_SHORT
    assert set(adjusted["entry_levels"]["short"]) == {"h4_ema20_ema60"}


def test_multi_timeframe_context_sets_h1_pullback_setup_from_source_without_reason_text() -> None:
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "score": 88,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "reasons": ("base long direction",),
    }
    context = {
        "summary": "context",
        "daily_bias": "NEUTRAL",
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "entry_levels": {
            "long": {
                "h1_ema20_ema60": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                },
            },
        },
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["setup_type"] == SETUP_H1_PULLBACK_LONG
    assert set(adjusted["entry_levels"]["long"]) == {"h1_ema20_ema60"}


def test_multi_timeframe_context_sets_oi_valley_setup_from_source_without_reason_text() -> None:
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "candidate_action": SignalAction.ENTRY_LONG.value,
        "score": 82,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "reasons": ("base long direction",),
    }
    context = {
        "summary": "context",
        "daily_bias": "NEUTRAL",
        "h4_structure": {"state": "BREAKOUT_UP"},
        "h4_oi_valley": {"state": "CONFIRMED"},
        "h1_trigger": {"direction": "LONG", "state": "FAKE_BREAKDOWN"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "entry_levels": {
            "long": {
                "sweep_reclaim_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                },
            },
        },
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["setup_type"] == SETUP_OI_VALLEY_REVERSAL_LONG
    assert set(adjusted["entry_levels"]["long"]) == {"sweep_reclaim_support"}
    assert adjusted["h4_direction"] == "LONG"
    assert adjusted["oi_valley_long_eligible"] is True


def test_ema55_oi_valley_supports_long_only_after_h4_direction_is_long() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 82,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "price": 100.0,
            "reasons": ("base long direction",),
            "vetoes": (),
        },
        {
            "summary": "context",
            "daily_bias": "NEUTRAL",
            "h4_structure": {
                "state": "BOX_LOWER_HALF",
                "direction": "LONG",
                "structure_type": "ASCENDING_SUPPORT",
            },
            "h4_oi_valley": {"state": "EXHAUSTION"},
            "h4_oi_valley_long": {"state": "CONFIRMED"},
            "h1_trigger": {
                "direction": "LONG",
                "state": "RETEST",
            },
            "h1_pullback": {"direction": "NONE", "state": "WAIT"},
            "entry_levels": {
                "long": {
                    "h4_ema55_reclaim": {
                        "low": 99.0,
                        "high": 101.0,
                        "price": 100.0,
                    },
                    "h1_support": {
                        "low": 98.0,
                        "high": 99.0,
                        "price": 98.5,
                    },
                },
            },
        },
    )

    assert adjusted["oi_valley_long_eligible"] is True
    assert adjusted["oi_valley_direction_gate_reason"] == ""
    assert adjusted["setup_type"] == SETUP_OI_VALLEY_REVERSAL_LONG
    assert adjusted["entry_timeframe_override"] == "4h"
    assert set(adjusted["entry_levels"]["long"]) == {
        "h4_ema55_reclaim"
    }
    assert adjusted["score_evidence_families"]["DERIVATIVES"] == 6


@pytest.mark.parametrize(
    "h4_structure",
    [
        {"state": "BOX_UPPER_HALF"},
        {"state": "BOX_LOWER_HALF", "direction": "NEUTRAL"},
        {"state": "UNKNOWN"},
    ],
)
def test_oi_valley_does_not_take_over_other_long_setup_without_h4_long(
    h4_structure: dict[str, object],
) -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 88,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "price": 100.0,
            "setup_type": SETUP_OI_VALLEY_REVERSAL_LONG,
            "reasons": ("base long direction",),
            "vetoes": (),
        },
        {
            "summary": "context",
            "daily_bias": "NEUTRAL",
            "h4_structure": h4_structure,
            "h4_oi_valley": {"state": "EXHAUSTION"},
            "h4_oi_valley_long": {"state": "CONFIRMED"},
            "h1_trigger": {"direction": "NONE", "state": "WAIT"},
            "h1_pullback": {
                "direction": "LONG",
                "state": "HEALTHY_PULLBACK",
            },
            "entry_levels": {
                "long": {
                    "h1_support": {
                        "low": 99.0,
                        "high": 101.0,
                        "price": 100.0,
                    },
                    "h4_ema55_reclaim": {
                        "low": 99.0,
                        "high": 101.0,
                        "price": 100.0,
                    },
                },
            },
        },
    )

    assert adjusted["h4_direction"] == "NEUTRAL"
    assert adjusted["oi_valley_long_eligible"] is False
    assert (
        adjusted["oi_valley_direction_gate_reason"]
        == "4h direction is not long"
    )
    assert adjusted["setup_type"] == SETUP_H1_PULLBACK_LONG
    assert "h4_ema55_reclaim" not in adjusted["entry_levels"]["long"]
    assert set(adjusted["entry_levels"]["long"]) == {"h1_support"}
    assert "DERIVATIVES" not in adjusted["score_evidence_families"]
    assert "entry_timeframe_override" not in adjusted
    assert adjusted["action"] == SignalAction.ENTRY_LONG.value


def test_h4_short_direction_blocks_long_and_oi_valley_cannot_override_it() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 100,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "price": 100.0,
            "reasons": ("base long direction",),
            "vetoes": (),
        },
        {
            "summary": "context",
            "daily_bias": "NEUTRAL",
            "h4_structure": {
                "state": "BOX_UPPER_HALF",
                "direction": "SHORT",
                "structure_type": "DESCENDING_RESISTANCE",
            },
            "h4_oi_valley": {"state": "EXHAUSTION"},
            "h4_oi_valley_long": {"state": "CONFIRMED"},
            "h1_trigger": {
                "direction": "LONG",
                "state": "RETEST",
            },
            "h1_pullback": {"direction": "NONE", "state": "WAIT"},
            "entry_levels": {
                "long": {
                    "breakout_retest": {
                        "low": 99.0,
                        "high": 101.0,
                        "price": 100.0,
                    },
                    "h4_ema55_reclaim": {
                        "low": 99.0,
                        "high": 101.0,
                        "price": 100.0,
                    },
                },
            },
        },
    )

    assert adjusted["action"] == SignalAction.WATCH.value
    assert adjusted["direction_gate"] == "SHORT_ONLY"
    assert adjusted["oi_valley_long_eligible"] is False
    assert adjusted["setup_type"] != SETUP_OI_VALLEY_REVERSAL_LONG
    assert "DERIVATIVES" not in adjusted["score_evidence_families"]
    assert "h4_ema55_reclaim" not in adjusted["entry_levels"]["long"]


def test_separated_ema20_and_ema60_do_not_create_a_bridge_entry_zone() -> None:
    indicator = indicator_snapshot(
        close=90.93,
        atr=1.0,
        ema20=94.0,
    )

    zone = _ema20_ema60_band(
        indicator,
        {"ema60": 91.5},
    )

    assert zone is None


def test_h1_pullback_setup_allows_split_ema60_entry_without_reason_text() -> None:
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "price": 91.5,
        "setup_type": SETUP_H1_PULLBACK_LONG,
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "entry_levels": {
            "long": {
                "h1_ema60": {
                    "low": 91.2,
                    "high": 91.8,
                    "price": 91.5,
                },
            },
        },
        "reasons": (),
    }

    timing, reason = _signal_entry_timing(signal)

    assert timing == "GOOD"
    assert reason == "已到优势入场区"


def test_oi_context_does_not_become_a_standalone_suggested_entry_zone() -> None:
    levels = {
        "short": {
            "h1_resistance": {
                "low": 93.8,
                "high": 94.8,
                "price": 94.3,
            },
            "oi_distribution": {
                "low": 90.15,
                "high": 91.71,
                "price": 90.93,
            },
        }
    }
    selected = _scored_entry_levels(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "signal_timeframe": "1h",
            "reasons": ("4h OI drained while price stalled",),
        },
        levels,
    )

    assert selected == {}


def test_ma_cluster_breakdown_is_direction_evidence_not_a_short_entry_zone() -> None:
    levels = {
        "short": {
            "ma_cluster_breakdown": {
                "low": 0.42585,
                "high": 0.473416,
                "price": 0.449633,
            },
            "h1_resistance": {
                "low": 0.41743,
                "high": 0.42457,
                "price": 0.421,
            },
        },
    }

    selected = _scored_entry_levels(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "signal_timeframe": "1h",
            "reasons": ("MA cluster breakdown down",),
        },
        levels,
    )

    assert selected == {}


def test_high_distribution_handoff_distinguishes_weak_price_from_healthy_recovery() -> None:
    start = datetime(2026, 6, 29, 12, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + timedelta(hours=index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        )
        for index in range(20)
    ]
    candles[14] = Candle(
        timestamp=candles[14].timestamp,
        open=108.0,
        high=110.0,
        low=108.0,
        close=109.0,
        volume=1800.0,
    )
    candles[19] = Candle(
        timestamp=candles[19].timestamp,
        open=106.0,
        high=106.5,
        low=104.5,
        close=105.0,
        volume=1200.0,
    )
    oi_values = [
        1080.0 + index
        for index in range(20)
    ]
    oi_values[14:20] = [
        1100.0,
        1090.0,
        1094.0,
        1098.0,
        1100.0,
        1102.0,
    ]
    ratios = [1.18 for _ in candles]
    ratios[14] = 1.20
    ratios[19] = 1.25
    indicators = [
        indicator_snapshot(
            close=candle.close,
            ema20=106.0,
            open_interest=oi,
            long_short_ratio=ratio,
        )
        for candle, oi, ratio in zip(candles, oi_values, ratios)
    ]

    handoff = _high_distribution_handoff(candles, indicators)

    assert handoff["state"] == "CONFIRMED"
    assert handoff["oi_rebuilt"]
    assert handoff["price_failed"]
    assert handoff["ratio_shifted_to_longs"]

    healthy_candles = list(candles)
    healthy_candles[19] = Candle(
        timestamp=candles[19].timestamp,
        open=108.5,
        high=109.8,
        low=108.2,
        close=109.5,
        volume=1200.0,
    )
    healthy_indicators = [
        indicator_snapshot(
            close=candle.close,
            ema20=106.0,
            open_interest=oi,
            long_short_ratio=1.19 if index == 19 else ratio,
        )
        for index, (candle, oi, ratio) in enumerate(
            zip(healthy_candles, oi_values, ratios)
        )
    ]
    assert (
        _high_distribution_handoff(
            healthy_candles,
            healthy_indicators,
        )["state"]
        == "WATCH"
    )


def test_high_distribution_handoff_vetoes_new_long_and_waits_for_stage_two_short() -> None:
    handoff = {"state": "CONFIRMED"}
    context = {
        "daily_bias": "NEUTRAL",
        "h4_structure": {"state": "BOX_LOWER_HALF"},
        "h1_structure": {},
        "h4_ma_cluster": {},
        "h1_ma_cluster": {},
        "h4_oi": {"state": "NORMAL"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "h1_ema_reliability": {"state": "RELIABLE"},
        "high_distribution_handoff": handoff,
        "distribution_short": {
            "active": False,
            "descending_trendline_zone": None,
        },
        "m15_precision": {},
        "entry_levels": {"long": {}, "short": {}},
        "summary": "test",
    }
    adjusted_long = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "score": 96,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "reasons": (),
            "vetoes": (),
        },
        context,
    )

    assert (
        "high distribution handoff complete; avoid new long"
        in adjusted_long["vetoes"]
    )
    assert (
        _distribution_short_stage(
            {
                "action": SignalAction.ENTRY_SHORT.value,
                "trend_state": "TREND_SHORT",
            },
            context,
        )
        == DISTRIBUTION_STAGE_DESCENDING
    )


def test_profitable_long_exits_after_high_distribution_handoff_completes() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
        )
    )
    position.metadata["max_favorable_distance"] = 2.1

    reason = _high_distribution_handoff_exit_reason(
        position,
        {"high_distribution_handoff": {"state": "CONFIRMED"}},
    )

    assert reason == "take profit: high distribution handoff complete"


def test_four_hour_oi_valley_requires_retail_carry_capitulation_and_rebuild() -> None:
    start = datetime(2026, 6, 25, tzinfo=UTC)
    closes = [100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 87.5, 88.0, 89.0]
    oi_values = [1000.0, 990.0, 980.0, 970.0, 960.0, 950.0, 900.0, 902.0, 906.0, 912.0]
    ratios = [1.0, 1.05, 1.10, 1.16, 1.24, 1.32, 1.36, 1.38, 1.40, 1.42]
    candles = [
        Candle(
            timestamp=start + timedelta(hours=4 * index),
            open=close + 0.5,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1000.0,
        )
        for index, close in enumerate(closes)
    ]
    indicators = [
        indicator_snapshot(
            close=close,
            open_interest=oi,
            long_short_ratio=ratio,
        )
        for close, oi, ratio in zip(closes, oi_values, ratios)
    ]

    valley = _four_hour_oi_valley(
        candles,
        indicators,
        AppSettings(),
    )

    assert valley["state"] == "CONFIRMED"
    assert valley["retail_long_carry"]
    assert valley["carry_price_change_pct"] < 0
    assert valley["carry_oi_change_pct"] < 0
    assert valley["carry_long_short_ratio_change_pct"] > 0
    assert valley["flush_pct"] == pytest.approx((950.0 - 900.0) / 950.0)
    assert valley["rebuild_pct"] == pytest.approx((912.0 - 900.0) / 900.0)

    no_rebuild_indicators = list(indicators)
    no_rebuild_indicators[-1] = indicator_snapshot(
        close=closes[-1],
        open_interest=905.0,
        long_short_ratio=ratios[-1],
    )
    assert (
        _four_hour_oi_valley(
            candles,
            no_rebuild_indicators,
            AppSettings(),
        )["state"]
        == "WATCH"
    )

    oi_did_not_decline = [
        indicator_snapshot(
            close=close,
            open_interest=oi,
            long_short_ratio=ratio,
        )
        for close, oi, ratio in zip(
            closes,
            [900.0, 910.0, 920.0, 930.0, 940.0, 950.0, 900.0, 902.0, 906.0, 912.0],
            ratios,
        )
    ]
    no_retail_carry = _four_hour_oi_valley(
        candles,
        oi_did_not_decline,
        AppSettings(),
    )
    assert no_retail_carry["state"] == "EXHAUSTION"
    assert not no_retail_carry["retail_long_carry"]


def test_gradual_four_hour_oi_valley_blocks_short_without_single_bar_flush() -> None:
    start = datetime(2026, 7, 10, tzinfo=UTC)
    closes = [100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 89.0, 88.0, 88.5, 90.0]
    oi_values = [1000.0, 988.0, 976.0, 964.0, 952.0, 940.0, 928.0, 916.0, 920.0, 930.0]
    ratios = [1.0, 1.04, 1.08, 1.12, 1.18, 1.24, 1.30, 1.34, 1.36, 1.38]
    candles = [
        Candle(
            timestamp=start + timedelta(hours=4 * index),
            open=close + 0.5,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1000.0,
        )
        for index, close in enumerate(closes)
    ]
    indicators = [
        indicator_snapshot(
            close=close,
            open_interest=oi,
            long_short_ratio=ratio,
        )
        for close, oi, ratio in zip(closes, oi_values, ratios)
    ]

    valley = _four_hour_oi_valley(candles, indicators, AppSettings())

    assert valley["flush_pct"] < AppSettings().strategy.smart_money_oi_flush
    assert valley["cumulative_flush_pct"] > AppSettings().strategy.smart_money_oi_flush
    assert valley["state"] == "CONFIRMED"
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "score": 110,
            "trend_state": "TREND_SHORT",
            "trend_stage_phase": "LATE",
            "risk_state": "NORMAL",
            "reasons": (),
            "vetoes": (),
            "price": 90.0,
        },
        {
            "daily_bias": "BEAR",
            "h4_structure": {"state": "BOX_LOWER_HALF"},
            "h4_oi_valley": valley,
            "h1_trigger": {"direction": "SHORT", "state": "RETEST"},
            "h1_pullback": {"direction": "SHORT", "state": "HEALTHY_PULLBACK"},
                "entry_levels": {
                    "short": {
                        "h1_ema20_ema60": {"low": 89.5, "high": 90.5, "price": 90.0},
                    }
                },
            "summary": "test",
        },
    )
    assert (
        "4h OI valley confirmed; low-area short chasing is blocked"
        in adjusted["vetoes"]
    )


def test_four_hour_oi_valley_absorption_builds_ema55_reversal_long() -> None:
    start = datetime(2026, 7, 4, tzinfo=UTC)
    closes = [100.0] * 48 + [98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 87.0, 88.0, 89.0, 91.0, 98.0, 100.0]
    lows = [close - 0.8 for close in closes]
    lows[54] = 84.0
    lows[56] = 85.0
    lows[59] = 96.0
    candles = [
        Candle(
            timestamp=start + timedelta(hours=4 * index),
            open=close + (1.0 if index in {54, 56} else 0.2),
            high=close + 2.0,
            low=lows[index],
            close=close,
            volume=1000.0,
        )
        for index, close in enumerate(closes)
    ]
    oi_values = [1000.0] * 54 + [900.0, 902.0, 906.0, 910.0, 918.0, 930.0]
    indicators = [
        indicator_snapshot(
            close=close,
            atr=2.0,
            open_interest=oi,
            long_short_ratio=1.5,
        )
        for close, oi in zip(closes, oi_values)
    ]
    valley = {
        "state": "EXHAUSTION",
        "flush_timestamp": candles[54].timestamp.isoformat(),
        "rebuild_pct": (930.0 - 900.0) / 900.0,
        "price_recovered_after_valley": True,
    }

    setup = _four_hour_oi_valley_long_setup(
        candles,
        indicators,
        valley,
        AppSettings(),
    )

    assert setup["state"] == "CONFIRMED"
    assert setup["lower_wick_count"] >= 2
    assert setup["oi_rebuilding"]
    assert setup["long_short_ratio_stable"]
    assert setup["ema55_reclaimed"]
    assert setup["stop_anchor"] < setup["floor_price"]


def test_ema55_oi_valley_is_supporting_evidence_and_cannot_override_short() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "candidate_action": SignalAction.ENTRY_SHORT.value,
            "score": 120,
            "trend_state": "TREND_SHORT",
            "risk_state": "NORMAL",
            "price": 100.0,
            "rsi14": 55.0,
            "reasons": ("base short direction",),
            "vetoes": (),
        },
        {
            "daily_bias": "NEUTRAL",
            "h4_structure": {"state": "BOX_LOWER_HALF"},
            "h4_oi_valley": {"state": "EXHAUSTION"},
            "h4_oi_valley_long": {
                "state": "CONFIRMED",
                "long_short_ratio": 1.5,
            },
            "h1_trigger": {"direction": "SHORT", "state": "RETEST"},
            "h1_pullback": {"direction": "NONE", "state": "WAIT"},
            "entry_levels": {
                "long": {
                    "h4_ema55_reclaim": {
                        "low": 99.0,
                        "high": 101.0,
                        "price": 100.0,
                    }
                }
            },
            "summary": "test",
        },
    )

    assert adjusted["action"] == SignalAction.WATCH.value
    assert adjusted["candidate_action"] == SignalAction.ENTRY_SHORT.value
    assert adjusted["score"] < AUTO_ENTRY_MIN_SCORE
    assert adjusted["oi_valley_long_eligible"] is False
    assert (
        adjusted["oi_valley_direction_gate_reason"]
        == "4h direction is not long"
    )
    assert (
        "4h direction is not long; OI valley remains observation-only"
        in adjusted["reasons"]
    )


def test_oi_valley_reversal_stop_and_early_invalidation_use_4h_structure() -> None:
    signal = {
        "setup_type": SETUP_OI_VALLEY_REVERSAL_LONG,
        "h4_oi_valley_long": {"stop_anchor": 94.0},
    }
    stop, basis = _refine_stop_with_setup_structure(
        PositionSide.LONG,
        98.0,
        100.0,
        signal,
        {},
        indicator_snapshot(close=100.0, atr=2.0),
        timeframe="4h",
    )
    assert stop == 94.0
    assert basis == "4h_oi_valley_floor_structure"

    position = Position(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
        opened_at=datetime(2026, 7, 14, tzinfo=UTC),
        stop_price=94.0,
        take_profit_1=106.0,
        take_profit_2=112.0,
        metadata={
            "entry_context": {
                "setup_type": SETUP_OI_VALLEY_REVERSAL_LONG,
            }
        },
    )
    reason = _oi_valley_long_invalidation_reason(
        position,
        {
            "h4_oi_valley_long": {
                "current_close": 97.0,
                "floor_price": 94.5,
                "ema55": 99.0,
                "ema55_buffer": 1.0,
                "current_oi_change_pct": 0.02,
            }
        },
    )
    assert reason == "stop loss: 4h closed below EMA55 while OI increased; new shorts likely"


def test_confirmed_four_hour_oi_valley_exits_short_and_needs_wick_for_long() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    short_position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "SHORT",
            margin_usdt=100,
            leverage=5,
            stop_loss=102.0,
            take_profit_1=98.0,
            take_profit_2=96.0,
        )
    )
    assert _oi_valley_short_exit_reason(
        short_position,
        {"h4_oi_valley": {"state": "CONFIRMED"}},
    ) is None

    short_position.metadata["max_favorable_distance"] = 2.1
    assert _oi_valley_short_exit_reason(
        short_position,
        {"h4_oi_valley": {"state": "CONFIRMED"}},
    ) == "take profit: 4h OI valley formed; downside trend exhausted"

    context = {
        "daily_bias": "NEUTRAL",
        "h4_structure": {
            "state": "BOX_LOWER_HALF",
            "direction": "LONG",
            "structure_type": "ASCENDING_SUPPORT",
        },
        "h1_structure": {"support": 99.0, "support_zone_low": 98.5, "support_zone_high": 99.5},
        "h4_ma_cluster": {},
        "h1_ma_cluster": {},
        "h4_oi": {"state": "NORMAL"},
        "h4_oi_valley": {"state": "CONFIRMED"},
        "h1_trigger": {"direction": "LONG", "state": "FAKE_BREAKDOWN"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "h1_ema_reliability": {"state": "RELIABLE"},
        "high_distribution_handoff": {"state": "WATCH"},
        "distribution_short": {"active": False},
        "m15_precision": {},
        "entry_levels": {
            "long": {
                "sweep_reclaim_support": {
                    "low": 98.5,
                    "high": 99.5,
                    "price": 99.0,
                }
            },
            "short": {},
        },
        "summary": "test",
    }
    adjusted_long = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "score": 82,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "reasons": (
                "downside sweep reclaimed support; stop-run filter favors long",
            ),
            "vetoes": (),
            "price": 99.0,
        },
        context,
    )
    assert (
        "4h OI valley formed after retail capitulation; downside wick reclaimed support"
        in adjusted_long["reasons"]
    )
    assert "sweep_reclaim_support" in adjusted_long["entry_levels"]["long"]

    adjusted_short = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "score": 96,
            "trend_state": "TREND_SHORT",
            "trend_stage_phase": "LATE",
            "risk_state": "NORMAL",
            "reasons": (),
            "vetoes": (),
            "price": 99.0,
        },
        context,
    )
    assert (
        "4h OI valley confirmed; low-area short chasing is blocked"
        in adjusted_short["vetoes"]
    )


def test_oi_valley_does_not_block_confirmed_short_at_structural_resistance() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "score": 110,
            "trend_state": "TREND_SHORT",
            "risk_state": "NORMAL",
            "reasons": (),
            "vetoes": (),
            "price": 102.8,
        },
        {
            "daily_bias": "BEAR",
            "h4_structure": {"state": "BOX_LOWER_HALF"},
            "h4_oi_valley": {"state": "CONFIRMED"},
            "h1_trigger": {
                "direction": "SHORT",
                "state": "FAKE_BREAKOUT",
                "rejection_high": 103.2,
            },
            "h1_pullback": {"direction": "NONE", "state": "WAIT"},
            "distribution_short": {
                "active": True,
                "descending_trendline_zone": {
                    "low": 102.0,
                    "high": 103.0,
                    "price": 102.5,
                    "anchor_count": 3,
                },
            },
            "entry_levels": {
                "short": {
                    "descending_high_trendline": {
                        "low": 102.0,
                        "high": 103.0,
                        "price": 102.5,
                    }
                }
            },
            "summary": "test",
        },
    )

    assert not any(
        "low-area short chasing" in reason
        for reason in adjusted["vetoes"]
    )
    assert adjusted["entry_timing"] == "GOOD"


def test_oi_valley_short_block_releases_after_fresh_oi_backed_breakdown() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "score": 105,
            "trend_state": "ONE_WAY_DOWN",
            "risk_state": "NORMAL",
            "oi_change": 0.02,
            "reasons": ("market structure confirms short",),
            "vetoes": (),
            "price": 98.8,
        },
        {
            "daily_bias": "BEAR",
            "h4_structure": {"state": "BREAKDOWN_DOWN"},
            "h4_oi_valley": {"state": "EXHAUSTION"},
            "h1_trigger": {"direction": "SHORT", "state": "BREAKDOWN"},
            "h1_pullback": {"direction": "NONE", "state": "WAIT"},
            "distribution_short": {"active": False},
            "entry_levels": {
                "short": {
                    "breakdown_retest": {
                        "low": 98.5,
                        "high": 99.2,
                        "price": 98.85,
                    }
                }
            },
            "summary": "test",
        },
    )

    assert not any(
        "low-area short chasing" in reason
        for reason in adjusted["vetoes"]
    )


def test_distribution_short_lifecycle_has_three_distinct_stages() -> None:
    signal = {
        "smart_money_phase": "DISTRIBUTION_EXIT",
        "trend_state": "TREND_SHORT",
    }
    base_context = {
        "distribution_short": {
            "active": True,
            "range_high_zone": {"low": 94.0, "high": 95.0, "price": 94.5},
            "descending_trendline_zone": None,
        },
        "h4_structure": {"state": "BOX_LOWER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h4_oi": {"state": "DELEVERAGE_WAIT"},
    }

    assert _distribution_short_stage(signal, base_context) == DISTRIBUTION_STAGE_RANGE

    descending_context = {
        **base_context,
        "distribution_short": {
            **base_context["distribution_short"],
            "descending_trendline_zone": {
                "low": 92.5,
                "high": 93.2,
                "price": 92.85,
            },
        },
    }
    assert (
        _distribution_short_stage(signal, descending_context)
        == DISTRIBUTION_STAGE_DESCENDING
    )

    markdown_context = {
        **descending_context,
        "h4_structure": {"state": "BREAKDOWN_DOWN"},
        "h4_oi": {"state": "DELEVERAGE_BREAKDOWN"},
    }
    assert (
        _distribution_short_stage(signal, markdown_context)
        == DISTRIBUTION_STAGE_MARKDOWN
    )


def test_distribution_structure_projects_the_latest_lower_high_line() -> None:
    start = datetime(2026, 6, 28, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + timedelta(hours=index),
            open=98.0,
            high=100.0,
            low=96.0,
            close=98.0,
            volume=1000.0,
        )
        for index in range(20)
    ]
    candles[5] = Candle(
        timestamp=candles[5].timestamp,
        open=106.0,
        high=110.0,
        low=104.0,
        close=106.0,
        volume=1000.0,
    )
    candles[10] = Candle(
        timestamp=candles[10].timestamp,
        open=103.0,
        high=106.0,
        low=101.0,
        close=103.0,
        volume=1000.0,
    )
    candles[15] = Candle(
        timestamp=candles[15].timestamp,
        open=101.0,
        high=103.0,
        low=99.0,
        close=101.0,
        volume=1000.0,
    )
    structure = _distribution_short_structure(
        candles,
        [indicator_snapshot(close=98.0, atr=1.0) for _ in candles],
        {"drop_from_high_pct": -0.10},
        AppSettings(),
    )

    assert structure["active"]
    line = structure["descending_trendline_zone"]
    assert line["price"] == pytest.approx(100.2)
    assert line["low"] == pytest.approx(99.8)
    assert line["high"] == pytest.approx(100.6)
    assert line["anchor_count"] == 3


def _descending_h4_market() -> tuple[list[Candle], list[IndicatorSnapshot]]:
    start = datetime(2026, 7, 19, tzinfo=UTC)
    candles: list[Candle] = []
    anchor_highs = {3: 110.0, 9: 105.0, 15: 100.0}
    for index in range(22):
        high = anchor_highs.get(index, 101.0 - index * 0.3)
        if index == 21:
            high = 95.2
        close = high - 0.3
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=index * 4),
                open=close - 0.2,
                high=high,
                low=high - 2.0,
                close=close,
                volume=1000.0,
            )
        )
    indicators = [
        indicator_snapshot(
            close=candle.close,
            atr=1.0,
            timestamp=candle.timestamp,
        )
        for candle in candles
    ]
    return candles, indicators


def test_four_hour_structure_separates_descending_direction_from_box_location() -> None:
    candles, indicators = _descending_h4_market()

    structure = _four_hour_structure(
        candles,
        indicators,
        AppSettings(),
    )

    assert structure["state"] == "BOX_LOWER_HALF"
    assert structure["direction"] == "SHORT"
    assert structure["location"] == "RESISTANCE"
    assert structure["structure_type"] == "DESCENDING_RESISTANCE"
    assert structure["descending_anchor_count"] == 3
    assert structure["descending_trendline_zone"]["price"] == pytest.approx(
        95.0
    )


def test_descending_trendline_break_needs_a_closed_retest_before_turning_long() -> None:
    candles, indicators = _descending_h4_market()
    current = candles[-1]
    candles[-1] = Candle(
        timestamp=current.timestamp,
        open=95.2,
        high=96.5,
        low=94.8,
        close=96.0,
        volume=current.volume,
    )
    indicators[-1] = indicator_snapshot(
        close=96.0,
        atr=1.0,
        timestamp=current.timestamp,
    )

    pending = _four_hour_structure(
        candles,
        indicators,
        AppSettings(),
    )

    assert pending["direction"] == "NEUTRAL"
    assert pending["structure_type"] == "DESCENDING_BREAKOUT_PENDING"

    previous = candles[-2]
    candles[-2] = Candle(
        timestamp=previous.timestamp,
        open=95.5,
        high=96.8,
        low=94.8,
        close=96.5,
        volume=previous.volume,
    )
    indicators[-2] = indicator_snapshot(
        close=96.5,
        atr=1.0,
        timestamp=previous.timestamp,
    )
    candles[-1] = Candle(
        timestamp=current.timestamp,
        open=95.6,
        high=95.7,
        low=94.9,
        close=95.1,
        volume=current.volume,
    )
    indicators[-1] = indicator_snapshot(
        close=95.1,
        atr=1.0,
        timestamp=current.timestamp,
    )

    confirmed = _four_hour_structure(
        candles,
        indicators,
        AppSettings(),
    )

    assert confirmed["direction"] == "LONG"
    assert confirmed["location"] == "SUPPORT"
    assert confirmed["structure_type"] == "DESCENDING_BREAKOUT_RETEST"


def test_two_anchor_distribution_projection_needs_resistance_confluence() -> None:
    distribution = {
        "descending_trendline_zone": {
            "low": 102.0,
            "high": 103.0,
            "price": 102.5,
            "anchor_count": 2,
        }
    }

    assert not _distribution_projection_supported(
        distribution,
        {
            "short": {
                "h1_resistance": {
                    "low": 104.0,
                    "high": 105.0,
                    "price": 104.5,
                }
            }
        },
    )
    assert _distribution_projection_supported(
        distribution,
        {
            "short": {
                "h1_resistance": {
                    "low": 102.7,
                    "high": 103.4,
                    "price": 103.0,
                }
            }
        },
    )


def test_distribution_stages_switch_the_only_eligible_short_entry_zone() -> None:
    levels = {
        "short": {
            "distribution_range_high": {
                "low": 94.0,
                "high": 95.0,
                "price": 94.5,
            },
            "descending_high_trendline": {
                "low": 92.5,
                "high": 93.2,
                "price": 92.85,
            },
            "h1_boll_mid": {
                "low": 89.5,
                "high": 90.5,
                "price": 90.0,
            },
            "h1_ema20_ema60": {
                "low": 89.8,
                "high": 90.8,
                "price": 90.3,
            },
        }
    }
    base_signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "smart_money_phase": "DISTRIBUTION_EXIT",
        "reasons": (
            "smart money distribution: repeated upper wicks with OI falling after markup",
        ),
    }

    stage_1 = _scored_entry_levels(
        {
            **base_signal,
            "distribution_short_stage": DISTRIBUTION_STAGE_RANGE,
        },
        levels,
    )
    assert set(stage_1["short"]) == {"distribution_range_high"}

    stage_2 = _scored_entry_levels(
        {
            **base_signal,
            "distribution_short_stage": DISTRIBUTION_STAGE_DESCENDING,
        },
        levels,
    )
    assert set(stage_2["short"]) == {"descending_high_trendline"}

    stage_3 = _scored_entry_levels(
        {
            **base_signal,
            "distribution_short_stage": DISTRIBUTION_STAGE_MARKDOWN,
        },
        levels,
    )
    assert set(stage_3["short"]) == {
        "h1_boll_mid",
        "h1_ema20_ema60",
    }


def test_distribution_stage_two_cannot_be_overridden_by_generic_four_hour_ema() -> None:
    selected = _scored_entry_levels(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "trend_state": "ONE_WAY_DOWN",
            "entry_timeframe_override": "4h",
            "distribution_short_stage": DISTRIBUTION_STAGE_DESCENDING,
            "setup_type": SETUP_H4_PULLBACK_SHORT,
        },
        {
            "short": {
                "descending_high_trendline": {
                    "low": 102.0,
                    "high": 103.0,
                    "price": 102.5,
                },
                "h4_ema20_ema60": {
                    "low": 99.0,
                    "high": 100.0,
                    "price": 99.5,
                },
            }
        },
    )

    assert selected == {
        "short": {
            "descending_high_trendline": {
                "low": 102.0,
                "high": 103.0,
                "price": 102.5,
            }
        }
    }


def test_distribution_stage_two_waits_for_rejection_at_descending_high() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 110,
        "trend_state": "ONE_WAY_DOWN",
        "risk_state": "NORMAL",
        "price": 102.8,
        "distribution_short_stage": DISTRIBUTION_STAGE_DESCENDING,
        "entry_levels": {
            "short": {
                "descending_high_trendline": {
                    "low": 102.0,
                    "high": 103.0,
                    "price": 102.5,
                }
            }
        },
    }

    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["h1_pullback"] = {
        "direction": "SHORT",
        "state": "HEALTHY_PULLBACK",
    }
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["h1_trigger"] = {
        "direction": "SHORT",
        "state": "FAKE_BREAKOUT",
        "rejection_high": 103.2,
    }
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"


def test_distribution_stage_two_wick_must_touch_and_close_below_projection() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 110,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "price": 102.4,
        "distribution_short_stage": DISTRIBUTION_STAGE_DESCENDING,
        "distribution_short": {
            "descending_trendline_zone": {
                "low": 102.0,
                "high": 103.0,
                "price": 102.5,
            }
        },
        "entry_levels": {
            "short": {
                "descending_high_trendline": {
                    "low": 102.0,
                    "high": 103.0,
                    "price": 102.5,
                }
            }
        },
        "large_wick_rejections": {
            "upper": {"high": 101.8, "close": 101.2},
        },
    }

    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "WAIT"

    signal["large_wick_rejections"] = {
        "upper": {"high": 103.2, "close": 102.4},
    }
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"


def test_stage_one_distribution_short_is_penalized_and_tiny() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 100,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "smart_money_phase": "DISTRIBUTION_EXIT",
        "reasons": (
            "smart money distribution: repeated upper wicks with OI falling after markup",
        ),
        "vetoes": (),
    }
    context = {
        "daily_bias": "NEUTRAL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_structure": {},
        "h4_ma_cluster": {},
        "h1_ma_cluster": {},
        "h4_oi": {
            "state": "DELEVERAGE_WAIT",
            "drop_from_high_pct": -0.10,
        },
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "distribution_short": {
            "active": True,
            "range_high_zone": {
                "low": 94.0,
                "high": 95.0,
                "price": 94.5,
            },
            "descending_trendline_zone": None,
        },
        "m15_precision": {},
        "entry_levels": {
            "short": {
                "distribution_range_high": {
                    "low": 94.0,
                    "high": 95.0,
                    "price": 94.5,
                },
                "h1_ema20_ema60": {
                    "low": 90.0,
                    "high": 91.0,
                    "price": 90.5,
                },
            }
        },
        "summary": "test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["distribution_short_stage"] == DISTRIBUTION_STAGE_RANGE
    assert adjusted["leverage_cap"] == 3
    assert adjusted["margin_factor"] == 0.25
    assert adjusted["score"] <= 92
    assert set(adjusted["entry_levels"]["short"]) == {
        "distribution_range_high"
    }


def test_distribution_stage_stop_sits_above_the_active_price_structure() -> None:
    signal = {
        "distribution_short_stage": DISTRIBUTION_STAGE_DESCENDING,
        "h1_trigger": {
            "direction": "SHORT",
            "state": "RETEST",
            "rejection_high": 93.4,
        },
        "entry_levels": {
            "short": {
                "descending_high_trendline": {
                    "low": 92.5,
                    "high": 93.2,
                    "price": 92.85,
                }
            }
        },
    }

    refined = _refine_stop_with_distribution_stage(
        PositionSide.SHORT,
        98.0,
        92.8,
        signal,
        indicator_snapshot(close=92.8, atr=1.0),
    )

    assert refined == pytest.approx(93.9136)
    assert refined > 93.4


def test_new_signal_version_does_not_reset_entry_position_inside_zone() -> None:
    signal = {
        "timestamp": "2026-07-01T00:00:00+00:00",
        "action": SignalAction.ENTRY_LONG.value,
        "score": 90,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
    }
    _update_entry_position_fields(signal)
    _update_entry_position_fields(signal)
    assert signal["entry_timing"] == "GOOD"

    signal["timestamp"] = "2026-07-01T01:00:00+00:00"
    _update_entry_position_fields(signal)

    assert signal["entry_timing"] == "GOOD"


def test_non_entry_signal_timing_is_not_excellent() -> None:
    timing, reason = _signal_entry_timing({"action": SignalAction.WATCH.value, "score": 90})
    assert timing == "BLOCK"
    assert "directional entry signal not established" in reason

    timing, reason = _signal_entry_timing({"action": SignalAction.NO_TRADE.value, "score": 90})
    assert timing == "BLOCK"
    assert "no trade signal" in reason


def test_entry_position_only_compares_price_with_zone_in_late_stage() -> None:
    late_long = {
        "action": "ENTRY_LONG",
        "score": 120,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "price": 100.0,
        "rsi14": 93.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }

    timing, reason = _signal_entry_timing(late_long)

    assert timing == "GOOD"
    assert reason == "已到优势入场区"
    assert _auto_signal_allowed(late_long)


def test_entry_reward_r_requires_enough_target_space() -> None:
    signal = {
        "h1_structure": {
            "resistance_zone_low": 101.0,
            "resistance": 101.2,
            "resistance_zone_high": 101.4,
        }
    }

    assert _entry_reward_r(signal, PositionSide.LONG, price=100.0, stop=99.0) == pytest.approx(1.2)
    assert _entry_reward_r(signal, PositionSide.LONG, price=100.0, stop=99.5) == pytest.approx(2.4)


def test_entry_reward_r_uses_target_from_entry_timeframe() -> None:
    signal = {
        "h1_structure": {"resistance": 101.0},
        "h4_structure": {"resistance": 108.0},
    }

    assert _entry_reward_r(
        signal,
        PositionSide.LONG,
        price=100.0,
        stop=99.0,
        timeframe="1h",
    ) == 1.0
    assert _entry_reward_r(
        signal,
        PositionSide.LONG,
        price=100.0,
        stop=96.0,
        timeframe="4h",
    ) == 2.0


def test_entry_reward_r_uses_planned_target_when_structure_is_unavailable() -> None:
    assert _entry_reward_r(
        {},
        PositionSide.LONG,
        price=100.0,
        stop=99.0,
        timeframe="1h",
        planned_target=101.8,
    ) == pytest.approx(1.8)


def test_entry_reward_r_uses_actual_planned_tp2_over_nearer_structure() -> None:
    assert _entry_reward_r(
        {"h1_structure": {"resistance": 100.1}},
        PositionSide.LONG,
        price=100.0,
        stop=99.0,
        timeframe="1h",
        planned_target=101.2,
    ) == pytest.approx(1.2)


def test_exit_plan_preserves_one_point_two_r_after_trading_costs() -> None:
    take_profit_1, take_profit_2 = _take_profits_for_final_stop(
        {"h1_structure": {"resistance": 100.1}},
        PositionSide.LONG,
        100.0,
        99.0,
        timeframe="1h",
        round_trip_cost_rate=0.001,
    )

    assert take_profit_1 == pytest.approx(100.8)
    assert _entry_reward_r(
        {},
        PositionSide.LONG,
        price=100.0,
        stop=99.0,
        timeframe="1h",
        planned_target=take_profit_2,
        round_trip_cost_rate=0.001,
    ) == pytest.approx(1.2)


def test_15m_take_profit_is_rebuilt_from_final_15m_stop() -> None:
    signal = {
        "h1_structure": {"resistance": 120.0},
    }

    take_profit_1, take_profit_2 = _take_profits_for_final_stop(
        signal,
        PositionSide.LONG,
        100.0,
        98.0,
        timeframe="15m",
    )

    assert take_profit_1 == 101.6
    assert take_profit_2 == 104.0
    assert _entry_reward_r(
        signal,
        PositionSide.LONG,
        100.0,
        98.0,
        timeframe="15m",
        planned_target=take_profit_2,
    ) == 2.0


@pytest.mark.parametrize(
    ("side", "target", "expected_tp1", "expected_tp2"),
    (
        (PositionSide.LONG, 100.01, 100.01, 101.2),
        (PositionSide.SHORT, 99.99, 99.99, 98.8),
    ),
)
def test_near_structure_target_becomes_partial_not_full_exit(
    side: PositionSide,
    target: float,
    expected_tp1: float,
    expected_tp2: float,
) -> None:
    structure = (
        {"resistance": target}
        if side == PositionSide.LONG
        else {"support": target}
    )

    take_profit_1, take_profit_2 = _take_profits_for_final_stop(
        {"h1_structure": structure},
        side,
        100.0,
        99.0 if side == PositionSide.LONG else 101.0,
        timeframe="1h",
    )

    assert take_profit_1 == pytest.approx(expected_tp1)
    assert take_profit_2 == pytest.approx(expected_tp2)
    assert (
        _exit_plan_error(
            side,
            100.0,
            99.0 if side == PositionSide.LONG else 101.0,
            take_profit_1,
            take_profit_2,
        )
        is None
    )


def test_1h_take_profit_does_not_borrow_a_4h_structure_target() -> None:
    signal = {
        "h4_structure": {"support": 95.0},
    }

    take_profit_1, take_profit_2 = _take_profits_for_final_stop(
        signal,
        PositionSide.SHORT,
        100.0,
        101.0,
        timeframe="1h",
    )

    assert take_profit_1 == 99.2
    assert take_profit_2 == 98.0


def test_exit_plan_postcondition_validates_stop_and_take_profit_direction() -> None:
    assert _exit_plan_error(
        PositionSide.LONG,
        100.0,
        99.0,
        101.2,
        102.0,
    ) is None
    assert _exit_plan_error(
        PositionSide.SHORT,
        100.0,
        101.0,
        98.8,
        98.0,
    ) is None
    assert "invalid stop loss" in str(_exit_plan_error(
        PositionSide.LONG,
        100.0,
        101.0,
        102.0,
        103.0,
    ))
    assert "take profit 1 must be above entry price" == _exit_plan_error(
        PositionSide.LONG,
        100.0,
        99.0,
        99.5,
        102.0,
    )


def test_entry_timeframe_follows_zone_actually_hit() -> None:
    signal = {
        "score": 95,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "smart_money_phase": "SHORT_SQUEEZE_MARKUP",
        "m15_precision": {
            "pullback": "M15_LONG_PULLBACK",
            "long_stop_anchor": 98.0,
            "trend": "UP",
        },
        "entry_levels": {
            "long": {
                "m15_ema20_ema60": {
                    "low": 99.8,
                    "high": 100.2,
                    "price": 100.0,
                },
                "h1_support": {
                    "low": 97.8,
                    "high": 98.2,
                    "price": 98.0,
                },
                "h4_support": {
                    "low": 94.5,
                    "high": 95.5,
                    "price": 95.0,
                },
            }
        }
    }

    assert _entry_timeframe_for_signal(
        signal,
        PositionSide.LONG,
        100.0,
    ) == "15m"
    assert _entry_timeframe_for_signal(
        signal,
        PositionSide.LONG,
        98.0,
    ) == "1h"
    assert _entry_timeframe_for_signal(
        signal,
        PositionSide.LONG,
        95.0,
    ) == "4h"
    short_m15_only = {
        "entry_levels": {
            "short": {
                "m15_ema20_ema60": {
                    "low": 99.8,
                    "high": 100.2,
                    "price": 100.0,
                }
            }
        }
    }
    assert _entry_timeframe_for_signal(
        short_m15_only,
        PositionSide.SHORT,
        100.0,
    ) == "1h"


def test_repeated_1h_ema_sweeps_mark_the_timeframe_unreliable() -> None:
    start = datetime(2026, 6, 26, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + timedelta(hours=index),
            open=100.0,
            high=101.0,
            low=100.0,
            close=100.0,
            volume=1000.0,
        )
        for index in range(70)
    ]
    for index in (52, 60):
        candles[index] = Candle(
            timestamp=candles[index].timestamp,
            open=100.0,
            high=101.0,
            low=98.0,
            close=100.0,
            volume=1000.0,
        )
    reliability = _one_hour_ema_reliability(
        candles,
        [
            indicator_snapshot(
                close=100.0,
                atr=1.0,
                ema20=100.0,
            )
            for _ in candles
        ],
    )

    assert reliability["state"] == "UNRELIABLE"
    assert reliability["ema20_breach_episodes"] == 2
    assert reliability["ema60_breach_episodes"] == 1


def test_repeated_1h_ema_resistance_sweeps_mark_short_entries_unreliable() -> None:
    start = datetime(2026, 6, 26, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + timedelta(hours=index),
            open=100.0,
            high=100.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        )
        for index in range(70)
    ]
    for index in (52, 60):
        candles[index] = Candle(
            timestamp=candles[index].timestamp,
            open=100.0,
            high=102.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        )
    reliability = _one_hour_ema_reliability(
        candles,
        [
            indicator_snapshot(
                close=100.0,
                atr=1.0,
                ema20=100.0,
            )
            for _ in candles
        ],
    )

    assert reliability["short_state"] == "UNRELIABLE"
    assert reliability["ema20_resistance_breach_episodes"] == 2
    assert reliability["ema60_resistance_breach_episodes"] == 1


def test_unreliable_1h_long_promotes_entry_stop_and_target_to_4h() -> None:
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 92,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "smart_money_phase": "NONE",
        "reasons": (),
        "vetoes": (),
        "price": 95.0,
    }
    context = {
        "daily_bias": "NEUTRAL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_structure": {},
        "h4_ma_cluster": {},
        "h1_ma_cluster": {},
        "h4_oi": {"state": "NORMAL"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "h1_ema_reliability": {
            "state": "UNRELIABLE",
            "ema20_breach_episodes": 2,
            "ema60_breach_episodes": 1,
        },
        "distribution_short": {"active": False},
        "m15_precision": {},
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 98.0,
                    "high": 99.0,
                    "price": 98.5,
                },
                "h4_support": {
                    "low": 94.5,
                    "high": 95.5,
                    "price": 95.0,
                },
                "h4_boll_mid": {
                    "low": 94.6,
                    "high": 95.4,
                    "price": 95.0,
                },
                "h4_ema20_ema60": {
                    "low": 94.7,
                    "high": 95.3,
                    "price": 95.0,
                },
            }
        },
        "summary": "test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["entry_timeframe_override"] == "4h"
    assert set(adjusted["entry_levels"]["long"]) == {
        "h4_support",
        "h4_boll_mid",
        "h4_ema20_ema60",
    }
    assert (
        _entry_timeframe_for_signal(
            adjusted,
            PositionSide.LONG,
            95.0,
        )
        == "4h"
    )


def test_unreliable_1h_short_promotes_entry_stop_and_target_to_4h() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 92,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
        "price": 105.0,
    }
    context = {
        "daily_bias": "NEUTRAL",
        "h4_structure": {"state": "RANGE_MID"},
        "h1_structure": {},
        "h4_ma_cluster": {},
        "h1_ma_cluster": {},
        "h4_oi": {"state": "NORMAL"},
        "h4_oi_valley": {"state": "WATCH"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "h1_ema_reliability": {
            "short_state": "UNRELIABLE",
            "ema20_resistance_breach_episodes": 2,
            "ema60_resistance_breach_episodes": 1,
        },
        "distribution_short": {"active": False},
        "m15_precision": {},
        "entry_levels": {
            "short": {
                "h1_resistance": {
                    "low": 101.0,
                    "high": 102.0,
                    "price": 101.5,
                },
                "h4_resistance": {
                    "low": 104.5,
                    "high": 105.5,
                    "price": 105.0,
                },
                "h4_boll_mid": {
                    "low": 104.6,
                    "high": 105.4,
                    "price": 105.0,
                },
                "h4_ema20_ema60": {
                    "low": 104.7,
                    "high": 105.3,
                    "price": 105.0,
                },
            }
        },
        "summary": "test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["entry_timeframe_override"] == "4h"
    assert set(adjusted["entry_levels"]["short"]) == {
        "h4_resistance",
        "h4_boll_mid",
        "h4_ema20_ema60",
    }
    assert (
        _entry_timeframe_for_signal(
            adjusted,
            PositionSide.SHORT,
            105.0,
        )
        == "4h"
    )


def test_short_squeeze_long_keeps_the_15m_tactical_exception() -> None:
    signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 92,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "smart_money_phase": "SHORT_SQUEEZE_MARKUP",
        "reasons": (),
        "vetoes": (),
        "price": 100.0,
    }
    context = {
        "daily_bias": "NEUTRAL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_structure": {},
        "h4_ma_cluster": {},
        "h1_ma_cluster": {},
        "h4_oi": {"state": "NORMAL"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "h1_ema_reliability": {"state": "UNRELIABLE"},
        "distribution_short": {"active": False},
        "m15_precision": {
            "pullback": "M15_LONG_PULLBACK",
            "trend": "UP",
            "long_stop_anchor": 98.5,
        },
        "entry_levels": {
            "long": {
                "m15_ema20_ema60": {
                    "low": 99.5,
                    "high": 100.5,
                    "price": 100.0,
                },
                "h4_support": {
                    "low": 94.5,
                    "high": 95.5,
                    "price": 95.0,
                },
            }
        },
        "summary": "test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "entry_timeframe_override" not in adjusted
    assert (
        _entry_timeframe_for_signal(
            adjusted,
            PositionSide.LONG,
            100.0,
        )
        == "15m"
    )


def test_strategy_signal_timeframe_is_limited_to_1h_or_4h() -> None:
    assert _entry_signal_timeframe("15m") == "1h"
    assert _entry_signal_timeframe("1h") == "1h"
    assert _entry_signal_timeframe("4h") == "4h"
    assert _entry_signal_timeframe("1d") == "1h"


def test_auto_signal_waits_for_1h_4h_retest_in_one_way_down_short() -> None:
    signal = {
        "action": "ENTRY_SHORT",
        "score": 96,
        "trend_state": "ONE_WAY_DOWN",
        "risk_state": "NORMAL",
        "price": 254.0,
        "m15_precision": {"pullback": "M15_SHORT_PULLBACK", "short_stop_anchor": 263.0, "trend": "DOWN"},
        "entry_levels": {
            "short": {
                "h1_resistance": {"low": 260.0, "high": 262.0, "price": 261.0},
                "m15_ema20_ema60": {"low": 253.5, "high": 254.5, "price": 254.0},
            }
        },
    }
    timing, reason = _signal_entry_timing(signal)

    assert timing == "WAIT"
    assert reason == "等待 1H压力反抽区"
    assert not _auto_signal_allowed(signal)


def test_short_at_or_below_closed_h1_boll_lower_waits_for_bounce() -> None:
    signal = {
        "action": "ENTRY_SHORT",
        "score": 96,
        "trend_state": "ONE_WAY_DOWN",
        "risk_state": "NORMAL",
        "price": 97.8,
        "entry_guardrails": {"h1_boll_lower": 98.0},
        "entry_levels": {
            "short": {
                "breakdown_retest": {
                    "low": 97.5,
                    "high": 99.5,
                    "price": 98.5,
                },
            },
        },
    }

    timing, reason = _signal_entry_timing(signal)

    assert timing == "WAIT"
    assert reason == "等待 1H/4H压力反抽区"


def test_short_above_h1_boll_lower_can_use_scored_retest_zone() -> None:
    signal = {
        "action": "ENTRY_SHORT",
        "score": 96,
        "trend_state": "ONE_WAY_DOWN",
        "risk_state": "NORMAL",
        "price": 98.5,
        "entry_guardrails": {"h1_boll_lower": 98.0},
        "entry_levels": {
            "short": {
                "breakdown_retest": {
                    "low": 97.5,
                    "high": 99.5,
                    "price": 98.5,
                },
            },
        },
    }

    timing, _ = _signal_entry_timing(signal)

    assert timing == "GOOD"


def test_entry_position_does_not_duplicate_m15_stop_validation() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 96,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "smart_money_phase": "SHORT_SQUEEZE_MARKUP",
        "price": 100.0,
        "m15_precision": {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 99.6, "trend": "UP"},
        "entry_levels": {
            "long": {
                "m15_ema20_ema60": {"low": 99.5, "high": 100.5, "price": 100.0},
            }
        },
    }

    timing, reason = _signal_entry_timing(signal)

    assert timing == "GOOD"
    assert reason == "已到优势入场区"
    assert _auto_signal_allowed(signal)


def test_auto_trade_skips_m15_entry_when_stop_would_fall_back_to_wider_timeframe() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    symbol = "TESTUSDT"
    engine.latest_prices[symbol] = 100.0
    engine.latest_indicators[symbol] = [indicator_snapshot(close=100.0, atr=3.0)]
    engine.latest_timeframe_indicators[symbol] = {"1h": [indicator_snapshot(close=100.0, atr=3.0)]}
    engine.latest_timeframe_contexts[symbol] = {
        "m15_precision": {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 98.5, "trend": "UP"}
    }
    engine.latest_signals[symbol] = {
        "action": "ENTRY_LONG",
        "score": 96,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "smart_money_phase": "SHORT_SQUEEZE_MARKUP",
        "price": 100.0,
        "m15_precision": {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 98.5, "trend": "UP"},
        "entry_levels": {
            "long": {
                "m15_ema20_ema60": {"low": 99.5, "high": 100.5, "price": 100.0},
            }
        },
    }

    import asyncio

    asyncio.run(engine._auto_trade_once())

    assert not engine.account.positions


def test_auto_trade_uses_15m_only_for_short_squeeze_long() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    symbol = "TESTUSDT"
    m15_indicator = indicator_snapshot(close=100.0, atr=1.0)
    engine.latest_prices[symbol] = 100.0
    engine.latest_indicators[symbol] = [indicator_snapshot(close=100.0, atr=1.5)]
    engine.latest_timeframe_indicators[symbol] = {
        "15m": [m15_indicator],
        "1h": [indicator_snapshot(close=100.0, atr=1.5)],
    }
    precision = {
        "pullback": "M15_LONG_PULLBACK",
        "long_stop_anchor": 98.5,
        "trend": "UP",
    }
    engine.latest_timeframe_contexts[symbol] = {"m15_precision": precision}
    engine.latest_signals[symbol] = {
        "action": "ENTRY_LONG",
        "score": 96,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "SHORT_CROWD",
        "smart_money_phase": "SHORT_SQUEEZE_MARKUP",
        "price": 100.0,
        "m15_precision": precision,
        "entry_levels": {
            "long": {
                "m15_ema20_ema60": {
                    "low": 99.5,
                    "high": 100.5,
                    "price": 100.0,
                }
            }
        },
        "h1_structure": {"resistance": 105.0},
    }

    asyncio.run(engine._auto_trade_once())

    position = engine.account.positions[symbol]
    assert position.metadata["entry_context"]["stop_timeframe"] == "15m"
    assert position.stop_price == 98.5


def test_multi_timeframe_context_adjusts_score_and_veto() -> None:
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
    _update_entry_position_fields(adjusted)

    assert adjusted["score"] == 64
    assert "1h trigger opposes long entry" in adjusted["vetoes"]
    assert "high area without pullback confirmation; wait for 1h/4h pullback before long" in adjusted["vetoes"]


def test_large_timeframe_wick_rejections_use_close_ratio() -> None:
    timestamp = datetime.now(UTC)
    rejections = _large_timeframe_wick_rejections(
        {
            "1h": [
                Candle(
                    timestamp=timestamp,
                    open=100.0,
                    high=102.0,
                    low=99.0,
                    close=100.0,
                    volume=1_000.0,
                )
            ],
            "4h": [
                Candle(
                    timestamp=timestamp,
                    open=100.0,
                    high=101.0,
                    low=97.5,
                    close=100.0,
                    volume=1_000.0,
                )
            ],
        }
    )

    assert rejections["upper"]["timeframe"] == "1h"
    assert rejections["upper"]["ratio"] == pytest.approx(0.02)
    assert rejections["lower"]["timeframe"] == "4h"
    assert rejections["lower"]["ratio"] == pytest.approx(0.025)


def test_large_timeframe_wick_rejection_is_hard_veto_for_chasing() -> None:
    long_signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 100,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    long_context = {
        "daily_bias": "NEUTRAL",
        "large_wick_rejections": {
            "upper": {"timeframe": "1h", "ratio": 0.021},
        },
        "summary": "MTF: test",
    }

    long_adjusted = _apply_multi_timeframe_context(long_signal, long_context)

    assert "1H上插针收实2.1%，禁止追多" in long_adjusted["vetoes"]

    short_signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 100,
        "risk_state": "NORMAL",
        "reasons": (),
        "vetoes": (),
    }
    short_context = {
        "daily_bias": "NEUTRAL",
        "large_wick_rejections": {
            "lower": {"timeframe": "4h", "ratio": 0.025},
        },
        "summary": "MTF: test",
    }

    short_adjusted = _apply_multi_timeframe_context(short_signal, short_context)

    assert "4H下插针收实2.5%，禁止追空" in short_adjusted["vetoes"]


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
    _update_entry_position_fields(adjusted)

    assert adjusted["score"] == 95
    assert "4h structure supports upside" not in adjusted["reasons"]
    assert adjusted["h4_direction"] == "NEUTRAL"
    assert adjusted["direction_gate"] == "NEUTRAL"
    assert "1h BOLL/EMA pullback held with clean risk" in adjusted["reasons"]
    assert not adjusted["vetoes"]


def test_descending_four_hour_resistance_blocks_long_without_flipping_short() -> None:
    candles, indicators = _descending_h4_market()
    h4 = _four_hour_structure(candles, indicators, AppSettings())
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "score": 100,
            "risk_state": "NORMAL",
            "price": 95.1,
            "reasons": (),
            "vetoes": (),
        },
        {
            "daily_bias": "NEUTRAL",
            "h4_structure": h4,
            "h1_trigger": {
                "direction": "LONG",
                "state": "RETEST",
            },
            "entry_levels": {
                "long": {
                    "breakout_retest": {
                        "low": 94.8,
                        "high": 95.3,
                        "price": 95.0,
                    }
                }
            },
            "summary": "MTF: FIL-like descending resistance",
        },
    )

    assert adjusted["action"] == SignalAction.WATCH.value
    assert adjusted["direction_gate"] == "SHORT_ONLY"
    assert adjusted["h4_structure_type"] == "DESCENDING_RESISTANCE"
    assert adjusted["direction_veto_reason"] in adjusted["vetoes"]
    assert adjusted["action"] != SignalAction.ENTRY_SHORT.value


def test_descending_four_hour_resistance_short_uses_only_its_primary_zone() -> None:
    candles, indicators = _descending_h4_market()
    h4 = _four_hour_structure(candles, indicators, AppSettings())
    descending_zone = h4["descending_trendline_zone"]
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "score": 90,
            "risk_state": "NORMAL",
            "price": 95.1,
            "reasons": (),
            "vetoes": (),
        },
        {
            "daily_bias": "NEUTRAL",
            "h4_structure": h4,
            "h1_trigger": {
                "direction": "SHORT",
                "state": "RETEST",
                "rejection_high": 95.3,
            },
            "entry_levels": {
                "short": {
                    "h4_descending_resistance": descending_zone,
                    "vwap_retest": {
                        "low": 93.0,
                        "high": 94.0,
                        "price": 93.5,
                    },
                }
            },
            "summary": "MTF: FIL-like descending resistance",
        },
    )

    assert adjusted["action"] == SignalAction.ENTRY_SHORT.value
    assert adjusted["setup_type"] == SETUP_H4_DESCENDING_RESISTANCE_SHORT
    assert set(adjusted["entry_levels"]["short"]) == {
        "h4_descending_resistance"
    }
    assert adjusted["entry_timeframe_override"] == "4h"
    assert adjusted["entry_state"] == "READY"
    assert adjusted["entry_trigger"] == "1H_RETEST"
    assert not adjusted["vetoes"]
    assert _auto_signal_allowed(adjusted)


def test_descending_four_hour_resistance_touch_alone_is_not_an_entry_trigger() -> None:
    candles, indicators = _descending_h4_market()
    h4 = _four_hour_structure(candles, indicators, AppSettings())
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "score": 100,
            "risk_state": "NORMAL",
            "price": 95.1,
            "reasons": (),
            "vetoes": (),
        },
        {
            "daily_bias": "NEUTRAL",
            "h4_structure": h4,
            "h1_trigger": {"direction": "NONE", "state": "WAIT"},
            "entry_levels": {
                "short": {
                    "h4_descending_resistance": h4[
                        "descending_trendline_zone"
                    ],
                }
            },
            "summary": "MTF: FIL-like descending resistance",
        },
    )

    assert adjusted["action"] == SignalAction.ENTRY_SHORT.value
    assert adjusted["entry_state"] == "POSITION_PENDING"
    assert adjusted["entry_timing"] != "GOOD"
    assert not _auto_signal_allowed(adjusted)


def test_descending_four_hour_resistance_accepts_m15_short_rejection() -> None:
    candles, indicators = _descending_h4_market()
    h4 = _four_hour_structure(candles, indicators, AppSettings())
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_SHORT.value,
            "score": 100,
            "risk_state": "NORMAL",
            "price": 95.1,
            "reasons": (),
            "vetoes": (),
        },
        {
            "daily_bias": "NEUTRAL",
            "h4_structure": h4,
            "h1_trigger": {"direction": "NONE", "state": "WAIT"},
            "m15_precision": {
                "trend": "DOWN",
                "pullback": "M15_SHORT_PULLBACK",
                "short_stop_anchor": 95.5,
            },
            "entry_levels": {
                "short": {
                    "h4_descending_resistance": h4[
                        "descending_trendline_zone"
                    ],
                }
            },
            "summary": "MTF: FIL-like descending resistance",
        },
    )

    assert adjusted["entry_state"] == "READY"
    assert adjusted["entry_trigger"] == "15M_M15_SHORT_PULLBACK"
    assert _auto_signal_allowed(adjusted)


def test_unconfirmed_four_hour_trendline_break_waits_for_closed_retest() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "score": 100,
            "risk_state": "NORMAL",
            "reasons": (),
            "vetoes": (),
        },
        {
            "h4_structure": {
                "state": "BOX_UPPER_HALF",
                "direction": "NEUTRAL",
                "location": "BREAKOUT",
                "structure_type": "DESCENDING_BREAKOUT_PENDING",
            },
            "h1_trigger": {
                "direction": "LONG",
                "state": "BREAKOUT",
            },
            "summary": "MTF: breakout pending retest",
        },
    )

    assert adjusted["action"] == SignalAction.WATCH.value
    assert adjusted["direction_gate"] == "WAIT_BREAKOUT_RETEST"
    assert (
        "4h trendline break is pending a closed-candle retest"
        in adjusted["vetoes"]
    )


def test_legacy_four_hour_breakdown_state_still_maps_to_short_direction() -> None:
    adjusted = _apply_multi_timeframe_context(
        {
            "action": SignalAction.ENTRY_LONG.value,
            "score": 100,
            "risk_state": "NORMAL",
            "reasons": (),
            "vetoes": (),
        },
        {
            "h4_structure": {"state": "BREAKDOWN_DOWN"},
            "h1_trigger": {"direction": "NONE", "state": "WAIT"},
            "summary": "MTF: legacy structure payload",
        },
    )

    assert adjusted["h4_direction"] == "SHORT"
    assert adjusted["direction_gate"] == "SHORT_ONLY"
    assert adjusted["action"] == SignalAction.WATCH.value


def test_high_score_without_core_entry_structure_is_capped_below_auto_entry() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 110,
        "risk_state": "NORMAL",
        "price": 100.0,
        "reasons": (
            "funding rate is not overheated for longs",
            "volume is acceptable but not strong",
        ),
        "vetoes": (),
    }
    context = {
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                },
            }
        },
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert adjusted["score"] == 79
    assert adjusted["action"] == SignalAction.WATCH.value
    assert adjusted["entry_levels"] == {}
    assert (
        "core entry structure absent; score capped below auto-entry threshold"
        in adjusted["reasons"]
    )


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
        "price": 100.0,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "LONG", "state": "RETEST"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "entry_levels": {
            "long": {
                "breakout_retest": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)
    _update_entry_position_fields(adjusted)

    assert "high area without pullback confirmation; wait for 1h/4h pullback before long" not in adjusted["vetoes"]
    assert _auto_signal_allowed(adjusted)


def test_one_way_high_area_allows_15m_boll_ema9_pullback() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 90,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "smart_money_phase": "SHORT_SQUEEZE_MARKUP",
        "price": 97.2,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "m15_precision": {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 95.8, "trend": "UP"},
        "entry_levels": {"long": {"m15_ema20_ema60": {"low": 96.8, "high": 97.6, "price": 97.2}}},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)
    _update_entry_position_fields(adjusted)

    assert "high area without pullback confirmation; wait for 1h/4h pullback before long" not in adjusted["vetoes"]
    assert "one-way uptrend 15m BOLL/EMA9 pullback confirmed; allow tactical long" in adjusted["reasons"]
    assert adjusted["m15_precision"]["long_stop_anchor"] == 95.8
    assert _auto_signal_allowed(adjusted)


def test_normal_rsi_overheated_waits_for_1h_4h_pullback() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 95,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "rsi14": 82,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_LOWER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "normal/chop trend RSI overheated; wait for 1h/4h pullback before long" in adjusted["vetoes"]
    assert not _auto_signal_allowed(adjusted)


def test_one_way_hot_rsi_requires_1h_or_15m_pullback() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 96,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "smart_money_phase": "SHORT_SQUEEZE_MARKUP",
        "rsi14": 88,
        "reasons": (),
        "vetoes": (),
    }
    no_pullback_context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_LOWER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "summary": "MTF: test",
    }
    pullback_context = {
        **no_pullback_context,
        "m15_precision": {"pullback": "M15_LONG_PULLBACK", "long_stop_anchor": 95.8, "trend": "UP"},
        "entry_levels": {"long": {"m15_ema20_ema60": {"low": 96.8, "high": 97.6, "price": 97.2}}},
    }
    signal = {**signal, "price": 97.2}

    blocked = _apply_multi_timeframe_context(signal, no_pullback_context)
    allowed = _apply_multi_timeframe_context(signal, pullback_context)
    _update_entry_position_fields(allowed)

    assert "one-way uptrend RSI hot without 1h/15m pullback; wait before long" in blocked["vetoes"]
    assert not _auto_signal_allowed(blocked)
    assert "one-way uptrend RSI hot, but 1h/15m pullback confirmed" in allowed["reasons"]
    assert _auto_signal_allowed(allowed)


def test_one_way_extreme_rsi_blocks_fresh_continuation_entry() -> None:
    signal = {
        "action": "ENTRY_SHORT",
        "score": 105,
        "trend_state": "ONE_WAY_DOWN",
        "risk_state": "NORMAL",
        "rsi14": 7,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BEAR",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "SHORT", "state": "HEALTHY_PULLBACK"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "one-way downtrend RSI below 8; skip fresh short and protect existing profit" in adjusted["vetoes"]
    assert not _auto_signal_allowed(adjusted)


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
    assert "MA cluster dense; wait for breakout or MA20 retest: price=1.29" in reasons
    assert not vetoes


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


def test_oi_deleverage_hold_long_is_only_an_event_until_valley_rebuild() -> None:
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

    assert "leverage_cap" not in adjusted
    assert "margin_factor" not in adjusted
    assert (
        "4h OI sharp drop is an event, not a confirmed OI valley; wait for OI rebuilding and downside-wick reclaim"
        in adjusted["reasons"]
    )


def test_oi_deleverage_with_long_short_ratio_rising_vetoes_long_during_retail_carry() -> None:
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
    assert (
        "4h OI dropped while long/short ratio rose; retail longs are carrying the decline"
        in adjusted["vetoes"]
    )
    assert not _auto_signal_allowed(adjusted)


def test_weak_oi_rebound_vetoes_long_pullback() -> None:
    signal = {
        "action": "ENTRY_LONG",
        "score": 98,
        "risk_state": "NORMAL",
        "volume_ratio": 0.7,
        "oi_change": -0.01,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h4_oi": {"state": "DELEVERAGE_HOLD_LONG", "drop_from_high_pct": -0.22, "rebound_pct": 0.0},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert "4h OI drained and volume is weak; EMA/BOLL bounce is not a clean long pullback" in adjusted["vetoes"]
    assert not _auto_signal_allowed(adjusted)


def test_weak_oi_rebound_with_ma_pressure_does_not_use_oi_to_improve_short() -> None:
    signal = {
        "action": "ENTRY_SHORT",
        "score": 82,
        "risk_state": "NORMAL",
        "volume_ratio": 0.7,
        "oi_change": -0.01,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BEAR",
        "h4_structure": {"state": "BOX_LOWER_HALF"},
        "h4_oi": {"state": "DELEVERAGE_WAIT", "drop_from_high_pct": -0.22, "rebound_pct": 0.0},
        "h1_trigger": {"direction": "NONE", "state": "WAIT"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "h1_ma_cluster": {"state": "RETEST_DOWN", "price": 198.6},
        "summary": "MTF: test",
    }

    adjusted = _apply_multi_timeframe_context(signal, context)

    assert (
        "4h OI sharp drop is not an OI valley and does not confirm a short entry"
        in adjusted["reasons"]
    )
    assert not any("OI drained" in reason for reason in adjusted["reasons"])
    assert adjusted["score"] > signal["score"]


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
    assert (
        "low area without 1h/4h resistance retest; wait for higher-timeframe bounce before short"
        in adjusted_short["vetoes"]
    )
    assert not _auto_signal_allowed(adjusted_short)


def test_oi_deleverage_breakdown_does_not_add_oi_short_score_after_failed_bounce() -> None:
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

    assert (
        "4h OI sharp drop is not an OI valley and does not confirm a short entry"
        in adjusted_short["reasons"]
    )
    assert not any("OI deleverage breakdown with failed bounce" in reason for reason in adjusted_short["reasons"])
    assert adjusted_short["score"] > short_signal["score"]


def test_fifteen_minute_precision_refines_stop_outside_anchor() -> None:
    assert _refine_stop_with_precision(PositionSide.LONG, 98.0, {"long_stop_anchor": 97.2}) == 97.2
    assert _refine_stop_with_precision(PositionSide.LONG, 95.0, {"long_stop_anchor": 97.2}) == 97.2
    assert _refine_stop_with_precision(PositionSide.SHORT, 102.0, {"short_stop_anchor": 103.1}) == 103.1
    assert _refine_stop_with_precision(PositionSide.SHORT, 105.0, {"short_stop_anchor": 103.1}) == 103.1


def test_fifteen_minute_precision_keeps_wider_stop_when_anchor_too_close() -> None:
    indicator = indicator_snapshot(close=100.0, atr=1.0)

    assert _refine_stop_with_precision(PositionSide.LONG, 96.0, {"long_stop_anchor": 99.6}, 100.0, indicator) == 96.0
    assert _refine_stop_with_precision(PositionSide.SHORT, 104.0, {"short_stop_anchor": 100.4}, 100.0, indicator) == 104.0
    assert _refine_stop_with_precision(PositionSide.LONG, 96.0, {"long_stop_anchor": 98.4}, 100.0, indicator) == 98.4
    assert _refine_stop_with_precision(PositionSide.SHORT, 104.0, {"short_stop_anchor": 101.6}, 100.0, indicator) == 101.6


def test_short_support_stop_waits_until_position_reached_one_r() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "SHORT",
            margin_usdt=100,
            leverage=5,
            stop_loss=105.0,
            take_profit_1=95.0,
            take_profit_2=90.0,
        )
    )
    indicator = indicator_snapshot(close=98.0, atr=1.0, ema20=100.0)

    _update_position_excursions(position, 98.0)
    _tighten_short_support_stop(position, 98.0, {}, indicator)

    assert position.stop_price == 105.0
    assert not position.metadata.get("short_support_protected")

    _update_position_excursions(position, 95.0)
    _tighten_short_support_stop(position, 95.0, {}, indicator)

    assert position.stop_price == pytest.approx(99.35)
    assert position.metadata["short_support_protected"]


def test_retest_structure_refines_short_stop_outside_resistance_and_ema60() -> None:
    context = {
        "h1_structure": {"resistance_zone_high": 198.6, "resistance": 198.0},
        "h4_structure": {"resistance": 205.0},
        "h1_trigger": {"direction": "SHORT", "state": "RETEST"},
        "h1_pullback": {"direction": "SHORT", "state": "HEALTHY_PULLBACK"},
        "h1_ma_cluster": {"state": "RETEST_DOWN", "upper": 197.8, "ema60": 198.6},
    }
    indicator = indicator_snapshot(close=190.3, atr=2.0)

    stop = _refine_stop_with_retest_structure(PositionSide.SHORT, 196.6, 190.3, context, indicator)

    assert stop >= 198.6


def test_1h_retest_stop_does_not_expand_to_4h_resistance() -> None:
    context = {
        "h1_structure": {
            "resistance_zone_high": 102.0,
            "resistance": 101.5,
        },
        "h4_structure": {"resistance": 112.0},
        "h1_trigger": {"direction": "SHORT", "state": "RETEST"},
    }
    indicator = indicator_snapshot(close=100.0, atr=1.0)

    stop = _refine_stop_with_retest_structure(
        PositionSide.SHORT,
        104.0,
        100.0,
        context,
        indicator,
        timeframe="1h",
    )

    assert 102.0 < stop < 112.0


def test_4h_retest_stop_uses_4h_resistance() -> None:
    context = {
        "h1_structure": {"resistance": 102.0},
        "h4_structure": {
            "resistance_zone_high": 112.0,
            "resistance": 111.0,
        },
    }
    indicator = indicator_snapshot(close=100.0, atr=2.0)

    stop = _refine_stop_with_retest_structure(
        PositionSide.SHORT,
        104.0,
        100.0,
        context,
        indicator,
        timeframe="4h",
    )

    assert stop > 112.0


def test_setup_structure_stop_prefers_setup_timeframe_over_entry_timeframe() -> None:
    context = {
        "h1_structure": {
            "resistance_zone_high": 102.0,
            "resistance": 101.5,
        },
        "h4_structure": {
            "resistance_zone_high": 112.0,
            "resistance": 111.0,
        },
    }
    indicator = indicator_snapshot(close=100.0, atr=2.0)

    stop, basis = _refine_stop_with_setup_structure(
        PositionSide.SHORT,
        104.0,
        100.0,
        {"setup_type": SETUP_H4_PULLBACK_SHORT},
        context,
        indicator,
        timeframe="1h",
    )

    assert stop > 112.0
    assert basis == "4h_structure"


def test_descending_four_hour_resistance_stop_sits_above_the_trendline_zone() -> None:
    stop, basis = _refine_stop_with_setup_structure(
        PositionSide.SHORT,
        96.0,
        95.1,
        {
            "setup_type": SETUP_H4_DESCENDING_RESISTANCE_SHORT,
            "h4_structure": {
                "descending_trendline_zone": {
                    "low": 94.6,
                    "high": 95.4,
                    "price": 95.0,
                }
            },
        },
        {},
        indicator_snapshot(close=95.1, atr=1.0),
        timeframe="4h",
    )

    assert stop > 95.4
    assert stop >= 95.1 * 1.012
    assert basis == "4h_descending_resistance_structure"


def test_setup_structure_stop_falls_back_when_structure_is_missing() -> None:
    stop, basis = _refine_stop_with_setup_structure(
        PositionSide.LONG,
        98.0,
        100.0,
        {"setup_type": SETUP_H1_PULLBACK_LONG},
        {},
        indicator_snapshot(close=100.0, atr=1.0),
        timeframe="1h",
    )

    assert stop == 98.0
    assert basis == "volatility_fallback"


def test_confirmed_entry_zone_provides_structure_stop_when_swing_is_missing() -> None:
    signal = {
        "action": SignalAction.ENTRY_SHORT.value,
        "score": 105,
        "trend_state": "TREND_SHORT",
        "risk_state": "NORMAL",
        "price": 100.0,
        "h1_trigger": {"direction": "SHORT", "state": "FAKE_BREAKOUT"},
        "entry_levels": {
            "short": {
                "h4_ema20_ema60": {
                    "low": 99.5,
                    "high": 100.5,
                    "price": 100.0,
                }
            }
        },
    }

    stop, basis = _refine_stop_with_entry_zone(
        PositionSide.SHORT,
        106.0,
        100.0,
        signal,
        indicator_snapshot(close=100.0, atr=1.0),
    )

    assert 100.5 < stop < 106.0
    assert basis == "entry_zone_structure"


def test_retest_structure_refines_long_stop_outside_support_and_ema60() -> None:
    context = {
        "h1_structure": {"support_zone_low": 95.0, "support": 95.4},
        "h4_structure": {"support": 92.0},
        "h1_trigger": {"direction": "LONG", "state": "RETEST"},
        "h1_pullback": {"direction": "LONG", "state": "HEALTHY_PULLBACK"},
        "h1_ma_cluster": {"state": "RETEST_UP", "lower": 95.2, "ema60": 95.0},
    }
    indicator = indicator_snapshot(close=100.0, atr=2.0)

    stop = _refine_stop_with_retest_structure(PositionSide.LONG, 98.8, 100.0, context, indicator)

    assert stop <= 95.0


def test_preferred_exit_indicator_uses_1h_4h_not_15m_for_targets() -> None:
    m15 = indicator_snapshot(close=99.0, atr=0.4)
    h1 = indicator_snapshot(close=100.0, atr=1.2)
    h4 = indicator_snapshot(close=101.0, atr=2.0)

    assert _preferred_exit_indicator({"15m": [m15], "1h": [h1], "4h": [h4]}, [m15]) is h1
    assert _preferred_exit_indicator({"15m": [m15], "4h": [h4]}, [m15]) is h4


def test_preferred_exit_indicator_uses_position_stop_timeframe() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    m15 = indicator_snapshot(close=99.0, atr=0.4)
    h1 = indicator_snapshot(close=100.0, atr=1.2)

    import asyncio

    position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=100,
            leverage=5,
            entry_context={"stop_basis": "15m_precision_structure"},
        )
    )

    assert position.metadata["entry_context"]["stop_timeframe"] == "15m"
    assert _preferred_exit_indicator({"15m": [m15], "1h": [h1]}, [m15], position) is m15


def test_preferred_exit_indicator_ignores_15m_stop_timeframe_for_short() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    m15 = indicator_snapshot(close=99.0, atr=0.4)
    h1 = indicator_snapshot(close=100.0, atr=1.2)

    import asyncio

    position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "SHORT",
            margin_usdt=100,
            leverage=5,
            entry_context={"stop_basis": "15m_precision_structure"},
        )
    )

    assert position.metadata["entry_context"]["stop_timeframe"] == "15m"
    assert _preferred_exit_indicator({"15m": [m15], "1h": [h1]}, [m15], position) is h1


def test_rotation_candidate_requires_one_way_volatility_and_clean_risk() -> None:
    good = {
        "action": "ENTRY_LONG",
        "score": 100,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }

    assert _rotation_candidate_allowed(good, indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5))
    assert not _rotation_candidate_allowed({**good, "score": 99}, indicator_snapshot())
    assert not _rotation_candidate_allowed({**good, "trend_state": "TREND_LONG"}, indicator_snapshot())
    assert not _rotation_candidate_allowed({**good, "risk_state": "LONG_CROWD"}, indicator_snapshot())
    assert not _rotation_candidate_allowed(good, indicator_snapshot(close=100.0, atr=0.5, volume_ratio=1.5))
    assert not _rotation_candidate_allowed(good, indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.0))
    wait_for_pullback = {
        **good,
        "price": 110.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }
    assert not _rotation_candidate_allowed(wait_for_pullback, indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5))


def test_auto_trade_caps_positions_and_prefers_highest_scores() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_prices = {f"TEST{idx}USDT": 100.0 for idx in range(6)}
    for idx, score in enumerate([84, 92, 85, 88, 86, 105]):
        engine.latest_signals[f"TEST{idx}USDT"] = {
            "action": "ENTRY_LONG",
            "score": score,
            "trend_state": "TREND_LONG",
            "price": 100.0,
            "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
            "h1_structure": {"resistance": 105.0},
        }

    import asyncio

    asyncio.run(engine._auto_trade_once())

    assert len(engine.account.positions) == 4
    assert "TEST0USDT" not in engine.account.positions
    assert set(engine.account.positions) == {
        "TEST1USDT",
        "TEST3USDT",
        "TEST4USDT",
        "TEST5USDT",
    }
    used_margin = sum(
        float(position.metadata["margin_usdt"])
        for position in engine.account.positions.values()
    )
    assert used_margin == pytest.approx(950.0, rel=0.01)
    assert float(engine.account.positions["TEST5USDT"].metadata["margin_usdt"]) == pytest.approx(380.0)
    assert {
        str(position.metadata["entry_context"]["entry_quality"])
        for position in engine.account.positions.values()
    } <= {"A", "S"}
    assert used_margin <= 1000 * engine.settings.risk.total_margin_limit


def test_auto_trade_turns_near_structure_target_into_partial_runner_plan() -> None:
    settings = AppSettings()
    settings.risk.max_open_positions = 1
    engine = PaperTradingEngine(settings, starting_balance=1000, market_data=FakeMarketData())
    engine.latest_prices = {"HIGHUSDT": 100.0, "NEXTUSDT": 100.0}
    common = {
        "action": "ENTRY_LONG",
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }
    engine.latest_signals["HIGHUSDT"] = {
        **common,
        "score": 95,
        "h1_structure": {"resistance_zone_low": 100.5, "resistance": 100.6},
    }
    engine.latest_signals["NEXTUSDT"] = {
        **common,
        "score": 90,
        "h1_structure": {"resistance_zone_low": 105.0, "resistance": 106.0},
    }

    asyncio.run(engine._auto_trade_once())

    assert set(engine.account.positions) == {"HIGHUSDT"}
    opened = engine.latest_signals["HIGHUSDT"]
    position = engine.account.positions["HIGHUSDT"]
    assert opened["entry_timing"] == "GOOD"
    assert not any("entry reward/risk" in reason for reason in opened.get("vetoes", ()))
    assert "最近结构目标空间不足；先分批止盈，剩余仓位目标不低于 1.20R" in opened.get("reasons", ())
    assert position.metadata["leverage"] <= 7
    assert position.metadata["entry_context"]["entry_reward_r"] == pytest.approx(1.2)
    assert position.metadata["entry_context"]["entry_structure_reward_r"] < 1.2


def test_auto_trade_waits_when_late_stage_structure_reward_is_too_low() -> None:
    settings = AppSettings()
    settings.risk.max_open_positions = 1
    engine = PaperTradingEngine(
        settings,
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine.latest_prices["LATEUSDT"] = 100.0
    engine.latest_signals["LATEUSDT"] = {
        "action": "ENTRY_LONG",
        "score": 95,
        "trend_state": "TREND_LONG",
        "trend_stage_phase": "LATE",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
        "h1_structure": {
            "resistance_zone_low": 100.5,
            "resistance": 100.6,
        },
    }

    asyncio.run(engine._auto_trade_once())

    assert not engine.account.positions
    assert (
        "trend late stage and structure reward below minimum; wait for a new pullback"
        in engine.latest_signals["LATEUSDT"]["vetoes"]
    )


def test_auto_trade_uses_generated_take_profit_when_structure_target_is_unavailable() -> None:
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        market_data=FakeMarketData(),
    )
    engine.latest_prices["TESTUSDT"] = 100.0
    engine.latest_signals["TESTUSDT"] = {
        "action": "ENTRY_LONG",
        "score": 95,
        "trend_state": "TREND_LONG",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {
            "long": {
                "h1_support": {
                    "low": 99.0,
                    "high": 101.0,
                    "price": 100.0,
                }
            }
        },
    }

    asyncio.run(engine._auto_trade_once())

    position = engine.account.positions["TESTUSDT"]
    assert position.take_profit_1 > position.entry_price
    assert position.take_profit_2 > position.take_profit_1
    assert position.metadata["entry_context"]["entry_quality"] != "S"
    assert "entry reward/risk target unavailable" not in engine.latest_signals["TESTUSDT"]["vetoes"]


def test_auto_trade_does_not_publish_partially_cleared_vetoes() -> None:
    class BlockingMarkPriceMarket(FakeMarketData):
        def __init__(self) -> None:
            super().__init__()
            self.price_requested = asyncio.Event()
            self.release_price = asyncio.Event()

        async def mark_prices(self, symbols):
            self.price_requested.set()
            await self.release_price.wait()
            return {symbol: 100.0 for symbol in symbols}

    async def scenario() -> None:
        market = BlockingMarkPriceMarket()
        engine = PaperTradingEngine(
            AppSettings(),
            starting_balance=1000,
            market_data=market,
        )
        prior_veto = "entry reward/risk 0.80R below minimum 1.20R"
        engine.latest_signals["TESTUSDT"] = {
            "action": "ENTRY_LONG",
            "score": 95,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "price": 100.0,
            "entry_timing": "BLOCK",
            "entry_timing_reason": prior_veto,
            "vetoes": (prior_veto,),
            "entry_levels": {
                "long": {
                    "h1_support": {
                        "low": 99.0,
                        "high": 101.0,
                        "price": 100.0,
                    }
                }
            },
            "h1_structure": {
                "resistance_zone_low": 100.5,
                "resistance": 100.6,
            },
        }

        task = asyncio.create_task(engine._auto_trade_once())
        await asyncio.wait_for(market.price_requested.wait(), timeout=1)

        assert engine.latest_signals["TESTUSDT"]["vetoes"] == (prior_veto,)
        assert engine.latest_signals["TESTUSDT"]["entry_timing_reason"] == prior_veto

        market.release_price.set()
        await asyncio.wait_for(task, timeout=1)

        assert not any(
            str(reason).startswith("entry reward/risk ")
            for reason in engine.latest_signals["TESTUSDT"].get("vetoes", ())
        )
        assert "TESTUSDT" in engine.account.positions

    asyncio.run(scenario())


def test_auto_trade_marks_ready_candidates_blocked_after_position_capacity_is_filled() -> None:
    settings = AppSettings()
    settings.risk.max_open_positions = 1
    engine = PaperTradingEngine(settings, starting_balance=1000, market_data=FakeMarketData())
    engine.latest_prices = {"BESTUSDT": 100.0, "SECONDUSDT": 100.0}
    for symbol, score in (("BESTUSDT", 95), ("SECONDUSDT", 90)):
        engine.latest_signals[symbol] = {
            "action": "ENTRY_LONG",
            "score": score,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "price": 100.0,
            "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
            "h1_structure": {"resistance_zone_low": 105.0, "resistance": 106.0},
        }

    asyncio.run(engine._auto_trade_once())

    assert set(engine.account.positions) == {"BESTUSDT"}
    assert "position capacity full: 1 open positions" in engine.latest_signals["SECONDUSDT"]["vetoes"]


def test_auto_trade_pauses_altcoin_entries_when_btc_4h_is_extreme() -> None:
    market = FakeMarketData()
    market.btc_4h_extreme = True
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=market)
    engine.latest_prices = {"TESTUSDT": 100.0}
    engine.latest_signals["TESTUSDT"] = {
        "action": "ENTRY_LONG",
        "score": 95,
        "trend_state": "ONE_WAY_UP",
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }

    import asyncio

    asyncio.run(engine._auto_trade_once())

    assert not engine.account.positions
    assert engine.latest_signals["TESTUSDT"]["entry_timing"] == "GOOD"
    assert "BTC 4h extreme volatility; pause new altcoin entries" in engine.latest_signals["TESTUSDT"]["vetoes"]


def test_auto_trade_rotates_by_efficiency_instead_of_grade_hierarchy() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_prices = {f"TEST{idx}USDT": 100.0 for idx in range(6)}

    import asyncio

    for idx in range(5):
        asyncio.run(engine.open_position(
            f"TEST{idx}USDT",
            "LONG",
            margin_usdt=50,
            leverage=5,
            stop_loss=99.0,
        ))
        engine.account.positions[f"TEST{idx}USDT"].opened_at = datetime.now(UTC) - timedelta(hours=1)
        engine.latest_signals[f"TEST{idx}USDT"] = {
            "score": 105 if idx == 0 else 85 + idx,
            "action": "WATCH",
            "trend_state": "CHOP" if idx == 0 else "TREND_LONG",
            "risk_state": "NORMAL",
        }
        engine.latest_indicators[f"TEST{idx}USDT"] = [indicator_snapshot(close=100.0, atr=0.6, volume_ratio=1.0, oi_change=0.0)]
    engine.latest_signals["TEST5USDT"] = {
        "action": "ENTRY_LONG",
        "score": 100,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
        "h1_structure": {"resistance": 105.0},
    }
    engine.latest_indicators["TEST5USDT"] = [indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5)]

    asyncio.run(engine._auto_trade_once())

    assert "TEST0USDT" not in engine.account.positions
    assert "TEST5USDT" in engine.account.positions
    assert len(engine.account.positions) == 5
    assert engine.account.fills[-2].reason == "rotation exit: efficiency rotation; symbol=TEST5USDT score=100 current_score=105"


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
