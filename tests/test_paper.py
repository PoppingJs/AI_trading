from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from ai_trading.api import create_app
from ai_trading.config import AppSettings
from ai_trading.models import Candle, IndicatorSnapshot, PositionSide, SignalAction
from ai_trading.paper import (
    MARKET_PRICE_STALE_SECONDS,
    PaperStateError,
    PaperTradingEngine,
    _clear_transient_auto_entry_blocks,
    _closed_candles,
    _adaptive_exits,
    _apply_multi_timeframe_context,
    _auto_entry_prerequisite_blocks,
    _auto_signal_allowed,
    _confirmed_structure_exit_reason,
    _daily_bias_margin_factor,
    _daily_pnl_payload,
    _entry_reward_r,
    _entry_signal_timeframe,
    _entry_timeframe_for_signal,
    _entry_stop_error,
    _exit_plan_error,
    _leverage_for_signal,
    _margin_for_signal,
    _merge_candles,
    _ma_cluster_signal_adjustment,
    _pnl_history_payload,
    _profit_drawdown_exit_reason,
    _required_entry_timeframes,
    _preferred_exit_indicator,
    _protect_confirmed_breakout_position,
    _precision_stop_allowed,
    _refine_stop_with_ma_cluster,
    _refine_stop_with_precision,
    _refine_stop_with_retest_structure,
    _refine_take_profit_with_ma_cluster,
    _pyramid_allowed,
    _rotation_candidate_allowed,
    _risk_exit_reason,
    _signal_entry_timing,
    _stop_exit_reason,
    _structure_take_profit_reason,
    _update_position_excursions,
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
                self.quote_volume = 100_000_000
                self.price_change_percent = 5.0
                self.last_price = 100.0
                self.high_price = 105.0
                self.low_price = 95.0
                self.open_price = 99.0

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
    assert close_fill["entry_position"] == "手动开仓≈100"
    assert not engine.account.positions
    assert engine.status()["equity"] > 1000


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
        "1H支撑回踩≈99-101；1H BOLL中轨回踩≈99.5-100.5；实际开仓≈100"
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

    restored.latest_prices["BTCUSDT"] = 110.0
    asyncio.run(restored.close_position("BTCUSDT", reason="test close"))
    restored_again = PaperTradingEngine(AppSettings(), starting_balance=500, market_data=FakeMarketData(), state_path=state_path)
    assert restored_again.status()["positions"] == []
    assert restored_again.status()["fills"][-1]["action"] == "CLOSE"


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
    assert "current funding rate data is stale for more than 15 minutes" in vetoes


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

    _protect_confirmed_breakout_position(position, 105.0, signal, indicator_snapshot(close=105.0, atr=1.0, ema20=104.0))

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


def test_auto_top30_universe_refreshes_symbols() -> None:
    engine = PaperTradingEngine(AppSettings(), starting_balance=1000, market_data=FakeMarketData())
    engine.latest_signals["STALEUSDT"] = {"score": 99}
    engine.account.latest_signals["STALEUSDT"] = {"score": 99}

    import asyncio

    assert engine.symbols == ["AUTO_TOP30"]
    asyncio.run(engine.refresh_universe_if_needed())

    assert len(engine.symbols) == 30
    assert "BTCUSDT" not in engine.symbols
    assert "ETHUSDT" not in engine.symbols
    assert "SOLUSDT" not in engine.symbols
    assert engine.symbols[0] == "TEST0USDT"
    assert "STALEUSDT" not in engine.latest_signals
    assert "STALEUSDT" not in engine.account.latest_signals


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
    assert _auto_signal_allowed({**base_signal, "score": 82, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({**base_signal, "score": 78, "risk_state": "FUNDING_HOT"})
    assert not _auto_signal_allowed({**base_signal, "score": 81, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({**base_signal, "score": 74, "risk_state": "NORMAL"})
    assert not _auto_signal_allowed({**base_signal, "score": 90, "risk_state": "NORMAL", "vetoes": ("1h trigger opposes long entry",)})
    assert not _auto_signal_allowed({"score": 90, "risk_state": "NORMAL"})

    assert _margin_for_signal(90, 1000) == 280
    assert _margin_for_signal(80, 1000) == 230
    assert _margin_for_signal(76, 1000) == 180
    assert _margin_for_signal(90, 1000, 950, 5) == 190
    assert _margin_for_signal(76, 1000, 950, 5) == 180


def test_auto_entry_prerequisites_explain_score_direction_and_timing_blocks() -> None:
    wait_signal = {
        "action": SignalAction.ENTRY_LONG.value,
        "score": 81,
        "risk_state": "NORMAL",
        "price": 106.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }

    blocks = _auto_entry_prerequisite_blocks(wait_signal)

    assert "final score 81 below auto-entry minimum 82" in blocks
    assert any(reason.startswith("current entry position is not excellent:") for reason in blocks)
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
    assert "scored short entry zone" in reason
    assert not _auto_signal_allowed(mid_zone_short)

    resistance_retest = {**mid_zone_short, "price": 261.0}
    timing, reason = _signal_entry_timing(resistance_retest)

    assert timing == "GOOD"
    assert "scored short entry zone" in reason
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
    assert "scored long entry zone" in reason
    assert not _auto_signal_allowed(signal)

    signal["price"] = 100.2
    timing, reason = _signal_entry_timing(signal)
    assert timing == "GOOD"
    assert "entry zone" in reason
    assert _auto_signal_allowed(signal)


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
    assert "scored long entry zone" in reason
    assert _auto_signal_allowed(late_long)


def test_entry_reward_r_requires_enough_target_space() -> None:
    signal = {
        "h1_structure": {
            "resistance_zone_low": 101.0,
            "resistance": 101.2,
            "resistance_zone_high": 101.4,
        }
    }

    assert _entry_reward_r(signal, PositionSide.LONG, price=100.0, stop=99.0) == 1.0
    assert _entry_reward_r(signal, PositionSide.LONG, price=100.0, stop=99.5) == 2.0


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
    assert "scored short entry zone" in reason
    assert not _auto_signal_allowed(signal)


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
    assert "scored long entry zone" in reason
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
        "price": 100.0,
        "reasons": (),
        "vetoes": (),
    }
    context = {
        "daily_bias": "BULL",
        "h4_structure": {"state": "BOX_UPPER_HALF"},
        "h1_trigger": {"direction": "LONG", "state": "RETEST"},
        "h1_pullback": {"direction": "NONE", "state": "WAIT"},
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
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


def test_weak_oi_rebound_with_ma_pressure_improves_short() -> None:
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

    assert "OI drained, rebound volume weak, and 1h/MA resistance rejected; short candidate improved" in adjusted["reasons"]
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
    assert _refine_stop_with_precision(PositionSide.LONG, 95.0, {"long_stop_anchor": 97.2}) == 97.2
    assert _refine_stop_with_precision(PositionSide.SHORT, 102.0, {"short_stop_anchor": 103.1}) == 103.1
    assert _refine_stop_with_precision(PositionSide.SHORT, 105.0, {"short_stop_anchor": 103.1}) == 103.1


def test_fifteen_minute_precision_keeps_wider_stop_when_anchor_too_close() -> None:
    indicator = indicator_snapshot(close=100.0, atr=1.0)

    assert _refine_stop_with_precision(PositionSide.LONG, 96.0, {"long_stop_anchor": 99.6}, 100.0, indicator) == 96.0
    assert _refine_stop_with_precision(PositionSide.SHORT, 104.0, {"short_stop_anchor": 100.4}, 100.0, indicator) == 104.0
    assert _refine_stop_with_precision(PositionSide.LONG, 96.0, {"long_stop_anchor": 98.4}, 100.0, indicator) == 98.4
    assert _refine_stop_with_precision(PositionSide.SHORT, 104.0, {"short_stop_anchor": 101.6}, 100.0, indicator) == 101.6


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
        "score": 92,
        "trend_state": "ONE_WAY_UP",
        "risk_state": "NORMAL",
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
    }

    assert _rotation_candidate_allowed(good, indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5))
    assert _rotation_candidate_allowed({**good, "score": 89}, indicator_snapshot())
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
    for idx, score in enumerate([81, 92, 82, 88, 83, 95]):
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

    assert len(engine.account.positions) == 5
    assert "TEST0USDT" not in engine.account.positions
    assert set(engine.account.positions) == {"TEST1USDT", "TEST2USDT", "TEST3USDT", "TEST4USDT", "TEST5USDT"}
    used_margin = sum(float(position.metadata["margin_usdt"]) for position in engine.account.positions.values())
    assert 760 <= used_margin <= 790
    assert engine.status()["available_balance"] >= 200
    assert engine.account.positions["TEST5USDT"].metadata["margin_usdt"] >= 180


def test_auto_trade_backfills_slot_when_higher_score_candidate_has_low_reward_risk() -> None:
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

    assert set(engine.account.positions) == {"NEXTUSDT"}
    blocked = engine.latest_signals["HIGHUSDT"]
    assert blocked["entry_timing"] == "GOOD"
    assert any("entry reward/risk" in reason for reason in blocked["vetoes"])


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

        assert any(
            str(reason).startswith("entry reward/risk ")
            for reason in engine.latest_signals["TESTUSDT"]["vetoes"]
        )

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
        "price": 100.0,
        "entry_levels": {"long": {"h1_support": {"low": 99.0, "high": 101.0, "price": 100.0}}},
        "h1_structure": {"resistance": 105.0},
    }
    engine.latest_indicators["TEST5USDT"] = [indicator_snapshot(close=100.0, atr=1.0, volume_ratio=1.5)]

    asyncio.run(engine._auto_trade_once())

    assert "TEST0USDT" not in engine.account.positions
    assert "TEST5USDT" in engine.account.positions
    assert len(engine.account.positions) == 5
    assert engine.account.fills[-2].reason == "rotation exit: trend invalidated; symbol=TEST5USDT score=95 current_score=75"


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
