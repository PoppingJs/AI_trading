from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.backtest import (
    BacktestEngine,
    LegacyBacktestEngine,
    PortfolioBacktestResult,
    _ReplayMarketData,
    _aggregate_candles,
    _conservative_intrabar_path,
    _replay_intrabar_price,
    _visible_candles,
    run_portfolio_backtest,
)
from ai_trading.config import AppSettings, RiskSettings, StrategySettings
from ai_trading.indicators import build_indicators
from ai_trading.models import Candle, DerivativesSnapshot, Position, PositionSide, SignalAction
from ai_trading.risk import (
    AccountRiskSnapshot,
    PortfolioRiskGate,
    RiskManager,
    TradePlan,
    current_open_risk_usdt,
    risk_factor_for_quality,
)
from ai_trading.paper import PaperTradingEngine
from ai_trading.strategy import CompositeStrategy


def test_risk_sizes_by_loss_amount_not_leverage() -> None:
    candles, derivatives = _market()
    indicators = build_indicators(candles, derivatives)
    signal = CompositeStrategy(StrategySettings(score_threshold=65)).generate_signal("ETHUSDT", candles, indicators)
    if signal.action != SignalAction.ENTRY_LONG:
        latest = indicators[-1]
        signal = signal.__class__(
            symbol="ETHUSDT",
            timestamp=latest.timestamp,
            action=SignalAction.ENTRY_LONG,
            regime=signal.regime,
            score=80,
            indicators=latest,
        )

    decision = RiskManager(RiskSettings(risk_per_trade=0.005)).plan_entry(
        signal,
        candles,
        equity=10_000,
        open_positions=[],
        daily_pnl=0,
        consecutive_losses=0,
        leverage=5,
    )

    assert decision.allowed
    risk_amount = abs(signal.indicators.close - decision.stop_price) * decision.quantity
    assert risk_amount <= 50.01


def test_portfolio_gate_keeps_wide_and_narrow_stops_at_same_loss_budget() -> None:
    gate = PortfolioRiskGate(RiskSettings(risk_per_trade=0.01))
    account = AccountRiskSnapshot(
        equity=1200.0,
        available_balance=1200.0,
        used_margin=0.0,
    )
    wide = gate.evaluate(
        TradePlan(
            symbol="WIDEUSDT",
            side=PositionSide.LONG,
            entry_price=0.01239,
            stop_price=0.01089,
            take_profit_1=0.01389,
            take_profit_2=0.01539,
            leverage=5,
        ),
        account,
    )
    narrow = gate.evaluate(
        TradePlan(
            symbol="NARROWUSDT",
            side=PositionSide.LONG,
            entry_price=1.848,
            stop_price=1.801,
            take_profit_1=1.895,
            take_profit_2=1.942,
            leverage=5,
        ),
        account,
    )

    assert wide.allowed and narrow.allowed
    assert wide.margin_required < narrow.margin_required
    assert wide.planned_risk_usdt == pytest.approx(12.0)
    assert narrow.planned_risk_usdt == pytest.approx(12.0)


def test_portfolio_gate_enforces_open_risk_and_actual_position_margin() -> None:
    existing = Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        quantity=10.0,
        opened_at=datetime.now(UTC),
        stop_price=98.0,
        take_profit_1=102.0,
        take_profit_2=104.0,
        metadata={"margin_usdt": 100.0, "leverage": 10},
    )
    settings = RiskSettings(
        risk_per_trade=0.01,
        single_symbol_margin_limit=0.20,
        total_margin_limit=0.35,
        total_open_risk_limit=0.03,
    )
    gate = PortfolioRiskGate(settings)
    account = AccountRiskSnapshot(
        equity=1000.0,
        available_balance=900.0,
        used_margin=100.0,
        open_positions=(existing,),
    )
    decision = gate.evaluate(
        TradePlan(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            entry_price=100.0,
            stop_price=102.0,
            take_profit_1=98.0,
            take_profit_2=96.0,
            leverage=10,
        ),
        account,
    )

    assert current_open_risk_usdt((existing,)) == pytest.approx(20.0)
    assert decision.allowed
    assert decision.planned_risk_usdt == pytest.approx(10.0)
    assert decision.open_risk_after_usdt == pytest.approx(30.0)
    assert decision.margin_required == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("snapshot_changes", "blocked_code"),
    [
        ({"daily_loss_locked": True}, "DAILY_LOSS_LIMIT"),
        ({"weekly_loss_locked": True}, "WEEKLY_LOSS_LIMIT"),
        ({"drawdown_locked": True}, "MAX_DRAWDOWN"),
        ({"consecutive_losses": 3}, "CONSECUTIVE_LOSSES"),
    ],
)
def test_portfolio_gate_blocks_account_level_circuit_breakers(
    snapshot_changes: dict[str, object],
    blocked_code: str,
) -> None:
    values = {
        "equity": 1000.0,
        "available_balance": 1000.0,
        "used_margin": 0.0,
        **snapshot_changes,
    }
    decision = PortfolioRiskGate().evaluate(
        TradePlan(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
            leverage=5,
        ),
        AccountRiskSnapshot(**values),
    )

    assert not decision.allowed
    assert decision.blocked_code == blocked_code


def test_quality_can_only_reduce_configured_trade_risk() -> None:
    assert risk_factor_for_quality("S") == 1.0
    assert risk_factor_for_quality("A") == 1.0
    assert risk_factor_for_quality("B") < 1.0


def test_backtest_runs_to_completion() -> None:
    candles, derivatives = _market()

    result = BacktestEngine(
        symbol="ETHUSDT",
        starting_equity=10_000,
        strategy_settings=StrategySettings(score_threshold=65),
    ).run(candles, derivatives)

    assert result.ending_equity > 0
    assert result.max_drawdown >= 0
    assert result.total_return > -1


def test_legacy_mode_preserves_the_frozen_backtest_baseline() -> None:
    candles, derivatives = _market()
    direct = LegacyBacktestEngine(
        symbol="ETHUSDT",
        starting_equity=10_000,
        strategy_settings=StrategySettings(score_threshold=65),
    ).run(candles, derivatives)
    compatibility = BacktestEngine(
        symbol="ETHUSDT",
        starting_equity=10_000,
        strategy_settings=StrategySettings(score_threshold=65),
        mode="legacy",
    ).run(candles, derivatives)

    assert compatibility == direct


def test_conservative_ohlc_path_visits_adverse_extreme_first() -> None:
    candle = Candle(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        open=100.0,
        high=103.0,
        low=97.0,
        close=101.0,
        volume=1000.0,
    )

    assert [price for _, price in _conservative_intrabar_path(
        candle,
        PositionSide.LONG,
    )] == [97.0, 103.0]
    assert [price for _, price in _conservative_intrabar_path(
        candle,
        PositionSide.SHORT,
    )] == [103.0, 97.0]


def test_same_bar_stop_and_target_uses_stop_first_at_trigger_price() -> None:
    replay_market = _ReplayMarketData()
    engine = PaperTradingEngine(
        AppSettings(),
        starting_balance=1000,
        symbols=["TESTUSDT"],
        market_data=replay_market,  # type: ignore[arg-type]
    )
    engine.latest_prices["TESTUSDT"] = 100.0
    position = asyncio.run(
        engine.open_position(
            "TESTUSDT",
            "LONG",
            margin_usdt=50,
            leverage=5,
            stop_loss=98.0,
            take_profit_1=102.0,
            take_profit_2=104.0,
        )
    )

    _replay_intrabar_price(engine, replay_market, "TESTUSDT", 97.0)
    _replay_intrabar_price(engine, replay_market, "TESTUSDT", 103.0)

    assert position.symbol not in engine.account.positions
    close_fills = [
        fill for fill in engine.account.fills
        if fill.action in {"CLOSE", "PARTIAL_CLOSE"}
    ]
    assert len(close_fills) == 1
    assert close_fills[0].action == "CLOSE"
    assert close_fills[0].price == 98.0
    assert close_fills[0].reason.startswith("stop loss")


def test_higher_timeframe_aggregation_drops_incomplete_future_bar() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + timedelta(minutes=15 * index),
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.5 + index,
            volume=10.0,
        )
        for index in range(5)
    ]
    hourly = _aggregate_candles(
        candles,
        base_seconds=15 * 60,
        target_seconds=60 * 60,
    )

    assert len(hourly) == 1
    assert hourly[0].open == 100.0
    assert hourly[0].close == 103.5
    assert _visible_candles(
        hourly,
        "1h",
        start + timedelta(minutes=59),
    ) == []
    assert _visible_candles(
        hourly,
        "1h",
        start + timedelta(hours=1),
    ) == hourly


def test_portfolio_backtest_returns_one_shared_account_result() -> None:
    candles, derivatives = _market()
    result = run_portfolio_backtest(
        {
            "ETHUSDT": (candles, derivatives),
            "SOLUSDT": (candles, derivatives),
        },
        starting_equity=1000.0,
    )

    assert isinstance(result, PortfolioBacktestResult)
    assert result.starting_equity == 1000.0
    assert result.ending_equity > 0
    assert "production-parity strategy path" in result.notes


def test_closed_bar_signal_fills_at_next_open_not_previous_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + timedelta(minutes=15 * index),
            open=(110.0 if index == 4 else 100.0),
            high=(111.0 if index == 4 else 101.0),
            low=(109.0 if index == 4 else 99.0),
            close=100.0,
            volume=1000.0,
        )
        for index in range(8)
    ]

    def publish_signal(engine: PaperTradingEngine, symbol: str) -> bool:
        if engine._now() < start + timedelta(hours=1):
            engine.latest_signals.pop(symbol, None)
            return False
        signal = {
            "timestamp": engine._now().isoformat(),
            "action": SignalAction.ENTRY_LONG.value,
            "candidate_action": SignalAction.ENTRY_LONG.value,
            "score": 95,
            "trend_state": "TREND_LONG",
            "risk_state": "NORMAL",
            "price": engine.latest_prices[symbol],
            "reasons": (),
            "vetoes": (),
            "entry_levels": {
                "long": {
                    "h1_support": {
                        "low": 100.0,
                        "high": 120.0,
                        "price": 110.0,
                    }
                }
            },
            "h1_structure": {"resistance": 120.0},
        }
        engine.latest_signals[symbol] = signal
        engine.account.latest_signals[symbol] = dict(signal)
        return True

    monkeypatch.setattr(
        PaperTradingEngine,
        "_publish_symbol_from_cache",
        publish_signal,
    )
    result = run_portfolio_backtest(
        {"GAPUSDT": (candles, None)},
        starting_equity=1000.0,
    )
    open_fills = [
        fill
        for fill in result.fills
        if fill.action == "OPEN"
    ]

    assert open_fills
    assert open_fills[0].price == pytest.approx(110.0 * 1.0003)
    assert open_fills[0].price != pytest.approx(100.0 * 1.0003)
    assert result.ending_equity - result.starting_equity == pytest.approx(
        sum(result.per_symbol_pnl.values())
    )


def _market() -> tuple[list[Candle], list[DerivativesSnapshot]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    derivatives: list[DerivativesSnapshot] = []
    price = 100.0
    oi = 10_000.0
    for idx in range(180):
        previous = price
        if idx < 80:
            price += 0.04
        else:
            price += 0.16 - (0.03 if idx % 9 == 0 else 0)
        timestamp = start + timedelta(minutes=15 * idx)
        candles.append(
            Candle(
                timestamp=timestamp,
                open=previous,
                high=max(price, previous) + 0.7,
                low=min(price, previous) - 0.7,
                close=price,
                volume=1_000 + (idx % 20) * 40,
            )
        )
        oi += 12 if idx > 80 else 2
        derivatives.append(DerivativesSnapshot(timestamp=timestamp, open_interest=oi, long_short_ratio=1.1, funding_rate=0.0001))
    return candles, derivatives
