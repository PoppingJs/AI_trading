from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.backtest import BacktestEngine
from ai_trading.config import RiskSettings, StrategySettings
from ai_trading.indicators import build_indicators
from ai_trading.models import Candle, DerivativesSnapshot, SignalAction
from ai_trading.risk import RiskManager
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
