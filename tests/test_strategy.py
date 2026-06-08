from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from ai_trading.config import StrategySettings
from ai_trading.indicators import build_indicators
from ai_trading.models import Candle, DerivativesSnapshot, SignalAction
from ai_trading.strategy import CompositeStrategy


def test_strategy_blocks_overheated_long() -> None:
    candles, derivatives = _trending_market()
    indicators = build_indicators(candles, derivatives)
    current = indicators[-1]
    indicators[-1] = replace(
        current,
        rsi14=82,
        long_short_ratio=2.5,
        funding_rate=0.001,
        boll_upper=current.close - 1,
    )

    signal = CompositeStrategy().generate_signal("BTCUSDT", candles, indicators)

    assert signal.action == SignalAction.NO_TRADE
    assert "RSI overheated for long entry" in signal.vetoes
    assert "long side overcrowded" in signal.vetoes


def test_strategy_emits_long_when_score_is_strong() -> None:
    candles, derivatives = _trending_market()
    indicators = build_indicators(candles, derivatives)
    previous = indicators[-2]
    current = indicators[-1]
    indicators[-2] = replace(previous, close=(previous.boll_mid or previous.close) - 0.1)
    indicators[-1] = replace(
        current,
        close=current.boll_mid + 0.2 if current.boll_mid else current.close,
        rsi14=58,
        volume_ratio=1.4,
        oi_change=0.003,
        long_short_ratio=1.2,
        funding_rate=0.0001,
    )

    signal = CompositeStrategy(StrategySettings(score_threshold=70)).generate_signal("BTCUSDT", candles, indicators)

    assert signal.action == SignalAction.ENTRY_LONG
    assert signal.score >= 70


def _trending_market() -> tuple[list[Candle], list[DerivativesSnapshot]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    derivatives: list[DerivativesSnapshot] = []
    price = 100.0
    oi = 10_000.0
    for idx in range(140):
        previous = price
        price += 0.15 + (idx % 5) * 0.01
        timestamp = start + timedelta(minutes=15 * idx)
        candles.append(
            Candle(
                timestamp=timestamp,
                open=previous,
                high=price + 0.8,
                low=previous - 0.8,
                close=price,
                volume=1_000 + idx * 3,
            )
        )
        oi += 15
        derivatives.append(DerivativesSnapshot(timestamp=timestamp, open_interest=oi, long_short_ratio=1.2, funding_rate=0.0001))
    return candles, derivatives
