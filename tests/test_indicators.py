from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.indicators import atr, build_indicators, ema, rsi, sma
from ai_trading.models import Candle, DerivativesSnapshot


def test_sma_and_ema_seed_after_window() -> None:
    values = [float(value) for value in range(1, 6)]

    assert sma(values, 3) == [None, None, 2.0, 3.0, 4.0]
    assert ema(values, 3)[2] == 2.0


def test_rsi_reaches_high_value_in_uptrend() -> None:
    values = [float(value) for value in range(1, 30)]

    assert rsi(values, 14)[-1] == 100.0


def test_build_indicators_aligns_derivatives_and_oi_change() -> None:
    candles = _candles(130)
    derivatives = [
        DerivativesSnapshot(
            timestamp=candle.timestamp,
            open_interest=10_000 + idx * 10,
            long_short_ratio=1.1,
            funding_rate=0.0001,
            taker_buy_sell_ratio=1.08,
            taker_buy_volume=108.0,
            taker_sell_volume=100.0,
            top_account_long_short_ratio=1.03,
            top_position_long_short_ratio=1.12,
        )
        for idx, candle in enumerate(candles)
    ]

    indicators = build_indicators(candles, derivatives)

    assert len(indicators) == len(candles)
    assert indicators[-1].ema20 is not None
    assert indicators[-1].ma100 is not None
    assert indicators[-1].oi_change is not None
    assert indicators[-1].long_short_ratio == 1.1
    assert indicators[-1].taker_buy_sell_ratio == 1.08
    assert indicators[-1].taker_buy_volume == 108.0
    assert indicators[-1].taker_sell_volume == 100.0
    assert indicators[-1].top_account_long_short_ratio == 1.03
    assert indicators[-1].top_position_long_short_ratio == 1.12
    assert indicators[-1].kc_mid is not None
    assert indicators[-1].kc_upper is not None
    assert indicators[-1].kc_lower is not None
    assert indicators[-1].quote_flow is not None
    assert indicators[-1].quote_flow_ratio is not None


def test_atr_is_positive() -> None:
    values = atr(_candles(30), 14)

    assert values[-1] is not None
    assert values[-1] > 0


def _candles(count: int) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    price = 100.0
    for idx in range(count):
        previous = price
        price += 0.2
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * idx),
                open=previous,
                high=price + 1,
                low=previous - 1,
                close=price,
                volume=1_000 + idx,
            )
        )
    return candles
