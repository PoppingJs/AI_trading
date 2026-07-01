from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from ai_trading.config import StrategySettings
from ai_trading.indicators import build_indicators
from ai_trading.models import Candle, DerivativesSnapshot, SignalAction
from ai_trading.strategy import CompositeStrategy, MarketStructure, _volume_breakout_retest_confirmation


def test_strategy_defers_overheated_rsi_to_multi_timeframe_context() -> None:
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
    assert "RSI overheated for long entry" not in signal.vetoes
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

    signal = CompositeStrategy(StrategySettings(score_threshold=70, strict_trend_entry=False)).generate_signal("BTCUSDT", candles, indicators)

    assert signal.action == SignalAction.ENTRY_LONG
    assert signal.score >= 70


def test_strategy_blocks_chasing_after_upper_wick_sweep() -> None:
    candles, derivatives = _trending_market()
    indicators = build_indicators(candles, derivatives)
    latest = candles[-1]
    candles[-1] = replace(latest, open=130.0, high=135.0, low=129.0, close=130.4, volume=5_000)
    current = indicators[-1]
    indicators[-1] = replace(
        current,
        close=130.4,
        ema20=125.0,
        ema50=123.0,
        ma100=120.0,
        boll_mid=124.0,
        boll_upper=132.0,
        boll_lower=118.0,
        rsi14=62,
        atr14=2.0,
        volume_ratio=1.6,
        oi_change=0.003,
        long_short_ratio=1.2,
        funding_rate=0.0001,
    )

    signal = CompositeStrategy(StrategySettings(score_threshold=70, strict_trend_entry=False)).generate_signal("BTCUSDT", candles, indicators)

    assert "upper wick sweep rejected; avoid chasing long" in signal.vetoes


def test_strategy_rewards_market_structure_breakout() -> None:
    candles, derivatives = _trending_market()
    indicators = build_indicators(candles, derivatives)
    settings = StrategySettings(score_threshold=999, watch_threshold=1, strict_trend_entry=False)
    previous_score = CompositeStrategy(settings).generate_signal("BTCUSDT", candles, indicators).score
    resistance = max(candle.high for candle in candles[-21:-1])
    latest = candles[-1]
    candles[-1] = replace(latest, close=resistance + 1.2, high=resistance + 1.5, low=resistance - 0.2, volume=5_000)
    current = indicators[-1]
    indicators[-1] = replace(current, close=resistance + 1.2, volume_ratio=1.5, atr14=1.0)

    signal = CompositeStrategy(settings).generate_signal("BTCUSDT", candles, indicators)

    assert signal.score >= previous_score + 8
    assert "market structure confirms long: breakout or retest held" in signal.reasons


def test_strategy_rewards_vwap_pullback_as_score_not_filter() -> None:
    candles, derivatives = _trending_market()
    indicators = build_indicators(candles, derivatives)
    settings = StrategySettings(score_threshold=999, watch_threshold=1, strict_trend_entry=False)
    previous_score = CompositeStrategy(settings).generate_signal("BTCUSDT", candles, indicators).score
    candles[-1] = replace(candles[-1], low=118.8, close=120.0)
    current = indicators[-1]
    indicators[-1] = replace(
        current,
        close=120.0,
        vwap=119.1,
        atr14=2.0,
        rsi14=58,
        long_short_ratio=1.2,
        funding_rate=0.0001,
    )

    score, reasons, vetoes = CompositeStrategy(settings)._score_long(candles, indicators)

    assert score > 0
    assert "VWAP pullback held; average cost support favors long" in reasons
    assert "VWAP pullback held; average cost support favors long" not in vetoes


def test_strategy_rewards_resistance_grind_breakout_short_squeeze() -> None:
    candles, derivatives = _trending_market()
    indicators = build_indicators(candles, derivatives)
    settings = StrategySettings(score_threshold=999, watch_threshold=1, strict_trend_entry=False)
    resistance = max(candle.high for candle in candles[-21:-6])
    for offset in range(5, 0, -1):
        idx = len(candles) - offset
        candles[idx] = replace(
            candles[idx],
            open=resistance - 0.45,
            high=resistance + 0.15,
            low=resistance - 0.90,
            close=resistance - 0.20,
            volume=3_000,
        )
        indicators[idx] = replace(indicators[idx], close=resistance - 0.20, atr14=1.0, volume_ratio=1.2)
    latest = candles[-1]
    candles[-1] = replace(latest, close=resistance + 1.1, high=resistance + 1.4, low=resistance - 0.3, volume=5_000)
    indicators[-1] = replace(indicators[-1], close=resistance + 1.1, atr14=1.0, volume_ratio=1.5)

    signal = CompositeStrategy(settings).generate_signal("BTCUSDT", candles, indicators)

    assert "market structure: resistance grind broke upward, shorts may be squeezed" in signal.reasons


def test_strategy_rewards_volume_breakout_quiet_retest_and_restart() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    derivatives: list[DerivativesSnapshot] = []
    oi = 10_000.0
    for idx in range(60):
        timestamp = start + timedelta(minutes=15 * idx)
        candles.append(Candle(timestamp=timestamp, open=100.0, high=100.5, low=99.5, close=100.0, volume=1_000))
        oi += 10
        derivatives.append(DerivativesSnapshot(timestamp=timestamp, open_interest=oi, long_short_ratio=1.1, funding_rate=0.0001))

    breakout_idx = len(candles) - 8
    candles[breakout_idx] = replace(candles[breakout_idx], open=100.2, high=102.2, low=100.1, close=101.6, volume=3_000)
    for offset in range(7, 1, -1):
        idx = len(candles) - offset
        candles[idx] = replace(candles[idx], open=101.2, high=101.5, low=100.7, close=101.0, volume=800)
    candles[-1] = replace(candles[-1], open=101.0, high=102.5, low=100.8, close=102.1, volume=3_200)
    indicators = build_indicators(candles, derivatives)
    settings = StrategySettings(score_threshold=999, watch_threshold=1, strict_trend_entry=False)
    indicators = list(indicators)
    indicators[breakout_idx] = replace(indicators[breakout_idx], atr14=1.0, volume_ratio=1.6)
    for offset in range(7, 1, -1):
        idx = len(candles) - offset
        indicators[idx] = replace(indicators[idx], atr14=1.0, volume_ratio=0.7)
    indicators[-1] = replace(indicators[-1], atr14=1.0, volume_ratio=1.3)

    score, reason = _volume_breakout_retest_confirmation(
        candles,
        indicators,
        MarketStructure(resistance=100.5),
        "LONG",
        settings,
    )

    assert score == 14
    assert reason == "volume pattern confirms long: breakout volume, quiet retest, renewed buying"


def test_strategy_rewards_washout_with_oi_drop_and_key_level_reclaim() -> None:
    candles, derivatives = _trending_market()
    indicators = build_indicators(candles, derivatives)
    settings = StrategySettings(score_threshold=999, watch_threshold=1, strict_trend_entry=False)
    support = min(candle.low for candle in candles[-21:-1])
    latest = candles[-1]
    candles[-1] = replace(latest, open=support + 0.5, high=support + 1.3, low=support - 1.5, close=support + 0.7, volume=5_000)
    indicators[-1] = replace(
        indicators[-1],
        close=support + 0.7,
        ema20=support + 0.2,
        boll_mid=support + 0.2,
        atr14=1.0,
        volume_ratio=1.5,
        oi_change=-0.006,
        rsi14=55,
        long_short_ratio=1.1,
        funding_rate=0.0001,
    )

    signal = CompositeStrategy(settings).generate_signal("BTCUSDT", candles, indicators)

    assert "washout confirmed: downside wick swept support, OI dropped, close reclaimed key level" in signal.reasons


def test_base_timeframe_oi_flush_does_not_claim_a_four_hour_oi_valley() -> None:
    candles, derivatives = _trending_market()
    candles = _rewrite_recent_prices(candles, start_price=118.0, step=0.12, lower_wicks=True)
    indicators = build_indicators(candles, derivatives)
    indicators = _rewrite_recent_derivatives(
        indicators,
        oi_values=[9_700, 9_850, 10_120, 10_350, 10_200, 9_880, 9_520, 9_420, 9_500, 9_560, 9_610, 9_660, 9_700, 9_760, 9_820, 9_900],
        ratio_start=1.15,
        ratio_step=0.0,
    )

    cycle = CompositeStrategy().smart_money_cycle(candles, indicators)

    assert cycle.phase == "NEUTRAL"
    assert cycle.long_bias == 0
    assert cycle.short_veto is None


def test_smart_money_detects_trapped_longs_markdown() -> None:
    candles, derivatives = _trending_market()
    candles = _rewrite_recent_prices(candles, start_price=126.0, step=-0.16, lower_wicks=False)
    indicators = build_indicators(candles, derivatives)
    indicators = _rewrite_recent_derivatives(
        indicators,
        oi_values=[10_000, 10_030, 10_060, 10_080, 10_110, 10_140, 10_170, 10_200, 10_230, 10_260, 10_290, 10_320, 10_350, 10_380, 10_410, 10_440],
        ratio_start=1.10,
        ratio_step=0.015,
    )

    cycle = CompositeStrategy().smart_money_cycle(candles, indicators)

    assert cycle.phase == "TRAPPED_LONGS_MARKDOWN"
    assert cycle.short_bias > cycle.long_bias
    assert cycle.long_veto == "trapped longs are increasing while price falls"


def _trending_market() -> tuple[list[Candle], list[DerivativesSnapshot]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    derivatives: list[DerivativesSnapshot] = []
    price = 100.0
    oi = 10_000.0
    for idx in range(240):
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


def _rewrite_recent_prices(candles: list[Candle], *, start_price: float, step: float, lower_wicks: bool) -> list[Candle]:
    out = list(candles)
    for offset in range(16):
        idx = len(out) - 16 + offset
        open_price = start_price + step * offset
        close = open_price + step * 0.7
        body_high = max(open_price, close)
        body_low = min(open_price, close)
        if lower_wicks and offset in {3, 11}:
            high = body_high + 0.25
            low = body_low - 1.2
        elif not lower_wicks and offset in {4, 10}:
            high = body_high + 1.2
            low = body_low - 0.25
        else:
            high = body_high + 0.35
            low = body_low - 0.35
        out[idx] = replace(out[idx], open=open_price, high=high, low=low, close=close, volume=2_000)
    return out


def _rewrite_recent_derivatives(
    indicators,
    *,
    oi_values: list[float],
    ratio_start: float,
    ratio_step: float,
):
    out = list(indicators)
    for offset, oi in enumerate(oi_values):
        idx = len(out) - len(oi_values) + offset
        out[idx] = replace(
            out[idx],
            open_interest=oi,
            volume_ratio=1.5,
            atr14=1.0,
            long_short_ratio=ratio_start + ratio_step * offset,
        )
    return out
