from __future__ import annotations

from collections.abc import Sequence

from ai_trading.models import Candle, DerivativesSnapshot, IndicatorSnapshot


def sma(values: Sequence[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float | None] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        out.append(running / window if idx >= window - 1 else None)
    return out


def ema(values: Sequence[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < window:
        return out

    seed = sum(values[:window]) / window
    out[window - 1] = seed
    multiplier = 2 / (window + 1)
    previous = seed
    for idx in range(window, len(values)):
        previous = (values[idx] - previous) * multiplier + previous
        out[idx] = previous
    return out


def rolling_std(values: Sequence[float], window: int) -> list[float | None]:
    out: list[float | None] = []
    for idx in range(len(values)):
        if idx < window - 1:
            out.append(None)
            continue
        sample = values[idx - window + 1 : idx + 1]
        mean = sum(sample) / window
        variance = sum((value - mean) ** 2 for value in sample) / window
        out.append(variance**0.5)
    return out


def bollinger(values: Sequence[float], window: int, stddev: float) -> tuple[list[float | None], list[float | None], list[float | None]]:
    mid = sma(values, window)
    std = rolling_std(values, window)
    upper: list[float | None] = []
    lower: list[float | None] = []
    for mean, deviation in zip(mid, std, strict=True):
        if mean is None or deviation is None:
            upper.append(None)
            lower.append(None)
            continue
        upper.append(mean + stddev * deviation)
        lower.append(mean - stddev * deviation)
    return mid, upper, lower


def rsi(values: Sequence[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) <= window:
        return out

    gains: list[float] = []
    losses: list[float] = []
    for idx in range(1, window + 1):
        change = values[idx] - values[idx - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    out[window] = _rsi_from_avgs(avg_gain, avg_loss)

    for idx in range(window + 1, len(values)):
        change = values[idx] - values[idx - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (window - 1)) + gain) / window
        avg_loss = ((avg_loss * (window - 1)) + loss) / window
        out[idx] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def atr(candles: Sequence[Candle], window: int) -> list[float | None]:
    if len(candles) < 2:
        return [None] * len(candles)
    true_ranges: list[float] = [candles[0].high - candles[0].low]
    for idx in range(1, len(candles)):
        candle = candles[idx]
        previous_close = candles[idx - 1].close
        true_ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    return ema(true_ranges, window)


def build_indicators(
    candles: Sequence[Candle],
    derivatives: Sequence[DerivativesSnapshot] | None = None,
    *,
    ema_fast: int = 20,
    ema_slow: int = 50,
    ma_trend: int = 100,
    bollinger_window: int = 20,
    bollinger_stddev: float = 2.0,
    rsi_window: int = 14,
    atr_window: int = 14,
    volume_window: int = 20,
    keltner_window: int = 20,
    keltner_atr_multiplier: float = 2.0,
    qps_window: int = 20,
) -> list[IndicatorSnapshot]:
    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]
    typical_prices = [(candle.high + candle.low + candle.close) / 3 for candle in candles]
    ema_fast_values = ema(closes, ema_fast)
    ema_slow_values = ema(closes, ema_slow)
    ema_trend_values = ema(closes, 200)
    ma_trend_values = sma(closes, ma_trend)
    boll_mid, boll_upper, boll_lower = bollinger(closes, bollinger_window, bollinger_stddev)
    rsi_values = rsi(closes, rsi_window)
    atr_values = atr(candles, atr_window)
    volume_sma = sma(volumes, volume_window)
    kc_mid_values = ema(typical_prices, keltner_window)
    quote_flows = _quote_flow_per_second(candles, typical_prices)
    quote_flow_sma = sma(quote_flows, qps_window)

    derivatives_by_time = {item.timestamp: item for item in derivatives or []}
    previous_oi: float | None = None
    vwap_day = None
    vwap_price_volume = 0.0
    vwap_volume = 0.0
    out: list[IndicatorSnapshot] = []

    for idx, candle in enumerate(candles):
        candle_day = candle.timestamp.date()
        if candle_day != vwap_day:
            vwap_day = candle_day
            vwap_price_volume = 0.0
            vwap_volume = 0.0
        typical_price = (candle.high + candle.low + candle.close) / 3
        vwap_price_volume += typical_price * candle.volume
        vwap_volume += candle.volume
        vwap_value = vwap_price_volume / vwap_volume if vwap_volume else None
        volume_average = volume_sma[idx]
        volume_ratio = candle.volume / volume_average if volume_average else None
        atr_value = atr_values[idx]
        kc_mid = kc_mid_values[idx]
        kc_upper = kc_mid + atr_value * keltner_atr_multiplier if kc_mid is not None and atr_value is not None else None
        kc_lower = kc_mid - atr_value * keltner_atr_multiplier if kc_mid is not None and atr_value is not None else None
        quote_flow = quote_flows[idx]
        quote_flow_average = quote_flow_sma[idx]
        quote_flow_ratio = quote_flow / quote_flow_average if quote_flow_average else None
        ema_slow_value = ema_slow_values[idx]
        ema_slope = None
        if idx >= 3 and ema_slow_value is not None and ema_slow_values[idx - 3] is not None:
            ema_slope = (ema_slow_value - ema_slow_values[idx - 3]) / ema_slow_values[idx - 3]

        derivative = derivatives_by_time.get(candle.timestamp)
        open_interest = None
        oi_change = None
        long_short_ratio = None
        funding_rate = None
        if derivative is not None:
            open_interest = derivative.open_interest
            long_short_ratio = derivative.long_short_ratio
            funding_rate = derivative.funding_rate
            if derivative.open_interest is not None and previous_oi:
                oi_change = (derivative.open_interest - previous_oi) / previous_oi
            if derivative.open_interest is not None:
                previous_oi = derivative.open_interest

        out.append(
            IndicatorSnapshot(
                timestamp=candle.timestamp,
                close=candle.close,
                ema20=ema_fast_values[idx],
                ema50=ema_slow_values[idx],
                ema200=ema_trend_values[idx],
                ma100=ma_trend_values[idx],
                boll_mid=boll_mid[idx],
                boll_upper=boll_upper[idx],
                boll_lower=boll_lower[idx],
                rsi14=rsi_values[idx],
                atr14=atr_value,
                volume_sma20=volume_average,
                volume_ratio=volume_ratio,
                ema50_slope=ema_slope,
                vwap=vwap_value,
                kc_mid=kc_mid,
                kc_upper=kc_upper,
                kc_lower=kc_lower,
                quote_flow=quote_flow,
                quote_flow_ratio=quote_flow_ratio,
                open_interest=open_interest,
                oi_change=oi_change,
                long_short_ratio=long_short_ratio,
                funding_rate=funding_rate,
            )
        )
    return out


def _quote_flow_per_second(candles: Sequence[Candle], typical_prices: Sequence[float]) -> list[float]:
    out: list[float] = []
    for idx, candle in enumerate(candles):
        if idx == 0:
            seconds = 60.0
        else:
            seconds = max((candle.timestamp - candles[idx - 1].timestamp).total_seconds(), 1.0)
        out.append(typical_prices[idx] * candle.volume / seconds)
    return out


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))
