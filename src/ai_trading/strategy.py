from __future__ import annotations

from collections.abc import Sequence

from ai_trading.config import StrategySettings
from ai_trading.models import Candle, IndicatorSnapshot, MarketRegime, SignalAction, StrategySignal


class CompositeStrategy:
    """EMA/MA/BOLL/VOL/OI/long-short/RSI/funding scoring strategy."""

    def __init__(self, settings: StrategySettings | None = None) -> None:
        self.settings = settings or StrategySettings()

    def generate_signal(
        self,
        symbol: str,
        candles: Sequence[Candle],
        indicators: Sequence[IndicatorSnapshot],
    ) -> StrategySignal:
        if len(candles) < self.settings.ma_trend or len(indicators) != len(candles):
            return StrategySignal(
                symbol=symbol,
                timestamp=candles[-1].timestamp,
                action=SignalAction.NO_TRADE,
                regime=MarketRegime.INSUFFICIENT_DATA,
                score=0,
                reasons=("not enough candles for MA trend filter",),
            )

        current = indicators[-1]
        if not _has_required_indicator_values(current):
            return StrategySignal(
                symbol=symbol,
                timestamp=current.timestamp,
                action=SignalAction.NO_TRADE,
                regime=MarketRegime.INSUFFICIENT_DATA,
                score=0,
                reasons=("latest candle lacks required indicators",),
                indicators=current,
            )

        regime = self._detect_regime(current)
        long_score, long_reasons, long_vetoes = self._score_long(candles, indicators)
        short_score, short_reasons, short_vetoes = self._score_short(candles, indicators)

        if long_vetoes and short_vetoes:
            return StrategySignal(
                symbol=symbol,
                timestamp=current.timestamp,
                action=SignalAction.NO_TRADE,
                regime=MarketRegime.OVERCROWDED,
                score=max(long_score, short_score),
                vetoes=tuple(sorted(set(long_vetoes + short_vetoes))),
                reasons=("both sides failed hard filters",),
                indicators=current,
            )

        if not long_vetoes and long_score >= self.settings.score_threshold and long_score >= short_score:
            return StrategySignal(
                symbol=symbol,
                timestamp=current.timestamp,
                action=SignalAction.ENTRY_LONG,
                regime=regime,
                score=long_score,
                reasons=tuple(long_reasons),
                indicators=current,
            )

        if not short_vetoes and short_score >= self.settings.score_threshold:
            return StrategySignal(
                symbol=symbol,
                timestamp=current.timestamp,
                action=SignalAction.ENTRY_SHORT,
                regime=regime,
                score=short_score,
                reasons=tuple(short_reasons),
                indicators=current,
            )

        best_score = max(long_score if not long_vetoes else 0, short_score if not short_vetoes else 0)
        if best_score >= self.settings.watch_threshold:
            reasons = long_reasons if long_score >= short_score else short_reasons
            return StrategySignal(
                symbol=symbol,
                timestamp=current.timestamp,
                action=SignalAction.WATCH,
                regime=regime,
                score=best_score,
                vetoes=tuple(long_vetoes if long_score >= short_score else short_vetoes),
                reasons=tuple(reasons),
                indicators=current,
            )

        return StrategySignal(
            symbol=symbol,
            timestamp=current.timestamp,
            action=SignalAction.NO_TRADE,
            regime=regime,
            score=best_score,
            vetoes=tuple(long_vetoes if long_score >= short_score else short_vetoes),
            reasons=("score below trading threshold",),
            indicators=current,
        )

    def exit_signal(self, position_side: str, indicators: IndicatorSnapshot) -> StrategySignal | None:
        if position_side == "LONG":
            if indicators.ema50 is not None and indicators.close < indicators.ema50:
                return StrategySignal(
                    symbol="",
                    timestamp=indicators.timestamp,
                    action=SignalAction.EXIT_LONG,
                    regime=self._detect_regime(indicators),
                    score=100,
                    reasons=("long invalidated: close below EMA50",),
                    indicators=indicators,
                )
            if indicators.rsi14 is not None and indicators.rsi14 > 75 and (indicators.volume_ratio or 0) >= self.settings.volume_extreme_ratio:
                return StrategySignal(
                    symbol="",
                    timestamp=indicators.timestamp,
                    action=SignalAction.EXIT_LONG,
                    regime=self._detect_regime(indicators),
                    score=90,
                    reasons=("long take profit: RSI overheated with extreme volume",),
                    indicators=indicators,
                )
        if position_side == "SHORT":
            if indicators.ema50 is not None and indicators.close > indicators.ema50:
                return StrategySignal(
                    symbol="",
                    timestamp=indicators.timestamp,
                    action=SignalAction.EXIT_SHORT,
                    regime=self._detect_regime(indicators),
                    score=100,
                    reasons=("short invalidated: close above EMA50",),
                    indicators=indicators,
                )
            if indicators.rsi14 is not None and indicators.rsi14 < 25 and (indicators.volume_ratio or 0) >= self.settings.volume_extreme_ratio:
                return StrategySignal(
                    symbol="",
                    timestamp=indicators.timestamp,
                    action=SignalAction.EXIT_SHORT,
                    regime=self._detect_regime(indicators),
                    score=90,
                    reasons=("short take profit: RSI oversold with extreme volume",),
                    indicators=indicators,
                )
        return None

    def _detect_regime(self, current: IndicatorSnapshot) -> MarketRegime:
        if _any_none(current.ema20, current.ema50, current.boll_mid, current.boll_upper, current.boll_lower, current.rsi14):
            return MarketRegime.INSUFFICIENT_DATA
        assert current.ema20 is not None
        assert current.ema50 is not None
        assert current.boll_mid is not None
        assert current.boll_upper is not None
        assert current.boll_lower is not None

        boll_width = (current.boll_upper - current.boll_lower) / current.boll_mid if current.boll_mid else 0
        ema_gap = abs(current.ema20 - current.ema50) / current.close
        rsi_mid = current.rsi14 is not None and 45 <= current.rsi14 <= 55
        if ema_gap < 0.003 and boll_width < 0.025 and rsi_mid:
            return MarketRegime.CHOP
        if current.long_short_ratio is not None:
            if current.long_short_ratio >= self.settings.long_short_overcrowded_long:
                return MarketRegime.OVERCROWDED
            if current.long_short_ratio <= self.settings.long_short_overcrowded_short:
                return MarketRegime.OVERCROWDED
        if current.ema20 > current.ema50 and (current.ema50_slope or 0) > 0:
            return MarketRegime.TREND_LONG
        if current.ema20 < current.ema50 and (current.ema50_slope or 0) < 0:
            return MarketRegime.TREND_SHORT
        return MarketRegime.CHOP

    def _score_long(self, candles: Sequence[Candle], indicators: Sequence[IndicatorSnapshot]) -> tuple[int, list[str], list[str]]:
        current = indicators[-1]
        previous = indicators[-2]
        score = 0
        reasons: list[str] = []
        vetoes = self._common_long_vetoes(current)

        if current.ema20 and current.ema50 and current.ma100:
            if current.close > current.ema50 and current.ema20 > current.ema50 and (current.ema50_slope or 0) > 0:
                score += 20
                reasons.append("EMA20 above EMA50 and EMA50 rising")
            if current.close > current.ma100:
                score += 5
                reasons.append("price above MA100 trend filter")

        if current.boll_mid and current.ema20 and current.atr14:
            reclaimed_mid = previous.close < (previous.boll_mid or previous.close) and current.close > current.boll_mid
            near_pullback_zone = min(abs(current.close - current.ema20), abs(current.close - current.boll_mid)) <= current.atr14 * self.settings.pullback_tolerance_atr
            not_extended = (current.close - current.ema20) <= current.atr14 * self.settings.max_extension_atr
            if (reclaimed_mid or near_pullback_zone) and not_extended:
                score += 15
                reasons.append("close confirmed near EMA20/BOLL mid without chasing upper band")

        if current.volume_ratio is not None:
            if self.settings.volume_min_ratio <= current.volume_ratio <= self.settings.volume_extreme_ratio:
                score += 15
                reasons.append("volume confirms move without extreme blow-off")
            elif current.volume_ratio >= 1.0:
                score += 7
                reasons.append("volume is acceptable but not strong")

        if current.oi_change is not None:
            if self.settings.oi_mild_change_min <= current.oi_change < self.settings.oi_extreme_change:
                score += 15
                reasons.append("open interest rising mildly with price")
            elif abs(current.oi_change) < self.settings.oi_mild_change_min:
                score += 6
                reasons.append("open interest stable")

        if current.long_short_ratio is not None and current.long_short_ratio < self.settings.long_short_overcrowded_long:
            score += 10
            reasons.append("long/short ratio is not overcrowded long")

        if current.rsi14 is not None and 45 <= current.rsi14 <= 68:
            score += 10
            reasons.append("RSI in healthy long-trend range")

        if current.funding_rate is not None and current.funding_rate < self.settings.funding_hot_long:
            score += 10
            reasons.append("funding rate is not overheated for longs")

        return score, reasons, vetoes

    def _score_short(self, candles: Sequence[Candle], indicators: Sequence[IndicatorSnapshot]) -> tuple[int, list[str], list[str]]:
        current = indicators[-1]
        previous = indicators[-2]
        score = 0
        reasons: list[str] = []
        vetoes = self._common_short_vetoes(current)

        if current.ema20 and current.ema50 and current.ma100:
            if current.close < current.ema50 and current.ema20 < current.ema50 and (current.ema50_slope or 0) < 0:
                score += 20
                reasons.append("EMA20 below EMA50 and EMA50 falling")
            if current.close < current.ma100:
                score += 5
                reasons.append("price below MA100 trend filter")

        if current.boll_mid and current.ema20 and current.atr14:
            failed_mid = previous.close > (previous.boll_mid or previous.close) and current.close < current.boll_mid
            near_retest_zone = min(abs(current.close - current.ema20), abs(current.close - current.boll_mid)) <= current.atr14 * self.settings.pullback_tolerance_atr
            not_extended = (current.ema20 - current.close) <= current.atr14 * self.settings.max_extension_atr
            if (failed_mid or near_retest_zone) and not_extended:
                score += 15
                reasons.append("close confirmed failed retest near EMA20/BOLL mid")

        if current.volume_ratio is not None:
            if self.settings.volume_min_ratio <= current.volume_ratio <= self.settings.volume_extreme_ratio:
                score += 15
                reasons.append("volume confirms sell pressure without capitulation chase")
            elif current.volume_ratio >= 1.0:
                score += 7
                reasons.append("volume is acceptable but not strong")

        if current.oi_change is not None:
            if self.settings.oi_mild_change_min <= current.oi_change < self.settings.oi_extreme_change:
                score += 15
                reasons.append("open interest rising mildly with falling price")
            elif abs(current.oi_change) < self.settings.oi_mild_change_min:
                score += 6
                reasons.append("open interest stable")

        if current.long_short_ratio is not None and current.long_short_ratio > self.settings.long_short_overcrowded_short:
            score += 10
            reasons.append("long/short ratio is not overcrowded short")

        if current.rsi14 is not None and 32 <= current.rsi14 <= 55:
            score += 10
            reasons.append("RSI in healthy short-trend range")

        if current.funding_rate is not None and current.funding_rate > self.settings.funding_hot_short:
            score += 10
            reasons.append("funding rate is not overheated for shorts")

        return score, reasons, vetoes

    def _common_long_vetoes(self, current: IndicatorSnapshot) -> list[str]:
        vetoes: list[str] = []
        if current.rsi14 is not None and current.rsi14 > 75:
            vetoes.append("RSI overheated for long entry")
        if current.long_short_ratio is not None and current.long_short_ratio >= self.settings.long_short_overcrowded_long:
            vetoes.append("long side overcrowded")
        if current.funding_rate is not None and current.funding_rate >= self.settings.funding_hot_long:
            vetoes.append("funding too hot for long entry")
        if current.oi_change is not None and current.oi_change >= self.settings.oi_extreme_change:
            vetoes.append("open interest spike risks liquidation sweep")
        if current.boll_upper is not None and current.close > current.boll_upper:
            vetoes.append("price closed above upper BOLL; no chase")
        return vetoes

    def _common_short_vetoes(self, current: IndicatorSnapshot) -> list[str]:
        vetoes: list[str] = []
        if current.rsi14 is not None and current.rsi14 < 25:
            vetoes.append("RSI oversold for short entry")
        if current.long_short_ratio is not None and current.long_short_ratio <= self.settings.long_short_overcrowded_short:
            vetoes.append("short side overcrowded")
        if current.funding_rate is not None and current.funding_rate <= self.settings.funding_hot_short:
            vetoes.append("funding too negative for short entry")
        if current.oi_change is not None and current.oi_change >= self.settings.oi_extreme_change:
            vetoes.append("open interest spike risks liquidation sweep")
        if current.boll_lower is not None and current.close < current.boll_lower:
            vetoes.append("price closed below lower BOLL; no chase")
        return vetoes


def _has_required_indicator_values(current: IndicatorSnapshot) -> bool:
    return not _any_none(
        current.ema20,
        current.ema50,
        current.ma100,
        current.boll_mid,
        current.boll_upper,
        current.boll_lower,
        current.rsi14,
        current.atr14,
        current.volume_ratio,
    )


def _any_none(*values: object) -> bool:
    return any(value is None for value in values)
