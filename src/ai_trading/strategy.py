from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from ai_trading.config import StrategySettings
from ai_trading.models import Candle, IndicatorSnapshot, MarketRegime, PositionSide, SignalAction, StrategySignal


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
        best_is_long = long_score >= short_score
        best_score = long_score if best_is_long else short_score
        best_reasons = long_reasons if best_is_long else short_reasons
        best_vetoes = long_vetoes if best_is_long else short_vetoes
        direction = PositionSide.LONG if best_is_long else PositionSide.SHORT
        signal_regime = (
            MarketRegime.OVERCROWDED
            if long_vetoes and short_vetoes
            else regime
        )

        if best_score >= self.settings.score_threshold:
            return StrategySignal(
                symbol=symbol,
                timestamp=current.timestamp,
                action=(
                    SignalAction.ENTRY_LONG
                    if best_is_long
                    else SignalAction.ENTRY_SHORT
                ),
                regime=signal_regime,
                score=best_score,
                direction=direction,
                vetoes=tuple(best_vetoes),
                reasons=tuple(best_reasons),
                indicators=current,
            )

        if best_score >= self.settings.watch_threshold:
            return StrategySignal(
                symbol=symbol,
                timestamp=current.timestamp,
                action=SignalAction.WATCH,
                regime=signal_regime,
                score=best_score,
                direction=direction,
                vetoes=tuple(best_vetoes),
                reasons=tuple(best_reasons),
                indicators=current,
            )

        return StrategySignal(
            symbol=symbol,
            timestamp=current.timestamp,
            action=SignalAction.NO_TRADE,
            regime=signal_regime,
            score=best_score,
            direction=direction,
            vetoes=tuple(best_vetoes),
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

    def smart_money_cycle(self, candles: Sequence[Candle], indicators: Sequence[IndicatorSnapshot]) -> "SmartMoneyCycle":
        return _detect_smart_money_cycle(candles, indicators, self.settings)

    def trend_state(self, position_side: str | None, indicators: Sequence[IndicatorSnapshot]) -> str:
        if not indicators:
            return "CHOP"
        current = indicators[-1]
        regime = self._detect_regime(current)
        if regime == MarketRegime.CHOP:
            return "CHOP"
        if self.strong_trend_mode(position_side or ("LONG" if regime == MarketRegime.TREND_LONG else "SHORT"), indicators):
            return "ONE_WAY_UP" if regime == MarketRegime.TREND_LONG else "ONE_WAY_DOWN"
        if regime == MarketRegime.TREND_LONG:
            return "TREND_LONG"
        if regime == MarketRegime.TREND_SHORT:
            return "TREND_SHORT"
        return "CHOP"

    def risk_state(self, indicators: Sequence[IndicatorSnapshot]) -> str:
        if not indicators:
            return "NORMAL"
        current = indicators[-1]
        ratio_surge = _ratio_change(indicators[-4:])
        if current.oi_change is not None and abs(current.oi_change) >= self.settings.oi_extreme_change:
            return "OI_ABNORMAL"
        if current.funding_rate is not None and (
            current.funding_rate >= self.settings.funding_hot_long or current.funding_rate <= self.settings.funding_hot_short
        ):
            return "FUNDING_HOT"
        if current.long_short_ratio is not None:
            if current.long_short_ratio >= self.settings.long_short_overcrowded_long or ratio_surge >= 0.08:
                return "LONG_CROWD"
            if current.long_short_ratio <= self.settings.long_short_overcrowded_short or ratio_surge <= -0.08:
                return "SHORT_CROWD"
        return "NORMAL"

    def strong_trend_mode(self, position_side: str, indicators: Sequence[IndicatorSnapshot]) -> bool:
        if len(indicators) < 6:
            return False
        current = indicators[-1]
        previous = indicators[-2]
        if _any_none(current.ema20, current.ema50, current.atr14, current.volume_ratio):
            return False
        assert current.ema20 is not None
        assert current.ema50 is not None
        assert previous.ema20 is not None or previous.ema20 is None
        ema_gap_expanding = previous.ema20 is not None and previous.ema50 is not None and abs(current.ema20 - current.ema50) > abs(previous.ema20 - previous.ema50)
        oi_ok = current.oi_change is None or current.oi_change > -self.settings.oi_extreme_change
        volume_ok = current.volume_ratio is None or current.volume_ratio >= 0.8
        funding_ok_long = current.funding_rate is None or current.funding_rate < self.settings.funding_hot_long
        funding_ok_short = current.funding_rate is None or current.funding_rate > self.settings.funding_hot_short
        recent = indicators[-5:]
        closes_above_ema20 = sum(1 for item in recent if item.ema20 is not None and item.close >= item.ema20)
        closes_below_ema20 = sum(1 for item in recent if item.ema20 is not None and item.close <= item.ema20)
        boll_expanding = _boll_width(current) > _boll_width(indicators[-3])
        rsi_long_ok = current.rsi14 is None or 55 <= current.rsi14 <= 72
        rsi_short_ok = current.rsi14 is None or 28 <= current.rsi14 <= 45
        if position_side == "LONG":
            checks = [
                current.close >= current.ema20 and current.ema20 > current.ema50,
                (current.ema50_slope or 0) > 0,
                ema_gap_expanding,
                closes_above_ema20 >= 4,
                boll_expanding,
                oi_ok,
                volume_ok,
                funding_ok_long,
                rsi_long_ok,
            ]
            return sum(bool(item) for item in checks) >= 7
        if position_side == "SHORT":
            checks = [
                current.close <= current.ema20 and current.ema20 < current.ema50,
                (current.ema50_slope or 0) < 0,
                ema_gap_expanding,
                closes_below_ema20 >= 4,
                boll_expanding,
                oi_ok,
                volume_ok,
                funding_ok_short,
                rsi_short_ok,
            ]
            return sum(bool(item) for item in checks) >= 7
        return False

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
        cycle = self.smart_money_cycle(candles, indicators)
        structure = _market_structure(candles, indicators, self.settings)
        strict_vetoes = self._strict_long_vetoes(candles, indicators) if self.settings.strict_trend_entry else []
        score = 0
        reasons: list[str] = []
        vetoes = self._common_long_vetoes(current) + strict_vetoes
        sweep_veto = _sweep_veto(candles[-1], current, side="LONG", settings=self.settings)
        if sweep_veto:
            vetoes.append(sweep_veto)

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

        latest_candle = candles[-1]
        if current.vwap is not None and current.atr14 is not None and current.atr14 > 0:
            if latest_candle.low <= current.vwap + current.atr14 * self.settings.vwap_near_atr and current.close >= current.vwap:
                score += 6
                reasons.append("VWAP pullback held; average cost support favors long")
            elif current.close > current.vwap + current.atr14 * self.settings.vwap_extension_atr:
                score -= 6
                reasons.append("price extended far above VWAP; chasing long risk")

        if current.kc_mid is not None and current.atr14 is not None and current.atr14 > 0:
            if latest_candle.low <= current.kc_mid + current.atr14 * self.settings.keltner_near_atr and current.close >= current.kc_mid:
                score += 5
                reasons.append("KC mid pullback held; volatility channel support favors long")

        if current.quote_flow_ratio is not None:
            if self.settings.qps_min_ratio <= current.quote_flow_ratio <= self.settings.qps_extreme_ratio and current.close >= previous.close:
                score += 5
                reasons.append("QPS quote flow accelerates with price; traded value confirms long")
            elif current.quote_flow_ratio > self.settings.qps_extreme_ratio and current.close < previous.close:
                reasons.append("QPS blow-off without price follow-through; long risk")

        if structure.long_confirmed:
            score += 12
            reasons.append("market structure confirms long: breakout or retest held")

        if structure.long_breakout_after_grind:
            score += 10
            reasons.append("market structure: resistance grind broke upward, shorts may be squeezed")

        volume_score, volume_reason = _volume_breakout_retest_confirmation(candles, indicators, structure, "LONG", self.settings)
        if volume_reason:
            score += volume_score
            reasons.append(volume_reason)

        if _long_washout_confirmed(candles[-1], current, structure, self.settings):
            score += 14
            reasons.append("washout confirmed: downside wick swept support, OI dropped, close reclaimed key level")

        if _lower_sweep_reclaimed(candles[-1], current, self.settings):
            score += 8
            reasons.append("downside sweep reclaimed support; stop-run filter favors long")

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

        if cycle.long_bias > 0:
            score += cycle.long_bias
            reasons.append(cycle.reason)
        if cycle.long_veto:
            vetoes.append(cycle.long_veto)

        return score, reasons, vetoes

    def _score_short(self, candles: Sequence[Candle], indicators: Sequence[IndicatorSnapshot]) -> tuple[int, list[str], list[str]]:
        current = indicators[-1]
        previous = indicators[-2]
        cycle = self.smart_money_cycle(candles, indicators)
        structure = _market_structure(candles, indicators, self.settings)
        strict_vetoes = self._strict_short_vetoes(candles, indicators) if self.settings.strict_trend_entry else []
        score = 0
        reasons: list[str] = []
        vetoes = self._common_short_vetoes(current) + strict_vetoes
        sweep_veto = _sweep_veto(candles[-1], current, side="SHORT", settings=self.settings)
        if sweep_veto:
            vetoes.append(sweep_veto)

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

        latest_candle = candles[-1]
        if current.vwap is not None and current.atr14 is not None and current.atr14 > 0:
            if latest_candle.high >= current.vwap - current.atr14 * self.settings.vwap_near_atr and current.close <= current.vwap:
                score += 6
                reasons.append("VWAP retest rejected; average cost resistance favors short")
            elif current.close < current.vwap - current.atr14 * self.settings.vwap_extension_atr:
                score -= 6
                reasons.append("price extended far below VWAP; chasing short risk")

        if current.kc_mid is not None and current.atr14 is not None and current.atr14 > 0:
            if latest_candle.high >= current.kc_mid - current.atr14 * self.settings.keltner_near_atr and current.close <= current.kc_mid:
                score += 5
                reasons.append("KC mid retest rejected; volatility channel resistance favors short")

        if current.quote_flow_ratio is not None:
            if self.settings.qps_min_ratio <= current.quote_flow_ratio <= self.settings.qps_extreme_ratio and current.close <= previous.close:
                score += 5
                reasons.append("QPS quote flow accelerates with price; traded value confirms short")
            elif current.quote_flow_ratio > self.settings.qps_extreme_ratio and current.close > previous.close:
                reasons.append("QPS blow-off without price follow-through; short risk")

        if structure.short_confirmed:
            score += 12
            reasons.append("market structure confirms short: breakdown or retest failed")

        if structure.short_breakdown_after_grind:
            score += 10
            reasons.append("market structure: support grind broke downward, longs may be liquidated")

        volume_score, volume_reason = _volume_breakout_retest_confirmation(candles, indicators, structure, "SHORT", self.settings)
        if volume_reason:
            score += volume_score
            reasons.append(volume_reason)

        if _short_washout_confirmed(candles[-1], current, structure, self.settings):
            score += 14
            reasons.append("washout confirmed: upside wick swept resistance, OI dropped, close rejected key level")

        if _upper_sweep_rejected(candles[-1], current, self.settings):
            score += 8
            reasons.append("upside sweep rejected resistance; stop-run filter favors short")

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

        if cycle.short_bias > 0:
            score += cycle.short_bias
            reasons.append(cycle.reason)
        if cycle.short_veto:
            vetoes.append(cycle.short_veto)

        return score, reasons, vetoes

    def _common_long_vetoes(self, current: IndicatorSnapshot) -> list[str]:
        vetoes: list[str] = []
        if _atr_pct(current) >= self.settings.extreme_atr_pct:
            vetoes.append("extreme volatility: skip new long entry")
        if current.long_short_ratio is not None and current.long_short_ratio >= self.settings.long_short_overcrowded_long:
            vetoes.append("long side overcrowded")
        if current.funding_rate is not None and current.funding_rate >= self.settings.funding_hot_long:
            vetoes.append("funding too hot for long entry")
        if current.oi_change is not None and current.oi_change >= self.settings.oi_extreme_change:
            vetoes.append("open interest spike risks liquidation sweep")
        return vetoes

    def _strict_long_vetoes(self, candles: Sequence[Candle], indicators: Sequence[IndicatorSnapshot]) -> list[str]:
        current = indicators[-1]
        previous = indicators[-2]
        vetoes: list[str] = []
        if _any_none(current.ema20, current.ema50, current.ema200) or not (current.ema20 > current.ema50 > current.ema200):
            vetoes.append("strict long blocked: EMA20/EMA50/EMA200 not bullish")
        if current.boll_mid is None or previous.boll_mid is None or not (current.close > current.boll_mid and previous.close > previous.boll_mid):
            vetoes.append("strict long blocked: BOLL mid not confirmed twice")
        if current.rsi14 is None or not (52 <= current.rsi14 <= 72):
            vetoes.append("strict long blocked: RSI not in 52-72")
        if current.volume_ratio is None or current.volume_ratio < self.settings.volume_min_ratio:
            vetoes.append("strict long blocked: volume below 1.5x average")
        oi_4h_change = _oi_change_over_window(indicators, self.settings.smart_money_window)
        if oi_4h_change is None or oi_4h_change < self.settings.oi_4h_entry_min:
            vetoes.append("strict long blocked: 4h OI increase below 3%")
        if current.long_short_ratio is None or current.long_short_ratio < self.settings.top_long_short_long_min:
            vetoes.append("strict long blocked: top long/short ratio below 1.1")
        if current.funding_rate is None or not (self.settings.funding_long_min <= current.funding_rate <= self.settings.funding_long_max):
            vetoes.append("strict long blocked: funding outside long range")
        if _ema_gap_too_small(current):
            vetoes.append("strict long blocked: EMA lines are too compressed")
        if current.rsi14 is not None and 48 <= current.rsi14 <= 52:
            vetoes.append("strict long blocked: RSI neutral zone")
        if _recent_amplitude(candles[-1]) > 0.05:
            vetoes.append("strict long blocked: 1h candle amplitude above 5%")
        return vetoes

    def _common_short_vetoes(self, current: IndicatorSnapshot) -> list[str]:
        vetoes: list[str] = []
        if _atr_pct(current) >= self.settings.extreme_atr_pct:
            vetoes.append("extreme volatility: skip new short entry")
        if current.long_short_ratio is not None and current.long_short_ratio <= self.settings.long_short_overcrowded_short:
            vetoes.append("short side overcrowded")
        if current.funding_rate is not None and current.funding_rate <= self.settings.funding_hot_short:
            vetoes.append("funding too negative for short entry")
        if current.oi_change is not None and current.oi_change >= self.settings.oi_extreme_change:
            vetoes.append("open interest spike risks liquidation sweep")
        return vetoes

    def _strict_short_vetoes(self, candles: Sequence[Candle], indicators: Sequence[IndicatorSnapshot]) -> list[str]:
        current = indicators[-1]
        previous = indicators[-2]
        vetoes: list[str] = []
        if _any_none(current.ema20, current.ema50, current.ema200) or not (current.ema20 < current.ema50 < current.ema200):
            vetoes.append("strict short blocked: EMA20/EMA50/EMA200 not bearish")
        if current.boll_mid is None or previous.boll_mid is None or not (current.close < current.boll_mid and previous.close < previous.boll_mid):
            vetoes.append("strict short blocked: BOLL mid not confirmed twice")
        if current.rsi14 is None or not (28 <= current.rsi14 <= 48):
            vetoes.append("strict short blocked: RSI not in 28-48")
        if current.volume_ratio is None or current.volume_ratio < self.settings.volume_min_ratio:
            vetoes.append("strict short blocked: volume below 1.5x average")
        oi_4h_change = _oi_change_over_window(indicators, self.settings.smart_money_window)
        if oi_4h_change is None or oi_4h_change < self.settings.oi_4h_entry_min:
            vetoes.append("strict short blocked: 4h OI increase below 3%")
        if current.long_short_ratio is None or current.long_short_ratio > self.settings.top_long_short_short_max:
            vetoes.append("strict short blocked: top long/short ratio above 0.9")
        if current.funding_rate is None or not (self.settings.funding_short_min <= current.funding_rate <= self.settings.funding_short_max):
            vetoes.append("strict short blocked: funding outside short range")
        if _ema_gap_too_small(current):
            vetoes.append("strict short blocked: EMA lines are too compressed")
        if current.rsi14 is not None and 48 <= current.rsi14 <= 52:
            vetoes.append("strict short blocked: RSI neutral zone")
        if _recent_amplitude(candles[-1]) > 0.05:
            vetoes.append("strict short blocked: 1h candle amplitude above 5%")
        return vetoes


@dataclass(frozen=True)
class SmartMoneyCycle:
    phase: str = "NEUTRAL"
    reason: str = "smart money cycle neutral"
    long_bias: int = 0
    short_bias: int = 0
    long_veto: str | None = None
    short_veto: str | None = None


@dataclass(frozen=True)
class MarketStructure:
    support: float | None = None
    resistance: float | None = None
    long_confirmed: bool = False
    short_confirmed: bool = False
    long_breakout_after_grind: bool = False
    short_breakdown_after_grind: bool = False


def _market_structure(
    candles: Sequence[Candle],
    indicators: Sequence[IndicatorSnapshot],
    settings: StrategySettings,
) -> MarketStructure:
    lookback = min(settings.structure_lookback, len(candles) - 1, len(indicators) - 1)
    if lookback < 6:
        return MarketStructure()
    current_candle = candles[-1]
    previous_candle = candles[-2]
    current = indicators[-1]
    atr = current.atr14 or 0.0
    buffer = atr * settings.structure_buffer_atr
    prior = candles[-lookback - 1 : -1]
    support = min(candle.low for candle in prior)
    resistance = max(candle.high for candle in prior)
    volume_ok = current.volume_ratio is None or current.volume_ratio >= 1.0
    recent = candles[-min(settings.structure_grind_bars + 1, len(candles) - 1) : -1]
    grind_tolerance = atr * settings.structure_grind_tolerance_atr
    grinding_resistance = _grinding_near_resistance(recent, resistance, grind_tolerance)
    grinding_support = _grinding_near_support(recent, support, grind_tolerance)

    long_breakout = current_candle.close > resistance + buffer and volume_ok
    long_retest = current_candle.low <= resistance + buffer and current_candle.close > resistance and previous_candle.close > resistance - buffer
    short_breakdown = current_candle.close < support - buffer and volume_ok
    short_retest = current_candle.high >= support - buffer and current_candle.close < support and previous_candle.close < support + buffer
    return MarketStructure(
        support=support,
        resistance=resistance,
        long_confirmed=long_breakout or long_retest,
        short_confirmed=short_breakdown or short_retest,
        long_breakout_after_grind=long_breakout and grinding_resistance,
        short_breakdown_after_grind=short_breakdown and grinding_support,
    )


def _grinding_near_resistance(candles: Sequence[Candle], resistance: float, tolerance: float) -> bool:
    if not candles or tolerance <= 0:
        return False
    touches = sum(1 for candle in candles if resistance - tolerance <= candle.high <= resistance + tolerance or resistance - tolerance <= candle.close <= resistance + tolerance)
    closes_below = sum(1 for candle in candles if candle.close <= resistance + tolerance)
    return touches >= max(2, len(candles) // 2) and closes_below >= len(candles) - 1


def _grinding_near_support(candles: Sequence[Candle], support: float, tolerance: float) -> bool:
    if not candles or tolerance <= 0:
        return False
    touches = sum(1 for candle in candles if support - tolerance <= candle.low <= support + tolerance or support - tolerance <= candle.close <= support + tolerance)
    closes_above = sum(1 for candle in candles if candle.close >= support - tolerance)
    return touches >= max(2, len(candles) // 2) and closes_above >= len(candles) - 1


def _volume_breakout_retest_confirmation(
    candles: Sequence[Candle],
    indicators: Sequence[IndicatorSnapshot],
    structure: MarketStructure,
    side: str,
    settings: StrategySettings,
) -> tuple[int, str | None]:
    if len(candles) < 8 or len(indicators) < 8:
        return 0, None
    current = candles[-1]
    previous = candles[-2]
    current_indicator = indicators[-1]
    atr = current_indicator.atr14
    if atr is None or atr <= 0:
        return 0, None
    buffer = atr * settings.structure_buffer_atr
    recent_start = max(1, len(candles) - 8)

    if side == "LONG":
        breakout: tuple[int, float, float] | None = None
        for idx in range(recent_start, len(candles) - 1):
            prior_window = candles[max(0, idx - settings.structure_lookback) : idx]
            if not prior_window:
                continue
            level = max(item.high for item in prior_window)
            ratio = indicators[idx].volume_ratio or 0.0
            if candles[idx].close > level + buffer and ratio >= settings.volume_breakout_ratio:
                breakout = (idx, ratio, level)
        if breakout is None:
            level = structure.resistance
            if (
                level is not None
                and current.close > level + buffer
                and (current_indicator.volume_ratio or 0.0) >= settings.volume_breakout_ratio
            ):
                return 6, "volume breakout above resistance; retest confirmation preferred"
            return 0, None
        breakout_idx, breakout_ratio, level = breakout
        pullback_held = any(
            candles[idx].low <= level + buffer * 1.25
            and candles[idx].close >= level - buffer
            and (indicators[idx].volume_ratio or 0.0) <= breakout_ratio * settings.volume_pullback_ratio
            for idx in range(breakout_idx + 1, len(candles) - 1)
        )
        restart_volume = (current_indicator.volume_ratio or 0.0) >= settings.volume_restart_ratio
        restart_up = current.close > max(previous.close, level)
        if pullback_held and restart_volume and restart_up:
            return 14, "volume pattern confirms long: breakout volume, quiet retest, renewed buying"
        if pullback_held:
            return 0, "breakout retest held quietly; waiting renewed buying volume"
        return 0, None

    breakdown: tuple[int, float, float] | None = None
    for idx in range(recent_start, len(candles) - 1):
        prior_window = candles[max(0, idx - settings.structure_lookback) : idx]
        if not prior_window:
            continue
        level = min(item.low for item in prior_window)
        ratio = indicators[idx].volume_ratio or 0.0
        if candles[idx].close < level - buffer and ratio >= settings.volume_breakout_ratio:
            breakdown = (idx, ratio, level)
    if breakdown is None:
        level = structure.support
        if (
            level is not None
            and current.close < level - buffer
            and (current_indicator.volume_ratio or 0.0) >= settings.volume_breakout_ratio
        ):
            return 6, "volume breakdown below support; retest confirmation preferred"
        return 0, None
    breakdown_idx, breakdown_ratio, level = breakdown
    retest_rejected = any(
        candles[idx].high >= level - buffer * 1.25
        and candles[idx].close <= level + buffer
        and (indicators[idx].volume_ratio or 0.0) <= breakdown_ratio * settings.volume_pullback_ratio
        for idx in range(breakdown_idx + 1, len(candles) - 1)
    )
    restart_volume = (current_indicator.volume_ratio or 0.0) >= settings.volume_restart_ratio
    restart_down = current.close < min(previous.close, level)
    if retest_rejected and restart_volume and restart_down:
        return 14, "volume pattern confirms short: breakdown volume, quiet retest, renewed selling"
    if retest_rejected:
        return 0, "breakdown retest rejected quietly; waiting renewed selling volume"
    return 0, None


def _sweep_veto(candle: Candle, indicator: IndicatorSnapshot, *, side: str, settings: StrategySettings) -> str | None:
    if indicator.atr14 is None or indicator.atr14 <= 0:
        return None
    upper_wick, lower_wick, close_position = _wick_profile(candle)
    volume_ok = indicator.volume_ratio is None or indicator.volume_ratio >= 1.1
    if side == "LONG" and upper_wick >= indicator.atr14 * settings.sweep_wick_atr and close_position <= 0.45 and volume_ok:
        return "upper wick sweep rejected; avoid chasing long"
    if side == "SHORT" and lower_wick >= indicator.atr14 * settings.sweep_wick_atr and close_position >= 0.55 and volume_ok:
        return "lower wick sweep reclaimed; avoid chasing short"
    return None


def _long_washout_confirmed(candle: Candle, indicator: IndicatorSnapshot, structure: MarketStructure, settings: StrategySettings) -> bool:
    if indicator.atr14 is None or indicator.atr14 <= 0 or structure.support is None:
        return False
    if indicator.oi_change is None or indicator.oi_change > -settings.wash_oi_drop_min:
        return False
    _, lower_wick, close_position = _wick_profile(candle)
    buffer = indicator.atr14 * settings.structure_buffer_atr
    swept_support = candle.low < structure.support - buffer
    reclaimed_support = indicator.close > structure.support + buffer * 0.25
    volume_ok = indicator.volume_ratio is None or indicator.volume_ratio >= 1.0
    return (
        swept_support
        and reclaimed_support
        and lower_wick >= indicator.atr14 * settings.sweep_wick_atr
        and close_position >= 0.55
        and volume_ok
    )


def _short_washout_confirmed(candle: Candle, indicator: IndicatorSnapshot, structure: MarketStructure, settings: StrategySettings) -> bool:
    if indicator.atr14 is None or indicator.atr14 <= 0 or structure.resistance is None:
        return False
    if indicator.oi_change is None or indicator.oi_change > -settings.wash_oi_drop_min:
        return False
    upper_wick, _, close_position = _wick_profile(candle)
    buffer = indicator.atr14 * settings.structure_buffer_atr
    swept_resistance = candle.high > structure.resistance + buffer
    rejected_resistance = indicator.close < structure.resistance - buffer * 0.25
    volume_ok = indicator.volume_ratio is None or indicator.volume_ratio >= 1.0
    return (
        swept_resistance
        and rejected_resistance
        and upper_wick >= indicator.atr14 * settings.sweep_wick_atr
        and close_position <= 0.45
        and volume_ok
    )


def _lower_sweep_reclaimed(candle: Candle, indicator: IndicatorSnapshot, settings: StrategySettings) -> bool:
    if indicator.atr14 is None or indicator.atr14 <= 0:
        return False
    _, lower_wick, close_position = _wick_profile(candle)
    near_mid = indicator.boll_mid is None or indicator.close >= indicator.boll_mid
    near_ema = indicator.ema20 is None or indicator.close >= indicator.ema20
    return lower_wick >= indicator.atr14 * settings.sweep_wick_atr and close_position >= 0.60 and near_mid and near_ema


def _upper_sweep_rejected(candle: Candle, indicator: IndicatorSnapshot, settings: StrategySettings) -> bool:
    if indicator.atr14 is None or indicator.atr14 <= 0:
        return False
    upper_wick, _, close_position = _wick_profile(candle)
    near_mid = indicator.boll_mid is None or indicator.close <= indicator.boll_mid
    near_ema = indicator.ema20 is None or indicator.close <= indicator.ema20
    return upper_wick >= indicator.atr14 * settings.sweep_wick_atr and close_position <= 0.40 and near_mid and near_ema


def _wick_profile(candle: Candle) -> tuple[float, float, float]:
    body_high = max(candle.open, candle.close)
    body_low = min(candle.open, candle.close)
    upper_wick = candle.high - body_high
    lower_wick = body_low - candle.low
    candle_range = candle.high - candle.low
    close_position = (candle.close - candle.low) / candle_range if candle_range > 0 else 0.5
    return upper_wick, lower_wick, close_position


def _detect_smart_money_cycle(
    candles: Sequence[Candle],
    indicators: Sequence[IndicatorSnapshot],
    settings: StrategySettings,
) -> SmartMoneyCycle:
    window = min(settings.smart_money_window, len(candles), len(indicators))
    if window < 8:
        return SmartMoneyCycle(reason="smart money cycle lacks enough candles")

    recent_candles = candles[-window:]
    recent_indicators = indicators[-window:]
    oi_values = [item.open_interest for item in recent_indicators if item.open_interest is not None]
    if len(oi_values) < max(6, window // 2):
        return SmartMoneyCycle(reason="smart money cycle lacks OI data")

    first_oi = oi_values[0]
    last_oi = oi_values[-1]
    min_oi = min(oi_values)
    max_oi = max(oi_values)
    if first_oi <= 0 or min_oi <= 0:
        return SmartMoneyCycle(reason="smart money cycle invalid OI data")

    oi_drop_from_high = (max_oi - last_oi) / max_oi if max_oi else 0.0
    oi_change_window = (last_oi - first_oi) / first_oi

    first_close = recent_candles[0].close
    last_close = recent_candles[-1].close
    price_change = (last_close - first_close) / first_close if first_close else 0.0
    upper_wicks = _count_wicks(recent_candles, recent_indicators, upper=True, settings=settings)
    volume_confirmed = _average_volume_ratio(recent_indicators[-5:]) >= settings.smart_money_volume_ratio
    ratio_change = _ratio_change(recent_indicators)

    oi_rising = oi_change_window >= settings.smart_money_oi_rebuild
    oi_falling_from_high = oi_drop_from_high >= settings.smart_money_oi_rebuild
    price_rising = price_change >= settings.smart_money_price_move
    price_falling = price_change <= -settings.smart_money_price_move
    upper_sweeps = upper_wicks >= settings.smart_money_min_wicks
    longs_getting_crowded = ratio_change >= 0.08
    shorts_getting_crowded = ratio_change <= -0.08

    if oi_rising and price_rising and shorts_getting_crowded and volume_confirmed:
        return SmartMoneyCycle(
            phase="SHORT_SQUEEZE_MARKUP",
            reason="short squeeze markup: price and OI rise while long/short ratio falls, shorts are being trapped",
            long_bias=14,
            short_veto="short crowd is vulnerable to a squeeze",
        )

    if price_rising and upper_sweeps and oi_falling_from_high:
        return SmartMoneyCycle(
            phase="DISTRIBUTION_EXIT",
            reason="smart money distribution: repeated upper wicks with OI falling after markup",
            short_bias=10,
            long_veto="smart money distribution after upper wick sweeps",
        )

    if price_falling and oi_change_window >= settings.smart_money_oi_trap and longs_getting_crowded:
        return SmartMoneyCycle(
            phase="TRAPPED_LONGS_MARKDOWN",
            reason="trapped longs markdown: price falls while OI and long/short ratio rise",
            short_bias=18,
            long_veto="trapped longs are increasing while price falls",
        )

    return SmartMoneyCycle()


def _count_wicks(
    candles: Sequence[Candle],
    indicators: Sequence[IndicatorSnapshot],
    *,
    upper: bool,
    settings: StrategySettings,
) -> int:
    count = 0
    for candle, indicator in zip(candles, indicators, strict=True):
        if indicator.atr14 is None or indicator.atr14 <= 0:
            continue
        body_high = max(candle.open, candle.close)
        body_low = min(candle.open, candle.close)
        wick = candle.high - body_high if upper else body_low - candle.low
        if wick >= indicator.atr14 * settings.smart_money_wick_atr:
            count += 1
    return count


def _average_volume_ratio(indicators: Sequence[IndicatorSnapshot]) -> float:
    values = [item.volume_ratio for item in indicators if item.volume_ratio is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ratio_change(indicators: Sequence[IndicatorSnapshot]) -> float:
    values = [item.long_short_ratio for item in indicators if item.long_short_ratio is not None]
    if len(values) < 2 or values[0] == 0:
        return 0.0
    return (values[-1] - values[0]) / values[0]


def _boll_width(indicator: IndicatorSnapshot) -> float:
    if indicator.boll_upper is None or indicator.boll_lower is None or not indicator.boll_mid:
        return 0.0
    return (indicator.boll_upper - indicator.boll_lower) / indicator.boll_mid


def _atr_pct(indicator: IndicatorSnapshot) -> float:
    if not indicator.close or not indicator.atr14:
        return 0.0
    return indicator.atr14 / indicator.close


def _oi_change_over_window(indicators: Sequence[IndicatorSnapshot], window: int) -> float | None:
    recent = indicators[-window:]
    values = [item.open_interest for item in recent if item.open_interest is not None]
    if len(values) < max(2, window // 2) or not values[0]:
        return None
    return (values[-1] - values[0]) / values[0]


def _ema_gap_too_small(current: IndicatorSnapshot) -> bool:
    if _any_none(current.ema20, current.ema50, current.ema200) or current.close == 0:
        return False
    assert current.ema20 is not None
    assert current.ema50 is not None
    assert current.ema200 is not None
    min_gap = min(abs(current.ema20 - current.ema50), abs(current.ema50 - current.ema200)) / current.close
    return min_gap < 0.003


def _recent_amplitude(candle: Candle) -> float:
    return (candle.high - candle.low) / candle.open if candle.open else 0.0


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
