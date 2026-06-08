from __future__ import annotations

from dataclasses import dataclass

from ai_trading.models import Candle, DerivativesSnapshot


@dataclass(frozen=True)
class DataQualityReport:
    symbol: str
    accepted: bool
    score: int
    reasons: tuple[str, ...]


class UniverseSelector:
    """Select Top USDT perpetuals by volume, then reject weak data quality."""

    def __init__(
        self,
        *,
        min_candles: int = 120,
        min_quote_volume: float = 50_000_000,
        max_missing_derivative_ratio: float = 0.08,
        max_flat_candle_ratio: float = 0.05,
    ) -> None:
        self.min_candles = min_candles
        self.min_quote_volume = min_quote_volume
        self.max_missing_derivative_ratio = max_missing_derivative_ratio
        self.max_flat_candle_ratio = max_flat_candle_ratio

    def evaluate(
        self,
        symbol: str,
        candles: list[Candle],
        derivatives: list[DerivativesSnapshot] | None,
        quote_volume: float,
    ) -> DataQualityReport:
        score = 100
        reasons: list[str] = []
        if quote_volume < self.min_quote_volume:
            score -= 25
            reasons.append("quote volume below threshold")
        if len(candles) < self.min_candles:
            score -= 35
            reasons.append("not enough candles")
        if candles:
            flat_ratio = sum(1 for candle in candles if candle.high == candle.low) / len(candles)
            if flat_ratio > self.max_flat_candle_ratio:
                score -= 25
                reasons.append("too many flat candles")
            if any(candle.close <= 0 or candle.volume < 0 for candle in candles):
                score -= 40
                reasons.append("invalid candle values")
        derivative_count = len(derivatives or [])
        if candles:
            missing_ratio = 1 - (derivative_count / len(candles))
            if missing_ratio > self.max_missing_derivative_ratio:
                score -= 20
                reasons.append("derivative data coverage too low")
        accepted = score >= 75
        if accepted:
            reasons.append("accepted")
        return DataQualityReport(symbol=symbol, accepted=accepted, score=max(score, 0), reasons=tuple(reasons))
