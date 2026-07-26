from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MarketRegime(str, Enum):
    TREND_LONG = "TREND_LONG"
    TREND_SHORT = "TREND_SHORT"
    CHOP = "CHOP"
    OVERCROWDED = "OVERCROWDED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SignalAction(str, Enum):
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    WATCH = "WATCH"
    NO_TRADE = "NO_TRADE"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


SETUP_M15_SQUEEZE_TACTICAL_LONG = "M15_SQUEEZE_TACTICAL_LONG"
SETUP_H1_PULLBACK_LONG = "H1_PULLBACK_LONG"
SETUP_H1_STRUCTURE_LONG = "H1_STRUCTURE_LONG"
SETUP_H4_PULLBACK_LONG = "H4_PULLBACK_LONG"
SETUP_OI_VALLEY_REVERSAL_LONG = "OI_VALLEY_REVERSAL_LONG"
SETUP_H1_PULLBACK_SHORT = "H1_PULLBACK_SHORT"
SETUP_H1_STRUCTURE_SHORT = "H1_STRUCTURE_SHORT"
SETUP_H4_PULLBACK_SHORT = "H4_PULLBACK_SHORT"
SETUP_H4_DESCENDING_RESISTANCE_SHORT = "H4_DESCENDING_RESISTANCE_SHORT"
SETUP_DISTRIBUTION_STAGE1_SHORT = "DISTRIBUTION_STAGE1_SHORT"
SETUP_DISTRIBUTION_STAGE2_SHORT = "DISTRIBUTION_STAGE2_SHORT"
SETUP_DISTRIBUTION_STAGE3_SHORT = "DISTRIBUTION_STAGE3_SHORT"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class DerivativesSnapshot:
    timestamp: datetime
    open_interest: float | None = None
    long_short_ratio: float | None = None
    funding_rate: float | None = None
    taker_buy_sell_ratio: float | None = None
    taker_buy_volume: float | None = None
    taker_sell_volume: float | None = None
    top_account_long_short_ratio: float | None = None
    top_position_long_short_ratio: float | None = None


@dataclass(frozen=True)
class IndicatorSnapshot:
    timestamp: datetime
    close: float
    ema20: float | None
    ema50: float | None
    ema200: float | None
    ma100: float | None
    boll_mid: float | None
    boll_upper: float | None
    boll_lower: float | None
    rsi14: float | None
    atr14: float | None
    volume_sma20: float | None
    volume_ratio: float | None
    ema50_slope: float | None
    vwap: float | None = None
    kc_mid: float | None = None
    kc_upper: float | None = None
    kc_lower: float | None = None
    quote_flow: float | None = None
    quote_flow_ratio: float | None = None
    open_interest: float | None = None
    oi_change: float | None = None
    long_short_ratio: float | None = None
    funding_rate: float | None = None
    taker_buy_sell_ratio: float | None = None
    taker_buy_volume: float | None = None
    taker_sell_volume: float | None = None
    top_account_long_short_ratio: float | None = None
    top_position_long_short_ratio: float | None = None


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    timestamp: datetime
    action: SignalAction
    regime: MarketRegime
    score: int
    direction: PositionSide | None = None
    vetoes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    indicators: IndicatorSnapshot | None = None
    setup_type: str = ""
    score_evidence_families: tuple[tuple[str, int], ...] = ()
    participant_flow_state: str = "NEUTRAL"
    participant_flow_score: int = 0
    participant_flow_confirmed_bars: int = 0
    participant_flow_reason: str = ""


@dataclass
class Position:
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    opened_at: datetime
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    remaining_fraction: float = 1.0
    first_tp_done: bool = False
    second_tp_done: bool = False
    bars_held: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity * self.remaining_fraction


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    quantity: float
    notional: float
    margin_required: float
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    opened_at: datetime
    closed_at: datetime
    pnl: float
    reason: str
