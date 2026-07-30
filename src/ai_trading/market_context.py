from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from statistics import median
from typing import Any, Mapping, Sequence

from ai_trading.models import Candle, IndicatorSnapshot


BASELINE_WINDOW = 60
MIN_BASELINE_SAMPLES = 20
MIN_LIQUIDITY_SAMPLES = 10
STATE_CONFIRMATIONS = 2
TREND_ENTER_PERCENTILE = 0.55
TREND_EXIT_PERCENTILE = 0.35
EXTREME_ENTER_PERCENTILE = 0.90
EXTREME_EXIT_PERCENTILE = 0.75
LIQUIDITY_VOLUME_ENTER_RATIO = 0.50
LIQUIDITY_VOLUME_EXIT_RATIO = 0.70
LIQUIDITY_SPREAD_HARD_BPS = 25.0
LIQUIDITY_STALE_SECONDS = 120
SYSTEM_SHOCK_ENTER_MULTIPLE = 3.0
SYSTEM_SHOCK_EXIT_MULTIPLE = 2.0
SYSTEM_HARD_RANGE_PCT = 0.12
SYSTEM_POOL_ENTER_RATIO = 0.35
SYSTEM_POOL_EXIT_RATIO = 0.20
MIN_SYSTEM_POOL_SYMBOLS = 5

UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LiquidityObservation:
    symbol: str
    quote_volume: float
    best_bid: float
    best_ask: float
    timestamp: datetime

    @property
    def spread_bps(self) -> float | None:
        mid = (self.best_bid + self.best_ask) / 2
        if (
            not all(
                math.isfinite(value)
                for value in (self.best_bid, self.best_ask, mid)
            )
            or not math.isfinite(self.quote_volume)
            or self.best_bid <= 0
            or self.best_ask <= self.best_bid
            or mid <= 0
        ):
            return None
        return (self.best_ask - self.best_bid) / mid * 10_000


@dataclass
class StateMemory:
    current: str = UNKNOWN
    pending: str | None = None
    confirm_count: int = 0
    changed_at: datetime | None = None
    last_observed_at: datetime | None = None

    def update(
        self,
        candidate: str,
        observed_at: datetime,
        *,
        immediate_states: frozenset[str] = frozenset(),
    ) -> str:
        observed_at = _aware(observed_at)
        if (
            self.last_observed_at is not None
            and observed_at <= self.last_observed_at
        ):
            return self.current
        self.last_observed_at = observed_at
        if candidate == self.current:
            self.pending = None
            self.confirm_count = 0
            return self.current
        if candidate in immediate_states:
            self.current = candidate
            self.pending = None
            self.confirm_count = 0
            self.changed_at = observed_at
            return self.current
        if candidate != self.pending:
            self.pending = candidate
            self.confirm_count = 1
            return self.current
        self.confirm_count += 1
        if self.confirm_count >= STATE_CONFIRMATIONS:
            self.current = candidate
            self.pending = None
            self.confirm_count = 0
            self.changed_at = observed_at
        return self.current

    def to_payload(self) -> dict[str, object]:
        return {
            "current": self.current,
            "pending": self.pending,
            "confirm_count": self.confirm_count,
            "changed_at": (
                self.changed_at.isoformat()
                if self.changed_at is not None
                else None
            ),
            "last_observed_at": (
                self.last_observed_at.isoformat()
                if self.last_observed_at is not None
                else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> StateMemory:
        return cls(
            current=str(payload.get("current") or UNKNOWN),
            pending=(
                str(payload["pending"])
                if payload.get("pending") is not None
                else None
            ),
            confirm_count=max(0, int(payload.get("confirm_count") or 0)),
            changed_at=_parse_datetime(payload.get("changed_at")),
            last_observed_at=_parse_datetime(
                payload.get("last_observed_at")
            ),
        )


@dataclass
class LiquidityHistory:
    volumes: list[float] = field(default_factory=list)
    spreads_bps: list[float] = field(default_factory=list)
    last_observed_at: datetime | None = None

    def append(self, observation: LiquidityObservation) -> None:
        observed_at = _aware(observation.timestamp)
        if (
            self.last_observed_at is not None
            and observed_at <= self.last_observed_at
        ):
            return
        spread_bps = observation.spread_bps
        if observation.quote_volume <= 0 or spread_bps is None:
            return
        self.volumes.append(float(observation.quote_volume))
        self.spreads_bps.append(float(spread_bps))
        self.volumes = self.volumes[-BASELINE_WINDOW:]
        self.spreads_bps = self.spreads_bps[-BASELINE_WINDOW:]
        self.last_observed_at = observed_at

    def to_payload(self) -> dict[str, object]:
        return {
            "volumes": self.volumes[-BASELINE_WINDOW:],
            "spreads_bps": self.spreads_bps[-BASELINE_WINDOW:],
            "last_observed_at": (
                self.last_observed_at.isoformat()
                if self.last_observed_at is not None
                else None
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> LiquidityHistory:
        return cls(
            volumes=[
                float(value)
                for value in list(payload.get("volumes") or [])
                if _finite_positive(value)
            ][-BASELINE_WINDOW:],
            spreads_bps=[
                float(value)
                for value in list(payload.get("spreads_bps") or [])
                if _finite_positive(value)
            ][-BASELINE_WINDOW:],
            last_observed_at=_parse_datetime(
                payload.get("last_observed_at")
            ),
        )


@dataclass
class MarketContextTracker:
    states: dict[str, dict[str, StateMemory]] = field(default_factory=dict)
    liquidity_histories: dict[str, LiquidityHistory] = field(
        default_factory=dict
    )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object] | None,
    ) -> MarketContextTracker:
        if not isinstance(payload, Mapping):
            return cls()
        states: dict[str, dict[str, StateMemory]] = {}
        raw_states = payload.get("states")
        if isinstance(raw_states, Mapping):
            for symbol, raw_symbol_states in raw_states.items():
                if not isinstance(raw_symbol_states, Mapping):
                    continue
                states[str(symbol).upper()] = {
                    str(name): StateMemory.from_payload(raw_memory)
                    for name, raw_memory in raw_symbol_states.items()
                    if isinstance(raw_memory, Mapping)
                }
        histories: dict[str, LiquidityHistory] = {}
        raw_histories = payload.get("liquidity_histories")
        if isinstance(raw_histories, Mapping):
            for symbol, raw_history in raw_histories.items():
                if isinstance(raw_history, Mapping):
                    histories[str(symbol).upper()] = (
                        LiquidityHistory.from_payload(raw_history)
                    )
        return cls(states=states, liquidity_histories=histories)

    def to_payload(self) -> dict[str, object]:
        return {
            "states": {
                symbol: {
                    name: memory.to_payload()
                    for name, memory in symbol_states.items()
                }
                for symbol, symbol_states in self.states.items()
            },
            "liquidity_histories": {
                symbol: history.to_payload()
                for symbol, history in self.liquidity_histories.items()
            },
        }

    @property
    def system_risk_state(self) -> str:
        return self._memory(
            "__SYSTEM__",
            "system_risk_state",
        ).current

    def update_system(
        self,
        btc_candles: Sequence[Candle],
        pool_candles: Mapping[str, Sequence[Candle]],
        observed_at: datetime,
    ) -> str:
        memory = self._memory("__SYSTEM__", "system_risk_state")
        candidate = system_risk_candidate(
            btc_candles,
            pool_candles,
            current_state=memory.current,
        )
        return memory.update(
            candidate,
            observed_at,
            immediate_states=frozenset({"STRESS", UNKNOWN}),
        )

    def update_symbol(
        self,
        symbol: str,
        *,
        h4_indicators: Sequence[IndicatorSnapshot],
        crowding_indicators: Sequence[IndicatorSnapshot],
        h4_context: Mapping[str, object] | None,
        liquidity: LiquidityObservation | None,
        observed_at: datetime,
    ) -> dict[str, object]:
        symbol = symbol.upper()
        direction_memory = self._memory(symbol, "direction_state")
        direction_candidate_value = direction_candidate(
            h4_indicators,
            h4_context,
            current_state=direction_memory.current,
        )
        direction_as_of = (
            h4_indicators[-1].timestamp
            if h4_indicators
            else observed_at
        )
        direction_memory.update(
            direction_candidate_value,
            direction_as_of,
            immediate_states=frozenset({UNKNOWN}),
        )

        crowding_memory = self._memory(symbol, "crowding_state")
        crowding_candidate_value = crowding_candidate(
            crowding_indicators,
            current_state=crowding_memory.current,
        )
        crowding_as_of = (
            crowding_indicators[-1].timestamp
            if crowding_indicators
            else observed_at
        )
        crowding_memory.update(
            crowding_candidate_value,
            crowding_as_of,
            immediate_states=frozenset({UNKNOWN}),
        )

        liquidity_memory = self._memory(symbol, "liquidity_state")
        history = self.liquidity_histories.setdefault(
            symbol,
            LiquidityHistory(),
        )
        liquidity_candidate_value = liquidity_candidate(
            liquidity,
            history,
            current_state=liquidity_memory.current,
            observed_at=observed_at,
        )
        liquidity_as_of = (
            liquidity.timestamp
            if liquidity is not None
            and liquidity_candidate_value != UNKNOWN
            else observed_at
        )
        liquidity_memory.update(
            liquidity_candidate_value,
            liquidity_as_of,
            immediate_states=frozenset({"THIN", UNKNOWN}),
        )
        if liquidity is not None:
            history.append(liquidity)

        system_memory = self._memory(
            "__SYSTEM__",
            "system_risk_state",
        )
        return {
            "direction_state": direction_memory.current,
            "crowding_state": crowding_memory.current,
            "liquidity_state": liquidity_memory.current,
            "system_risk_state": system_memory.current,
            "as_of": _aware(observed_at).isoformat(),
        }

    def _memory(self, symbol: str, name: str) -> StateMemory:
        symbol_states = self.states.setdefault(symbol.upper(), {})
        return symbol_states.setdefault(name, StateMemory())


def direction_candidate(
    indicators: Sequence[IndicatorSnapshot],
    h4_context: Mapping[str, object] | None,
    *,
    current_state: str,
) -> str:
    strengths = [
        abs(item.ema20 - item.ema50) / item.atr14
        for item in indicators[-(BASELINE_WINDOW + 1) :]
        if (
            item.ema20 is not None
            and item.ema50 is not None
            and item.atr14 is not None
            and item.atr14 > 0
        )
    ]
    if len(strengths) < MIN_BASELINE_SAMPLES + 1:
        return UNKNOWN
    current_strength = strengths[-1]
    percentile = percentile_rank(strengths[:-1], current_strength)
    direction = _h4_direction(h4_context, indicators[-1])
    required = (
        TREND_EXIT_PERCENTILE
        if (
            current_state == "TREND_LONG"
            and direction == "LONG"
        )
        or (
            current_state == "TREND_SHORT"
            and direction == "SHORT"
        )
        else TREND_ENTER_PERCENTILE
    )
    if direction == "LONG" and percentile >= required:
        return "TREND_LONG"
    if direction == "SHORT" and percentile >= required:
        return "TREND_SHORT"
    return "RANGE"


def crowding_candidate(
    indicators: Sequence[IndicatorSnapshot],
    *,
    current_state: str,
) -> str:
    recent = indicators[-(BASELINE_WINDOW + 1) :]
    ratio_values = [
        float(item.long_short_ratio)
        for item in recent
        if item.long_short_ratio is not None
        and item.long_short_ratio > 0
    ]
    funding_values = [
        float(item.funding_rate)
        for item in recent
        if item.funding_rate is not None
    ]
    ratio_signal = _crowding_signal(
        ratio_values,
        current_state=current_state,
    )
    funding_signal = _crowding_signal(
        funding_values,
        current_state=current_state,
    )
    signals = {signal for signal in (ratio_signal, funding_signal) if signal}
    if not signals:
        if (
            len(ratio_values) < MIN_BASELINE_SAMPLES + 1
            and len(funding_values) < MIN_BASELINE_SAMPLES + 1
        ):
            return UNKNOWN
        return "NORMAL"
    if len(signals) > 1:
        return UNKNOWN
    return signals.pop()


def liquidity_candidate(
    observation: LiquidityObservation | None,
    history: LiquidityHistory,
    *,
    current_state: str,
    observed_at: datetime,
) -> str:
    if observation is None:
        return UNKNOWN
    observed_at = _aware(observed_at)
    observation_at = _aware(observation.timestamp)
    if (
        (observed_at - observation_at).total_seconds()
        > LIQUIDITY_STALE_SECONDS
    ):
        return UNKNOWN
    spread_bps = observation.spread_bps
    if observation.quote_volume <= 0 or spread_bps is None:
        return UNKNOWN
    if (
        len(history.volumes) < MIN_LIQUIDITY_SAMPLES
        or len(history.spreads_bps) < MIN_LIQUIDITY_SAMPLES
    ):
        return UNKNOWN
    volume_baseline = median(history.volumes[-BASELINE_WINDOW:])
    if volume_baseline <= 0:
        return UNKNOWN
    volume_ratio = observation.quote_volume / volume_baseline
    spread_percentile = percentile_rank(
        history.spreads_bps[-BASELINE_WINDOW:],
        spread_bps,
    )
    if current_state == "NORMAL":
        if (
            volume_ratio < LIQUIDITY_VOLUME_ENTER_RATIO
            or spread_percentile >= EXTREME_ENTER_PERCENTILE
            or spread_bps >= LIQUIDITY_SPREAD_HARD_BPS
        ):
            return "THIN"
        return "NORMAL"
    if (
        volume_ratio >= LIQUIDITY_VOLUME_EXIT_RATIO
        and spread_percentile <= EXTREME_EXIT_PERCENTILE
        and spread_bps < LIQUIDITY_SPREAD_HARD_BPS
    ):
        return "NORMAL"
    return "THIN"


def system_risk_candidate(
    btc_candles: Sequence[Candle],
    pool_candles: Mapping[str, Sequence[Candle]],
    *,
    current_state: str,
) -> str:
    btc_multiple = range_multiple(btc_candles)
    if btc_multiple is None:
        return UNKNOWN
    latest_btc = btc_candles[-1]
    btc_range_pct = (
        (latest_btc.high - latest_btc.low) / latest_btc.open
        if latest_btc.open > 0
        else 0.0
    )
    if btc_range_pct >= SYSTEM_HARD_RANGE_PCT:
        return "STRESS"
    shock_threshold = (
        SYSTEM_SHOCK_EXIT_MULTIPLE
        if current_state == "STRESS"
        else SYSTEM_SHOCK_ENTER_MULTIPLE
    )
    if btc_multiple >= shock_threshold:
        return "STRESS"
    stressed = 0
    eligible = 0
    for symbol, candles in pool_candles.items():
        if symbol.upper() == "BTCUSDT":
            continue
        multiple = range_multiple(candles)
        if multiple is None:
            continue
        eligible += 1
        if multiple >= shock_threshold:
            stressed += 1
    if eligible < MIN_SYSTEM_POOL_SYMBOLS:
        return UNKNOWN
    pool_ratio = stressed / eligible
    pool_threshold = (
        SYSTEM_POOL_EXIT_RATIO
        if current_state == "STRESS"
        else SYSTEM_POOL_ENTER_RATIO
    )
    if pool_ratio >= pool_threshold:
        return "STRESS"
    return "NORMAL"


def range_multiple(candles: Sequence[Candle]) -> float | None:
    recent = list(candles[-(MIN_BASELINE_SAMPLES + 2) :])
    if len(recent) < MIN_BASELINE_SAMPLES + 2:
        return None
    ranges: list[float] = []
    for previous, current in zip(recent, recent[1:]):
        true_range = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        if true_range > 0:
            ranges.append(true_range)
    if len(ranges) < MIN_BASELINE_SAMPLES + 1:
        return None
    baseline = median(ranges[:-1])
    if baseline <= 0:
        return None
    return ranges[-1] / baseline


def percentile_rank(values: Sequence[float], current: float) -> float:
    finite = sorted(
        float(value)
        for value in values
        if math.isfinite(float(value))
    )
    if not finite:
        return 0.5
    equal = sum(
        1
        for value in finite
        if math.isclose(value, current, rel_tol=1e-9, abs_tol=1e-12)
    )
    below = sum(
        1
        for value in finite
        if value < current
        and not math.isclose(
            value,
            current,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    )
    return (below + equal * 0.5) / len(finite)


def _h4_direction(
    context: Mapping[str, object] | None,
    current: IndicatorSnapshot,
) -> str:
    h4 = (
        context.get("h4_structure")
        if isinstance(context, Mapping)
        else None
    )
    if isinstance(h4, Mapping):
        direction = str(h4.get("direction") or "").upper()
        if direction in {"LONG", "SHORT"}:
            return direction
        state = str(h4.get("state") or "").upper()
        if state == "BREAKOUT_UP":
            return "LONG"
        if state == "BREAKDOWN_DOWN":
            return "SHORT"
    if (
        current.ema20 is not None
        and current.ema50 is not None
        and current.ema50_slope is not None
    ):
        if current.ema20 > current.ema50 and current.ema50_slope > 0:
            return "LONG"
        if current.ema20 < current.ema50 and current.ema50_slope < 0:
            return "SHORT"
    return "NEUTRAL"


def _crowding_signal(
    values: Sequence[float],
    *,
    current_state: str,
) -> str | None:
    if len(values) < MIN_BASELINE_SAMPLES + 1:
        return None
    current = values[-1]
    rank = percentile_rank(values[:-1], current)
    long_threshold = (
        EXTREME_EXIT_PERCENTILE
        if current_state == "LONG_CROWDED"
        else EXTREME_ENTER_PERCENTILE
    )
    short_threshold = (
        1 - EXTREME_EXIT_PERCENTILE
        if current_state == "SHORT_CROWDED"
        else 1 - EXTREME_ENTER_PERCENTILE
    )
    if rank >= long_threshold:
        return "LONG_CROWDED"
    if rank <= short_threshold:
        return "SHORT_CROWDED"
    return None


def _finite_positive(value: object) -> bool:
    try:
        parsed = float(value)
        return math.isfinite(parsed) and parsed > 0
    except (TypeError, ValueError):
        return False


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return _aware(parsed)
