from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StrategySettings:
    score_threshold: int = 85
    watch_threshold: int = 65
    strict_trend_entry: bool = False
    volume_window: int = 20
    volume_min_ratio: float = 1.2
    volume_extreme_ratio: float = 2.8
    ema_fast: int = 20
    ema_slow: int = 50
    ma_trend: int = 100
    bollinger_window: int = 20
    bollinger_stddev: float = 2.0
    rsi_window: int = 14
    atr_window: int = 14
    pullback_tolerance_atr: float = 0.6
    max_extension_atr: float = 2.2
    oi_mild_change_min: float = 0.001
    oi_extreme_change: float = 0.05
    oi_4h_entry_min: float = 0.03
    smart_money_window: int = 16
    smart_money_oi_flush: float = 0.035
    smart_money_oi_rebuild: float = 0.012
    smart_money_oi_trap: float = 0.018
    smart_money_price_move: float = 0.012
    smart_money_wick_atr: float = 0.7
    smart_money_min_wicks: int = 2
    smart_money_volume_ratio: float = 1.2
    structure_lookback: int = 20
    structure_buffer_atr: float = 0.4
    structure_grind_bars: int = 4
    structure_grind_tolerance_atr: float = 0.8
    vwap_near_atr: float = 0.45
    vwap_extension_atr: float = 2.6
    keltner_window: int = 20
    keltner_atr_multiplier: float = 2.0
    keltner_near_atr: float = 0.35
    qps_window: int = 20
    qps_min_ratio: float = 1.15
    qps_extreme_ratio: float = 4.0
    volume_breakout_ratio: float = 1.3
    volume_pullback_ratio: float = 0.75
    volume_restart_ratio: float = 1.15
    sweep_wick_atr: float = 0.9
    wash_oi_drop_min: float = 0.003
    extreme_atr_pct: float = 0.06
    long_short_overcrowded_long: float = 2.2
    long_short_overcrowded_short: float = 0.45
    top_long_short_long_min: float = 1.1
    top_long_short_short_max: float = 0.9
    funding_long_min: float = -0.0001
    funding_long_max: float = 0.0005
    funding_short_min: float = -0.0005
    funding_short_max: float = 0.0001
    funding_hot_long: float = 0.0005
    funding_hot_short: float = -0.0005


@dataclass
class RiskSettings:
    leverage_default: int = 5
    leverage_max: int = 10
    risk_per_trade: float = 0.01
    single_symbol_margin_limit: float = 0.10
    total_margin_limit: float = 0.95
    total_open_risk_limit: float = 0.04
    max_open_positions: int = 5
    daily_loss_limit: float = 0.0
    weekly_loss_limit: float = 0.0
    max_drawdown_circuit_breaker: float = 0.0
    max_consecutive_losses: int = 0
    cooldown_hours: int = 0
    atr_stop_buffer: float = 0.5
    time_stop_bars: int = 5
    first_take_profit_r: float = 1.0
    second_take_profit_r: float = 2.0
    first_take_profit_fraction: float = 0.35
    second_take_profit_fraction: float = 0.35
    trailing_remainder_fraction: float = 0.20
    trailing_activation_r: float = 2.0
    trailing_lock_r: float = 0.5


@dataclass
class ExecutionSettings:
    paper_trading: bool = True
    taker_fee_rate: float = 0.0004
    slippage_rate: float = 0.0003


@dataclass
class AppSettings:
    symbols_mode: str = "top20_usdt_perp"
    symbol_rank_by: str = "quote_volume_24h"
    timeframes: list[str] = field(default_factory=lambda: ["15m", "1h"])
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str | Path = "config/strategy.yaml") -> AppSettings:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = _load_yaml(handle.read())
    return AppSettings(
        symbols_mode=raw.get("symbols_mode", "top20_usdt_perp"),
        symbol_rank_by=raw.get("symbol_rank_by", "quote_volume_24h"),
        timeframes=list(raw.get("timeframes", ["15m", "1h"])),
        strategy=StrategySettings(**raw.get("strategy", {})),
        risk=RiskSettings(**raw.get("risk", {})),
        execution=ExecutionSettings(**raw.get("execution", {})),
    )


def _load_yaml(content: str) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _minimal_yaml(content)
    loaded = yaml.safe_load(content) or {}
    if not isinstance(loaded, dict):
        raise ValueError("strategy config must be a mapping")
    return loaded


def _minimal_yaml(content: str) -> dict[str, Any]:
    """Small YAML subset parser for this project's default config.

    It supports top-level mappings, one-level nested mappings, and simple lists.
    PyYAML is still preferred when installed.
    """
    root: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            current_section = None
            current_list_key = None
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                root[key] = _coerce(value)
            else:
                root[key] = {}
                current_section = key
                current_list_key = key
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_list_key:
            if not isinstance(root[current_list_key], list):
                root[current_list_key] = []
            root[current_list_key].append(_coerce(stripped[2:].strip()))
            continue
        if current_section:
            key, _, value = stripped.partition(":")
            root[current_section][key.strip()] = _coerce(value.strip())
    return root


def _coerce(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
