from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StrategySettings:
    score_threshold: int = 75
    watch_threshold: int = 60
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
    long_short_overcrowded_long: float = 2.2
    long_short_overcrowded_short: float = 0.45
    funding_hot_long: float = 0.0005
    funding_hot_short: float = -0.0005


@dataclass
class RiskSettings:
    leverage_default: int = 5
    leverage_max: int = 10
    risk_per_trade: float = 0.005
    single_symbol_margin_limit: float = 0.10
    total_margin_limit: float = 0.35
    max_open_positions: int = 3
    daily_loss_limit: float = 0.02
    max_consecutive_losses: int = 3
    cooldown_hours: int = 6
    atr_stop_buffer: float = 0.5
    time_stop_bars: int = 5
    first_take_profit_r: float = 1.0
    second_take_profit_r: float = 2.0
    first_take_profit_fraction: float = 0.35
    second_take_profit_fraction: float = 0.35


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
