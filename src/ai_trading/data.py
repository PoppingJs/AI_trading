from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.models import Candle, DerivativesSnapshot


def load_candles_csv(path: str | Path) -> list[Candle]:
    """Load candles from CSV columns: timestamp,open,high,low,close,volume."""
    rows: list[Candle] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Candle(
                    timestamp=_parse_timestamp(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return rows


def load_derivatives_csv(path: str | Path) -> list[DerivativesSnapshot]:
    """Load derivative data from CSV columns: timestamp,open_interest,long_short_ratio,funding_rate."""
    rows: list[DerivativesSnapshot] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                DerivativesSnapshot(
                    timestamp=_parse_timestamp(row["timestamp"]),
                    open_interest=_optional_float(row.get("open_interest")),
                    long_short_ratio=_optional_float(row.get("long_short_ratio")),
                    funding_rate=_optional_float(row.get("funding_rate")),
                )
            )
    return rows


def _parse_timestamp(value: str) -> datetime:
    if value.isdigit():
        raw = int(value)
        seconds = raw / 1000 if raw > 10_000_000_000 else raw
        return datetime.fromtimestamp(seconds, tz=UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _optional_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
