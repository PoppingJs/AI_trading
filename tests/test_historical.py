from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.historical import (
    HistoricalDataset,
    HistoricalSymbolData,
    SHANGHAI,
    _floor_time,
    _inclusive_end_date,
    analyze_replay_failures,
)
from ai_trading.models import Candle, PositionSide
from ai_trading.paper import PaperFill


def test_failure_analysis_identifies_wick_stop_recovery() -> None:
    opened = datetime(2026, 7, 10, 1, tzinfo=UTC)
    closed = opened + timedelta(minutes=30)
    candles = (
        Candle(
            timestamp=closed,
            open=100.0,
            high=101.0,
            low=94.0,
            close=98.0,
            volume=1_000_000.0,
        ),
    )
    dataset = HistoricalDataset(
        start=opened,
        end=closed + timedelta(hours=1),
        universe=("TESTUSDT",),
        symbols={
            "TESTUSDT": HistoricalSymbolData(
                candles={"15m": candles},
                derivatives={},
            )
        },
        context={},
    )
    opening = _fill(
        timestamp=opened,
        action="OPEN",
        price=100.0,
        realized_pnl=0.0,
        opened_at=opened,
    )
    closing = _fill(
        timestamp=closed,
        action="CLOSE",
        price=95.0,
        realized_pnl=-5.0,
        opened_at=opened,
        reason="stop loss: structure invalidated",
        closed_at=closed,
    )

    analysis = analyze_replay_failures([opening, closing], dataset)

    assert analysis["metrics"]["completed"] == 1
    assert analysis["metrics"]["win_rate"] == 0.0
    lifecycle = analysis["lifecycles"][0]
    assert lifecycle["failure_cause"] == "插针扫损"
    assert "收盘重新回到止损内侧" in lifecycle["failure_evidence"]
    assert analysis["failure_causes"][0]["pnl"] < 0
    symbol_summary = analysis["symbol_summaries"][0]
    assert symbol_summary["symbol"] == "TESTUSDT"
    assert symbol_summary["losses"] == 1
    assert symbol_summary["causes"] == [{"cause": "插针扫损", "count": 1}]
    assert "收盘重新回到止损内侧" in symbol_summary["evidence"][0]


def test_open_lifecycle_is_excluded_from_win_rate() -> None:
    opened = datetime(2026, 7, 10, 1, tzinfo=UTC)
    dataset = HistoricalDataset(
        start=opened,
        end=opened + timedelta(days=1),
        universe=("TESTUSDT",),
        symbols={},
        context={},
    )

    analysis = analyze_replay_failures(
        [_fill(timestamp=opened, action="OPEN", price=100.0, realized_pnl=0.0, opened_at=opened)],
        dataset,
    )

    assert analysis["metrics"]["completed"] == 0
    assert analysis["metrics"]["win_rate"] == 0.0
    assert "没有完成交易" in analysis["failure_summary"]


def test_current_day_end_is_floored_to_completed_base_interval() -> None:
    now = datetime(2026, 7, 13, 14, 33, 41, tzinfo=SHANGHAI)

    boundary = _floor_time(now, "15m")

    assert boundary.astimezone(SHANGHAI) == datetime(2026, 7, 13, 14, 30, tzinfo=SHANGHAI)
    assert _inclusive_end_date(boundary).isoformat() == "2026-07-13"


def _fill(
    *,
    timestamp: datetime,
    action: str,
    price: float,
    realized_pnl: float,
    opened_at: datetime,
    reason: str = "auto strategy score=90",
    closed_at: datetime | None = None,
) -> PaperFill:
    return PaperFill(
        timestamp=timestamp,
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        action=action,
        price=price,
        entry_price=100.0,
        quantity=1.0,
        realized_pnl=realized_pnl,
        fee=0.0,
        reason=reason,
        leverage=5,
        margin_usdt=20.0,
        stop_price=95.0,
        take_profit_1=105.0,
        take_profit_2=110.0,
        opened_at=opened_at,
        closed_at=closed_at,
        planned_risk_usdt=5.0,
    )
