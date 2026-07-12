from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.analytics import analyze_trade_lifecycles
from ai_trading.models import PositionSide
from ai_trading.paper import PaperFill


def _fill(timestamp: datetime, action: str, *, pnl: float = 0.0) -> PaperFill:
    return PaperFill(
        timestamp=timestamp,
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        action=action,
        price=105.0 if action == "CLOSE" else 100.0,
        entry_price=100.0,
        quantity=2.0,
        realized_pnl=pnl,
        fee=0.1,
        reason="take profit target" if action == "CLOSE" else "breakout",
        leverage=5,
        margin_usdt=40.0,
        stop_price=95.0,
        take_profit_1=104.0,
        take_profit_2=108.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=timestamp if action == "CLOSE" else None,
        setup_type="突破回踩",
        entry_quality="A",
        planned_risk_usdt=10.0,
        mae_r=0.3,
        mfe_r=1.2,
    )


def test_trade_lifecycle_analysis_groups_open_and_close() -> None:
    opened = datetime(2026, 1, 1, tzinfo=UTC)
    analysis = analyze_trade_lifecycles(
        [_fill(opened, "OPEN"), _fill(opened + timedelta(hours=3), "CLOSE", pnl=10.0)],
        starting_equity=1_000.0,
    )

    metrics = analysis["metrics"]
    lifecycle = analysis["lifecycles"][0]
    assert metrics["completed"] == 1
    assert metrics["win_rate"] == 1.0
    assert metrics["net_pnl"] == 9.9
    assert lifecycle["realized_r"] == 0.99
    assert lifecycle["exit_category"] == "止盈"
    assert lifecycle["setup_type"] == "突破回踩"
