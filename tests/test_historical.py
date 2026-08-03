from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.historical import (
    HistoricalDataset,
    HistoricalReplayResult,
    HistoricalSymbolData,
    SHANGHAI,
    _completed_trade_payloads,
    _floor_time,
    _has_replay_entry_candidate,
    _inclusive_end_date,
    _progress_snapshot,
    _result_payload,
    _replay_signal_refresh_due,
    _symbol_from_payload,
    _symbol_payload,
    analyze_replay_failures,
)
from ai_trading.models import Candle, DerivativesSnapshot, PositionSide
from ai_trading.paper import PaperFill


class _ReplayCandidatePaper:
    def __init__(self, signal: dict[str, object]) -> None:
        self.latest_signals = {"TESTUSDT": signal}
        self.account = type("Account", (), {"positions": {}})()


def test_replay_refreshes_pipeline4_research_and_startup_candidates() -> None:
    for score, mode in ((65, "TREND_RESEARCH"), (70, "TREND_STARTUP")):
        paper = _ReplayCandidatePaper(
            {
                "entry_pipeline_version": 4,
                "candidate_action": "ENTRY_LONG",
                "decision_action": "WATCH",
                "entry_mode": mode,
                "score": score,
            }
        )

        assert _has_replay_entry_candidate(paper) is True


def test_replay_keeps_legacy_75_threshold_for_old_pipeline() -> None:
    assert _has_replay_entry_candidate(
        _ReplayCandidatePaper(
            {
                "entry_pipeline_version": 3,
                "candidate_action": "ENTRY_LONG",
                "score": 74,
            }
        )
    ) is False


def test_v8_replay_recomputes_signal_on_every_closed_15m_bar() -> None:
    at_1030 = datetime(2026, 7, 30, 10, 30, tzinfo=UTC)

    assert _replay_signal_refresh_due(
        at_1030,
        score_model_version=8,
    ) is True
    assert _replay_signal_refresh_due(
        at_1030,
        score_model_version=7,
    ) is False


def test_historical_cache_round_trips_optional_participant_fields() -> None:
    timestamp = datetime(2026, 7, 10, 1, tzinfo=UTC)
    data = HistoricalSymbolData(
        candles={},
        derivatives={
            "1h": (
                DerivativesSnapshot(
                    timestamp=timestamp,
                    open_interest=10_000.0,
                    long_short_ratio=1.1,
                    funding_rate=0.0001,
                    taker_buy_sell_ratio=1.08,
                    taker_buy_volume=108.0,
                    taker_sell_volume=100.0,
                    top_account_long_short_ratio=1.03,
                    top_position_long_short_ratio=1.12,
                ),
            )
        },
    )

    restored = _symbol_from_payload(_symbol_payload(data))
    row = restored.derivatives["1h"][0]

    assert row.taker_buy_sell_ratio == 1.08
    assert row.taker_buy_volume == 108.0
    assert row.taker_sell_volume == 100.0
    assert row.top_account_long_short_ratio == 1.03
    assert row.top_position_long_short_ratio == 1.12


def test_historical_cache_still_reads_legacy_four_column_derivatives() -> None:
    restored = _symbol_from_payload(
        {
            "candles": {},
            "derivatives": {
                "1h": [
                    [
                        "2026-07-10T01:00:00+00:00",
                        10_000.0,
                        1.1,
                        0.0001,
                    ]
                ]
            },
        }
    )
    row = restored.derivatives["1h"][0]

    assert row.open_interest == 10_000.0
    assert row.taker_buy_sell_ratio is None
    assert row.top_position_long_short_ratio is None


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
    symbol_summary = analysis["symbol_summaries"][0]
    assert symbol_summary["symbol"] == "TESTUSDT"
    assert "插针扫损1笔" in symbol_summary["text"]
    assert "收盘重新回到止损内侧" in symbol_summary["text"]
    assert "改进建议" in symbol_summary["text"]
    assert "lifecycles" not in analysis
    assert "failure_causes" not in analysis


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


def test_completed_trade_and_win_rate_share_full_lifecycle_pnl() -> None:
    first_opened = datetime(2026, 7, 10, 1, tzinfo=UTC)
    second_opened = first_opened + timedelta(hours=2)
    fills = [
        _fill(
            timestamp=first_opened,
            action="OPEN",
            price=100.0,
            realized_pnl=0.0,
            opened_at=first_opened,
            fee=1.0,
            quantity=10.0,
            margin_usdt=100.0,
        ),
        _fill(
            timestamp=first_opened + timedelta(minutes=15),
            action="PARTIAL_CLOSE",
            price=103.0,
            realized_pnl=12.0,
            opened_at=first_opened,
            fee=0.2,
            quantity=4.0,
            margin_usdt=40.0,
        ),
        _fill(
            timestamp=first_opened + timedelta(minutes=30),
            action="FUNDING",
            price=101.0,
            realized_pnl=-1.0,
            opened_at=first_opened,
            quantity=6.0,
            margin_usdt=0.0,
        ),
        _fill(
            timestamp=first_opened + timedelta(minutes=45),
            action="CLOSE",
            price=99.0,
            realized_pnl=-4.0,
            opened_at=first_opened,
            fee=0.3,
            quantity=6.0,
            margin_usdt=60.0,
            closed_at=first_opened + timedelta(minutes=45),
        ),
        _fill(
            timestamp=second_opened,
            action="OPEN",
            price=100.0,
            realized_pnl=0.0,
            opened_at=second_opened,
            fee=1.0,
            quantity=5.0,
            margin_usdt=50.0,
        ),
        _fill(
            timestamp=second_opened + timedelta(minutes=30),
            action="CLOSE",
            price=99.0,
            realized_pnl=-2.0,
            opened_at=second_opened,
            fee=0.2,
            quantity=5.0,
            margin_usdt=50.0,
            closed_at=second_opened + timedelta(minutes=30),
        ),
    ]
    dataset = HistoricalDataset(
        start=first_opened,
        end=second_opened + timedelta(hours=1),
        universe=("TESTUSDT",),
        symbols={},
        context={},
    )

    trades = _completed_trade_payloads(fills)
    analysis = analyze_replay_failures(fills, dataset)

    assert len(trades) == 2
    by_opened_at = {row["opened_at"]: row for row in trades}
    first_trade = by_opened_at[first_opened.isoformat()]
    second_trade = by_opened_at[second_opened.isoformat()]
    assert first_trade["realized_pnl"] == 6.0
    assert first_trade["fee"] == 1.5
    assert first_trade["partials"] == 1
    assert second_trade["realized_pnl"] == -3.0
    assert analysis["metrics"]["completed"] == len(trades)
    assert analysis["metrics"]["win_rate"] == 0.5
    assert sum(float(row["realized_pnl"]) > 0 for row in trades) == 1

    status = {
        "equity": 1003.0,
        "available_balance": 1003.0,
        "used_margin": 0.0,
        "realized_pnl": 3.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 3.0,
        "total_pnl_pct": 0.003,
        "fees_paid": 2.7,
        "positions": [],
        "fills": [{"action": "CLOSE", "realized_pnl": -999.0}],
    }
    snapshot = _progress_snapshot(status, fills, [], 1000.0)
    replay = HistoricalReplayResult(
        starting_equity=1000.0,
        ending_equity=1003.0,
        equity_curve=(),
        fills=tuple(fills),
        final_status=status,
        max_drawdown=0.0,
        total_return=0.003,
        per_symbol_pnl={"TESTUSDT": 3.0},
        notes=(),
    )
    result = _result_payload(replay, dataset, analysis)

    assert snapshot["summary"]["trade_count"] == 2
    assert snapshot["summary"]["win_rate"] == 0.5
    assert snapshot["summary"]["closed_trade_pnl"] == 3.0
    assert snapshot["summary"]["open_trade_pnl"] == 0.0
    assert snapshot["account"]["fills"] == trades
    assert result["summary"]["trade_count"] == 2
    assert result["summary"]["win_rate"] == 0.5
    assert result["summary"]["closed_trade_pnl"] == 3.0
    assert result["summary"]["open_trade_pnl"] == 0.0
    assert result["account"]["fills"] == trades


def test_explicit_trade_cycle_id_is_the_authoritative_lifecycle_key() -> None:
    opened = datetime(2026, 7, 10, 1, tzinfo=UTC)
    cycle_id = "TESTUSDT:cycle-001"
    fills = [
        _fill(
            timestamp=opened,
            action="OPEN",
            price=100.0,
            realized_pnl=0.0,
            opened_at=opened,
            fee=0.5,
            trade_cycle_id=cycle_id,
        ),
        _fill(
            timestamp=opened + timedelta(minutes=15),
            action="PARTIAL_CLOSE",
            price=102.0,
            realized_pnl=3.0,
            opened_at=opened + timedelta(seconds=1),
            trade_cycle_id=cycle_id,
        ),
        _fill(
            timestamp=opened + timedelta(minutes=30),
            action="CLOSE",
            price=99.0,
            realized_pnl=-1.0,
            opened_at=opened + timedelta(seconds=2),
            closed_at=opened + timedelta(minutes=30),
            trade_cycle_id=cycle_id,
        ),
    ]

    trades = _completed_trade_payloads(fills)

    assert len(trades) == 1
    assert trades[0]["id"] == cycle_id
    assert trades[0]["trade_cycle_id"] == cycle_id
    assert trades[0]["partials"] == 1
    assert trades[0]["realized_pnl"] == 1.5


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
    fee: float = 0.0,
    quantity: float = 1.0,
    margin_usdt: float = 20.0,
    trade_cycle_id: str = "",
) -> PaperFill:
    return PaperFill(
        timestamp=timestamp,
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        action=action,
        price=price,
        entry_price=100.0,
        quantity=quantity,
        realized_pnl=realized_pnl,
        fee=fee,
        reason=reason,
        leverage=5,
        margin_usdt=margin_usdt,
        stop_price=95.0,
        take_profit_1=105.0,
        take_profit_2=110.0,
        opened_at=opened_at,
        closed_at=closed_at,
        planned_risk_usdt=5.0,
        trade_cycle_id=trade_cycle_id,
    )
