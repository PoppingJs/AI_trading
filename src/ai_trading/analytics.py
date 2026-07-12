from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Iterable

from ai_trading.paper import PaperFill


def analyze_trade_lifecycles(
    fills: Iterable[PaperFill],
    *,
    starting_equity: float | None = None,
) -> dict[str, object]:
    ordered = sorted(fills, key=lambda fill: fill.timestamp)
    groups: dict[tuple[str, datetime], list[PaperFill]] = {}
    for fill in ordered:
        groups.setdefault((fill.symbol, fill.opened_at), []).append(fill)

    lifecycles = [
        lifecycle
        for key, items in groups.items()
        if (lifecycle := _lifecycle_payload(key, items)) is not None
    ]
    lifecycles.sort(key=lambda item: str(item["closed_at"]), reverse=True)
    chronological = sorted(lifecycles, key=lambda item: str(item["closed_at"]))
    cumulative = 0.0
    curve: list[dict[str, object]] = []
    for item in chronological:
        cumulative += float(item["pnl"])
        curve.append(
            {
                "timestamp": item["closed_at"],
                "value": cumulative,
            }
        )

    pnls = [float(item["pnl"]) for item in lifecycles]
    realized_rs = [float(item["realized_r"]) for item in lifecycles]
    winners = [value for value in pnls if value > 0]
    losers = [value for value in pnls if value < 0]
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    fees = sum(fill.fee for fill in ordered)
    metrics = {
        "completed": len(lifecycles),
        "win_rate": len(winners) / len(pnls) if pnls else 0.0,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else gross_profit
            if gross_profit > 0
            else 0.0
        ),
        "average_r": mean(realized_rs) if realized_rs else 0.0,
        "fees": fees,
        "average_holding_seconds": mean(
            float(item["holding_seconds"]) for item in lifecycles
        ) if lifecycles else 0.0,
        "average_mae_r": mean(
            float(item["mae_r"]) for item in lifecycles
        ) if lifecycles else 0.0,
        "average_mfe_r": mean(
            float(item["mfe_r"]) for item in lifecycles
        ) if lifecycles else 0.0,
        "net_pnl": sum(pnls),
        "return_pct": (
            sum(pnls) / starting_equity
            if starting_equity and starting_equity > 0
            else 0.0
        ),
    }
    setups = _group_performance(lifecycles, "setup_type")
    qualities = _group_performance(lifecycles, "entry_quality")
    sides = _group_performance(lifecycles, "side")
    exits = Counter(str(item["exit_category"]) for item in lifecycles)
    diagnostics = _diagnostics(lifecycles, setups, metrics)
    return {
        "metrics": metrics,
        "lifecycles": lifecycles,
        "cumulative_curve": curve,
        "setup_performance": setups,
        "quality_performance": qualities,
        "side_performance": sides,
        "exit_distribution": [
            {"reason": reason, "count": count}
            for reason, count in exits.most_common()
        ],
        "diagnostics": diagnostics,
    }


def _lifecycle_payload(
    key: tuple[str, datetime],
    fills: list[PaperFill],
) -> dict[str, object] | None:
    opens = [fill for fill in fills if fill.action == "OPEN"]
    closes = [
        fill for fill in fills
        if fill.action in {"PARTIAL_CLOSE", "CLOSE"}
    ]
    if not opens or not any(fill.action == "CLOSE" for fill in closes):
        return None
    opening = opens[0]
    closing = next(
        fill for fill in reversed(closes)
        if fill.action == "CLOSE"
    )
    realized = sum(
        fill.realized_pnl
        for fill in fills
        if fill.action in {"PARTIAL_CLOSE", "CLOSE", "FUNDING"}
    )
    entry_fees = sum(
        fill.fee for fill in fills if fill.action in {"OPEN", "ADD"}
    )
    pnl = realized - entry_fees
    planned_risk = opening.planned_risk_usdt
    if planned_risk <= 0:
        planned_risk = (
            abs(opening.entry_price - opening.stop_price)
            * opening.quantity
        )
    setup_type = opening.setup_type or _setup_from_entry_text(
        opening.entry_position
    )
    return {
        "id": f"{key[0]}-{key[1].isoformat()}",
        "symbol": opening.symbol,
        "side": opening.side.value,
        "leverage": opening.leverage,
        "setup_type": setup_type or "未分类",
        "entry_quality": opening.entry_quality or "-",
        "opened_at": opening.opened_at.isoformat(),
        "closed_at": (closing.closed_at or closing.timestamp).isoformat(),
        "entry_price": opening.entry_price,
        "exit_price": closing.price,
        "stop_price": opening.stop_price,
        "take_profit_1": opening.take_profit_1,
        "take_profit_2": opening.take_profit_2,
        "planned_risk_usdt": planned_risk,
        "pnl": pnl,
        "realized_r": pnl / planned_risk if planned_risk > 0 else 0.0,
        "mae_r": max((fill.mae_r for fill in closes), default=0.0),
        "mfe_r": max((fill.mfe_r for fill in closes), default=0.0),
        "holding_seconds": max(
            ((closing.closed_at or closing.timestamp) - opening.opened_at).total_seconds(),
            0.0,
        ),
        "exit_reason": closing.reason,
        "exit_category": _exit_category(closing.reason),
        "entry_position": opening.entry_position,
        "adds": sum(fill.action == "ADD" for fill in fills),
        "partials": sum(fill.action == "PARTIAL_CLOSE" for fill in fills),
        "timeline": [
            {
                "timestamp": fill.timestamp.isoformat(),
                "action": fill.action,
                "price": fill.price,
                "quantity": fill.quantity,
                "pnl": fill.realized_pnl,
                "reason": fill.reason,
                "stop_price": fill.stop_price,
                "take_profit_1": fill.take_profit_1,
                "take_profit_2": fill.take_profit_2,
            }
            for fill in fills
        ],
    }


def _group_performance(
    lifecycles: list[dict[str, object]],
    field: str,
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for item in lifecycles:
        groups.setdefault(str(item.get(field) or "未分类"), []).append(item)
    result: list[dict[str, object]] = []
    for name, items in groups.items():
        pnls = [float(item["pnl"]) for item in items]
        rs = [float(item["realized_r"]) for item in items]
        result.append(
            {
                "name": name,
                "count": len(items),
                "pnl": sum(pnls),
                "average_r": mean(rs) if rs else 0.0,
                "win_rate": sum(value > 0 for value in pnls) / len(pnls),
            }
        )
    return sorted(result, key=lambda item: float(item["pnl"]), reverse=True)


def _diagnostics(
    lifecycles: list[dict[str, object]],
    setups: list[dict[str, object]],
    metrics: dict[str, object],
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    losing_setups = [item for item in setups if float(item["pnl"]) < 0]
    if losing_setups:
        worst = losing_setups[-1]
        diagnostics.append(
            {
                "level": "high",
                "title": f"{worst['name']} 贡献为负",
                "detail": f"{worst['count']} 笔，平均 {float(worst['average_r']):+.2f}R",
            }
        )
    stopped = [
        item for item in lifecycles
        if str(item["exit_category"]) == "止损"
    ]
    swept = [
        item for item in stopped
        if float(item["mfe_r"]) >= 1.0
    ]
    if swept:
        diagnostics.append(
            {
                "level": "high",
                "title": "部分止损交易曾达到明显浮盈",
                "detail": f"{len(swept)} 笔止损交易的 MFE 不低于 1R",
            }
        )
    if float(metrics["fees"]) > abs(float(metrics["net_pnl"])) * 0.25:
        diagnostics.append(
            {
                "level": "medium",
                "title": "手续费占比较高",
                "detail": f"累计手续费 {float(metrics['fees']):.2f} U",
            }
        )
    profitable_setups = [item for item in setups if float(item["pnl"]) > 0]
    if profitable_setups:
        best = profitable_setups[0]
        diagnostics.append(
            {
                "level": "positive",
                "title": f"{best['name']} 是当前优势 Setup",
                "detail": f"平均 {float(best['average_r']):+.2f}R，胜率 {float(best['win_rate']):.1%}",
            }
        )
    if not diagnostics:
        diagnostics.append(
            {
                "level": "neutral",
                "title": "样本仍不足",
                "detail": "积累更多完整交易生命周期后再判断稳定优势",
            }
        )
    return diagnostics[:8]


def _setup_from_entry_text(text: str) -> str:
    if not text:
        return ""
    return text.split("；", 1)[0].split("≈", 1)[0][:32]


def _exit_category(reason: str) -> str:
    lowered = reason.lower()
    if "target" in lowered or "take profit" in lowered or "止盈" in reason:
        return "止盈"
    if "stop loss" in lowered or "止损" in reason:
        return "止损"
    if "rotation" in lowered or "轮动" in reason:
        return "轮动"
    if "time" in lowered or "时间" in reason:
        return "时间退出"
    if "structure" in lowered or "结构" in reason:
        return "结构退出"
    return "其他"
