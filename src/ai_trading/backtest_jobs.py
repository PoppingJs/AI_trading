from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from ai_trading.analytics import analyze_trade_lifecycles
from ai_trading.backtest import (
    LegacyBacktestEngine,
    ProductionPortfolioBacktestEngine,
)
from ai_trading.binance import BinanceFuturesMarketData
from ai_trading.cli import _synthetic_market
from ai_trading.config import AppSettings
from ai_trading.data import load_candles_csv, load_derivatives_csv
from ai_trading.models import Candle, DerivativesSnapshot


@dataclass
class BacktestJob:
    id: str
    request: dict[str, Any]
    status: str = "QUEUED"
    progress: int = 0
    stage: str = "等待执行"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    cancel_requested: bool = False

    def payload(self, *, include_result: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "request": self.request,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "result": self.result if include_result else None,
        }


class BacktestJobManager:
    def __init__(
        self,
        settings: AppSettings,
        *,
        data_root: str | Path = "data/backtests",
    ) -> None:
        self.settings = settings
        self.data_root = Path(data_root).resolve()
        self._jobs: dict[str, BacktestJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ai-trading-backtest",
        )

    def submit(self, request: dict[str, Any]) -> BacktestJob:
        job = BacktestJob(id=uuid4().hex[:12], request=dict(request))
        with self._lock:
            self._jobs[job.id] = job
            self._futures[job.id] = self._executor.submit(self._run, job.id)
        return job

    def get(self, job_id: str) -> BacktestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[BacktestJob]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )[:limit]

    def cancel(self, job_id: str) -> BacktestJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.cancel_requested = True
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                job.status = "CANCELLED"
                job.stage = "已取消"
                job.finished_at = datetime.now(UTC)
            elif job.status in {"QUEUED", "RUNNING"}:
                job.stage = "等待安全停止"
            return job

    def datasets(self) -> list[dict[str, object]]:
        if not self.data_root.exists():
            return []
        results: list[dict[str, object]] = []
        for candles_path in sorted(self.data_root.rglob("*candles.csv")):
            try:
                relative = candles_path.relative_to(self.data_root)
            except ValueError:
                continue
            dataset_id = str(relative).replace("\\", "/")
            derivatives_path = candles_path.with_name(
                candles_path.name.replace("candles.csv", "derivatives.csv")
            )
            results.append(
                {
                    "id": dataset_id,
                    "label": relative.parent.name
                    if relative.name == "candles.csv"
                    else relative.stem.replace("_candles", ""),
                    "has_derivatives": derivatives_path.exists(),
                }
            )
        return results

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        stage: str | None = None,
    ) -> BacktestJob:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0, min(progress, 100))
            if stage is not None:
                job.stage = stage
            return job

    def _run(self, job_id: str) -> None:
        job = self._update(
            job_id,
            status="RUNNING",
            progress=2,
            stage="准备历史数据",
        )
        job.started_at = datetime.now(UTC)
        try:
            candles, derivatives = self._load_market_data(job)
            if job.cancel_requested:
                self._finish_cancelled(job)
                return
            if len(candles) < 20:
                raise ValueError("历史K线不足，至少需要20根")
            self._update(job_id, progress=30, stage="回放交易系统")
            result = self._execute(job, candles, derivatives)
            if job.cancel_requested:
                self._finish_cancelled(job)
                return
            self._update(job_id, progress=92, stage="生成分析报告")
            with self._lock:
                job.result = result
                job.status = "COMPLETED"
                job.progress = 100
                job.stage = "已完成"
                job.finished_at = datetime.now(UTC)
        except Exception as exc:  # noqa: BLE001 - job errors are user-facing
            with self._lock:
                job.status = "FAILED"
                job.stage = "执行失败"
                job.error = str(exc)
                job.finished_at = datetime.now(UTC)

    def _load_market_data(
        self,
        job: BacktestJob,
    ) -> tuple[list[Candle], list[DerivativesSnapshot] | None]:
        source = str(job.request.get("data_source") or "demo")
        if source == "demo":
            self._update(job.id, progress=20, stage="生成演示数据")
            return _synthetic_market()
        if source == "local":
            dataset = str(job.request.get("dataset") or "")
            return self._load_local_dataset(dataset)
        if source == "binance":
            return asyncio.run(self._download_binance(job))
        raise ValueError(f"不支持的数据来源：{source}")

    def _load_local_dataset(
        self,
        dataset: str,
    ) -> tuple[list[Candle], list[DerivativesSnapshot] | None]:
        path = (self.data_root / dataset).resolve()
        if self.data_root not in path.parents or not path.is_file():
            raise ValueError("本地数据集不存在或路径不安全")
        candles = load_candles_csv(path)
        derivatives_path = path.with_name(
            path.name.replace("candles.csv", "derivatives.csv")
        )
        derivatives = (
            load_derivatives_csv(derivatives_path)
            if derivatives_path.exists()
            else None
        )
        return candles, derivatives

    async def _download_binance(
        self,
        job: BacktestJob,
    ) -> tuple[list[Candle], None]:
        symbol = str(job.request.get("symbol") or "BTCUSDT").upper()
        interval = str(job.request.get("base_timeframe") or "15m")
        start = _parse_date(str(job.request.get("start_date") or ""))
        end = _parse_date(str(job.request.get("end_date") or ""), end=True)
        if end <= start:
            raise ValueError("结束日期必须晚于开始日期")
        if end - start > timedelta(days=90):
            raise ValueError("在线下载单次最多支持90天，请缩短日期范围")
        client = BinanceFuturesMarketData()
        candles: list[Candle] = []
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        try:
            while cursor < end_ms:
                if job.cancel_requested:
                    break
                batch = await client.klines(
                    symbol,
                    interval,
                    limit=1000,
                    start_time_ms=cursor,
                    end_time_ms=end_ms,
                )
                if not batch:
                    break
                candles.extend(batch)
                next_cursor = int(batch[-1].timestamp.timestamp() * 1000) + 1
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                elapsed = (cursor - int(start.timestamp() * 1000)) / max(
                    end_ms - int(start.timestamp() * 1000),
                    1,
                )
                self._update(
                    job.id,
                    progress=5 + int(min(elapsed, 1.0) * 20),
                    stage=f"下载 {symbol} 历史K线",
                )
        finally:
            await client.aclose()
        deduped = {
            candle.timestamp: candle
            for candle in candles
            if start <= candle.timestamp < end
        }
        return [deduped[key] for key in sorted(deduped)], None

    def _execute(
        self,
        job: BacktestJob,
        candles: list[Candle],
        derivatives: list[DerivativesSnapshot] | None,
    ) -> dict[str, Any]:
        symbol = str(job.request.get("symbol") or "DEMOUSDT").upper()
        initial = float(job.request.get("starting_equity") or 10_000.0)
        mode = str(job.request.get("mode") or "production")
        if mode == "legacy":
            result = LegacyBacktestEngine(
                symbol=symbol,
                starting_equity=initial,
                strategy_settings=self.settings.strategy,
                risk_settings=self.settings.risk,
                execution_settings=self.settings.execution,
            ).run(candles, derivatives)
            trades = [asdict(trade) for trade in result.trades]
            return {
                "mode": "legacy",
                "symbol": symbol,
                "summary": {
                    "starting_equity": result.starting_equity,
                    "ending_equity": result.ending_equity,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trade_count": len(result.trades),
                },
                "trades": _json_safe(trades),
                "notes": list(result.notes),
                "analysis": None,
                "equity_curve": [],
            }

        portfolio = ProductionPortfolioBacktestEngine(
            starting_equity=initial,
            settings=self.settings,
        ).run({symbol: (candles, derivatives)})
        analysis = analyze_trade_lifecycles(
            portfolio.fills,
            starting_equity=initial,
        )
        return {
            "mode": "production",
            "symbol": symbol,
            "summary": {
                "starting_equity": portfolio.starting_equity,
                "ending_equity": portfolio.ending_equity,
                "total_return": portfolio.total_return,
                "max_drawdown": portfolio.max_drawdown,
                "win_rate": portfolio.win_rate,
                "trade_count": len(analysis["lifecycles"]),
                "profit_factor": analysis["metrics"]["profit_factor"],
                "average_r": analysis["metrics"]["average_r"],
                "fees": analysis["metrics"]["fees"],
            },
            "analysis": _json_safe(analysis),
            "equity_curve": _json_safe(_downsample_curve(portfolio.equity_curve)),
            "per_symbol_pnl": portfolio.per_symbol_pnl,
            "notes": list(portfolio.notes),
        }

    def _finish_cancelled(self, job: BacktestJob) -> None:
        with self._lock:
            job.status = "CANCELLED"
            job.stage = "已取消"
            job.finished_at = datetime.now(UTC)


def _parse_date(value: str, *, end: bool = False) -> datetime:
    if not value:
        now = datetime.now(UTC)
        return now if end else now - timedelta(days=30)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if end and len(value) <= 10:
        parsed += timedelta(days=1)
    return parsed.astimezone(UTC)


def _downsample_curve(
    curve: tuple[tuple[datetime, float], ...],
    limit: int = 600,
) -> list[dict[str, object]]:
    if not curve:
        return []
    step = max(len(curve) // limit, 1)
    sampled = list(curve[::step])
    if sampled[-1] != curve[-1]:
        sampled.append(curve[-1])
    return [
        {"timestamp": timestamp.isoformat(), "equity": equity}
        for timestamp, equity in sampled
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value
