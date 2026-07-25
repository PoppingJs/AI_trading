from __future__ import annotations

import asyncio
import gzip
import json
import math
from bisect import bisect_right
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any, Callable, Iterable
from uuid import uuid4

from ai_trading.binance import BinanceFuturesMarketData, FuturesSymbol
from ai_trading.config import AppSettings
from ai_trading.models import Candle, DerivativesSnapshot, PositionSide
from ai_trading.paper import (
    AUTO_UNIVERSE_EXCLUDED_SYMBOLS,
    AUTO_UNIVERSE_SYMBOL,
    AUTO_ENTRY_MIN_SCORE,
    CANDIDATE_MIN_QUOTE_VOLUME,
    PAPER_DEFAULT_BALANCE,
    PaperFill,
    PaperTradingEngine,
    SUPPORTED_TIMEFRAMES,
)


BASE_TIMEFRAME = "15m"
SIGNAL_TIMEFRAME = "1h"
UNIVERSE_SIZE = 50
UNIVERSE_DISCOVERY_LIMIT = 80
MAX_REPLAY_DAYS = 30
VISIBLE_HISTORY_LIMIT = 500
KLINE_WARMUP_BARS = 240
DERIVATIVE_WARMUP_DAYS = 7
MIN_USABLE_SYMBOLS = 10
SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")

_TIMEFRAME_SECONDS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


@dataclass(frozen=True)
class HistoricalSymbolData:
    candles: dict[str, tuple[Candle, ...]]
    derivatives: dict[str, tuple[DerivativesSnapshot, ...]]


@dataclass(frozen=True)
class HistoricalDataset:
    start: datetime
    end: datetime
    universe: tuple[str, ...]
    symbols: dict[str, HistoricalSymbolData]
    context: dict[str, HistoricalSymbolData]
    skipped: tuple[dict[str, str], ...] = ()
    downloaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class HistoricalReplayResult:
    starting_equity: float
    ending_equity: float
    equity_curve: tuple[tuple[datetime, float], ...]
    fills: tuple[PaperFill, ...]
    final_status: dict[str, object]
    max_drawdown: float
    total_return: float
    per_symbol_pnl: dict[str, float]
    notes: tuple[str, ...]


@dataclass
class HistoricalReplayJob:
    id: str
    request: dict[str, str]
    status: str = "QUEUED"
    progress: int = 0
    stage: str = "等待执行"
    virtual_time: datetime | None = None
    cache_hit: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    cancel_requested: bool = False

    def payload(self, *, include_result: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "request": self.request,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "virtual_time": self.virtual_time.isoformat() if self.virtual_time else None,
            "cache_hit": self.cache_hit,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "snapshot": self.snapshot,
            "result": self.result if include_result else None,
        }


class HistoricalReplayManager:
    """One production-parity historical replay worker with one bounded dataset cache."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        cache_root: str | Path = "data/historical_cache",
        market_data_factory: Callable[[], BinanceFuturesMarketData] = BinanceFuturesMarketData,
    ) -> None:
        self.settings = settings
        self.cache_root = Path(cache_root).resolve()
        self.market_data_factory = market_data_factory
        self._jobs: dict[str, HistoricalReplayJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ai-trading-historical-replay",
        )

    def submit(self, request: dict[str, str]) -> HistoricalReplayJob:
        start, end = _validated_range(request.get("start_date", ""), request.get("end_date", ""))
        normalized = {
            "start_date": start.astimezone(SHANGHAI).date().isoformat(),
            "end_date": (end - timedelta(microseconds=1)).astimezone(SHANGHAI).date().isoformat(),
        }
        job = HistoricalReplayJob(id=uuid4().hex[:12], request=normalized)
        with self._lock:
            active = any(item.status in {"QUEUED", "RUNNING"} for item in self._jobs.values())
            if active:
                raise ValueError("已有历史回测正在运行，请等待完成或先取消")
            self._jobs[job.id] = job
            self._futures[job.id] = self._executor.submit(self._run, job.id)
        return job

    def get(self, job_id: str) -> HistoricalReplayJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> HistoricalReplayJob | None:
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

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str) -> None:
        job = self._update(job_id, status="RUNNING", progress=1, stage="准备历史行情")
        job.started_at = datetime.now(UTC)
        try:
            start, end = _validated_range(job.request["start_date"], job.request["end_date"])
            dataset = self._load_cached_dataset(start, end)
            if dataset is None:
                dataset = asyncio.run(self._download_dataset(job, start, end))
                if job.cancel_requested:
                    self._finish_cancelled(job)
                    return
                self._save_cached_dataset(dataset)
            else:
                job.cache_hit = True
                self._update(job.id, progress=28, stage="已读取最近一次本地行情缓存")

            if len(dataset.symbols) < MIN_USABLE_SYMBOLS:
                reason_summary = _skipped_reason_summary(dataset.skipped)
                raise ValueError(
                    f"可完整回放币种仅 {len(dataset.symbols)} 个，少于最低要求 {MIN_USABLE_SYMBOLS} 个；"
                    f"{reason_summary or '请缩短到币安仍保留 OI/多空比的最近日期'}"
                )
            if job.cancel_requested:
                self._finish_cancelled(job)
                return

            self._update(job.id, progress=32, stage="使用实时交易状态机重放")
            replay = HistoricalReplayEngine(
                settings=self.settings,
                starting_equity=PAPER_DEFAULT_BALANCE,
            ).run(
                dataset,
                progress=lambda timestamp, ratio, snapshot: self._replay_progress(
                    job.id,
                    timestamp,
                    ratio,
                    snapshot,
                ),
                cancelled=lambda: job.cancel_requested,
            )
            if job.cancel_requested:
                self._finish_cancelled(job)
                return

            self._update(job.id, progress=94, stage="生成失败根因总结")
            analysis = analyze_replay_failures(replay.fills, dataset)
            result = _result_payload(replay, dataset, analysis)
            with self._lock:
                job.result = result
                job.status = "COMPLETED"
                job.progress = 100
                job.stage = "已完成"
                job.virtual_time = dataset.end
                job.finished_at = datetime.now(UTC)
        except Exception as exc:  # noqa: BLE001 - task errors are user-facing
            with self._lock:
                job.status = "FAILED"
                job.stage = "执行失败"
                job.error = str(exc)
                job.finished_at = datetime.now(UTC)

    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        stage: str | None = None,
        virtual_time: datetime | None = None,
    ) -> HistoricalReplayJob:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0, min(progress, 100))
            if stage is not None:
                job.stage = stage
            if virtual_time is not None:
                job.virtual_time = virtual_time
            return job

    def _replay_progress(
        self,
        job_id: str,
        timestamp: datetime,
        ratio: float,
        snapshot: dict[str, Any],
    ) -> None:
        progress = 32 + int(max(0.0, min(ratio, 1.0)) * 60)
        with self._lock:
            job = self._jobs[job_id]
            job.progress = progress
            job.stage = f"历史时间 {timestamp.astimezone(SHANGHAI).strftime('%Y-%m-%d %H:%M')}"
            job.virtual_time = timestamp
            job.snapshot = snapshot

    def _finish_cancelled(self, job: HistoricalReplayJob) -> None:
        with self._lock:
            job.status = "CANCELLED"
            job.stage = "已取消"
            job.finished_at = datetime.now(UTC)

    async def _download_dataset(
        self,
        job: HistoricalReplayJob,
        start: datetime,
        end: datetime,
    ) -> HistoricalDataset:
        client = self.market_data_factory()
        skipped: list[dict[str, str]] = []
        try:
            discovered = await client.top_usdt_perpetuals(limit=UNIVERSE_DISCOVERY_LIMIT)
            universe = _current_top50(discovered)
            if len(universe) < MIN_USABLE_SYMBOLS:
                raise ValueError("当前符合实时交易过滤条件的 USDT 永续合约不足")

            symbols: dict[str, HistoricalSymbolData] = {}
            total = len(universe)
            completed = 0
            semaphore = asyncio.Semaphore(3)

            async def download_one(symbol: str) -> None:
                nonlocal completed
                if job.cancel_requested:
                    return
                try:
                    async with semaphore:
                        data = await _download_symbol(
                            client,
                            symbol,
                            start,
                            end,
                            include_derivatives=True,
                        )
                    reason = _symbol_data_problem(data, start, end)
                    if reason:
                        skipped.append({"symbol": symbol, "reason": reason})
                    else:
                        symbols[symbol] = data
                except Exception as exc:  # noqa: BLE001 - one bad listing should not hide the report
                    skipped.append({"symbol": symbol, "reason": str(exc)})
                finally:
                    completed += 1
                    self._update(
                        job.id,
                        progress=3 + int(completed / max(total, 1) * 22),
                        stage=f"下载历史行情与衍生品 {completed}/{total}",
                    )

            await asyncio.gather(*(download_one(symbol) for symbol in universe))
            symbols = {
                symbol: symbols[symbol]
                for symbol in universe
                if symbol in symbols
            }

            context: dict[str, HistoricalSymbolData] = {}
            if "BTCUSDT" not in symbols:
                try:
                    context["BTCUSDT"] = await _download_symbol(
                        client,
                        "BTCUSDT",
                        start,
                        end,
                        include_derivatives=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped.append({"symbol": "BTCUSDT", "reason": f"BTC风控上下文缺失：{exc}"})
            return HistoricalDataset(
                start=start,
                end=end,
                universe=tuple(symbols),
                symbols=symbols,
                context=context,
                skipped=tuple(skipped),
            )
        finally:
            await client.aclose()

    @property
    def _cache_path(self) -> Path:
        return self.cache_root / "latest_market_dataset.json.gz"

    def _load_cached_dataset(self, start: datetime, end: datetime) -> HistoricalDataset | None:
        path = self._cache_path
        if not path.is_file():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            dataset = _dataset_from_payload(payload)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        exact_match = dataset.start == start and dataset.end == end
        current_day_cache_match = (
            dataset.start == start
            and dataset.end <= end
            and _inclusive_end_date(dataset.end) == _inclusive_end_date(end) == datetime.now(SHANGHAI).date()
        )
        if not exact_match and not current_day_cache_match:
            return None
        return dataset

    def _save_cached_dataset(self, dataset: HistoricalDataset) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        target = self._cache_path
        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(_dataset_payload(dataset), handle, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(target)


class HistoricalReplayEngine:
    """Drive PaperTradingEngine with a point-in-time historical market adapter."""

    def __init__(self, *, settings: AppSettings, starting_equity: float) -> None:
        self.settings = settings
        self.starting_equity = starting_equity

    def run(
        self,
        dataset: HistoricalDataset,
        *,
        progress: Callable[[datetime, float, dict[str, Any]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> HistoricalReplayResult:
        return asyncio.run(self._run_async(dataset, progress=progress, cancelled=cancelled))

    async def _run_async(
        self,
        dataset: HistoricalDataset,
        *,
        progress: Callable[[datetime, float, dict[str, Any]], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> HistoricalReplayResult:
        timeline = sorted(
            {
                candle.timestamp
                for data in dataset.symbols.values()
                for candle in data.candles.get(BASE_TIMEFRAME, ())
                if dataset.start <= candle.timestamp < dataset.end
            }
        )
        if not timeline:
            raise ValueError("所选时间范围没有可回放的15分钟K线")

        clock = _ReplayClock(timeline[0])
        market = _ReplayMarketData()
        universe = list(dataset.universe)
        paper = PaperTradingEngine(
            self.settings,
            starting_balance=self.starting_equity,
            symbols=[AUTO_UNIVERSE_SYMBOL],
            interval=SIGNAL_TIMEFRAME,
            market_data=market,  # type: ignore[arg-type]
            clock=clock,
            fill_price_resolver=lambda price, side, entering: _slipped(
                price,
                side,
                self.settings.execution.slippage_rate,
                entering=entering,
            ),
        )
        paper._universe_symbols = list(universe)
        paper.symbols = list(universe)
        paper._candidate_symbols = []
        paper.auto_trade = True
        curve: list[tuple[datetime, float]] = []
        charged_funding: set[tuple[str, datetime]] = set()
        base_lookup = {
            symbol: {candle.timestamp: candle for candle in data.candles[BASE_TIMEFRAME]}
            for symbol, data in dataset.symbols.items()
        }

        for index, event_time in enumerate(timeline):
            if cancelled and cancelled():
                break
            active = {
                symbol: lookup[event_time]
                for symbol, lookup in base_lookup.items()
                if event_time in lookup
            }
            if not active:
                continue

            clock.value = event_time
            _publish_context_market(market, dataset, event_time)
            for symbol, candle in active.items():
                paper._remember_mark_price(symbol, candle.open)
                market.prices[symbol] = candle.open
            paper._account_risk_snapshot(_paper_risk_status(paper), now=clock.value)
            paper._manage_open_positions()
            paper._refresh_live_entry_timing()
            await paper._auto_trade_once()
            _append_equity_point(paper, clock.value, curve)

            for phase, offset in (("adverse", 1 / 3), ("favorable", 2 / 3)):
                clock.value = event_time + timedelta(seconds=_TIMEFRAME_SECONDS[BASE_TIMEFRAME] * offset)
                for symbol, candle in active.items():
                    position = paper.account.positions.get(symbol)
                    side = position.side if position is not None else None
                    signal = paper.latest_signals.get(symbol, {})
                    candidate_action = str(signal.get("candidate_action") or signal.get("action") or "")
                    intrabar_price = _intrabar_phase_price(
                        candle,
                        side,
                        phase,
                        candidate_action=candidate_action,
                    )
                    paper._remember_mark_price(symbol, intrabar_price)
                    market.prices[symbol] = intrabar_price
                    if position is not None:
                        paper._manage_open_positions()
                if _has_replay_entry_candidate(paper):
                    paper._refresh_live_entry_timing()
                    await paper._auto_trade_once()
                _append_equity_point(paper, clock.value, curve)

            clock.value = event_time + timedelta(seconds=_TIMEFRAME_SECONDS[BASE_TIMEFRAME] - 1)
            for symbol, candle in active.items():
                paper._remember_mark_price(symbol, candle.close)
                market.prices[symbol] = candle.close
            paper._manage_open_positions()
            if _has_replay_entry_candidate(paper):
                paper._refresh_live_entry_timing()
                await paper._auto_trade_once()

            close_time = event_time + timedelta(seconds=_TIMEFRAME_SECONDS[BASE_TIMEFRAME])
            clock.value = close_time
            signal_timeframe_closed = close_time.minute == 0
            for symbol in active:
                _publish_symbol_history(
                    paper,
                    market,
                    symbol,
                    dataset.symbols[symbol],
                    close_time,
                    publish_signal=signal_timeframe_closed,
                )
                _apply_replay_funding(
                    paper,
                    symbol,
                    paper._timeframe_derivatives.get(symbol, {}),
                    close_time,
                    charged_funding,
                )
            if signal_timeframe_closed:
                paper._rebalance_auto_signal_pools(fresh_symbols=set(active))
            paper._manage_open_positions()
            if signal_timeframe_closed:
                paper._record_account_snapshots(close_time)
            _append_equity_point(paper, close_time, curve)

            if progress and (index % 4 == 0 or index == len(timeline) - 1):
                status = paper.status()
                progress(
                    close_time,
                    (index + 1) / len(timeline),
                    _progress_snapshot(
                        status,
                        paper.account.fills,
                        curve,
                        self.starting_equity,
                    ),
                )

        final_time = min(dataset.end, timeline[-1] + timedelta(seconds=_TIMEFRAME_SECONDS[BASE_TIMEFRAME]))
        clock.value = final_time
        _append_equity_point(paper, final_time, curve)
        status = paper.status()
        ending_equity = float(status["equity"])
        max_drawdown = _max_drawdown(curve, self.starting_equity)
        per_symbol = _per_symbol_pnl(paper.account.fills)
        return HistoricalReplayResult(
            starting_equity=self.starting_equity,
            ending_equity=ending_equity,
            equity_curve=tuple(curve),
            fills=tuple(paper.account.fills),
            final_status=status,
            max_drawdown=max_drawdown,
            total_return=(ending_equity - self.starting_equity) / self.starting_equity,
            per_symbol_pnl=per_symbol,
            notes=(
                "使用实时模拟交易 PaperTradingEngine 的同一决策、风险、持仓和出场状态机",
                "标的池为首次运行时固定的当前 Top50，不代表历史 Top50",
                "信号最早在下一市场事件成交，同一K线冲突采用不利路径",
                "结束时未平仓仓位按市值计入总权益，不制造强制平仓成交",
            ),
        )


@dataclass
class _ReplayClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


class _ReplayMarketData:
    def __init__(self) -> None:
        self.candles: dict[str, dict[str, list[Candle]]] = {}
        self.prices: dict[str, float] = {}

    async def klines(
        self,
        symbol: str,
        interval: str = BASE_TIMEFRAME,
        *,
        limit: int = 500,
        **_: object,
    ) -> list[Candle]:
        return list(self.candles.get(symbol.upper(), {}).get(interval, []))[-limit:]

    async def mark_prices(self, symbols: Iterable[str] | None = None) -> dict[str, float]:
        wanted = set(symbols or self.prices)
        return {symbol: price for symbol, price in self.prices.items() if symbol in wanted}

    async def aclose(self) -> None:
        return None


async def _download_symbol(
    client: BinanceFuturesMarketData,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    include_derivatives: bool,
) -> HistoricalSymbolData:
    candles: dict[str, tuple[Candle, ...]] = {}
    derivatives: dict[str, tuple[DerivativesSnapshot, ...]] = {}
    for timeframe in SUPPORTED_TIMEFRAMES:
        seconds = _TIMEFRAME_SECONDS[timeframe]
        warmup_start = start - timedelta(seconds=seconds * KLINE_WARMUP_BARS)
        candles[timeframe] = tuple(
            await _download_klines(client, symbol, timeframe, warmup_start, end)
        )

    if include_derivatives:
        derivative_start = max(
            start - timedelta(days=DERIVATIVE_WARMUP_DAYS),
            _floor_time(datetime.now(UTC) - timedelta(days=30), BASE_TIMEFRAME),
        )
        funding = await _download_funding(client, symbol, derivative_start, end)
        for timeframe in ("15m", "1h", "4h"):
            derivatives[timeframe] = tuple(
                await _download_derivatives(
                    client,
                    symbol,
                    timeframe,
                    derivative_start,
                    end,
                    funding,
                )
            )
    return HistoricalSymbolData(candles=candles, derivatives=derivatives)


async def _download_klines(
    client: BinanceFuturesMarketData,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[Candle]:
    cursor = _to_ms(start)
    end_ms = _to_ms(end)
    rows: dict[datetime, Candle] = {}
    while cursor < end_ms:
        batch = await client.klines(
            symbol,
            timeframe,
            limit=1500,
            start_time_ms=cursor,
            end_time_ms=end_ms,
        )
        if not batch:
            break
        for candle in batch:
            if start <= candle.timestamp < end:
                rows[candle.timestamp] = candle
        next_cursor = _to_ms(batch[-1].timestamp) + _TIMEFRAME_SECONDS[timeframe] * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1500:
            break
    return [rows[key] for key in sorted(rows)]


async def _download_derivatives(
    client: BinanceFuturesMarketData,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    funding: dict[datetime, float],
) -> list[DerivativesSnapshot]:
    cursor = _to_ms(start)
    end_ms = _to_ms(end)
    oi: dict[datetime, float] = {}
    ratios: dict[datetime, float] = {}
    taker: dict[datetime, tuple[float, float, float]] = {}
    top_accounts: dict[datetime, float] = {}
    top_positions: dict[datetime, float] = {}
    step_ms = _TIMEFRAME_SECONDS[timeframe] * 1000
    while cursor < end_ms:
        optional_calls = [
            _optional_derivatives_history(
                client,
                method_name,
                symbol,
                timeframe,
                cursor,
                end_ms,
            )
            for method_name in (
                "taker_buy_sell_volume",
                "top_trader_account_ratio",
                "top_trader_position_ratio",
            )
        ]
        oi_batch, ratio_batch, taker_batch, top_account_batch, top_position_batch = await asyncio.gather(
            client.open_interest_history(
                symbol,
                timeframe,
                limit=500,
                start_time_ms=cursor,
                end_time_ms=end_ms,
            ),
            client.global_long_short_ratio(
                symbol,
                timeframe,
                limit=500,
                start_time_ms=cursor,
                end_time_ms=end_ms,
            ),
            *optional_calls,
        )
        if not oi_batch or not ratio_batch:
            break
        oi.update(oi_batch)
        ratios.update(ratio_batch)
        taker.update(taker_batch)
        top_accounts.update(top_account_batch)
        top_positions.update(top_position_batch)
        latest = min(max(oi_batch), max(ratio_batch))
        next_cursor = _to_ms(latest) + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(oi_batch) < 500 and len(ratio_batch) < 500:
            break

    funding_times = sorted(funding)
    snapshots: list[DerivativesSnapshot] = []
    for timestamp in sorted(set(oi) & set(ratios)):
        if not (start <= timestamp < end):
            continue
        position = bisect_right(funding_times, timestamp) - 1
        funding_rate = funding[funding_times[position]] if position >= 0 else None
        snapshots.append(
            DerivativesSnapshot(
                timestamp=timestamp,
                open_interest=oi[timestamp],
                long_short_ratio=ratios[timestamp],
                funding_rate=funding_rate,
                taker_buy_sell_ratio=(
                    taker[timestamp][0]
                    if timestamp in taker
                    else None
                ),
                taker_buy_volume=(
                    taker[timestamp][1]
                    if timestamp in taker
                    else None
                ),
                taker_sell_volume=(
                    taker[timestamp][2]
                    if timestamp in taker
                    else None
                ),
                top_account_long_short_ratio=top_accounts.get(timestamp),
                top_position_long_short_ratio=top_positions.get(timestamp),
            )
        )
    return snapshots


async def _optional_derivatives_history(
    client: BinanceFuturesMarketData,
    method_name: str,
    symbol: str,
    timeframe: str,
    start_time_ms: int,
    end_time_ms: int,
) -> dict:
    """Load an optional derivatives series without breaking old/fake clients."""

    method = getattr(client, method_name, None)
    if not callable(method):
        return {}
    try:
        result = await method(
            symbol,
            timeframe,
            limit=500,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
    except Exception:
        return {}
    return result if isinstance(result, dict) else {}


async def _download_funding(
    client: BinanceFuturesMarketData,
    symbol: str,
    start: datetime,
    end: datetime,
) -> dict[datetime, float]:
    cursor = _to_ms(start)
    end_ms = _to_ms(end)
    rows: dict[datetime, float] = {}
    while cursor < end_ms:
        batch = await client.funding_rates(
            symbol,
            limit=1000,
            start_time_ms=cursor,
            end_time_ms=end_ms,
        )
        if not batch:
            break
        rows.update(batch)
        latest = max(batch)
        next_cursor = _to_ms(latest) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
    return rows


def analyze_replay_failures(
    fills: Iterable[PaperFill],
    dataset: HistoricalDataset,
) -> dict[str, object]:
    ordered = sorted(fills, key=lambda fill: fill.timestamp)
    groups = _fill_lifecycle_groups(ordered)

    lifecycles: list[dict[str, object]] = []
    for key, items in groups.items():
        lifecycle = _lifecycle_payload(key, items, dataset)
        if lifecycle is not None:
            lifecycles.append(lifecycle)
    lifecycles.sort(key=lambda row: str(row["closed_at"]), reverse=True)

    completed = len(lifecycles)
    winners = [row for row in lifecycles if float(row["pnl"]) > 0]
    losers = [row for row in lifecycles if float(row["pnl"]) < 0]
    gross_profit = sum(float(row["pnl"]) for row in winners)
    gross_loss = abs(sum(float(row["pnl"]) for row in losers))
    fees = sum(fill.fee for fill in ordered)
    causes: dict[str, dict[str, float]] = {}
    for row in losers:
        cause = str(row["failure_cause"])
        item = causes.setdefault(cause, {"count": 0.0, "pnl": 0.0})
        item["count"] += 1
        item["pnl"] += float(row["pnl"])
    ranked_causes = sorted(
        (
            {"cause": cause, "count": int(values["count"]), "pnl": values["pnl"]}
            for cause, values in causes.items()
        ),
        key=lambda item: float(item["pnl"]),
    )
    summary = _failure_summary(ranked_causes, completed, len(losers))
    return {
        "metrics": {
            "completed": completed,
            "win_rate": len(winners) / completed if completed else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss else gross_profit,
            "average_r": mean(float(row["realized_r"]) for row in lifecycles) if lifecycles else 0.0,
            "fees": fees,
            "closed_trade_pnl": sum(float(row["pnl"]) for row in lifecycles),
        },
        "failure_summary": summary,
        "symbol_summaries": _symbol_analysis_summaries(lifecycles),
    }


def _fill_lifecycle_groups(
    fills: Iterable[PaperFill],
) -> dict[str, list[PaperFill]]:
    groups: dict[str, list[PaperFill]] = {}
    for fill in fills:
        cycle_id = fill.trade_cycle_id or f"{fill.symbol}:{fill.opened_at.isoformat()}"
        groups.setdefault(cycle_id, []).append(fill)
    return groups


def _completed_trade_payloads(
    fills: Iterable[PaperFill],
) -> list[dict[str, object]]:
    """Aggregate one open-to-final-close lifecycle into one trade record."""
    ordered = sorted(fills, key=lambda fill: fill.timestamp)
    rows = [
        row
        for key, items in _fill_lifecycle_groups(ordered).items()
        if (row := _completed_trade_payload(key, items)) is not None
    ]
    rows.sort(key=lambda row: str(row["closed_at"]), reverse=True)
    return rows


def _completed_trade_payload(
    key: str,
    fills: list[PaperFill],
) -> dict[str, object] | None:
    entries = [fill for fill in fills if fill.action in {"OPEN", "ADD"}]
    closes = [fill for fill in fills if fill.action in {"PARTIAL_CLOSE", "CLOSE"}]
    opens = [fill for fill in entries if fill.action == "OPEN"]
    if not opens or not any(fill.action == "CLOSE" for fill in closes):
        return None

    opening = opens[0]
    closing = next(fill for fill in reversed(closes) if fill.action == "CLOSE")
    realized = sum(
        fill.realized_pnl
        for fill in fills
        if fill.action in {"PARTIAL_CLOSE", "CLOSE", "FUNDING"}
    )
    entry_fees = sum(fill.fee for fill in entries)
    pnl = realized - entry_fees
    entry_quantity = sum(fill.quantity for fill in entries)
    closed_quantity = sum(fill.quantity for fill in closes)
    entry_notional = sum(fill.price * fill.quantity for fill in entries)
    exit_notional = sum(fill.price * fill.quantity for fill in closes)
    committed_margin = sum(fill.margin_usdt for fill in entries)
    exit_time = closing.closed_at or closing.timestamp

    return {
        "id": key,
        "trade_cycle_id": key,
        "symbol": opening.symbol,
        "side": opening.side.value,
        "action": "CLOSE",
        "leverage": opening.leverage,
        "entry_price": (
            entry_notional / entry_quantity if entry_quantity > 0 else opening.entry_price
        ),
        "price": exit_notional / closed_quantity if closed_quantity > 0 else closing.price,
        "quantity": entry_quantity,
        "realized_pnl": pnl,
        "fee": sum(fill.fee for fill in fills),
        "margin_usdt": committed_margin,
        "return_pct": pnl / committed_margin if committed_margin > 0 else 0.0,
        "stop_price": closing.stop_price,
        "take_profit_1": closing.take_profit_1,
        "take_profit_2": closing.take_profit_2,
        "opened_at": opening.opened_at.isoformat(),
        "closed_at": exit_time.isoformat(),
        "entry_position": opening.entry_position,
        "reason": closing.reason,
        "primary_setup": opening.primary_setup or opening.setup_type,
        "supporting_evidence": list(opening.supporting_evidence),
        "entry_trigger": opening.entry_trigger,
        "validation_state": closing.validation_state or opening.validation_state,
        "soft_stop_price": opening.soft_stop_price or opening.stop_price,
        "hard_stop_price": opening.hard_stop_price or opening.stop_price,
        "exit_category": closing.exit_category or _exit_category(closing.reason),
        "adds": sum(fill.action == "ADD" for fill in fills),
        "partials": sum(fill.action == "PARTIAL_CLOSE" for fill in fills),
    }


def _symbol_analysis_summaries(
    lifecycles: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for lifecycle in lifecycles:
        grouped.setdefault(str(lifecycle["symbol"]), []).append(lifecycle)

    summaries: list[dict[str, object]] = []
    for symbol, rows in grouped.items():
        losers = [row for row in rows if float(row["pnl"]) < 0]
        causes: dict[str, int] = {}
        evidence: list[str] = []
        for row in losers:
            cause = str(row.get("failure_cause") or "未分类亏损")
            causes[cause] = causes.get(cause, 0) + 1
            detail = str(row.get("failure_evidence") or "").strip()
            if detail and detail not in evidence:
                evidence.append(detail)
        pnl = sum(float(row["pnl"]) for row in rows)
        wins = sum(float(row["pnl"]) > 0 for row in rows)
        ranked = sorted(causes.items(), key=lambda item: (-item[1], item[0]))
        issue_text = "、".join(f"{cause}{count}笔" for cause, count in ranked)
        recommendations = list(
            dict.fromkeys(_recommendation_for_cause(cause) for cause, _ in ranked)
        )
        if not recommendations:
            recommendations.append("当前完成交易未出现亏损，维持现有入场与退出规则并继续积累样本")
        parts = [
            f"完成{len(rows)}笔，胜{wins}负{len(losers)}",
            f"净收益{pnl:+.2f} U",
            f"主要问题：{issue_text}" if issue_text else "本区间未发现亏损交易",
        ]
        if evidence:
            parts.append(f"关键观察：{'；'.join(evidence[:3])}")
        parts.append(f"改进建议：{'；'.join(recommendations)}")
        summaries.append({"symbol": symbol, "pnl": pnl, "text": "；".join(parts)})
    summaries.sort(key=lambda row: (float(row["pnl"]), str(row["symbol"])))
    return summaries


def _recommendation_for_cause(cause: str) -> str:
    recommendations = {
        "方向错误": "提高日线与4小时方向一致性要求，方向冲突时放弃入场",
        "入场过早": "等待价格进入建议区并完成低周期确认后再开仓",
        "插针扫损": "复核止损是否需要结构外缓冲或收盘确认，避免单根影线直接扫出",
        "盈利回吐": "达到计划盈利后分批锁盈，并按规则上移保护止损",
        "轮动负贡献": "提高换仓标的相对优势门槛，优势不足时不轮动",
        "结构失效": "保留结构止损，并复核入场时结构是否已经接近失效",
        "时间退出": "检查持仓超时前是否长期缺乏顺向推进，减少低效率占仓",
        "未分类亏损": "复核该笔方向、入场位置和退出过程，补充可重复的失败分类",
    }
    return recommendations.get(cause, "复核同类交易的方向、入场位置与退出节奏")


def _lifecycle_payload(
    key: str,
    fills: list[PaperFill],
    dataset: HistoricalDataset,
) -> dict[str, object] | None:
    opens = [fill for fill in fills if fill.action == "OPEN"]
    closes = [fill for fill in fills if fill.action in {"PARTIAL_CLOSE", "CLOSE"}]
    completed_trade = _completed_trade_payload(key, fills)
    if not opens or completed_trade is None:
        return None
    opening = opens[0]
    closing = next(fill for fill in reversed(closes) if fill.action == "CLOSE")
    pnl = float(completed_trade["realized_pnl"])
    planned_risk = opening.planned_risk_usdt or abs(opening.entry_price - opening.stop_price) * opening.quantity
    exit_time = closing.closed_at or closing.timestamp
    exit_category = _exit_category(closing.reason)
    failure_cause, evidence = _classify_failure(
        opening,
        closing,
        pnl,
        exit_category,
        dataset,
    )
    return {
        "id": key,
        "trade_cycle_id": key,
        "symbol": opening.symbol,
        "side": opening.side.value,
        "leverage": opening.leverage,
        "setup_type": opening.setup_type or "未分类",
        "primary_setup": opening.primary_setup or opening.setup_type,
        "supporting_evidence": list(opening.supporting_evidence),
        "entry_trigger": opening.entry_trigger,
        "validation_state": closing.validation_state or opening.validation_state,
        "soft_stop_price": opening.soft_stop_price or opening.stop_price,
        "hard_stop_price": opening.hard_stop_price or opening.stop_price,
        "entry_quality": opening.entry_quality or "-",
        "opened_at": opening.opened_at.isoformat(),
        "closed_at": exit_time.isoformat(),
        "entry_price": completed_trade["entry_price"],
        "exit_price": completed_trade["price"],
        "stop_price": opening.stop_price,
        "take_profit_1": opening.take_profit_1,
        "take_profit_2": opening.take_profit_2,
        "planned_risk_usdt": planned_risk,
        "pnl": pnl,
        "realized_r": pnl / planned_risk if planned_risk > 0 else 0.0,
        "mae_r": max((fill.mae_r for fill in closes), default=0.0),
        "mfe_r": max((fill.mfe_r for fill in closes), default=0.0),
        "holding_seconds": max((exit_time - opening.opened_at).total_seconds(), 0.0),
        "exit_reason": closing.reason,
        "exit_category": exit_category,
        "failure_cause": failure_cause if pnl < 0 else "",
        "failure_evidence": evidence if pnl < 0 else "",
        "entry_position": opening.entry_position,
        "adds": sum(fill.action == "ADD" for fill in fills),
        "partials": sum(fill.action == "PARTIAL_CLOSE" for fill in fills),
    }


def _classify_failure(
    opening: PaperFill,
    closing: PaperFill,
    pnl: float,
    exit_category: str,
    dataset: HistoricalDataset,
) -> tuple[str, str]:
    if pnl >= 0:
        return "", ""
    if exit_category == "轮动":
        return "轮动负贡献", "换仓退出后形成实际亏损，应对比换出与换入标的后续表现"
    if closing.mfe_r >= 1.0:
        return "盈利回吐", f"持仓最大浮盈达到 {closing.mfe_r:.2f}R，最终以亏损退出"
    if exit_category == "止损":
        candle = _candle_at(dataset, opening.symbol, closing.timestamp)
        if candle is not None:
            recovered = (
                candle.close > opening.stop_price
                if opening.side == PositionSide.LONG
                else candle.close < opening.stop_price
            )
            if recovered:
                return "插针扫损", "止损在15分钟影线内触发，但该K线收盘重新回到止损内侧"
        if closing.mfe_r < 0.35:
            return "方向判断失败", f"入场后最大有利波动仅 {closing.mfe_r:.2f}R，行情未验证入场方向"
        return "入场时机过早", f"方向曾产生 {closing.mfe_r:.2f}R 有利波动，但先触发了结构止损"
    if exit_category == "结构退出":
        return "结构判断失效", "持仓结构确认失效后亏损退出"
    if exit_category == "时间退出":
        return "行情未按预期展开", "规定持仓时间内未形成足够趋势延续"
    return "执行后走势不符", closing.reason or "亏损退出"


def _failure_summary(causes: list[dict[str, object]], completed: int, losses: int) -> str:
    if completed == 0:
        return "所选区间没有完成交易，无法判断胜率和失败根因。请先查看信号表中的否决原因与数据覆盖情况。"
    if not causes:
        return f"共完成 {completed} 笔交易，没有亏损交易；仍需扩大不同市场阶段样本，避免仅凭单一区间下结论。"
    top = causes[:3]
    details = "；".join(
        f"{item['cause']} {item['count']}笔，贡献 {float(item['pnl']):.2f}U"
        for item in top
    )
    return (
        f"共完成 {completed} 笔，亏损 {losses} 笔。主要失败来源：{details}。"
        "建议优先修改亏损贡献最大的现有逻辑分支，再用同一行情缓存重新回测，不增加新的前端调参项。"
    )


def _result_payload(
    replay: HistoricalReplayResult,
    dataset: HistoricalDataset,
    analysis: dict[str, object],
) -> dict[str, Any]:
    metrics = analysis["metrics"]
    account = dict(replay.final_status)
    account["fills"] = _completed_trade_payloads(replay.fills)
    total_pnl = replay.ending_equity - replay.starting_equity
    closed_trade_pnl = float(metrics.get("closed_trade_pnl") or 0.0)
    return _json_safe(
        {
            "summary": {
                "starting_equity": replay.starting_equity,
                "ending_equity": replay.ending_equity,
                "total_return": replay.total_return,
                "total_pnl": total_pnl,
                "closed_trade_pnl": closed_trade_pnl,
                "open_trade_pnl": total_pnl - closed_trade_pnl,
                "max_drawdown": replay.max_drawdown,
                "win_rate": metrics["win_rate"],
                "trade_count": metrics["completed"],
                "profit_factor": metrics["profit_factor"],
                "average_r": metrics["average_r"],
                "fees": replay.final_status.get("fees_paid", 0.0),
            },
            "period": {
                "start": dataset.start.astimezone(SHANGHAI).date().isoformat(),
                "end": (dataset.end - timedelta(microseconds=1)).astimezone(SHANGHAI).date().isoformat(),
            },
            "universe": list(dataset.universe),
            "universe_note": "首次运行时固定的当前Top50，非历史Top50；同日期命中缓存后沿用该名单。",
            "skipped_symbols": list(dataset.skipped),
            "equity_curve": _downsample_curve(replay.equity_curve),
            "account": account,
            "analysis": analysis,
            "per_symbol_pnl": replay.per_symbol_pnl,
            "notes": replay.notes,
        }
    )


def _publish_symbol_history(
    paper: PaperTradingEngine,
    market: _ReplayMarketData,
    symbol: str,
    data: HistoricalSymbolData,
    close_time: datetime,
    *,
    publish_signal: bool,
) -> None:
    visible_candles = {
        timeframe: _visible_candles(rows, timeframe, close_time)
        for timeframe, rows in data.candles.items()
    }
    visible_derivatives = {
        timeframe: _visible_derivatives(
            data.derivatives.get(timeframe, ()),
            visible_candles.get(timeframe, []),
        )
        for timeframe in visible_candles
    }
    paper._timeframe_candles[symbol] = visible_candles
    paper._timeframe_derivatives[symbol] = visible_derivatives
    market.candles[symbol] = visible_candles
    if publish_signal:
        paper._publish_symbol_from_cache(symbol)


def _publish_context_market(market: _ReplayMarketData, dataset: HistoricalDataset, event_time: datetime) -> None:
    for symbol, data in dataset.context.items():
        market.candles[symbol] = {
            timeframe: _visible_candles(rows, timeframe, event_time)
            for timeframe, rows in data.candles.items()
        }
    if "BTCUSDT" in dataset.symbols:
        data = dataset.symbols["BTCUSDT"]
        market.candles["BTCUSDT"] = {
            timeframe: _visible_candles(rows, timeframe, event_time)
            for timeframe, rows in data.candles.items()
        }


def _visible_candles(rows: tuple[Candle, ...], timeframe: str, event_time: datetime) -> list[Candle]:
    duration = timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])
    end = bisect_right(rows, event_time, key=lambda candle: candle.timestamp + duration)
    return list(rows[max(0, end - VISIBLE_HISTORY_LIMIT):end])


def _visible_derivatives(
    rows: tuple[DerivativesSnapshot, ...],
    candles: list[Candle],
) -> list[DerivativesSnapshot]:
    wanted = {candle.timestamp for candle in candles}
    return [row for row in rows if row.timestamp in wanted][-VISIBLE_HISTORY_LIMIT:]


def _intrabar_phase_price(
    candle: Candle,
    side: PositionSide | None,
    phase: str,
    *,
    candidate_action: str,
) -> float:
    if side == PositionSide.SHORT:
        adverse, favorable = candle.high, candle.low
    elif side == PositionSide.LONG:
        adverse, favorable = candle.low, candle.high
    elif candidate_action == "ENTRY_LONG":
        adverse, favorable = candle.high, candle.low
    elif candidate_action == "ENTRY_SHORT":
        adverse, favorable = candle.low, candle.high
    else:
        adverse, favorable = candle.low, candle.high
    return adverse if phase == "adverse" else favorable


def _has_replay_entry_candidate(paper: PaperTradingEngine) -> bool:
    return any(
        int(signal.get("score") or 0) >= AUTO_ENTRY_MIN_SCORE
        and str(signal.get("candidate_action") or signal.get("action") or "")
        in {"ENTRY_LONG", "ENTRY_SHORT"}
        for symbol, signal in paper.latest_signals.items()
        if symbol not in paper.account.positions
    )


def _apply_replay_funding(
    paper: PaperTradingEngine,
    symbol: str,
    derivatives: dict[str, list[DerivativesSnapshot]],
    event_time: datetime,
    charged: set[tuple[str, datetime]],
) -> None:
    if event_time.minute != 0 or event_time.hour % 8 != 0:
        return
    key = (symbol, event_time)
    position = paper.account.positions.get(symbol)
    snapshots = derivatives.get("1h") or derivatives.get("15m") or []
    if key in charged or position is None or not snapshots:
        return
    rate = snapshots[-1].funding_rate
    if rate is None:
        return
    mark = paper.latest_prices.get(symbol, position.entry_price)
    quantity = position.quantity * max(position.remaining_fraction, 0.0)
    realized = (-1 if position.side == PositionSide.LONG else 1) * mark * quantity * rate
    paper.account.wallet_balance += realized
    paper.account.realized_pnl += realized
    paper.account.fills.append(
        PaperFill(
            timestamp=event_time,
            symbol=symbol,
            side=position.side,
            action="FUNDING",
            price=mark,
            entry_price=position.entry_price,
            quantity=quantity,
            realized_pnl=realized,
            fee=0.0,
            reason=f"funding settlement rate={rate}",
            leverage=int(position.metadata.get("leverage", 1)),
            margin_usdt=0.0,
            stop_price=position.stop_price,
            take_profit_1=position.take_profit_1,
            take_profit_2=position.take_profit_2,
            opened_at=position.opened_at,
            closed_at=event_time,
        )
    )
    charged.add(key)


def _append_equity_point(
    paper: PaperTradingEngine,
    timestamp: datetime,
    curve: list[tuple[datetime, float]],
) -> None:
    equity = _paper_equity(paper)
    point = (_as_utc(timestamp), equity)
    if curve and curve[-1][0] == point[0]:
        curve[-1] = point
    else:
        curve.append(point)


def _progress_snapshot(
    status: dict[str, object],
    fills: Iterable[PaperFill],
    curve: list[tuple[datetime, float]],
    starting_equity: float,
) -> dict[str, Any]:
    """Return the small, strategy-signal-free snapshot used while replaying."""
    completed_trades = _completed_trade_payloads(fills)
    account_keys = (
        "equity",
        "available_balance",
        "used_margin",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "total_pnl_pct",
        "fees_paid",
        "positions",
    )
    account = {key: status.get(key) for key in account_keys}
    account["fills"] = completed_trades
    completed_pnl = [float(row["realized_pnl"]) for row in completed_trades]

    equity_value = status.get("equity")
    equity = float(equity_value if equity_value is not None else starting_equity)
    total_pnl = equity - starting_equity
    closed_trade_pnl = sum(completed_pnl)
    return _json_safe(
        {
            "summary": {
                "win_rate": (
                    sum(value > 0 for value in completed_pnl) / len(completed_pnl)
                    if completed_pnl
                    else 0.0
                ),
                "trade_count": len(completed_pnl),
                "max_drawdown": _max_drawdown(curve, starting_equity),
                "total_pnl": total_pnl,
                "closed_trade_pnl": closed_trade_pnl,
                "open_trade_pnl": total_pnl - closed_trade_pnl,
                "total_return": total_pnl / starting_equity if starting_equity else 0.0,
            },
            "account": account,
        }
    )


def _paper_equity(paper: PaperTradingEngine) -> float:
    unrealized = 0.0
    for position in paper.account.positions.values():
        mark = paper.latest_prices.get(position.symbol, position.entry_price)
        quantity = position.quantity * max(position.remaining_fraction, 0.0)
        if position.side == PositionSide.LONG:
            unrealized += (mark - position.entry_price) * quantity
        else:
            unrealized += (position.entry_price - mark) * quantity
    return paper.account.wallet_balance + unrealized


def _paper_risk_status(paper: PaperTradingEngine) -> dict[str, float]:
    equity = _paper_equity(paper)
    used_margin = sum(
        float(position.metadata.get("margin_usdt", 0.0))
        for position in paper.account.positions.values()
    )
    return {
        "equity": equity,
        "available_balance": max(equity - used_margin, 0.0),
        "used_margin": used_margin,
    }


def _max_drawdown(curve: list[tuple[datetime, float]], starting_equity: float) -> float:
    peak = starting_equity
    maximum = 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak if peak else 0.0)
    return maximum


def _per_symbol_pnl(fills: Iterable[PaperFill]) -> dict[str, float]:
    result: dict[str, float] = {}
    for fill in fills:
        value = -fill.fee if fill.action in {"OPEN", "ADD"} else fill.realized_pnl
        result[fill.symbol] = result.get(fill.symbol, 0.0) + value
    return result


def _symbol_data_problem(data: HistoricalSymbolData, start: datetime, end: datetime) -> str | None:
    base = [candle for candle in data.candles.get(BASE_TIMEFRAME, ()) if start <= candle.timestamp < end]
    if not base:
        return "所选时期尚未上市或没有15分钟K线"
    required = max(int((end - start).total_seconds() // _TIMEFRAME_SECONDS[BASE_TIMEFRAME]) - 2, 1)
    if len(base) < required * 0.95:
        return f"15分钟K线覆盖不足：{len(base)}/{required}"
    for timeframe in ("1h", "4h"):
        rows = [row for row in data.derivatives.get(timeframe, ()) if start <= row.timestamp < end]
        if not rows:
            return f"{timeframe} OI/多空比历史不可用"
        if rows[0].timestamp > start + timedelta(seconds=_TIMEFRAME_SECONDS[timeframe] * 2):
            return f"{timeframe} OI/多空比未覆盖回测开始时间"
        if rows[-1].timestamp < end - timedelta(seconds=_TIMEFRAME_SECONDS[timeframe] * 2):
            return f"{timeframe} OI/多空比未覆盖回测结束时间"
        if not any(row.funding_rate is not None for row in rows):
            return "历史资金费率不可用"
    return None


def _skipped_reason_summary(rows: Iterable[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "未知数据错误")
        counts[reason] = counts.get(reason, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:2]
    return "；".join(f"{reason}（{count}个）" for reason, count in ranked)


def _current_top50(rows: Iterable[FuturesSymbol]) -> list[str]:
    return [
        item.symbol.upper()
        for item in rows
        if item.symbol.upper() not in AUTO_UNIVERSE_EXCLUDED_SYMBOLS
        and item.quote_volume >= CANDIDATE_MIN_QUOTE_VOLUME
    ][:UNIVERSE_SIZE]


def _validated_range(start_value: str, end_value: str) -> tuple[datetime, datetime]:
    if not start_value or not end_value:
        raise ValueError("开始日期和结束日期不能为空")
    start = _parse_date(start_value)
    end = _parse_date(end_value) + timedelta(days=1)
    today_end = (
        datetime.now(SHANGHAI)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    ).astimezone(UTC)
    if end > today_end:
        raise ValueError("结束日期不能晚于今天")
    end = min(end, _floor_time(datetime.now(UTC), BASE_TIMEFRAME))
    if end <= start:
        raise ValueError("所选区间还没有已完成的15分钟K线")
    if end - start > timedelta(days=MAX_REPLAY_DAYS):
        raise ValueError(f"为保证 OI/多空比一致，单次历史回测最多 {MAX_REPLAY_DAYS} 天")
    return start, end


def _floor_time(value: datetime, timeframe: str) -> datetime:
    seconds = _TIMEFRAME_SECONDS[timeframe]
    timestamp = int(value.astimezone(UTC).timestamp())
    return datetime.fromtimestamp(timestamp - timestamp % seconds, UTC)


def _inclusive_end_date(end: datetime):
    return (end.astimezone(SHANGHAI) - timedelta(microseconds=1)).date()


def _parse_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    return parsed.astimezone(UTC)


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


def _candle_at(dataset: HistoricalDataset, symbol: str, timestamp: datetime) -> Candle | None:
    data = dataset.symbols.get(symbol)
    if data is None:
        return None
    seconds = _TIMEFRAME_SECONDS[BASE_TIMEFRAME]
    slot = datetime.fromtimestamp(int(timestamp.timestamp() // seconds) * seconds, tz=UTC)
    return next((candle for candle in data.candles.get(BASE_TIMEFRAME, ()) if candle.timestamp == slot), None)


def _downsample_curve(
    curve: tuple[tuple[datetime, float], ...],
    limit: int = 600,
) -> list[dict[str, object]]:
    if not curve:
        return []
    step = max(math.ceil(len(curve) / limit), 1)
    sampled = list(curve[::step])
    if sampled[-1] != curve[-1]:
        sampled.append(curve[-1])
    return [{"timestamp": timestamp, "equity": equity} for timestamp, equity in sampled]


def _dataset_payload(dataset: HistoricalDataset) -> dict[str, object]:
    return {
        "schema": 1,
        "start": dataset.start.isoformat(),
        "end": dataset.end.isoformat(),
        "downloaded_at": dataset.downloaded_at.isoformat(),
        "universe": list(dataset.universe),
        "skipped": list(dataset.skipped),
        "symbols": {symbol: _symbol_payload(data) for symbol, data in dataset.symbols.items()},
        "context": {symbol: _symbol_payload(data) for symbol, data in dataset.context.items()},
    }


def _symbol_payload(data: HistoricalSymbolData) -> dict[str, object]:
    return {
        "candles": {
            timeframe: [
                [row.timestamp.isoformat(), row.open, row.high, row.low, row.close, row.volume]
                for row in rows
            ]
            for timeframe, rows in data.candles.items()
        },
        "derivatives": {
            timeframe: [
                [
                    row.timestamp.isoformat(),
                    row.open_interest,
                    row.long_short_ratio,
                    row.funding_rate,
                    row.taker_buy_sell_ratio,
                    row.taker_buy_volume,
                    row.taker_sell_volume,
                    row.top_account_long_short_ratio,
                    row.top_position_long_short_ratio,
                ]
                for row in rows
            ]
            for timeframe, rows in data.derivatives.items()
        },
    }


def _dataset_from_payload(payload: dict[str, object]) -> HistoricalDataset:
    if payload.get("schema") != 1:
        raise ValueError("历史缓存版本不兼容")
    return HistoricalDataset(
        start=_as_utc(datetime.fromisoformat(str(payload["start"]))),
        end=_as_utc(datetime.fromisoformat(str(payload["end"]))),
        downloaded_at=_as_utc(datetime.fromisoformat(str(payload["downloaded_at"]))),
        universe=tuple(str(symbol) for symbol in payload["universe"]),  # type: ignore[index]
        skipped=tuple(dict(row) for row in payload.get("skipped", [])),  # type: ignore[arg-type]
        symbols={
            str(symbol): _symbol_from_payload(data)
            for symbol, data in dict(payload["symbols"]).items()  # type: ignore[arg-type]
        },
        context={
            str(symbol): _symbol_from_payload(data)
            for symbol, data in dict(payload.get("context", {})).items()  # type: ignore[arg-type]
        },
    )


def _symbol_from_payload(raw: object) -> HistoricalSymbolData:
    payload = dict(raw)  # type: ignore[arg-type]
    candles = {
        str(timeframe): tuple(
            Candle(
                timestamp=_as_utc(datetime.fromisoformat(str(row[0]))),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
        )
        for timeframe, rows in dict(payload.get("candles", {})).items()  # type: ignore[arg-type]
    }
    derivatives = {
        str(timeframe): tuple(
            DerivativesSnapshot(
                timestamp=_as_utc(datetime.fromisoformat(str(row[0]))),
                open_interest=float(row[1]) if row[1] is not None else None,
                long_short_ratio=float(row[2]) if row[2] is not None else None,
                funding_rate=float(row[3]) if row[3] is not None else None,
                taker_buy_sell_ratio=(
                    float(row[4])
                    if len(row) > 4 and row[4] is not None
                    else None
                ),
                taker_buy_volume=(
                    float(row[5])
                    if len(row) > 5 and row[5] is not None
                    else None
                ),
                taker_sell_volume=(
                    float(row[6])
                    if len(row) > 6 and row[6] is not None
                    else None
                ),
                top_account_long_short_ratio=(
                    float(row[7])
                    if len(row) > 7 and row[7] is not None
                    else None
                ),
                top_position_long_short_ratio=(
                    float(row[8])
                    if len(row) > 8 and row[8] is not None
                    else None
                ),
            )
            for row in rows
        )
        for timeframe, rows in dict(payload.get("derivatives", {})).items()  # type: ignore[arg-type]
    }
    return HistoricalSymbolData(candles=candles, derivatives=derivatives)


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


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _to_ms(value: datetime) -> int:
    return int(_as_utc(value).timestamp() * 1000)


def _slipped(price: float, side: PositionSide, rate: float, *, entering: bool) -> float:
    if side == PositionSide.LONG:
        return price * (1 + rate if entering else 1 - rate)
    return price * (1 - rate if entering else 1 + rate)
