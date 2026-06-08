from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime, timedelta

from ai_trading.backtest import BacktestEngine
from ai_trading.config import load_settings
from ai_trading.data import load_candles_csv, load_derivatives_csv
from ai_trading.indicators import build_indicators
from ai_trading.models import Candle, DerivativesSnapshot
from ai_trading.strategy import CompositeStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="AI trading strategy research CLI")
    parser.add_argument("--config", default="config/strategy.yaml", help="Path to strategy YAML")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo backtest")
    parser.add_argument("--symbol", default="DEMOUSDT", help="Symbol name for reports")
    parser.add_argument("--candles-csv", help="CSV with timestamp,open,high,low,close,volume")
    parser.add_argument("--derivatives-csv", help="CSV with timestamp,open_interest,long_short_ratio,funding_rate")
    parser.add_argument("--equity", type=float, default=10_000.0, help="Starting equity")
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.candles_csv:
        candles = load_candles_csv(args.candles_csv)
        derivatives = load_derivatives_csv(args.derivatives_csv) if args.derivatives_csv else None
        result = BacktestEngine(
            symbol=args.symbol,
            starting_equity=args.equity,
            strategy_settings=settings.strategy,
            risk_settings=settings.risk,
            execution_settings=settings.execution,
        ).run(candles, derivatives)
        print(f"symbol={args.symbol}")
        print(f"ending_equity={result.ending_equity:.2f} return={result.total_return:.2%} trades={len(result.trades)}")
        print(f"max_drawdown={result.max_drawdown:.2%} win_rate={result.win_rate:.2%}")
        for trade in result.trades[-10:]:
            print(f"trade side={trade.side.value} entry={trade.entry_price:.4f} exit={trade.exit_price:.4f} pnl={trade.pnl:.2f} reason={trade.reason}")
        return

    if args.demo:
        candles, derivatives = _synthetic_market()
        indicators = build_indicators(
            candles,
            derivatives,
            ema_fast=settings.strategy.ema_fast,
            ema_slow=settings.strategy.ema_slow,
            ma_trend=settings.strategy.ma_trend,
            bollinger_window=settings.strategy.bollinger_window,
            bollinger_stddev=settings.strategy.bollinger_stddev,
            rsi_window=settings.strategy.rsi_window,
            atr_window=settings.strategy.atr_window,
            volume_window=settings.strategy.volume_window,
        )
        signal = CompositeStrategy(settings.strategy).generate_signal("DEMOUSDT", candles, indicators)
        result = BacktestEngine(
            symbol=args.symbol,
            starting_equity=args.equity,
            strategy_settings=settings.strategy,
            risk_settings=settings.risk,
            execution_settings=settings.execution,
        ).run(candles, derivatives)
        print(f"latest_signal={signal.action.value} score={signal.score} regime={signal.regime.value}")
        print(f"reasons={'; '.join(signal.reasons)}")
        print(f"ending_equity={result.ending_equity:.2f} return={result.total_return:.2%} trades={len(result.trades)}")
        print(f"max_drawdown={result.max_drawdown:.2%} win_rate={result.win_rate:.2%}")
        return

    parser.print_help()


def _synthetic_market() -> tuple[list[Candle], list[DerivativesSnapshot]]:
    candles: list[Candle] = []
    derivatives: list[DerivativesSnapshot] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    price = 100.0
    oi = 10_000_000.0
    for idx in range(180):
        wave = math.sin(idx / 8) * 0.35
        drift = 0.09 if idx > 70 else 0.02
        previous = price
        price = price + drift + wave * 0.08
        high = max(previous, price) * 1.004
        low = min(previous, price) * 0.996
        volume = 1_000_000 + (idx % 12) * 35_000
        if idx in {115, 130, 145}:
            volume *= 1.5
        timestamp = start + timedelta(minutes=15 * idx)
        candles.append(Candle(timestamp=timestamp, open=previous, high=high, low=low, close=price, volume=volume))
        oi *= 1.0015 if idx > 80 else 1.0002
        derivatives.append(
            DerivativesSnapshot(
                timestamp=timestamp,
                open_interest=oi,
                long_short_ratio=1.15,
                funding_rate=0.0001,
            )
        )
    return candles, derivatives


if __name__ == "__main__":
    main()
