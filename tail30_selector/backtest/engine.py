"""Simple backtest engine for tail30 strategy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from tail30_selector.datasource.base import DataSource
from tail30_selector.strategy.tail30 import Tail30Strategy


@dataclass
class BacktestResult:
    trade_date: date
    symbol: str
    next_open_return: float
    next_close_return: float


def run_backtest(strategy: Tail30Strategy, trade_date: date, universe: list[str]) -> list[BacktestResult]:
    results = strategy.run(trade_date, universe)
    selected = [r for r in results if r.step7_pass]
    backtest_results: list[BacktestResult] = []
    for item in selected:
        history = strategy.data_source.get_daily_history(trade_date, item.symbol, lookback=2)
        if len(history) < 2:
            continue
        prev = history.iloc[-2]
        current = history.iloc[-1]
        next_open_return = (current["close"] - prev["close"]) / prev["close"]
        next_close_return = next_open_return
        backtest_results.append(
            BacktestResult(
                trade_date=trade_date,
                symbol=item.symbol,
                next_open_return=float(next_open_return),
                next_close_return=float(next_close_return),
            )
        )
    return backtest_results
