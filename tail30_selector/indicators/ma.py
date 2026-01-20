"""Moving average indicators."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def slope(series: pd.Series) -> float:
    """Simple linear regression slope for a series."""
    y = series.dropna().values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    slope_val = np.polyfit(x, y, 1)[0]
    return float(slope_val)


def is_bullish_alignment(ma_values: dict[int, float]) -> bool:
    periods = sorted(ma_values.keys())
    return all(ma_values[periods[i]] > ma_values[periods[i + 1]] for i in range(len(periods) - 1))


def is_diverging(current: dict[int, float], past: dict[int, float]) -> bool:
    """Check if gaps are widening between moving averages."""
    periods = sorted(current.keys())
    current_gaps = [current[periods[i]] - current[periods[i + 1]] for i in range(len(periods) - 1)]
    past_gaps = [past[periods[i]] - past[periods[i + 1]] for i in range(len(periods) - 1)]
    return all(c > p for c, p in zip(current_gaps, past_gaps))


def build_ma_snapshot(close: pd.Series, periods: Iterable[int]) -> dict[int, float]:
    snapshot = {}
    for period in periods:
        ma_series = moving_average(close, period)
        snapshot[period] = float(ma_series.iloc[-1])
    return snapshot
