"""Intraday indicators: VWAP and relative strength."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Tuple

import pandas as pd

from tail30_selector.utils.time_utils import tail_window_times


@dataclass
class IntradayStrength:
    above_vwap_ratio_day: float
    above_vwap_ratio_tail: float
    outperformance_mean_tail: float
    outperformance_last_tail: float
    new_high_after_1430: bool
    pullback_support: bool
    pullback_reentry_time: pd.Timestamp | None


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    amount = df["amount"] if "amount" in df.columns else df["close"] * df["volume"]
    cumulative_amount = amount.cumsum()
    cumulative_volume = df["volume"].cumsum().replace(0, pd.NA)
    return cumulative_amount / cumulative_volume


def intraday_strength(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> IntradayStrength:
    stock_df = stock_df.copy()
    index_df = index_df.copy()
    stock_df["vwap"] = compute_vwap(stock_df)
    index_df["vwap"] = compute_vwap(index_df)
    stock_df = stock_df.set_index("datetime")
    index_df = index_df.set_index("datetime")
    aligned = stock_df.join(index_df[["close"]], rsuffix="_index", how="inner")
    if aligned.empty:
        raise ValueError("No aligned intraday data for stock and index.")
    aligned["above_vwap"] = aligned["close"] >= aligned["vwap"]
    above_vwap_ratio_day = float(aligned["above_vwap"].mean())
    tail_start, tail_end = tail_window_times()
    tail_mask = (aligned.index.time >= tail_start) & (aligned.index.time <= tail_end)
    tail_slice = aligned[tail_mask]
    if tail_slice.empty:
        raise ValueError("No intraday data in tail window.")
    above_vwap_ratio_tail = float(tail_slice["above_vwap"].mean())
    aligned["return"] = aligned["close"] / aligned["close"].iloc[0] - 1.0
    aligned["return_index"] = aligned["close_index"] / aligned["close_index"].iloc[0] - 1.0
    aligned["outperformance"] = aligned["return"] - aligned["return_index"]
    outperformance_mean_tail = float(tail_slice["outperformance"].mean())
    outperformance_last_tail = float(tail_slice["outperformance"].iloc[-1])
    pre_tail_max = aligned[aligned.index.time < tail_start]["close"].max()
    new_high_after_1430 = bool(tail_slice["close"].max() > pre_tail_max)
    pullback_support, reentry_time = _pullback_support(tail_slice)
    return IntradayStrength(
        above_vwap_ratio_day=above_vwap_ratio_day,
        above_vwap_ratio_tail=above_vwap_ratio_tail,
        outperformance_mean_tail=outperformance_mean_tail,
        outperformance_last_tail=outperformance_last_tail,
        new_high_after_1430=new_high_after_1430,
        pullback_support=pullback_support,
        pullback_reentry_time=reentry_time,
    )


def _pullback_support(tail_slice: pd.DataFrame) -> Tuple[bool, pd.Timestamp | None]:
    tail_slice = tail_slice.copy()
    tail_slice["above_vwap"] = tail_slice["close"] >= tail_slice["vwap"]
    below = tail_slice[~tail_slice["above_vwap"]]
    if below.empty:
        return True, None
    reentry = tail_slice[tail_slice.index > below.index[-1]]
    reentry = reentry[reentry["above_vwap"]]
    if reentry.empty:
        return False, None
    return True, reentry.index[0]
