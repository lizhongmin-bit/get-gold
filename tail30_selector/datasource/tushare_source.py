"""Tushare data source implementation (fallback)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from tail30_selector.datasource.base import DataSource, DataUniverse


@dataclass
class TushareConfig:
    token: Optional[str] = None
    index_symbol: str = "000001.SH"


class TushareSource:
    def __init__(self, config: Optional[TushareConfig] = None):
        self.config = config or TushareConfig()
        try:
            import tushare as ts  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Tushare is not installed. Please install tushare to use this datasource."
            ) from exc
        if not self.config.token:
            raise RuntimeError("Tushare token is required. Set TS_TOKEN env or config.")
        self.ts = ts
        self.pro = ts.pro_api(self.config.token)

    def get_universe(self, universe: str) -> DataUniverse:
        if universe != "custom":
            raise ValueError("Tushare source only supports custom universe in this demo.")
        raise NotImplementedError("Provide custom symbols when using Tushare source.")

    def get_daily_snapshot(self, trade_date: date, symbols: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Tushare daily snapshot not implemented in fallback mode.")

    def get_intraday(self, trade_date: date, symbol: str) -> pd.DataFrame:
        raise NotImplementedError("Tushare intraday data requires premium access.")

    def get_index_intraday(self, trade_date: date, index_symbol: str) -> pd.DataFrame:
        raise NotImplementedError("Tushare intraday index data requires premium access.")

    def get_daily_history(self, trade_date: date, symbol: str, lookback: int) -> pd.DataFrame:
        start = (trade_date.replace(day=1) - pd.Timedelta(days=lookback * 2)).strftime("%Y%m%d")
        df = self.pro.daily(ts_code=symbol, start_date=start)
        df = df.rename(columns={"trade_date": "date", "close": "close", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.sort_values("date")
        return df.tail(lookback)
