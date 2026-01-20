"""Base datasource abstraction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd


@dataclass
class DataUniverse:
    name: str
    symbols: list[str]


class DataSource(Protocol):
    """Protocol for data sources."""

    def get_universe(self, universe: str) -> DataUniverse:
        ...

    def get_daily_snapshot(self, trade_date: date, symbols: list[str]) -> pd.DataFrame:
        ...

    def get_intraday(self, trade_date: date, symbol: str) -> pd.DataFrame:
        ...

    def get_index_intraday(self, trade_date: date, index_symbol: str) -> pd.DataFrame:
        ...

    def get_daily_history(self, trade_date: date, symbol: str, lookback: int) -> pd.DataFrame:
        ...
