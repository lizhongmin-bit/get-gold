"""AkShare data source implementation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from tail30_selector.datasource.base import DataSource, DataUniverse


@dataclass
class AkShareConfig:
    index_symbol: str = "000001.SH"


class AkShareSource:
    """AkShare-backed datasource.

    Uses AkShare spot and minute data when available.
    """

    def __init__(self, config: Optional[AkShareConfig] = None):
        self.config = config or AkShareConfig()
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "AkShare is not installed. Please install akshare to use this datasource."
            ) from exc
        self.ak = ak

    def get_universe(self, universe: str) -> DataUniverse:
        if universe == "hs300":
            df = self.ak.index_stock_cons(symbol="000300")
            symbols = df["con_code"].astype(str).tolist()
            return DataUniverse(name="hs300", symbols=symbols)
        if universe == "all":
            df = self.ak.stock_zh_a_spot_em()
            symbols = df["代码"].astype(str).tolist()
            return DataUniverse(name="all", symbols=symbols)
        raise ValueError(f"Unsupported universe: {universe}")

    def get_daily_snapshot(self, trade_date: date, symbols: list[str]) -> pd.DataFrame:
        df = self.ak.stock_zh_a_spot_em()
        df = df[df["代码"].isin(symbols)]
        df = df.rename(
            columns={
                "代码": "symbol",
                "名称": "name",
                "涨跌幅": "pct_change",
                "成交量": "volume",
                "量比": "volume_ratio",
                "换手率": "turnover",
                "流通市值": "float_mktcap",
                "最新价": "close",
            }
        )
        df["trade_date"] = pd.to_datetime(trade_date)
        df["pct_change"] = df["pct_change"].astype(float) / 100.0
        df["turnover"] = df["turnover"].astype(float) / 100.0
        df["float_mktcap"] = df["float_mktcap"].astype(float)
        return df

    def get_intraday(self, trade_date: date, symbol: str) -> pd.DataFrame:
        try:
            df = self.ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="qfq")
        except Exception as exc:  # pragma: no cover - upstream variations
            raise RuntimeError(f"Failed to fetch intraday for {symbol}: {exc}") from exc
        rename_map = {
            "时间": "datetime",
            "datetime": "datetime",
            "收盘": "close",
            "close": "close",
            "成交量": "volume",
            "volume": "volume",
            "成交额": "amount",
            "amount": "amount",
        }
        df = df.rename(columns=rename_map)
        if "datetime" not in df.columns:
            raise RuntimeError(f"Intraday data missing datetime column for {symbol}.")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"].dt.date == trade_date]
        return df

    def get_index_intraday(self, trade_date: date, index_symbol: str) -> pd.DataFrame:
        symbols_to_try = [index_symbol]
        if index_symbol.endswith(".SH"):
            symbols_to_try.append(index_symbol.replace(".SH", ""))
        last_exc = None
        df = pd.DataFrame()
        for symbol in symbols_to_try:
            try:
                df = self.ak.index_zh_a_hist_min_em(symbol=symbol)
                if not df.empty:
                    break
            except Exception as exc:  # pragma: no cover - upstream variations
                last_exc = exc
        if df.empty and last_exc is not None:
            raise RuntimeError(f"Failed to fetch index intraday for {index_symbol}: {last_exc}") from last_exc
        df = df.rename(columns={"时间": "datetime", "收盘": "close", "成交量": "volume", "成交额": "amount"})
        if "datetime" not in df.columns:
            raise RuntimeError(f"Index intraday data missing datetime column for {index_symbol}.")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"].dt.date == trade_date]
        return df

    def get_daily_history(self, trade_date: date, symbol: str, lookback: int) -> pd.DataFrame:
        try:
            df = self.ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to fetch daily history for {symbol}: {exc}") from exc
        df = df.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume"})
        df["date"] = pd.to_datetime(df["date"])  # type: ignore[assignment]
        df = df.sort_values("date")
        if len(df) > lookback:
            df = df.tail(lookback)
        return df
