"""Command-line interface for tail30 selector."""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import Optional

from tail30_selector.backtest.engine import run_backtest
from tail30_selector.config import DEFAULT_CONFIG
from tail30_selector.datasource.akshare_source import AkShareSource
from tail30_selector.datasource.base import DataSource
from tail30_selector.datasource.tushare_source import TushareConfig, TushareSource
from tail30_selector.strategy.tail30 import Tail30Strategy
from tail30_selector.utils.logging import setup_logging
from tail30_selector.utils.report import to_dict, write_csv
from tail30_selector.utils.time_utils import most_recent_trading_date


RISK_WARNING = (
    "风险提示: 策略不是100%胜率，只是提高概率；有时候筛完一只都没有 => 空仓也是操作；"
    "要设置止盈止损，行情不对及时撤退；仅供学习交流，不构成投资建议。"
)


def build_data_source(source: str, token: Optional[str]) -> DataSource:
    if source == "akshare":
        return AkShareSource()
    if source == "tushare":
        token = token or os.getenv("TS_TOKEN")
        return TushareSource(TushareConfig(token=token))
    raise ValueError(f"Unsupported datasource: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail30 selector")
    parser.add_argument("--date", help="trade date YYYY-MM-DD")
    parser.add_argument("--mode", default="realtime", choices=["realtime", "backtest"])
    parser.add_argument("--universe", default="all", choices=["all", "hs300", "custom"])
    parser.add_argument("--datasource", default="akshare", choices=["akshare", "tushare"])
    parser.add_argument("--token", help="tushare token")
    args = parser.parse_args()

    logger = setup_logging(logging.INFO)
    trade_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else most_recent_trading_date()
    )
    data_source = build_data_source(args.datasource, args.token)
    universe_data = data_source.get_universe(args.universe)
    strategy = Tail30Strategy(data_source, DEFAULT_CONFIG)

    logger.info("Running tail30 selector on %s with universe %s", trade_date, args.universe)
    results = strategy.run(trade_date, universe_data.symbols)
    selected = [r for r in results if r.step7_pass]
    selected_sorted = sorted(selected, key=lambda x: x.pct_change or 0.0, reverse=True)

    for item in selected_sorted[: DEFAULT_CONFIG.max_output]:
        logger.info("%s %s -> %s", item.symbol, item.name, item.reason)

    rows = [to_dict(item) for item in selected_sorted]
    filename = f"selected_{trade_date.strftime('%Y%m%d')}.csv"
    output_path = write_csv(filename, rows)
    logger.info("Saved CSV to %s", output_path)
    logger.info(RISK_WARNING)

    if args.mode == "backtest":
        backtest_results = run_backtest(strategy, trade_date, universe_data.symbols)
        for result in backtest_results:
            logger.info(
                "Backtest %s: next open return %.2f%%", result.symbol, result.next_open_return * 100
            )


if __name__ == "__main__":
    main()
