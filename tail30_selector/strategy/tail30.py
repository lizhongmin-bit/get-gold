"""Tail30 strategy: implement stepwise filters from 14:30 onward."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from tail30_selector.config import StrategyConfig
from tail30_selector.datasource.base import DataSource
from tail30_selector.indicators.intraday import IntradayStrength, intraday_strength
from tail30_selector.indicators.ma import build_ma_snapshot, is_bullish_alignment, is_diverging, slope
from tail30_selector.indicators.volume_shape import VolumeShapeResult, analyze_volume_shape


@dataclass
class StockStepResult:
    symbol: str
    name: str
    step1_pass: bool
    step2_pass: bool
    step3_pass: bool
    step4_pass: bool
    step5_pass: bool
    step6_pass: bool
    step7_pass: bool
    pct_change: float | None = None
    volume_ratio: float | None = None
    turnover: float | None = None
    float_mktcap: float | None = None
    volume_shape: VolumeShapeResult | None = None
    ma_snapshot: dict[int, float] | None = None
    ma_slope: dict[int, float] | None = None
    bullish_alignment: bool = False
    ma_diverging: bool = False
    close_above_ma: bool = False
    intraday: IntradayStrength | None = None
    reason: str = ""
    notes: dict[str, Any] | None = None


class Tail30Strategy:
    def __init__(self, data_source: DataSource, config: StrategyConfig):
        self.data_source = data_source
        self.config = config

    def run(self, trade_date: date, universe: list[str]) -> list[StockStepResult]:
        snapshot = self.data_source.get_daily_snapshot(trade_date, universe)
        results: list[StockStepResult] = []
        for _, row in snapshot.iterrows():
            result = self._evaluate_symbol(trade_date, row)
            results.append(result)
        return results

    def _retry_call(self, func, *args, **kwargs):
        for attempt in range(self.config.retry_times + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - external data variance
                if attempt >= self.config.retry_times:
                    raise exc

    def _evaluate_symbol(self, trade_date: date, row: pd.Series) -> StockStepResult:
        pct_change = float(row.get("pct_change", 0.0))
        volume_ratio = float(row.get("volume_ratio", 0.0) or 0.0)
        turnover = float(row.get("turnover", 0.0) or 0.0)
        float_mktcap = float(row.get("float_mktcap", 0.0))
        symbol = str(row.get("symbol"))
        name = str(row.get("name", ""))

        reasons = []
        notes: dict[str, Any] = {}

        # Step1: 14:30后涨幅3%-5%过滤
        step1_pass = self.config.pct_change_range[0] <= pct_change <= self.config.pct_change_range[1]
        reasons.append(f"Step1涨幅{pct_change:.2%}是否在3%-5%: {step1_pass}")

        # Step2: 量比>=1筛人气
        if volume_ratio <= 0:
            try:
                history = self._retry_call(
                    self.data_source.get_daily_history,
                    trade_date,
                    symbol,
                    lookback=6,
                )
                if len(history) >= 2:
                    volume_ratio = float(history["volume"].iloc[-1] / history["volume"].iloc[:-1].mean())
                    notes["volume_ratio_estimated"] = True
            except Exception as exc:
                notes["volume_ratio_error"] = str(exc)
        # Step2: 量比>=1筛人气
        step2_pass = volume_ratio >= self.config.volume_ratio_min
        reasons.append(f"Step2量比{volume_ratio:.2f}是否>=1: {step2_pass}")

        # Step3: 换手率5%-10%
        if turnover <= 0 and "free_float_shares" in row:
            free_float = float(row.get("free_float_shares") or 0.0)
            if free_float > 0:
                turnover = float(row.get("volume", 0.0)) / free_float
                notes["turnover_estimated"] = True
        step3_pass = self.config.turnover_range[0] <= turnover <= self.config.turnover_range[1]
        reasons.append(f"Step3换手率{turnover:.2%}是否在5%-10%: {step3_pass}")

        # Step4: 流通市值50亿-100亿
        step4_pass = self.config.float_mktcap_range[0] <= float_mktcap <= self.config.float_mktcap_range[1]
        reasons.append(
            f"Step4流通市值{float_mktcap/1e8:.2f}亿是否在50-100亿: {step4_pass}"
        )

        step5_pass = False
        volume_shape = None
        step6_pass = False
        ma_snapshot = None
        ma_slope = None
        bullish_alignment = False
        ma_diverging = False
        close_above_ma = False
        step7_pass = False
        intraday_result = None

        if step1_pass and step2_pass and step3_pass and step4_pass:
            # Step5: 成交量台阶式放量判定
            history = self._retry_call(
                self.data_source.get_daily_history,
                trade_date,
                symbol,
                lookback=self.config.volume_shape_window,
            )
            volume_shape = analyze_volume_shape(
                history["volume"],
                min_seg=self.config.volume_shape_min_seg,
                k_candidates=self.config.volume_shape_k_candidates,
                delta_min=self.config.volume_shape_delta_min,
                delta_allow=self.config.volume_shape_delta_allow,
                sigma_max=self.config.volume_shape_sigma_max,
                cv_max=self.config.volume_shape_cv_max,
                sep_min=self.config.volume_shape_sep_min,
                gain_min=self.config.volume_shape_gain_min,
                recent_ratio_min=self.config.volume_shape_recent_ratio_min,
                jump_ratio_high=self.config.volume_shape_jump_ratio_high,
                jump_ratio_low=self.config.volume_shape_jump_ratio_low,
                max_jump_count=self.config.volume_shape_jump_max_count,
            )
            step5_pass = volume_shape.is_stepwise
            reasons.append(
                "Step5成交量形态台阶式放量: "
                f"{step5_pass} (K={volume_shape.k_star}, gain={volume_shape.gain:.2f}, "
                f"ratio={volume_shape.recent_vs_early_ratio:.2f})"
            )

            # Step6: 均线多头排列 + K线在均线之上
            close_series = history["close"]
            ma_snapshot = build_ma_snapshot(close_series, self.config.ma_periods)
            ma_slope = {
                period: slope(close_series.rolling(window=period).mean())
                for period in self.config.ma_periods
            }
            bullish_alignment = is_bullish_alignment(ma_snapshot)
            past_snapshot = build_ma_snapshot(
                close_series.iloc[:-1], self.config.ma_periods
            )
            ma_diverging = is_diverging(ma_snapshot, past_snapshot)
            close_above_ma = close_series.iloc[-1] > max(
                ma_snapshot[period] for period in [5, 10, 20]
            )
            slope_positive = all(val > 0 for val in ma_slope.values())
            step6_pass = bullish_alignment and ma_diverging and close_above_ma and slope_positive
            reasons.append(
                "Step6均线多头排列/上升发散/K线在均线上方: "
                f"{step6_pass} (排列={bullish_alignment},发散={ma_diverging},"
                f"K线={close_above_ma},斜率={slope_positive})"
            )

        if step1_pass and step2_pass and step3_pass and step4_pass and step5_pass and step6_pass:
            # Step7: 分时图强势 + 跑赢大盘验证
            try:
                intraday = self._retry_call(self.data_source.get_intraday, trade_date, symbol)
                index_intraday = self._retry_call(
                    self.data_source.get_index_intraday, trade_date, "000001.SH"
                )
                intraday_result = intraday_strength(intraday, index_intraday)
            except Exception as exc:
                notes["intraday_error"] = str(exc)
                intraday_result = None
        if intraday_result is not None:
            step7_pass = (
                intraday_result.above_vwap_ratio_day >= self.config.intraday_above_vwap_ratio_day
                and intraday_result.above_vwap_ratio_tail
                >= self.config.intraday_above_vwap_ratio_tail
                and intraday_result.outperformance_mean_tail > 0
                and intraday_result.outperformance_last_tail > 0
                and (not self.config.new_high_after_1430 or intraday_result.new_high_after_1430)
                and (not self.config.pullback_not_break or intraday_result.pullback_support)
            )
            reasons.append(
                "Step7分时强势/跑赢大盘: "
                f"{step7_pass} (日内在VWAP上方={intraday_result.above_vwap_ratio_day:.2f},"
                f"尾盘={intraday_result.above_vwap_ratio_tail:.2f},"
                f"尾盘跑赢={intraday_result.outperformance_last_tail:.2%})"
            )
        elif step1_pass and step2_pass and step3_pass and step4_pass and step5_pass and step6_pass:
            reasons.append("Step7分时强势/跑赢大盘: False (分钟数据缺失，已降级)")

        reason = "; ".join(reasons)
        return StockStepResult(
            symbol=symbol,
            name=name,
            step1_pass=step1_pass,
            step2_pass=step2_pass,
            step3_pass=step3_pass,
            step4_pass=step4_pass,
            step5_pass=step5_pass,
            step6_pass=step6_pass,
            step7_pass=step7_pass,
            pct_change=pct_change,
            volume_ratio=volume_ratio,
            turnover=turnover,
            float_mktcap=float_mktcap,
            volume_shape=volume_shape,
            ma_snapshot=ma_snapshot,
            ma_slope=ma_slope,
            bullish_alignment=bullish_alignment,
            ma_diverging=ma_diverging,
            close_above_ma=close_above_ma,
            intraday=intraday_result,
            reason=reason,
            notes=notes,
        )
