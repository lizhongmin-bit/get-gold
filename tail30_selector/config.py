"""Configuration for tail30 selector strategy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class StrategyConfig:
    """Default configuration aligned with original strategy thresholds."""

    pct_change_range: Tuple[float, float] = (0.03, 0.05)
    volume_ratio_min: float = 1.0
    turnover_range: Tuple[float, float] = (0.05, 0.10)
    float_mktcap_range: Tuple[float, float] = (5e9, 1e10)
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    ma_slope_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    ma_slope_required_ratio: float = 1.0
    intraday_above_vwap_ratio_day: float = 0.80
    intraday_above_vwap_ratio_tail: float = 0.90
    tail_window_start: str = "14:30"
    tail_window_end: str = "15:00"
    new_high_after_1430: bool = True
    pullback_not_break: bool = True
    volume_shape_window: int = 8
    volume_shape_min_seg: int = 3
    volume_shape_k_candidates: Tuple[int, ...] = (2, 3, 4)
    volume_shape_delta_min: float = 0.10
    volume_shape_delta_allow: float = 0.03
    volume_shape_sigma_max: float = 0.35
    volume_shape_cv_max: float = 0.6
    volume_shape_sep_min: float = 1.2
    volume_shape_gain_min: float = 0.15
    volume_shape_recent_ratio_min: float = 1.3
    volume_shape_jump_ratio_high: float = 2.2
    volume_shape_jump_ratio_low: float = 0.45
    volume_shape_jump_max_count: int = 1
    volume_shape_sign_switch_ratio_max: float = 0.5
    relaxed_volume_shape_enabled: bool = True
    relaxed_volume_shape_sigma_max: float = 0.6
    relaxed_volume_shape_cv_max: float = 1.0
    relaxed_volume_shape_sep_min: float = 0.8
    relaxed_volume_shape_gain_min: float = 0.05
    relaxed_volume_shape_recent_ratio_min: float = 1.1
    relaxed_volume_shape_jump_max_count: int = 2
    relaxed_volume_shape_sign_switch_ratio_max: float = 0.8
    volume_shape_skip_if_empty: bool = True
    max_output: int = 20
    retry_times: int = 2
    intraday_minute_freq: str = "1min"


DEFAULT_CONFIG = StrategyConfig()
