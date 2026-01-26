"""Volume shape identification: stepwise rise vs ECG-like volatility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeShapeResult:
    is_stepwise: bool
    k_star: int | None
    breakpoints: list[int]
    mu_list: list[float]
    delta_mu_list: list[float]
    sep_list: list[float]
    gain: float
    segment_std_list: list[float]
    segment_cv_list: list[float]
    jump_extreme_count: int
    sign_switch_count: int
    recent_vs_early_ratio: float
    fail_reasons: list[str]


def analyze_volume_shape(
    volume_series: pd.Series,
    min_seg: int = 3,
    k_candidates: tuple[int, ...] = (2, 3, 4),
    delta_min: float = 0.10,
    delta_allow: float = 0.03,
    sigma_max: float = 0.35,
    cv_max: float = 0.6,
    sep_min: float = 1.2,
    gain_min: float = 0.15,
    recent_ratio_min: float = 1.3,
    jump_ratio_high: float = 2.2,
    jump_ratio_low: float = 0.45,
    max_jump_count: int = 1,
    sign_switch_ratio_max: float = 0.5,
) -> VolumeShapeResult:
    volumes = volume_series.dropna().astype(float)
    fail_reasons: list[str] = []
    if len(volumes) < min_seg * min(k_candidates):
        return _build_result(False, fail_reasons=["insufficient_volume_history"])

    winsorized = _winsorize(volumes, 0.05, 0.95)
    x = np.log1p(winsorized.to_numpy())
    n = len(x)

    best = None
    for k in k_candidates:
        if k * min_seg > n:
            continue
        dp_result = _dp_piecewise_constant(x, k, min_seg)
        if dp_result is None:
            continue
        breakpoints, sse = dp_result
        bic = n * np.log(sse / n) + k * np.log(n)
        if best is None or bic < best["bic"]:
            best = {"k": k, "breakpoints": breakpoints, "sse": sse, "bic": bic}

    if best is None:
        return _build_result(False, fail_reasons=["no_valid_segmentation"])

    k_star = best["k"]
    breakpoints = best["breakpoints"]
    segments = _segments_from_breakpoints(breakpoints, n)
    mu_list = [float(np.mean(x[start:end])) for start, end in segments]
    delta_mu_list = [mu_list[i + 1] - mu_list[i] for i in range(len(mu_list) - 1)]
    segment_std_list = [float(np.std(x[start:end], ddof=0)) for start, end in segments]
    segment_cv_list = [
        float(np.std(volumes.iloc[start:end], ddof=0) / np.mean(volumes.iloc[start:end]))
        if np.mean(volumes.iloc[start:end]) > 0
        else 0.0
        for start, end in segments
    ]
    sep_list = _segment_separation(mu_list, segment_std_list, segments)

    sse_lin = _linear_sse(x)
    gain = float((sse_lin - best["sse"]) / sse_lin) if sse_lin > 0 else 0.0

    is_monotonic = _monotonic_check(mu_list, delta_mu_list, delta_min, delta_allow)
    if not is_monotonic:
        fail_reasons.append("non_monotonic_step")
    if any(std > sigma_max for std in segment_std_list):
        fail_reasons.append("segment_std_too_high")
    if any(cv > cv_max for cv in segment_cv_list):
        fail_reasons.append("segment_cv_too_high")
    if any(sep < sep_min for sep in sep_list):
        fail_reasons.append("segment_separation_too_low")
    if gain < gain_min:
        fail_reasons.append("gain_too_low")

    recent_vs_early_ratio = _recent_vs_early_ratio(volumes)
    if mu_list and mu_list[-1] != max(mu_list):
        fail_reasons.append("last_segment_not_highest")
    if recent_vs_early_ratio < recent_ratio_min:
        fail_reasons.append("recent_vs_early_ratio_too_low")

    jump_extreme_count = _jump_extreme_count(volumes, jump_ratio_high, jump_ratio_low)
    if jump_extreme_count > max_jump_count:
        fail_reasons.append("jump_extreme_too_many")

    sign_switch_count = _sign_switch_count(x)
    max_switches = int(sign_switch_ratio_max * (n - 2)) if n > 2 else 0
    if sign_switch_count > max_switches:
        fail_reasons.append("sign_switch_too_many")

    is_stepwise = len(fail_reasons) == 0
    return VolumeShapeResult(
        is_stepwise=is_stepwise,
        k_star=k_star,
        breakpoints=breakpoints,
        mu_list=mu_list,
        delta_mu_list=delta_mu_list,
        sep_list=sep_list,
        gain=gain,
        segment_std_list=segment_std_list,
        segment_cv_list=segment_cv_list,
        jump_extreme_count=jump_extreme_count,
        sign_switch_count=sign_switch_count,
        recent_vs_early_ratio=recent_vs_early_ratio,
        fail_reasons=fail_reasons,
    )


def _build_result(is_stepwise: bool, fail_reasons: list[str]) -> VolumeShapeResult:
    return VolumeShapeResult(
        is_stepwise=is_stepwise,
        k_star=None,
        breakpoints=[],
        mu_list=[],
        delta_mu_list=[],
        sep_list=[],
        gain=0.0,
        segment_std_list=[],
        segment_cv_list=[],
        jump_extreme_count=0,
        sign_switch_count=0,
        recent_vs_early_ratio=0.0,
        fail_reasons=fail_reasons,
    )


def _winsorize(series: pd.Series, low: float, high: float) -> pd.Series:
    lower = series.quantile(low)
    upper = series.quantile(high)
    return series.clip(lower, upper)


def _dp_piecewise_constant(
    x: np.ndarray, k: int, min_seg: int
) -> tuple[list[int], float] | None:
    n = len(x)
    cost = _segment_costs(x)
    dp = np.full((k + 1, n + 1), np.inf)
    prev = np.full((k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for seg in range(1, k + 1):
        for t in range(seg * min_seg, n + 1):
            for s in range((seg - 1) * min_seg, t - min_seg + 1):
                candidate = dp[seg - 1, s] + cost[s, t]
                if candidate < dp[seg, t]:
                    dp[seg, t] = candidate
                    prev[seg, t] = s

    if not np.isfinite(dp[k, n]):
        return None

    breakpoints = []
    t = n
    for seg in range(k, 0, -1):
        s = prev[seg, t]
        if s < 0:
            return None
        breakpoints.append(s)
        t = s
    breakpoints = sorted(breakpoints)
    return breakpoints + [n], float(dp[k, n])


def _segment_costs(x: np.ndarray) -> np.ndarray:
    n = len(x)
    cost = np.zeros((n + 1, n + 1))
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    cumsum_sq = np.cumsum(np.insert(x * x, 0, 0.0))
    for i in range(n):
        for j in range(i + 1, n + 1):
            length = j - i
            mean = (cumsum[j] - cumsum[i]) / length
            sse = (cumsum_sq[j] - cumsum_sq[i]) - 2 * mean * (cumsum[j] - cumsum[i]) + length * mean * mean
            cost[i, j] = sse
    return cost


def _segments_from_breakpoints(breakpoints: list[int], n: int) -> list[tuple[int, int]]:
    start = 0
    segments = []
    for bp in breakpoints:
        if bp <= start:
            continue
        segments.append((start, bp))
        start = bp
    if start < n:
        segments.append((start, n))
    return segments


def _segment_separation(
    mu_list: list[float],
    std_list: list[float],
    segments: list[tuple[int, int]],
) -> list[float]:
    seps = []
    for i in range(len(mu_list) - 1):
        len_i = segments[i][1] - segments[i][0]
        len_j = segments[i + 1][1] - segments[i + 1][0]
        denom = np.sqrt((std_list[i] ** 2) / len_i + (std_list[i + 1] ** 2) / len_j)
        seps.append(float((mu_list[i + 1] - mu_list[i]) / denom) if denom > 0 else 0.0)
    return seps


def _linear_sse(x: np.ndarray) -> float:
    n = len(x)
    t = np.arange(n)
    coef = np.polyfit(t, x, 1)
    fitted = coef[0] * t + coef[1]
    return float(np.sum((x - fitted) ** 2))


def _monotonic_check(
    mu_list: list[float],
    delta_mu_list: list[float],
    delta_min: float,
    delta_allow: float,
) -> bool:
    if not delta_mu_list:
        return False
    strict_threshold = np.log1p(delta_min)
    allow_threshold = -np.log1p(delta_allow)
    strict_ok = [delta >= strict_threshold for delta in delta_mu_list]
    strict_count = sum(strict_ok)
    if all(strict_ok):
        return True
    if len(delta_mu_list) >= 2 and strict_count >= 2:
        minor_violations = [delta >= allow_threshold for delta in delta_mu_list]
        return sum(minor_violations) >= len(delta_mu_list) - 1
    return False


def _recent_vs_early_ratio(volumes: pd.Series) -> float:
    if len(volumes) < 6:
        return 0.0
    recent = volumes.iloc[-3:].mean()
    early = volumes.iloc[:3].mean()
    return float(recent / early) if early > 0 else 0.0


def _jump_extreme_count(volumes: pd.Series, high: float, low: float) -> int:
    ratios = volumes.iloc[1:].to_numpy() / volumes.iloc[:-1].to_numpy()
    extreme = (ratios >= high) | (ratios <= low)
    return int(np.sum(extreme))


def _sign_switch_count(x: np.ndarray) -> int:
    diffs = np.diff(x)
    signs = np.sign(diffs)
    sign_switches = np.sum(signs[1:] * signs[:-1] < 0)
    return int(sign_switches)
