# select_leaders_intraday_tushare.py
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
from datetime import datetime, timedelta, date

import pandas as pd

import tushare as ts


# ----------------------------
# Config (可调参)
# ----------------------------
DEFAULT_TOPK = 5
DEFAULT_PREFILTER_N = 500

# 初筛（盘中按“当前涨幅”）
MIN_PCT_CHG = 2.0

# 指标阈值
STRONG_INTRADAY_TH = 0.25      # (high_today-last)/(high_today-low_today)
NEAR_HIGH_RATIO = 0.98         # last >= 10日最高 * 0.98
VOL_COMPLETION_TH = 1.2        # 14:30量能兑现度阈值（相对近5日均量*时间占比）
DIST_TO_LIMITUP_TH = 0.03      # 离涨停<=3%

# 历史窗口
DAILY_LOOKBACK_DAYS = 120      # 拉取最近多少交易日（日线）
LOOKBACK_K = 80                # 计算MA等用
NEW_STOCK_MIN_BARS = 60        # 新股过滤（近似：历史交易日数不足）

# 分钟频率（14:30建议 1MIN）
INTRADAY_FREQ = "1MIN"         # rt_min/rt_min_daily 需要大写：1MIN/5MIN/...

# 缓存
CACHE_DIR = "data_cache_ts"
CACHE_MAX_AGE_HOURS = 12

# 速率控制（防止触发频控）
SLEEP_EVERY = 20               # 每处理多少只睡一下
SLEEP_SECONDS = 0.7


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_cache_fresh(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime) <= timedelta(hours=max_age_hours)


def safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def code_to_ts(code6: str) -> str:
    """
    6位代码 -> ts_code
    规则：6开头通常SH，其余SZ（粗规则，对A股够用）
    """
    code6 = str(code6).strip()
    m = re.search(r"\d{6}", code6)
    if not m:
        return code6
    c = m.group(0)
    return f"{c}.SH" if c.startswith("6") else f"{c}.SZ"


def guess_limitup_ratio(ts_code: str) -> float:
    # 300/688 一般20%，其余10%（ST已过滤）
    if ts_code.startswith(("300", "688")):
        return 0.20
    return 0.10


def guess_limitup_threshold_pct(ts_code: str) -> float:
    return 19.8 if ts_code.startswith(("300", "688")) else 9.8


def trading_minutes_passed(now_dt: datetime) -> int:
    """
    A股交易时段：
      09:30-11:30 (120min)
      13:00-15:00 (120min)
    返回截至 now_dt 已经过的有效交易分钟数（0~240）
    """
    t = now_dt.time()
    d = now_dt.date()

    t0930 = datetime.combine(d, datetime.strptime("09:30", "%H:%M").time())
    t1130 = datetime.combine(d, datetime.strptime("11:30", "%H:%M").time())
    t1300 = datetime.combine(d, datetime.strptime("13:00", "%H:%M").time())
    t1500 = datetime.combine(d, datetime.strptime("15:00", "%H:%M").time())

    if now_dt <= t0930:
        return 0
    if now_dt <= t1130:
        return int((now_dt - t0930).total_seconds() // 60)
    if now_dt <= t1300:
        return 120
    if now_dt <= t1500:
        return 120 + int((now_dt - t1300).total_seconds() // 60)
    return 240


def time_ratio(now_dt: datetime) -> float:
    mins = trading_minutes_passed(now_dt)
    return max(1e-6, mins / 240.0)


# ----------------------------
# Tushare fetchers (with cache)
# ----------------------------
def pro_api(token: str):
    ts.set_token(token)
    return ts.pro_api()


def get_recent_trade_dates(pro, end_date_yyyymmdd: str, n: int) -> list[str]:
    """
    取最近n个开市日（包含end_date当日如果开市）
    """
    start_dt = datetime.strptime(end_date_yyyymmdd, "%Y%m%d") - timedelta(days=n * 3)
    start = start_dt.strftime("%Y%m%d")

    cal = pro.trade_cal(exchange="", start_date=start, end_date=end_date_yyyymmdd, is_open="1",
                        fields="cal_date,is_open")
    if cal is None or cal.empty:
        raise RuntimeError("trade_cal 拉取失败")
    dates = cal["cal_date"].tolist()
    dates = dates[-n:]
    return dates


def load_daily_by_trade_date(pro, trade_date: str) -> pd.DataFrame:
    """
    pro.daily(trade_date=YYYYMMDD) 一次取全市场日线（稳定、少请求）。
    """
    ensure_dir(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, f"daily_{trade_date}.parquet")

    if is_cache_fresh(cache_path, CACHE_MAX_AGE_HOURS):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    # fields尽量取全一点，便于后续扩展
    df = pro.daily(trade_date=trade_date, fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg")
    if df is None or df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ["open", "high", "low", "close", "vol", "amount", "pct_chg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return df


def load_intraday_minutes_rt_min_daily(pro, ts_code: str, day: date, freq: str) -> pd.DataFrame | None:
    """
    优先用 rt_min_daily（单票当日开盘以来分钟全量）:contentReference[oaicite:2]{index=2}
    """
    ensure_dir(CACHE_DIR)
    day_str = day.strftime("%Y%m%d")
    cache_path = os.path.join(CACHE_DIR, f"min_{ts_code.replace('.', '')}_{day_str}_{freq}.parquet")

    if is_cache_fresh(cache_path, CACHE_MAX_AGE_HOURS):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    # rt_min_daily：文档说存在该接口（当日分钟全量，单票）:contentReference[oaicite:3]{index=3}
    # SDK里通常是 pro.rt_min_daily(ts_code=..., freq=...)
    for attempt in range(3):
        try:
            df = pro.rt_min_daily(ts_code=ts_code, freq=freq)
            if df is None or df.empty:
                return None
            # 统一字段
            # 文档输出：ts_code,time,open,close,high,low,vol,amount :contentReference[oaicite:4]{index=4}
            df = df.rename(columns={"time": "time"})
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            for c in ["open", "close", "high", "low", "vol", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass
            return df
        except Exception:
            time.sleep(0.8 + attempt * 0.7)

    return None


def load_intraday_minutes_fallback_rt_min(pro, ts_code: str, day: date, freq: str) -> pd.DataFrame | None:
    """
    降级用 rt_min（可多票），这里我们单票调用以兼容
    """
    for attempt in range(3):
        try:
            df = pro.rt_min(ts_code=ts_code, freq=freq)
            if df is None or df.empty:
                return None
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            for c in ["open", "close", "high", "low", "vol", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            # 只保留当天（有时返回就只有当天）
            df = df[df["time"].dt.date == day].reset_index(drop=True)
            return df if not df.empty else None
        except Exception:
            time.sleep(0.8 + attempt * 0.7)
    return None


def load_intraday_minutes(pro, ts_code: str, now_dt: datetime, freq: str) -> pd.DataFrame | None:
    """
    先 rt_min_daily，失败再 rt_min
    """
    d = now_dt.date()
    df = load_intraday_minutes_rt_min_daily(pro, ts_code, d, freq)
    if df is not None and not df.empty:
        return df
    return load_intraday_minutes_fallback_rt_min(pro, ts_code, d, freq)


# ----------------------------
# Strategy computation
# ----------------------------
def build_daily_panel(pro, end_date_yyyymmdd: str) -> tuple[pd.DataFrame, str, str]:
    """
    拉取最近N个交易日全市场日线，形成一个panel用于：
    - 昨日收盘
    - MA5/MA10（截至昨日）
    - 近10日最高
    - 涨停基因（60日）
    返回：daily_all, prev_trade_date, last_trade_date
    """
    trade_dates = get_recent_trade_dates(pro, end_date_yyyymmdd, DAILY_LOOKBACK_DAYS)
    if len(trade_dates) < 20:
        raise RuntimeError("交易日数量不足，无法计算指标")

    last_trade_date = trade_dates[-1]     # 今天（若开市）
    prev_trade_date = trade_dates[-2]     # 昨日

    frames = []
    for td in trade_dates:
        df = load_daily_by_trade_date(pro, td)
        if df is not None and not df.empty:
            frames.append(df)

    daily_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if daily_all.empty:
        raise RuntimeError("日线数据拉取失败：daily_all为空")

    # trade_date已转datetime
    daily_all = daily_all.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return daily_all, prev_trade_date, last_trade_date


def compute_daily_indicators_upto_yday(daily_all: pd.DataFrame, now_dt: datetime) -> pd.DataFrame:
    """
    对每只股票计算截至昨日的：close_y, ma5_y, ma10_y, ma5_y_3ago, high10_y, avg_vol5_y, limitup_gene_60
    """
    # 截至昨日（去掉今天的日线，如果存在）
    daily_upto_y = daily_all[daily_all["trade_date"].dt.date < now_dt.date()].copy()
    if daily_upto_y.empty:
        return pd.DataFrame()

    def per_code(g: pd.DataFrame) -> dict:
        g = g.tail(LOOKBACK_K).copy()
        if len(g) < NEW_STOCK_MIN_BARS:
            return {}

        g["ma5"] = g["close"].rolling(5).mean()
        g["ma10"] = g["close"].rolling(10).mean()

        y = g.iloc[-1]
        close_y = safe_float(y["close"])
        ma5_y = safe_float(y["ma5"])
        ma10_y = safe_float(y["ma10"])

        ma5_y_3ago = None
        if g["ma5"].notna().sum() >= 8:
            ma5_y_3ago = safe_float(g["ma5"].iloc[-4])

        high10_y = safe_float(g["high"].tail(10).max())
        avg_vol5_y = safe_float(g["vol"].tail(5).mean())

        ts_code = y["ts_code"]
        th = guess_limitup_threshold_pct(ts_code)
        limitup_gene_60 = bool((g["pct_chg"].fillna(-999).tail(60) >= th).any())

        return {
            "ts_code": ts_code,
            "close_y": close_y,
            "ma5_y": ma5_y,
            "ma10_y": ma10_y,
            "ma5_y_3ago": ma5_y_3ago,
            "high10_y": high10_y,
            "avg_vol5_y": avg_vol5_y,
            "limitup_gene_60": limitup_gene_60,
            "bars": len(g),
        }

    rows = []
    for ts_code, g in daily_upto_y.groupby("ts_code", sort=False):
        r = per_code(g)
        if r:
            rows.append(r)

    ind = pd.DataFrame(rows)
    return ind


def aggregate_intraday_features(min_df: pd.DataFrame, now_dt: datetime) -> dict | None:
    """
    聚合当日分钟线（到当前时刻为止）：
    last_price/high_today/low_today/vol_sofar/amount_sofar
    """
    if min_df is None or min_df.empty:
        return None

    # 只保留 <= now_dt 的分钟
    m = min_df[min_df["time"] <= now_dt].copy()
    if m.empty:
        # 如果分钟线时间没有精确到秒，可能最后一根略晚，这里放宽取最后一根
        m = min_df.copy()

    high_today = safe_float(m["high"].max())
    low_today = safe_float(m["low"].min())
    last_price = safe_float(m["close"].iloc[-1])

    vol_sofar = safe_float(m["vol"].sum())
    amount_sofar = safe_float(m["amount"].sum())

    return {
        "high_today": high_today,
        "low_today": low_today,
        "last_price": last_price,
        "vol_sofar": vol_sofar,
        "amount_sofar": amount_sofar,
    }


def score_one(ts_code: str, name: str, intraday: dict, daily_ind_row: pd.Series, now_dt: datetime) -> dict | None:
    last_price = intraday.get("last_price")
    high_today = intraday.get("high_today")
    low_today = intraday.get("low_today")
    vol_sofar = intraday.get("vol_sofar")
    amount_sofar = intraday.get("amount_sofar")

    close_y = safe_float(daily_ind_row.get("close_y"))
    ma5_y = safe_float(daily_ind_row.get("ma5_y"))
    ma10_y = safe_float(daily_ind_row.get("ma10_y"))
    ma5_y_3ago = safe_float(daily_ind_row.get("ma5_y_3ago"))
    high10_y = safe_float(daily_ind_row.get("high10_y"))
    avg_vol5_y = safe_float(daily_ind_row.get("avg_vol5_y"))
    limitup_gene_60 = bool(daily_ind_row.get("limitup_gene_60"))

    if last_price is None or close_y is None or close_y <= 0:
        return None

    pct_chg = (last_price - close_y) / close_y * 100.0

    # 临收盘强势
    hl = None
    strong_ratio = None
    strong_intraday_flag = False
    if high_today is not None and low_today is not None:
        hl = high_today - low_today
        if hl and hl > 0:
            strong_ratio = (high_today - last_price) / hl
            strong_intraday_flag = (strong_ratio <= STRONG_INTRADAY_TH)

    # 趋势（截至昨日）
    trend_flag = False
    if close_y is not None and ma5_y is not None and ma10_y is not None:
        if close_y > ma5_y and ma5_y > ma10_y:
            if ma5_y_3ago is not None:
                trend_flag = (ma5_y > ma5_y_3ago)

    # 逼近10日高（截至昨日）
    near_high_flag = False
    if high10_y is not None and high10_y > 0:
        near_high_flag = (last_price >= high10_y * NEAR_HIGH_RATIO)

    # 量能兑现度（相对近5日均量*时间占比）
    tr = time_ratio(now_dt)
    vol_completion = None
    vol_completion_flag = False
    if vol_sofar is not None and avg_vol5_y is not None and avg_vol5_y > 0:
        expected = avg_vol5_y * tr
        if expected > 0:
            vol_completion = vol_sofar / expected
            vol_completion_flag = (vol_completion >= VOL_COMPLETION_TH)

    # 距离涨停（基于昨收估算）
    limit_ratio = guess_limitup_ratio(ts_code)
    limitup_price_est = close_y * (1.0 + limit_ratio)
    dist_to_limit = (limitup_price_est - last_price) / last_price if last_price > 0 else None
    dist_flag = (dist_to_limit is not None and dist_to_limit <= DIST_TO_LIMITUP_TH)

    # 一字板风险（简化）
    is_one_word = False
    if high_today is not None and low_today is not None and high_today == low_today:
        if dist_to_limit is not None and dist_to_limit <= 0.003:
            is_one_word = True

    strict_hit = all([
        pct_chg is not None and pct_chg >= MIN_PCT_CHG,
        strong_intraday_flag,
        trend_flag,
        (vol_completion_flag if vol_completion is not None else False),
        near_high_flag,
        limitup_gene_60,
        dist_flag,
        (not is_one_word),
    ])

    # 评分（可按你偏好改权重）
    score = 0
    score += 20 if strong_intraday_flag else 0
    score += 20 if trend_flag else 0
    score += 25 if (vol_completion_flag if vol_completion is not None else False) else 0
    score += 20 if near_high_flag else 0
    score += 15 if limitup_gene_60 else 0

    # 距离涨停越近越加分（最多20）
    if dist_to_limit is not None:
        if dist_to_limit <= 0.03:
            score += 20
        elif dist_to_limit >= 0.08:
            score += 0
        else:
            score += int(round(20 * (0.08 - dist_to_limit) / (0.08 - 0.03)))

    if is_one_word:
        score -= 20

    return {
        "ts_code": ts_code,
        "name": name,
        "score": int(score),
        "strict_hit": bool(strict_hit),
        "pct_chg": float(pct_chg),
        "last_price": last_price,
        "high_today": high_today,
        "low_today": low_today,
        "amount_sofar": amount_sofar,
        "vol_sofar": vol_sofar,
        "time_ratio": tr,
        "vol_completion": vol_completion,
        "strong_intraday_flag": strong_intraday_flag,
        "strong_ratio": strong_ratio,
        "trend_flag": trend_flag,
        "near_high_flag": near_high_flag,
        "limitup_gene_60": limitup_gene_60,
        "close_y": close_y,
        "ma5_y": ma5_y,
        "ma10_y": ma10_y,
        "high10_y": high10_y,
        "avg_vol5_y": avg_vol5_y,
        "limitup_price_est": limitup_price_est,
        "distance_to_limitup": dist_to_limit,
        "is_one_word": is_one_word,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.getenv("TUSHARE_TOKEN", ""),
                        help="Tushare Pro token；也可用环境变量 TUSHARE_TOKEN")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--prefilter_n", type=int, default=DEFAULT_PREFILTER_N)
    parser.add_argument("--date", type=str, default="",
                        help="回放模式：YYYYMMDD；不填则用当前系统时间（盘中14:30运行）")
    parser.add_argument("--freq", type=str, default=INTRADAY_FREQ, help="分钟频率：1MIN/5MIN/15MIN/...")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("请提供token：--token xxxxx 或设置环境变量 TUSHARE_TOKEN")

    pro = pro_api(args.token)

    # now_dt：实时盘中；回放则固定到 14:30
    now_dt = datetime.now()
    if args.date:
        d = datetime.strptime(args.date, "%Y%m%d").date()
        now_dt = datetime.combine(d, datetime.strptime("14:30", "%H:%M").time())

    end_date = now_dt.strftime("%Y%m%d")

    # 1) 拉最近交易日日线panel（全市场），算截至昨日指标
    daily_all, prev_trade_date, last_trade_date = build_daily_panel(pro, end_date)
    daily_ind = compute_daily_indicators_upto_yday(daily_all, now_dt)
    if daily_ind.empty:
        raise RuntimeError("无法得到截至昨日的指标（daily_ind为空）")

    # 2) 股票池：用stock_basic取名称、过滤ST（基本面表更稳定）
    stock_basic = pro.stock_basic(exchange="", list_status="L",
                                  fields="ts_code,name")
    if stock_basic is None or stock_basic.empty:
        raise RuntimeError("stock_basic 拉取失败")

    # 简单过滤ST（名称包含ST）
    stock_basic = stock_basic[~stock_basic["name"].astype(str).str.contains(r"ST|\*ST|退", regex=True)].copy()

    # 3) 构造盘中“初筛列表”：先不拉分钟线全市场（太重），而是：
    #    用 daily_ind（有昨收）+ 分钟线（只拉一遍/每票）来算当前涨幅 -> 取涨幅前N
    #    为了性能：先取一个中等规模的候选（比如 2000）再拉分钟线算涨幅
    #    这里用 daily_ind 作为可交易池（已过滤新股近似）
    pool = daily_ind.merge(stock_basic, on="ts_code", how="inner")

    # 先按昨日成交额/成交量（可用daily_all的prev_trade_date）做个轻量前置过滤，减少分钟请求
    prev_day_df = daily_all[daily_all["trade_date"].dt.strftime("%Y%m%d") == prev_trade_date].copy()
    prev_day_df = prev_day_df[["ts_code", "amount", "vol"]].rename(columns={"amount": "amount_y", "vol": "vol_y"})
    pool = pool.merge(prev_day_df, on="ts_code", how="left")

    # 取昨日日成交额靠前的一批（比如 2000）减少分钟请求
    pool = pool.sort_values("amount_y", ascending=False).head(max(args.prefilter_n * 4, 2000)).reset_index(drop=True)

    # 4) 对 pool 拉分钟线，计算盘中指标与评分
    results = []
    for i, row in pool.iterrows():
        ts_code = row["ts_code"]
        name = row["name"]

        min_df = load_intraday_minutes(pro, ts_code, now_dt, args.freq)
        if min_df is None or min_df.empty:
            continue

        intraday = aggregate_intraday_features(min_df, now_dt)
        if intraday is None:
            continue

        r = score_one(ts_code, name, intraday, row, now_dt)
        if r is None:
            continue

        # 初筛：盘中涨幅>MIN_PCT_CHG
        if r["pct_chg"] is None or r["pct_chg"] < MIN_PCT_CHG:
            continue

        results.append(r)

        # 速率控制
        if (i + 1) % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    if not results:
        print("没有得到候选结果。可能是：分钟权限不足/频控/过滤过严。")
        print("建议：降低 MIN_PCT_CHG 或减少 prefilter_n，或检查 rt_min_daily/rt_min 权限。")
        return

    out = pd.DataFrame(results)

    # 盘中涨幅优先筛前prefilter_n，再做Strict/Score TopK
    out = out.sort_values(["pct_chg", "score"], ascending=[False, False]).head(args.prefilter_n)
    out = out.sort_values(["strict_hit", "score", "pct_chg"], ascending=[False, False, False]).reset_index(drop=True)

    strict_df = out[out["strict_hit"]].copy()
    pick = strict_df.head(args.topk) if len(strict_df) >= args.topk else out.head(args.topk)

    ensure_dir("output")
    day_str = now_dt.strftime("%Y%m%d")
    out_path = os.path.join("output", f"leaders_ts_intraday_all_{day_str}.csv")
    pick_path = os.path.join("output", f"leaders_ts_intraday_top{args.topk}_{day_str}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    pick.to_csv(pick_path, index=False, encoding="utf-8-sig")

    print(f"\n=== Tushare盘中候选（{now_dt.strftime('%Y-%m-%d %H:%M')}，Top{args.topk}） ===")
    cols = [
        "ts_code", "name", "score", "strict_hit", "pct_chg", "last_price",
        "amount_sofar", "vol_completion", "strong_intraday_flag",
        "trend_flag", "near_high_flag", "limitup_gene_60",
        "distance_to_limitup", "is_one_word"
    ]
    print(pick[cols].to_string(index=False))

    print(f"\n已保存：\n- 全量候选: {out_path}\n- Top{args.topk}: {pick_path}\n")


if __name__ == "__main__":
    main()
