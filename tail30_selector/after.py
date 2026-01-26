#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A股盘后选股（未来3日更可能延续偏强）：
- 数据：Tushare Pro（daily / daily_basic / stock_basic / trade_cal；可选 anns_d）
- 指标：MA(5/10/20)、MACD、RSI、ATR、ADX、OBV、BOLL、量能/成交额、换手率/量比
- 逻辑：趋势过滤 + 量价承接 + 动量确认 + 风控阈值 +（可选）公告标题LLM排雷（豆包OpenAI兼容）

用法示例：
1) 默认用最近交易日：
   python after_close_stock_picker.py

2) 指定交易日：
   python after_close_stock_picker.py --trade_date 20260123

3) 启用公告LLM排雷（需要设置豆包环境变量，且尽量开通anns_d权限）：
   python after_close_stock_picker.py --use_llm
"""

import os
import json
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

import tushare as ts
import tushare.pro.client as ts_client



# =========================
# 全局配置（请按需修改）
# =========================

# -------- Tushare 配置 --------
TUSHARE_TOKEN = "1d75857c8ae77369e6314c076d8cc0cdfefde3f3fd0431e179ed916e"

# 如果你通过代理 / 私有网关访问 Tushare
# 例：
# "http://127.0.0.1:8080/api/tushare"
# "http://proxy.company.com:9000"
TUSHARE_API_URL = "http://api.tushare.pro"

# 是否启用 HTTP(S) 代理（如果是反向代理，一般不需要）
USE_HTTP_PROXY = False
HTTP_PROXY = "http://user:pass@proxy_host:port"
HTTPS_PROXY = "http://user:pass@proxy_host:port"


# -------- 豆包（OpenAI 兼容）配置 --------
USE_LLM_RISK_CHECK = False  # 是否启用公告/事件排雷

DOUBAO_BASE_URL = "https://xxx/v1"
DOUBAO_API_KEY = "你的豆包API Key"
DOUBAO_MODEL = "你的模型名"


# -------- 选股参数（可按风格微调）--------
LOOKBACK_DAYS = 70          # 回看交易日数（>=60）
TOP_N = 50                  # 输出股票数量
MIN_AMOUNT_K = 80000        # 最小成交额（千元，80000=8000万）
TURNOVER_MIN = 2.0
TURNOVER_MAX = 20.0
RSI_MIN = 50.0
RSI_MAX = 82.0
ADX_MIN = 20.0
MAX_PCT_CHG = 9.2           # 避免接近涨停的过热
ANN_LOOKBACK_DAYS = 10      # 公告回看天数



# -----------------------------
# 技术指标（纯pandas/numpy实现）
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = dif - dea
    return dif, dea, hist

def rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(high, low, close, period=14):
    # 方向运动
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_val

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val, plus_di, minus_di

def obv(close: pd.Series, vol: pd.Series):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * vol).cumsum()

def bollinger(close: pd.Series, period=20, n_std=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


# -----------------------------
# 豆包（OpenAI兼容）LLM排雷
# -----------------------------
def build_openai_client():
    """
    使用脚本顶部常量配置豆包（OpenAI 兼容）
    """
    if not USE_LLM_RISK_CHECK:
        raise RuntimeError("USE_LLM_RISK_CHECK=False，未启用 LLM")

    if not (DOUBAO_BASE_URL and DOUBAO_API_KEY and DOUBAO_MODEL):
        raise RuntimeError("请在脚本顶部配置完整的豆包参数")

    client = OpenAI(
        base_url=DOUBAO_BASE_URL,
        api_key=DOUBAO_API_KEY
    )
    return client, DOUBAO_MODEL


def llm_risk_check(client, model: str, ts_code: str, name: str, ann_titles: list[str]) -> dict:
    """
    输入：公告标题列表（近7-10天）
    输出：{"pass": bool, "risk_level": "low|mid|high", "reasons": [..]}
    """
    titles = ann_titles[:30]
    prompt = f"""
你是A股短线盘后选股的风控助手。请仅根据“公告标题”判断是否存在明显利空/高不确定性事件。
股票：{name}（{ts_code}）
公告标题（可能不全）：{json.dumps(titles, ensure_ascii=False)}

请识别以下风险（只要疑似就算风险）：
- 立案调查/监管处罚/重大诉讼仲裁/退市风险提示/ST相关
- 业绩预告大幅下修、亏损扩大、商誉减值等
- 大股东/高管减持、质押风险、强制平仓提示
- 重大资产重组失败/终止、重大合同违约、债务危机
- 其他可能导致未来1-3个交易日大幅波动或下跌的事件

输出严格JSON（不要额外文字）：
{{"pass": true/false, "risk_level": "low|mid|high", "reasons": ["..."]}}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你只输出严格JSON，不要多余文字。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()
    # 兜底解析
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM输出不是dict")
        data.setdefault("pass", True)
        data.setdefault("risk_level", "low")
        data.setdefault("reasons", [])
        return data
    except Exception:
        # 解析失败时，保守处理：不一票否决，但标记需要人工复核
        return {"pass": True, "risk_level": "mid", "reasons": ["LLM输出解析失败，建议人工复核"]}


# -----------------------------
# Tushare 拉取与选股逻辑
# -----------------------------
def get_latest_trade_date(pro, today: datetime) -> str:
    # 向前取近30天交易日历，找最后一个 is_open=1 且 <= today
    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(days=45)).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)
    cal = cal[cal["is_open"] == 1].sort_values("cal_date")
    if cal.empty:
        raise RuntimeError("trade_cal 未找到近期开市日期")
    return cal["cal_date"].iloc[-1]

def get_last_n_trade_dates(pro, end_trade_date: str, n: int) -> list[str]:
    end_dt = datetime.strptime(end_trade_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=200)  # 给足缓冲，避免节假日不够
    cal = pro.trade_cal(exchange="SSE",
                        start_date=start_dt.strftime("%Y%m%d"),
                        end_date=end_trade_date)
    dates = cal[cal["is_open"] == 1].sort_values("cal_date")["cal_date"].tolist()
    if len(dates) < n:
        raise RuntimeError(f"交易日数量不足：只有 {len(dates)} 天，需求 {n} 天")
    return dates[-n:]

def fetch_universe(pro) -> pd.DataFrame:
    # 只取正常上市
    df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,market,industry")
    # 过滤ST（用名称粗过滤，正式可再叠加其他规则）
    df["is_st_like"] = df["name"].astype(str).str.contains("ST|\\*ST|退", regex=True)
    return df

def fetch_history_daily(pro, trade_dates: list[str]) -> pd.DataFrame:
    frames = []
    for d in tqdm(trade_dates, desc="拉取daily历史(日线)"):
        df = pro.daily(trade_date=d)
        if df is None or df.empty:
            continue
        frames.append(df)
        time.sleep(0.02)  # 温和限速
    if not frames:
        raise RuntimeError("daily 历史拉取失败：无数据")
    out = pd.concat(frames, ignore_index=True)
    # 统一排序
    out = out.sort_values(["ts_code", "trade_date"])
    return out

def fetch_daily_basic(pro, trade_date: str) -> pd.DataFrame:
    # 单日全市场：最多6000条，一次足够
    df = pro.daily_basic(trade_date=trade_date,
                         fields="ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe_ttm,pb,total_mv,circ_mv")
    if df is None or df.empty:
        raise RuntimeError("daily_basic 拉取失败")
    return df

def try_fetch_anns_titles(pro, ts_codes: list[str], start_date: str, end_date: str) -> dict:
    """
    尝试用 anns_d 拉取公告标题（独立权限；可能会报错）
    返回：{ts_code: [title1, title2, ...]}
    """
    titles = {c: [] for c in ts_codes}
    try:
        anns = pro.anns_d(start_date=start_date, end_date=end_date,
                          fields="ann_date,ts_code,name,title")
        if anns is None or anns.empty:
            return titles
        anns = anns[anns["ts_code"].isin(ts_codes)].sort_values(["ts_code", "ann_date"], ascending=[True, False])
        for c, g in anns.groupby("ts_code"):
            titles[c] = g["title"].astype(str).head(30).tolist()
        return titles
    except Exception:
        # 没权限或接口异常，直接返回空
        return titles

def compute_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    输入：多股票历史日线（至少60个交易日）
    输出：每行=某股票某日，带各种特征
    """
    df = hist.copy()
    # 成交量单位：daily vol 是“手”，amount 是“千元”
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def per_stock(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("trade_date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]
        vol = g["vol"]

        # MA
        g["ma5"] = close.rolling(5).mean()
        g["ma10"] = close.rolling(10).mean()
        g["ma20"] = close.rolling(20).mean()

        # MACD
        dif, dea, hist_ = macd(close)
        g["macd_dif"] = dif
        g["macd_dea"] = dea
        g["macd_hist"] = hist_

        # RSI
        g["rsi14"] = rsi(close, 14)

        # ATR / ADX
        g["atr14"] = atr(high, low, close, 14)
        adx_val, plus_di, minus_di = adx(high, low, close, 14)
        g["adx14"] = adx_val
        g["pdi14"] = plus_di
        g["mdi14"] = minus_di

        # OBV
        g["obv"] = obv(close, vol)

        # BOLL
        mid, upper, lower = bollinger(close, 20, 2.0)
        g["boll_mid"] = mid
        g["boll_up"] = upper
        g["boll_low"] = lower

        # 量均线
        g["vol_ma5"] = vol.rolling(5).mean()

        return g

    df = df.groupby("ts_code", group_keys=False).apply(per_stock)
    return df

def pick_stocks(feat_df: pd.DataFrame, daily_basic_df: pd.DataFrame, universe: pd.DataFrame,
                trade_date: str,
                min_amount_k=80000,   # 千元：8e4=8,000万成交额（按你风格可调）
                turnover_min=2.0,
                turnover_max=20.0,
                rsi_min=50.0,
                rsi_max=82.0,
                adx_min=20.0,
                max_pct_chg=9.2
                ) -> pd.DataFrame:
    """
    输出候选池：带打分与原因
    """
    # 取最后一天
    last = feat_df[feat_df["trade_date"] == trade_date].copy()
    last = last.merge(daily_basic_df, on=["ts_code", "trade_date"], how="left")
    last = last.merge(universe[["ts_code", "name", "industry", "market", "is_st_like"]],
                      on="ts_code", how="left")

    # 近几天辅助（比如macd柱体递增）
    def last_n(g, n=3, col="macd_hist"):
        return g.sort_values("trade_date")[col].tail(n).tolist()

    hist3 = feat_df[feat_df["trade_date"].isin(
        sorted(feat_df["trade_date"].unique())[-3:]
    )].groupby("ts_code").apply(lambda g: last_n(g, 3, "macd_hist")).to_dict()

    last["macd_hist_3"] = last["ts_code"].map(hist3)

    # 条件：趋势（站上均线、多头）
    cond_trend = (
        (last["close"] > last["ma5"]) &
        (last["ma5"] > last["ma10"]) &
        (last["ma10"] > last["ma20"]) &
        (last["ma5"].notna()) & (last["ma20"].notna())
    )

    # 条件：量价承接（量>5日均量；成交额足够；避免极端）
    cond_volume = (
        (last["vol"] > last["vol_ma5"]) &
        (last["amount"] >= float(min_amount_k))
    )

    # 条件：动量（MACD多头、柱体不走弱；RSI强势区；BOLL偏强）
    def macd_hist_ok(x):
        if not isinstance(x, list) or len(x) < 3:
            return False
        return (x[-1] > 0) and (x[-1] >= x[-2] - 1e-9)  # 最近不明显缩短
    cond_macd = (last["macd_dif"] > last["macd_dea"]) & last["macd_hist_3"].apply(macd_hist_ok)

    cond_rsi = (last["rsi14"] >= rsi_min) & (last["rsi14"] <= rsi_max)

    # BOLL：收盘在中轨之上，且不“远离上轨太多”（避免过弱）
    cond_boll = (last["close"] > last["boll_mid"]) & (last["boll_up"].notna()) & ((last["boll_up"] - last["close"]) / last["boll_up"] <= 0.03)

    # 条件：趋势强度（ADX）
    cond_adx = (last["adx14"] >= adx_min)

    # 风控：不过热（避免一字/接近涨停的超热）
    cond_hot = (last["pct_chg"] <= max_pct_chg) & (last["pct_chg"] >= -2.0)

    # 换手率过滤（用 daily_basic 的 turnover_rate）
    cond_turnover = (last["turnover_rate"] >= turnover_min) & (last["turnover_rate"] <= turnover_max)

    # 排除ST类（仅用名称粗过滤）
    cond_not_st = (last["is_st_like"] == False)

    picked = last[cond_trend & cond_volume & cond_macd & cond_rsi & cond_boll & cond_adx & cond_hot & cond_turnover & cond_not_st].copy()

    # 简单打分（可按你的风格改权重）
    # 趋势：均线间距；动量：macd_hist；量：相对量；强度：adx；风险：rsi过热扣分
    picked["vol_rel"] = picked["vol"] / picked["vol_ma5"]
    picked["ma_spread"] = (picked["ma5"] - picked["ma20"]) / picked["ma20"]
    picked["score"] = (
        30 * picked["ma_spread"].clip(-0.05, 0.20) +
        20 * picked["vol_rel"].clip(0.8, 3.0) +
        20 * picked["adx14"].clip(0, 60) / 60 +
        20 * picked["macd_hist"].clip(-0.2, 0.6) +
        10 * (1 - ((picked["rsi14"] - 50).clip(0, 40) / 40))
    )

    cols = [
        "ts_code", "name", "industry", "market", "trade_date",
        "close", "pct_chg", "amount", "vol_rel",
        "turnover_rate", "volume_ratio", "pe_ttm", "pb", "circ_mv",
        "ma5", "ma10", "ma20",
        "macd_dif", "macd_dea", "macd_hist", "rsi14", "adx14",
        "boll_mid", "boll_up",
        "score"
    ]
    cols = [c for c in cols if c in picked.columns]
    picked = picked.sort_values("score", ascending=False)[cols]
    return picked

def main():
    # ========== 初始化 Tushare ==========
    if not TUSHARE_TOKEN:
        raise RuntimeError("请在脚本顶部配置 TUSHARE_TOKEN")

    if TUSHARE_API_URL:
        print(f"[INFO] 使用自定义 Tushare API URL: {TUSHARE_API_URL}")
        ts_client.DataApi._DataApi__http_url = TUSHARE_API_URL
    pro = ts.pro_api(TUSHARE_TOKEN)

    # 连通性自检（强烈建议保留）
    try:
        _ = pro.trade_cal(exchange="SSE", limit=1)
    except Exception as e:
        raise RuntimeError(f"Tushare API 连通失败，请检查代理或 API URL：{e}")




    # 1) 交易日
    today = datetime.now()
    trade_date = get_latest_trade_date(pro, today)
    trade_dates = get_last_n_trade_dates(pro, trade_date, LOOKBACK_DAYS)

    print(f"[INFO] trade_date={trade_date}, lookback={LOOKBACK_DAYS} trading days")

    # 2) 股票池
    universe = fetch_universe(pro)

    # 3) 历史行情（多日循环，单日全市场 <=6000条）
    hist = fetch_history_daily(pro, trade_dates)

    # 4) 当日 daily_basic（换手/量比/估值）
    db = fetch_daily_basic(pro, trade_date)

    # 5) 计算特征
    feat = compute_features(hist)

    # 6) 选股
    picked = pick_stocks(
        feat, db, universe, trade_date,
        min_amount_k=MIN_AMOUNT_K,
        turnover_min=TURNOVER_MIN,
        turnover_max=TURNOVER_MAX,
        rsi_min=RSI_MIN,
        rsi_max=RSI_MAX,
        adx_min=ADX_MIN,
        max_pct_chg=MAX_PCT_CHG
    )

    if picked.empty:
        print("[WARN] 未筛到满足条件的股票。你可以适当放宽阈值（如成交额/ADX/RSI等）。")
    else:
        picked = picked.head(TOP_N).copy()

    # 7) 可选：公告LLM排雷
    if USE_LLM_RISK_CHECK and not picked.empty:
        try:
            llm_client, model = build_openai_client()
            end_dt = datetime.strptime(trade_date, "%Y%m%d")
            start_dt = (end_dt - timedelta(days=ANN_LOOKBACK_DAYS)).strftime("%Y%m%d")
            ann_map = try_fetch_anns_titles(pro, picked["ts_code"].tolist(), start_dt, trade_date)

            passes = []
            risk_levels = []
            reasons = []

            print("[INFO] LLM公告排雷中（若anns_d无权限将自动跳过公告内容）...")
            for _, row in tqdm(picked.iterrows(), total=len(picked), desc="LLM排雷"):
                code = row["ts_code"]
                name = row["name"]
                titles = ann_map.get(code, [])
                # 若没有公告标题，也可以继续给LLM：它会更保守
                result = llm_risk_check(llm_client, model, code, name, titles)
                passes.append(bool(result.get("pass", True)))
                risk_levels.append(result.get("risk_level", "low"))
                reasons.append("; ".join(result.get("reasons", []))[:300])

                time.sleep(0.05)

            picked["llm_pass"] = passes
            picked["llm_risk"] = risk_levels
            picked["llm_reasons"] = reasons

            # 只保留通过的（你也可以改为“不过滤，只标注”）
            picked = picked[picked["llm_pass"] == True].copy()
            picked = picked.sort_values("score", ascending=False).head(TOP_N)

        except Exception as e:
            print(f"[WARN] LLM排雷未启用或失败：{e}")

    # 8) 输出
    out_path = f"picked_{trade_date}.csv"
    picked.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] 候选数量={len(picked)}，已输出：{out_path}")
    if not picked.empty:
        print(picked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
