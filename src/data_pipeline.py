# -*- coding: utf-8 -*-
"""
行业ETF数据下载
数据源：adata.fund.market.get_market_etf
输出：data/etf20/price.csv  —— date + N只ETF重建前复权收盘价

流程：
  1. 固定20只行业ETF列表（硬编码）
  2. 多线程并发下载，增量更新
  3. change_pct链式重建前复权价格（ETF无前复权参数）
  4. 自动检测EFFECTIVE_START（全员共有数据的最早日期）
  5. 有效数据不足80%的ETF自动排除
"""

import os
import time
import random
import logging
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import adata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 配置 ─────────────────────────────────────────────────────────
DATA_VARIANT   = "etf20"
FETCH_START    = "2019-01-01"   # 宽松起始，确保2019年上市的ETF有足够数据
MIN_DATA_RATIO = 0.80           # 有效数据比例低于此则排除该ETF
MAX_WORKERS    = 6
MAX_RETRY      = 4

INDUSTRY_ETF = [
    ("510050", "上证50"),
    ("510300", "沪深300"),
    ("510500", "中证500"),
    ("512880", "证券"),
    ("512800", "银行"),
    ("159928", "消费"),
    ("512010", "医疗"),
    ("512660", "军工"),
    ("512400", "有色金属"),
    ("512200", "地产"),
    ("512980", "传媒"),
    ("512690", "白酒"),
    ("512170", "医疗健康"),
    ("515050", "科技"),
    ("159995", "芯片"),
    ("515020", "汽车"),
    ("515880", "交通运输"),
    ("159825", "农业"),
    ("512090", "央企改革"),
    ("159996", "家电"),
]

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BASE_DIR)
OUTPUT_DIR   = os.path.join(_PROJECT_DIR, "data", DATA_VARIANT)
CACHE_DIR    = os.path.join(OUTPUT_DIR, "cache")
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, "price.csv")


# ── 缓存 I/O ─────────────────────────────────────────────────────

def _cache_path(code: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{code}.csv")


def _load_cache(code: str) -> Optional[pd.DataFrame]:
    path = _cache_path(code)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["trade_date"])
        return df if not df.empty else None
    except Exception:
        return None


def _save_cache(code: str, df: pd.DataFrame):
    df[["trade_date", "adj_close"]].to_csv(_cache_path(code), index=False)


# ── 下载原语 ─────────────────────────────────────────────────────

def _backoff(attempt: int):
    time.sleep(min(2 ** attempt + random.uniform(0, 1.5), 30))


def _fetch_etf(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    下载ETF行情，返回含 trade_date / close / change_pct 的 DataFrame。
    """
    for attempt in range(MAX_RETRY):
        try:
            df = adata.fund.market.get_market_etf(
                fund_code=code,
                start_date=start,
                end_date=end,
            )
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df[df["trade_date"] <= pd.Timestamp(end)]
                df = df.sort_values("trade_date").reset_index(drop=True)
                needed = {"trade_date", "close", "change_pct"}
                if not needed.issubset(df.columns):
                    log.warning(f"[{code}] 缺少列: {needed - set(df.columns)}，实际列: {list(df.columns)}")
                    return None
                df["change_pct"] = df["change_pct"].infer_objects(copy=False).fillna(0)
                return df[["trade_date", "close", "change_pct"]]
            log.warning(f"[{code}] 空数据 attempt {attempt + 1}")
        except Exception as e:
            log.warning(f"[{code}] attempt {attempt + 1}: {type(e).__name__}: {e}")
        _backoff(attempt)
    log.error(f"[{code}] 所有重试失败")
    return None


def _build_adj_close(raw_df: pd.DataFrame, base_price: Optional[float] = None) -> pd.Series:
    """
    用 change_pct 链式重建前复权价格。

    全量下载 (base_price=None)：
      以 close[0] 为锚点，change_pct[0] 忽略（无前日数据），从第1日起累积。
    增量更新 (base_price=last_adj_close)：
      以前日 adj_close 为锚点，链式累积所有新行的 change_pct。
    """
    cp = raw_df["change_pct"].fillna(0)

    if base_price is None:
        base_price = float(raw_df["close"].iloc[0])
        # 第0行固定为1.0（锚点），后续行累积
        factors = pd.concat(
            [pd.Series([1.0]), (1 + cp.iloc[1:] / 100)]
        ).reset_index(drop=True)
    else:
        factors = (1 + cp / 100).reset_index(drop=True)

    adj = base_price * factors.cumprod()
    adj.index = raw_df.index
    return adj.rename("adj_close")


# ── 增量下载 ─────────────────────────────────────────────────────

def download_etf(code: str, name: str) -> Optional[pd.Series]:
    """
    增量策略：
      - 无缓存   → 全量下载（从 FETCH_START）
      - 有缓存   → 拉增量，以缓存最后 adj_close 为锚点链式续接
    返回以 trade_date 为索引、name 为列名的 adj_close Series。
    """
    today = date.today().strftime("%Y-%m-%d")
    time.sleep(random.uniform(0, 0.5))

    cached = _load_cache(code)

    if cached is not None:
        last_dt  = cached["trade_date"].max()
        last_str = last_dt.strftime("%Y-%m-%d")

        if last_str >= today:
            return cached.set_index("trade_date")["adj_close"].rename(name)

        fetch_start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        log.info(f"[{name}({code})] 增量 {fetch_start} ~ {today}")
        new_raw = _fetch_etf(code, fetch_start, today)

        if new_raw is not None and not new_raw.empty:
            base = float(cached["adj_close"].iloc[-1])
            new_adj = _build_adj_close(new_raw, base_price=base)
            new_part = new_raw[["trade_date"]].copy()
            new_part["adj_close"] = new_adj.values
            merged = pd.concat([cached, new_part], ignore_index=True)
            merged = merged.drop_duplicates("trade_date").sort_values("trade_date")
        else:
            log.warning(f"[{name}({code})] 增量为空，沿用缓存")
            merged = cached
    else:
        log.info(f"[{name}({code})] 全量下载 {FETCH_START} ~ {today}")
        raw = _fetch_etf(code, FETCH_START, today)
        if raw is None:
            return None
        adj = _build_adj_close(raw)
        merged = raw[["trade_date"]].copy()
        merged["adj_close"] = adj.values
        merged = merged.sort_values("trade_date")

    _save_cache(code, merged)
    return merged.set_index("trade_date")["adj_close"].rename(name)


# ── 汇总 ─────────────────────────────────────────────────────────

def build_price_table() -> pd.DataFrame:
    series_list = []
    failed = []

    log.info(f"开始下载 {len(INDUSTRY_ETF)} 只行业ETF ...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_etf, code, name): name
            for code, name in INDUSTRY_ETF
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                s = future.result()
            except Exception as e:
                log.error(f"[{name}] 任务异常: {e}")
                s = None

            if s is not None and not s.empty:
                series_list.append(s)
            else:
                log.warning(f"[{name}] 下载失败，跳过")
                failed.append(name)

    if not series_list:
        raise RuntimeError("所有ETF下载失败，请检查网络或 adata 版本")

    df = pd.concat(series_list, axis=1).sort_index()

    # 自动检测 EFFECTIVE_START：所有ETF第一个有效日期的最大值
    first_dates = df.apply(lambda col: col.first_valid_index())
    effective_start = first_dates.max()
    log.info(f"EFFECTIVE_START: {effective_start.date()}（由最晚上市的ETF决定）")
    for col in df.columns:
        fd = first_dates[col]
        if fd is not None and fd > df.index.min():
            log.info(f"  {col} 起始日: {fd.date()}")

    # 从 EFFECTIVE_START 起，过滤有效数据不足 MIN_DATA_RATIO 的ETF
    sub = df.loc[effective_start:]
    total_rows = len(sub)
    valid_cols = [
        c for c in sub.columns
        if sub[c].notna().sum() / total_rows >= MIN_DATA_RATIO
    ]
    dropped = set(df.columns) - set(valid_cols)
    if dropped:
        log.warning(f"有效数据不足 {MIN_DATA_RATIO*100:.0f}%，已排除: {dropped}")

    df = sub[valid_cols].ffill().dropna(how="all")

    if failed:
        log.warning(f"下载失败（已跳过）: {failed}")

    return df


# ── 保存 / 质检 ─────────────────────────────────────────────────

def save_price_table(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = df.reset_index().rename(columns={"trade_date": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    log.info(f"已保存: {OUTPUT_PATH}")
    log.info(f"  shape : {out.shape}  ({out.shape[1]-1} 只ETF × {out.shape[0]} 交易日)")
    log.info(f"  日期范围: {out['date'].iloc[0]} ~ {out['date'].iloc[-1]}")


def check_quality(df: pd.DataFrame):
    log.info("=== 数据质量检查 ===")
    log.info(f"  ETF数量    : {df.shape[1]}")
    log.info(f"  交易日数   : {df.shape[0]}")
    log.info(f"  日期范围   : {df.index.min().date()} ~ {df.index.max().date()}")
    log.info(f"  总缺失率   : {df.isnull().mean().mean()*100:.2f}%")
    missing = df.isnull().sum()
    if missing.any():
        log.warning(f"  含缺失值的ETF:\n{missing[missing > 0]}")


if __name__ == "__main__":
    log.info("=== 行业ETF数据下载 ===")
    price_df = build_price_table()
    check_quality(price_df)
    save_price_table(price_df)
    log.info("=== 完成 ===")
