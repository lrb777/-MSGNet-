# -*- coding: utf-8 -*-
"""
数据预处理：价格 → MSGNet 可读的对数收益率矩阵

输入：data/etf20/price.csv       （adj_close，data_pipeline 输出）
输出：
  data/etf20/returns.csv         ← MSGNet 训练用（含 date 列）
  data/etf20/returns.npy         ← numpy 张量 [T-1, N]，供因子提取用

处理步骤：
  1. 对数收益率：log(p_t / p_{t-1})
  2. 截面 z-score：每个交易日对 N 只 ETF 做标准化，消除市场整体涨跌
  3. 极端值裁剪：±3σ 截断，防止黑天鹅日破坏标准化
"""

import os
import numpy as np
import pandas as pd

_BASE_DIR    = os.path.dirname(__file__)
DATA_VARIANT = "etf20"
OUTPUT_DIR   = os.path.join(_BASE_DIR, "..", "data", DATA_VARIANT)
INPUT_PATH   = os.path.join(OUTPUT_DIR, "price.csv")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "returns.csv")
OUTPUT_NPY   = os.path.join(OUTPUT_DIR, "returns.npy")

CLIP_SIGMA   = 3.0   # 截面 z-score 裁剪阈值


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    assert df.isnull().sum().sum() == 0, "原始价格有缺失值，请先检查 data_pipeline"
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """对数收益率，丢弃第一行 NaN"""
    return np.log(prices / prices.shift(1)).dropna()


def cross_section_zscore(returns: pd.DataFrame, clip: float = CLIP_SIGMA) -> pd.DataFrame:
    """
    每个交易日（行）对所有 ETF（列）做 z-score，再截断极端值。
    结果：每日截面均值≈0，标准差≈1。
    """
    mean = returns.mean(axis=1)
    std  = returns.std(axis=1).replace(0, 1)   # 防止全零行除零
    standardized = returns.sub(mean, axis=0).div(std, axis=0)
    return standardized.clip(-clip, clip)


def save_outputs(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV：保留 date 列，供 Dataset_Custom 直接读取
    df.reset_index().rename(columns={"date": "date"}).to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8"
    )

    # npy：纯数值矩阵 [T, N]
    np.save(OUTPUT_NPY, df.values.astype(np.float32))

    print(f"[SAVED] {OUTPUT_CSV}")
    print(f"[SAVED] {OUTPUT_NPY}")


def quality_report(raw: pd.DataFrame, processed: pd.DataFrame):
    print("\n=== 预处理质量报告 ===")
    print(f"  原始价格行数   : {len(raw)}")
    print(f"  收益率行数     : {len(processed)}  (减1行因 shift)")
    print(f"  ETF 数量       : {processed.shape[1]}")
    print(f"  日期范围       : {processed.index.min().date()} ~ {processed.index.max().date()}")
    print(f"  缺失值         : {processed.isnull().sum().sum()}")
    print(f"\n  截面统计（理想：均值≈0，标准差≈1）")
    daily_mean = processed.mean(axis=1)
    daily_std  = processed.std(axis=1)
    print(f"    每日均值  mean={daily_mean.mean():.4f}  std={daily_mean.std():.4f}")
    print(f"    每日标准差 mean={daily_std.mean():.4f}  std={daily_std.std():.4f}")
    print(f"\n  各 ETF 收益率统计（z-score 后）")
    desc = processed.describe().loc[["mean", "std", "min", "max"]]
    print(desc.T.to_string())


if __name__ == "__main__":
    print("=== ETF 数据预处理 ===\n")

    prices    = load_prices(INPUT_PATH)
    returns   = log_returns(prices)
    processed = cross_section_zscore(returns)

    quality_report(prices, processed)
    save_outputs(processed)

    print("\nDone.")
