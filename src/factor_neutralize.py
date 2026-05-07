# -*- coding: utf-8 -*-
"""
因子中性化：对 amplitude 做市场Beta中性化

原理：
  1. 用训练集（前60%）计算每只ETF相对等权市场的历史Beta
  2. 截面回归：amplitude = a + b × Beta + residual
  3. 残差即为剥离Beta暴露后的纯Alpha信号

输出：facts/runs/<run_id>/factors/neutralized/period_factors_neutralized.csv
      （替换 amplitude 列为残差，其余列不变）
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_VARIANT = "etf20"
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data", DATA_VARIANT)
from run_config import get_subdir

_FACTOR_DIR  = get_subdir("factors")
_NEUTRAL_DIR = os.path.join(_FACTOR_DIR, "neutralized")

TRAIN_RATIO  = 0.6   # 与 train.py 一致，Beta 用训练集计算


# ── 计算历史 Beta ─────────────────────────────────────────────────

def calc_betas(price_ret: pd.DataFrame) -> pd.Series:
    """
    每只ETF相对等权市场的 Beta（训练集内）。
    Beta_i = Cov(R_i, R_mkt) / Var(R_mkt)
    """
    n_train = int(len(price_ret) * TRAIN_RATIO)
    train   = price_ret.iloc[:n_train]
    mkt     = train.mean(axis=1)   # 等权市场

    betas = {}
    for col in train.columns:
        var_mkt = mkt.var()
        betas[col] = train[col].cov(mkt) / var_mkt if var_mkt > 0 else 1.0

    return pd.Series(betas, name="beta")


# ── 截面回归取残差 ────────────────────────────────────────────────

def neutralize(factor: pd.Series, risk: pd.Series) -> pd.Series:
    """
    factor = a + b × risk + residual
    返回残差，index 与 factor 对齐。
    """
    common = factor.index.intersection(risk.index)
    f = factor.reindex(common).values.reshape(-1, 1)
    r = risk.reindex(common).values.reshape(-1, 1)

    reg      = LinearRegression().fit(r, f)
    residual = f.flatten() - reg.predict(r).flatten()

    print(f"  回归 R² = {reg.score(r, f):.4f}  "
          f"（Beta解释了amplitude方差的{reg.score(r, f)*100:.1f}%）")

    return pd.Series(residual, index=common, name=factor.name)


# ── 诊断输出 ──────────────────────────────────────────────────────

def diagnose(original: pd.Series, neutralized: pd.Series, risk: pd.Series):
    print("\n  中性化前后对比:")
    print(f"    amplitude vs Beta 相关性  :"
          f"  中性化前 {original.corr(risk):+.4f}"
          f"  → 中性化后 {neutralized.corr(risk):+.4f}")
    print(f"\n  amplitude 分布变化:")
    df = pd.DataFrame({"原始": original, "中性化后": neutralized})
    print(df.describe().to_string())
    print(f"\n  各ETF Beta:")
    print(risk.sort_values().to_string())


# ── 主流程 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== amplitude Beta 中性化 ===\n")

    # 1. 加载价格收益率，计算 Beta
    price    = pd.read_csv(os.path.join(_DATA_DIR, "price.csv"),
                           parse_dates=["date"], index_col="date")
    price_ret = price.pct_change().dropna()
    betas    = calc_betas(price_ret)

    # 2. 加载 amplitude
    pf        = pd.read_csv(os.path.join(_FACTOR_DIR, "period_factors.csv"),
                            index_col="ETF")
    amplitude = pf["amplitude"]

    # 3. Beta 中性化
    print("回归结果:")
    amp_neutral = neutralize(amplitude, betas)

    # 4. 诊断
    diagnose(amplitude, amp_neutral, betas)

    # 5. 保存：替换 amplitude 列，其余因子列不变
    os.makedirs(_NEUTRAL_DIR, exist_ok=True)
    pf_out = pf.copy()
    pf_out["amplitude"] = amp_neutral.reindex(pf.index)
    out_path = os.path.join(_NEUTRAL_DIR, "period_factors_neutralized.csv")
    pf_out.to_csv(out_path)

    print(f"\n已保存: {out_path}")
    print("=== 完成 ===")
