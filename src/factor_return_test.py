# -*- coding: utf-8 -*-
"""
单因子收益显著性检验（Beta 控制版）

用途：
  在现有 IC 检验之外，估计每个 MSGNet 单因子的横截面因子收益序列。

改造原则：
  1. ETF20 截面太小，不引入行业哑变量，避免自由度被吃掉。
  2. 行业 ETF 本身就是研究对象，不做行业中性化。
  3. 只控制历史 Beta 暴露，目标因子先对 Beta 取残差，再参与回归。

回归口径：
  future_ret_{i,t,d} = a_t + b_t * beta_i + f_{k,t,d} * factor_resid_i + e_{i,t}

输出：
  facts/runs/<run_id>/validation/factor_return_report.csv
"""

import os
import numpy as np
import pandas as pd

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_VARIANT = "etf20"
_DATA_DIR = os.path.join(_PROJECT_DIR, "data", DATA_VARIANT)
from run_config import get_subdir

_FACTOR_DIR = get_subdir("factors")
_REPORT_DIR = get_subdir("validation")

PRED_HORIZONS = [1, 5, 10, 20]
TRAIN_RATIO = 0.6
ANNUAL_DAYS = 252
MIN_OBS = 8


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=1)
    if std == 0 or pd.isna(std):
        return s * 0.0
    return (s - s.mean()) / std


def load_price_returns() -> pd.DataFrame:
    price = pd.read_csv(os.path.join(_DATA_DIR, "price.csv"),
                        parse_dates=["date"], index_col="date")
    return price.pct_change().dropna(how="all")


def load_factors() -> pd.DataFrame:
    graph = pd.read_csv(os.path.join(_FACTOR_DIR, "graph_factors.csv"),
                        index_col="ETF")
    period = pd.read_csv(os.path.join(_FACTOR_DIR, "period_factors.csv"),
                         index_col="ETF")

    # softmax 行归一化导致 out_degree 恒约等于 1，无截面信息量。
    graph = graph[[c for c in graph.columns if "out_degree" not in c]]
    factors = pd.concat([graph, period], axis=1)
    return factors.loc[:, ~factors.columns.duplicated()]


def calc_static_beta(price_ret: pd.DataFrame) -> pd.Series:
    n_train = int(len(price_ret) * TRAIN_RATIO)
    train = price_ret.iloc[:n_train]
    market = train.mean(axis=1)
    market_var = market.var()

    betas = {}
    for col in train.columns:
        betas[col] = train[col].cov(market) / market_var if market_var > 0 else 1.0
    return pd.Series(betas, name="beta")


def neutralize_factor_to_beta(factor: pd.Series, beta: pd.Series) -> tuple[pd.Series, float]:
    common = factor.dropna().index.intersection(beta.dropna().index)
    y = zscore(factor.reindex(common))
    x = zscore(beta.reindex(common))
    design = np.column_stack([np.ones(len(common)), x.values])
    coef = np.linalg.lstsq(design, y.values, rcond=None)[0]
    residual = y - design @ coef
    residual = pd.Series(residual, index=common, name=factor.name)
    corr = y.corr(x)
    return zscore(residual), corr


def future_returns(price_ret: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return (1 + price_ret).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1


def factor_return_series(factor_resid: pd.Series,
                         beta: pd.Series,
                         price_ret: pd.DataFrame,
                         horizon: int) -> pd.Series:
    fut = future_returns(price_ret, horizon)
    common = factor_resid.index.intersection(beta.index).intersection(price_ret.columns)
    factor_x = zscore(factor_resid.reindex(common))
    beta_x = zscore(beta.reindex(common))

    coefs = {}
    for date, row in fut[common].iterrows():
        y = row.dropna()
        idx = y.index.intersection(common)
        if len(idx) < MIN_OBS:
            continue

        x_beta = beta_x.reindex(idx)
        x_factor = factor_x.reindex(idx)
        valid = y.index[x_beta.notna() & x_factor.notna()]
        if len(valid) < MIN_OBS:
            continue

        design = np.column_stack([
            np.ones(len(valid)),
            x_beta.reindex(valid).values,
            x_factor.reindex(valid).values,
        ])
        coef = np.linalg.lstsq(design, y.reindex(valid).values, rcond=None)[0]
        coefs[date] = coef[2]

    return pd.Series(coefs, name=f"factor_return_h{horizon}")


def newey_west_tvalue(s: pd.Series, max_lag: int) -> float:
    """
    Newey-West 均值 t 值。horizon>1 的未来收益窗口有重叠，
    普通 t 值会高估显著性，因此用 horizon-1 作为滞后阶数。
    """
    x = s.dropna().values
    n = len(x)
    if n < 2:
        return 0.0
    demeaned = x - x.mean()
    gamma0 = np.dot(demeaned, demeaned) / n
    lrv = gamma0
    lag = min(max_lag, n - 1)
    for j in range(1, lag + 1):
        gamma = np.dot(demeaned[j:], demeaned[:-j]) / n
        weight = 1 - j / (lag + 1)
        lrv += 2 * weight * gamma
    if lrv <= 0:
        return 0.0
    se_mean = np.sqrt(lrv / n)
    return x.mean() / se_mean if se_mean > 0 else 0.0


def summarize_factor_return(fr: pd.Series, horizon: int) -> dict:
    s = fr.dropna()
    per_day = s / horizon
    std = per_day.std(ddof=1)
    ann_ret = per_day.mean() * ANNUAL_DAYS
    ir = np.sqrt(ANNUAL_DAYS) * per_day.mean() / std if std > 0 else 0.0
    t_value = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 and s.std(ddof=1) > 0 else 0.0
    nw_t_value = newey_west_tvalue(s, max_lag=max(0, horizon - 1))

    return {
        "样本数": len(s),
        "单期因子收益均值": round(s.mean(), 6),
        "单期因子收益标准差": round(s.std(ddof=1), 6),
        "年化因子收益": round(ann_ret, 6),
        "因子IR": round(ir, 4),
        "t值": round(t_value, 4),
        "NW_t值": round(nw_t_value, 4),
        "正收益比例": round((s > 0).mean(), 4),
    }


def main():
    os.makedirs(_REPORT_DIR, exist_ok=True)
    price_ret = load_price_returns()
    factors = load_factors()
    beta = calc_static_beta(price_ret)

    print("=" * 70)
    print("单因子收益显著性检验（Beta 控制版）")
    print(f"ETF数量: {price_ret.shape[1]}  交易日: {price_ret.shape[0]}")
    print(f"预测期: {PRED_HORIZONS} 日")
    print("=" * 70)

    rows = []
    for factor_name in factors.columns:
        factor = factors[factor_name]
        factor_resid, beta_corr = neutralize_factor_to_beta(factor, beta)

        for horizon in PRED_HORIZONS:
            fr = factor_return_series(factor_resid, beta, price_ret, horizon)
            summary = summarize_factor_return(fr, horizon)
            significant = abs(summary["NW_t值"]) >= 2 and abs(summary["因子IR"]) >= 0.5
            tag = "PASS" if significant else ""
            print(
                f"{factor_name:24s} h={horizon:2d} "
                f"ann={summary['年化因子收益']:+.2%} "
                f"IR={summary['因子IR']:+.3f} "
                f"NW_t={summary['NW_t值']:+.3f} {tag}"
            )
            rows.append({
                "因子": factor_name,
                "预测期": horizon,
                "Beta相关性_中性化前": round(beta_corr, 4),
                **summary,
                "显著": significant,
            })

    report = pd.DataFrame(rows)
    out_path = os.path.join(_REPORT_DIR, "factor_return_report.csv")
    report.to_csv(out_path, index=False, encoding="utf-8-sig")

    passed = report[report["显著"]].sort_values(["因子IR"], key=lambda s: s.abs(), ascending=False)
    print("\n" + "=" * 70)
    print("显著因子筛选：|NW_t值| >= 2 且 |IR| >= 0.5")
    print("=" * 70)
    if passed.empty:
        print("暂无因子通过该显著性门槛。")
    else:
        print(passed[["因子", "预测期", "年化因子收益", "因子IR", "NW_t值", "正收益比例"]].to_string(index=False))
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
