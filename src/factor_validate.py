# -*- coding: utf-8 -*-
"""
阶段四：因子有效性验证

IC 检验：因子值与未来收益率的截面 Spearman 相关系数
分组分析：按因子值分5组，观察各组平均收益单调性

注：图结构因子和周期因子为静态因子（训练后固定），
    在时间截面上用跨ETF的截面相关验证其预测力。
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_VARIANT = "etf20"
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data", DATA_VARIANT)
from run_config import get_subdir

_FACTOR_DIR  = get_subdir("factors")
_REPORT_DIR  = get_subdir("validation")

PRED_HORIZONS = [1, 5, 10, 20]   # 验证多个预测期
MAIN_HORIZON  = 5                 # 主要验证周期（周频）
N_GROUPS      = 5                 # 分组数


# ── 数据加载 ────────────────────────────────────────────────────

def load_returns() -> pd.DataFrame:
    """IC 计算用：z-score 标准化的对数收益率（rank 排序不变，Spearman IC 有效）"""
    df = pd.read_csv(os.path.join(_DATA_DIR, "returns.csv"),
                     parse_dates=["date"], index_col="date")
    return df


def load_price_returns() -> pd.DataFrame:
    """分组收益计算用：从 price.csv 算真实百分比收益率"""
    price = pd.read_csv(os.path.join(_DATA_DIR, "price.csv"),
                        parse_dates=["date"], index_col="date")
    ret = price.pct_change().dropna()
    # 对齐到 returns.csv 的列（可能因数据过滤有差异）
    common = ret.columns.intersection(
        pd.read_csv(os.path.join(_DATA_DIR, "returns.csv"), nrows=0).columns.drop("date")
    )
    return ret[common]


def load_factors() -> dict:
    gf = pd.read_csv(os.path.join(_FACTOR_DIR, "graph_factors.csv"),
                     index_col="ETF")
    pf = pd.read_csv(os.path.join(_FACTOR_DIR, "period_factors.csv"),
                     index_col="ETF")

    # 剔除信息量为零的 out_degree（softmax行归一化，恒为1）
    gf = gf[[c for c in gf.columns if "out_degree" not in c]]

    return {"graph": gf, "period": pf}


# ── IC 计算 ─────────────────────────────────────────────────────

def rolling_ic(factor_vec: np.ndarray,
               returns: pd.DataFrame,
               horizon: int) -> pd.Series:
    """
    静态因子的滚动 IC：
      每个时间截面，计算 factor_vec [N] 与 当日未来 horizon 日累计收益 [N] 的 Spearman 相关。
    returns: [T, N]
    """
    etf_names = returns.columns.tolist()
    future_ret = (returns + 1).rolling(horizon).apply(
        lambda x: x.prod(), raw=True
    ).shift(-horizon) - 1   # 未来 horizon 日累计收益，向前对齐

    ic_series = {}
    for date, row in future_ret.iterrows():
        valid = row.dropna()
        if len(valid) < 5:
            continue
        f = pd.Series(factor_vec, index=etf_names).loc[valid.index]
        ic, _ = stats.spearmanr(f.values, valid.values)
        ic_series[date] = ic

    return pd.Series(ic_series, name=f"IC_h{horizon}")


def ic_summary(ic_series: pd.Series) -> dict:
    ic = ic_series.dropna()
    return {
        "IC均值":    round(ic.mean(), 4),
        "IC标准差":  round(ic.std(), 4),
        "IC_IR":    round(ic.mean() / ic.std() if ic.std() > 0 else 0, 4),
        "正向比例":  round((ic > 0).mean(), 4),
        "样本数":    len(ic),
    }


# ── 分组分析 ────────────────────────────────────────────────────

def group_return(factor_vec: np.ndarray,
                 price_ret: pd.DataFrame,
                 horizon: int,
                 n_groups: int = N_GROUPS) -> pd.Series:
    """按因子值分组，计算各组平均持有期复利收益（用真实价格收益率）"""
    common_stocks = price_ret.columns.tolist()
    factor = pd.Series(factor_vec,
                       index=pd.read_csv(  # 读取因子对应的股票名列表
                           os.path.join(_DATA_DIR, "returns.csv"), nrows=0
                       ).columns.drop("date").tolist())
    factor = factor.reindex(common_stocks).dropna()

    labels = pd.qcut(factor, n_groups, labels=[f"G{i+1}" for i in range(n_groups)],
                     duplicates="drop")

    # 未来 horizon 日复利收益
    future_ret = (1 + price_ret[factor.index]).rolling(horizon).apply(
        np.prod, raw=True
    ).shift(-horizon) - 1

    group_means = {}
    for g in sorted(labels.unique()):
        stocks_in_g = labels[labels == g].index
        group_means[str(g)] = future_ret[stocks_in_g].mean(axis=1).mean()

    return pd.Series(group_means).sort_index()


# ── 可视化 ──────────────────────────────────────────────────────

def plot_ic_series(ic_dict: dict, factor_name: str, save_path: str):
    fig, axes = plt.subplots(len(ic_dict), 1,
                             figsize=(12, 3 * len(ic_dict)), sharex=False)
    if len(ic_dict) == 1:
        axes = [axes]
    for ax, (horizon, ic_series) in zip(axes, ic_dict.items()):
        ic = ic_series.dropna()
        ax.bar(ic.index, ic.values, color=["#e74c3c" if v < 0 else "#2ecc71" for v in ic.values],
               alpha=0.7, width=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(ic.mean(), color="navy", linewidth=1.2, linestyle="--",
                   label=f"均值={ic.mean():.4f}")
        ax.set_title(f"{factor_name} | 预测期={horizon}日  IC_IR={ic.mean()/ic.std():.3f}")
        ax.legend(fontsize=8)
        ax.set_ylabel("IC")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_group_bar(group_ret: pd.Series, factor_name: str, horizon: int, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in group_ret.values]
    ax.bar(group_ret.index, group_ret.values * 100, color=colors, alpha=0.85)
    ax.set_title(f"{factor_name} | 分组收益（预测期={horizon}日）")
    ax.set_ylabel("平均收益率 (%)")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ── 主流程 ──────────────────────────────────────────────────────

def validate_factor(factor_name: str,
                    factor_vec: np.ndarray,
                    returns: pd.DataFrame,
                    price_ret: pd.DataFrame,
                    report_rows: list,
                    plot_dir: str):
    print(f"\n  [{factor_name}]")

    # 多预测期 IC（用 z-score returns，rank 不变）
    ic_dict = {}
    for h in PRED_HORIZONS:
        ic_s = rolling_ic(factor_vec, returns, h)
        ic_dict[h] = ic_s
        summary = ic_summary(ic_s)
        tag = "✅" if abs(summary["IC均值"]) > 0.03 and abs(summary["IC_IR"]) > 0.5 else "  "
        print(f"    h={h:2d}日: IC均值={summary['IC均值']:+.4f}  "
              f"IC_IR={summary['IC_IR']:+.4f}  "
              f"正向比={summary['正向比例']:.2%}  {tag}")
        report_rows.append({"因子": factor_name, "预测期": h, **summary})

    # IC 时序图
    plot_ic_series(ic_dict, factor_name,
                   os.path.join(plot_dir, f"ic_{factor_name}.png"))

    # 分组图（主预测期，用真实价格收益率）
    gr = group_return(factor_vec, price_ret, MAIN_HORIZON)
    plot_group_bar(gr, factor_name, MAIN_HORIZON,
                   os.path.join(plot_dir, f"group_{factor_name}.png"))


if __name__ == "__main__":
    os.makedirs(_REPORT_DIR, exist_ok=True)
    plot_dir = os.path.join(_REPORT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    returns   = load_returns()
    price_ret = load_price_returns()
    factors   = load_factors()

    print("=" * 60)
    print("因子有效性验证（IC 检验 + 分组分析）")
    print(f"股票数量: {returns.shape[1]}  交易日: {returns.shape[0]}")
    print(f"预测期: {PRED_HORIZONS} 日")
    print("=" * 60)

    report_rows = []

    # 图结构因子
    print("\n【图结构因子】")
    gf = factors["graph"]
    for col in gf.columns:
        validate_factor(col, gf[col].values, returns, price_ret, report_rows, plot_dir)

    # 周期因子
    print("\n【周期因子】")
    pf = factors["period"]
    for col in pf.columns:
        validate_factor(col, pf[col].values, returns, price_ret, report_rows, plot_dir)

    # 汇总报告
    report = pd.DataFrame(report_rows)
    report.to_csv(os.path.join(_REPORT_DIR, "ic_report.csv"), index=False)

    print("\n" + "=" * 60)
    print("筛选：IC均值 > 0.03 且 IC_IR > 0.5 的因子")
    print("=" * 60)
    passed = report[
        (report["IC均值"].abs() > 0.03) &
        (report["IC_IR"].abs() > 0.5) &
        (report["预测期"] == MAIN_HORIZON)
    ][["因子", "预测期", "IC均值", "IC_IR", "正向比例"]]

    if passed.empty:
        print("  暂无因子通过筛选标准，建议检查模型收敛情况或调整超参。")
    else:
        print(passed.to_string(index=False))

    print(f"\n报告已保存: {_REPORT_DIR}/ic_report.csv")
    print(f"图表已保存: {plot_dir}/")
