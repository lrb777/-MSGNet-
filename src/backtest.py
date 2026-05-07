# -*- coding: utf-8 -*-
"""
多空组合回测
因子：amplitude（反向，IC_IR=-1.386）+ period_stability（正向，IC_IR=+0.766）
      IC加权合成，h=10 IC绝对值作为权重
策略：做多高得分组（G1）+ 做空低得分组（G5），每5日换仓
"""

import os
import numpy as np
import pandas as pd
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
_OUT_DIR     = get_subdir("backtest")

# True = 使用 Beta 中性化后的 amplitude；False = 原始因子
USE_NEUTRALIZED = True

REBAL_FREQ   = 5     # 换仓频率（交易日）
TOP_PCT      = 0.20  # 多头/空头各取20%
COST_BPS     = 10    # 单边交易成本（bps），10bps = 0.1%
ANNUAL_DAYS  = 252

# IC 绝对值（h=10，用于加权合成）
IC_AMP   = 0.2738  # amplitude，反向因子（取负使用）
IC_PS    = 0.1525  # period_stability，正向因子


# ── 数据加载 ─────────────────────────────────────────────────────

def load_price_returns() -> pd.DataFrame:
    price = pd.read_csv(os.path.join(_DATA_DIR, "price.csv"),
                        parse_dates=["date"], index_col="date")
    return price.pct_change().dropna()


def load_factors() -> pd.Series:
    if USE_NEUTRALIZED:
        path = os.path.join(_FACTOR_DIR, "neutralized", "period_factors_neutralized.csv")
        print("  [因子] 使用 Beta 中性化后的 amplitude")
    else:
        path = os.path.join(_FACTOR_DIR, "period_factors.csv")
        print("  [因子] 使用原始 amplitude（未中性化）")

    pf = pd.read_csv(path, index_col="ETF")

    amp = pf["amplitude"]           # 反向因子（IC<0，取负）
    ps  = pf["period_stability"]    # 正向因子（IC>0，直接用）

    w_amp = IC_AMP / (IC_AMP + IC_PS)
    w_ps  = IC_PS  / (IC_AMP + IC_PS)
    composite = -amp * w_amp + ps * w_ps   # amplitude取负对齐方向

    # 截面 rank 标准化
    composite = composite.rank(pct=True)
    return composite


# ── 组合构建 ─────────────────────────────────────────────────────

def build_portfolio(factor: pd.Series, price_ret: pd.DataFrame):
    """
    基于静态因子构建多空组合。
    多头：因子值最高的 TOP_PCT 股票（经过反向后得分最高 = 原始低振幅）
    空头：因子值最低的 TOP_PCT 股票
    """
    common = factor.index.intersection(price_ret.columns)
    factor = factor.reindex(common).dropna()
    price_ret = price_ret[factor.index]

    n = len(factor)
    top_n = max(1, int(n * TOP_PCT))

    long_stocks  = factor.nlargest(top_n).index    # 高得分 = 低振幅 + 高周期稳定性
    short_stocks = factor.nsmallest(top_n).index   # 低得分 = 高振幅 + 低周期稳定性

    print(f"多头股票数: {len(long_stocks)}  空头股票数: {len(short_stocks)}")
    return long_stocks, short_stocks, price_ret


def calc_portfolio_returns(long_stocks, short_stocks, price_ret: pd.DataFrame,
                           rebal_freq: int = REBAL_FREQ) -> pd.Series:
    """
    等权多空组合日收益率（含换仓成本）。
    由于因子静态不变，换仓只在首日和每 rebal_freq 日执行（实际持仓不变，成本象征性扣除）。
    """
    long_ret  = price_ret[long_stocks].mean(axis=1)
    short_ret = price_ret[short_stocks].mean(axis=1)
    ls_ret    = long_ret - short_ret

    # 扣除换仓成本（每 rebal_freq 日双边各 COST_BPS bps）
    cost_per_day = (COST_BPS * 2 / 10000) / rebal_freq
    ls_ret = ls_ret - cost_per_day

    return ls_ret


# ── 绩效指标 ─────────────────────────────────────────────────────

def performance(ret: pd.Series, label: str = "") -> dict:
    cum   = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    years = len(ret) / ANNUAL_DAYS
    ann_ret  = (1 + total) ** (1 / years) - 1
    ann_vol  = ret.std() * np.sqrt(ANNUAL_DAYS)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    drawdown = (cum / cum.cummax() - 1)
    max_dd   = drawdown.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (ret > 0).mean()

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  区间          : {ret.index[0].date()} ~ {ret.index[-1].date()}")
    print(f"  总收益        : {total:+.2%}")
    print(f"  年化收益      : {ann_ret:+.2%}")
    print(f"  年化波动      : {ann_vol:.2%}")
    print(f"  Sharpe        : {sharpe:.3f}")
    print(f"  最大回撤      : {max_dd:.2%}")
    print(f"  Calmar        : {calmar:.3f}")
    print(f"  日胜率        : {win_rate:.2%}")

    return dict(区间=f"{ret.index[0].date()}~{ret.index[-1].date()}",
                总收益=f"{total:+.2%}", 年化收益=f"{ann_ret:+.2%}",
                年化波动=f"{ann_vol:.2%}", Sharpe=f"{sharpe:.3f}",
                最大回撤=f"{max_dd:.2%}", Calmar=f"{calmar:.3f}",
                日胜率=f"{win_rate:.2%}")


# ── 可视化 ───────────────────────────────────────────────────────

def plot_results(ls_ret: pd.Series, long_ret: pd.Series, short_ret: pd.Series,
                 price_ret: pd.DataFrame, test_start: pd.Timestamp):
    os.makedirs(_OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # 1. 累计净值（全期 + 测试期标注）
    ax = axes[0]
    cum_ls    = (1 + ls_ret).cumprod()
    cum_long  = (1 + long_ret).cumprod()
    cum_short = (1 + short_ret).cumprod()
    cum_mkt   = (1 + price_ret.mean(axis=1)).cumprod()

    cum_ls.plot(ax=ax, label="多空组合", color="#e74c3c", linewidth=1.5)
    cum_long.plot(ax=ax, label="多头（低振幅+高稳定性）", color="#2ecc71", linewidth=1.2, linestyle="--")
    cum_short.plot(ax=ax, label="空头（高振幅+低稳定性）", color="#e67e22", linewidth=1.2, linestyle="--")
    cum_mkt.plot(ax=ax, label="等权市场", color="#95a5a6", linewidth=1.0, linestyle=":")
    ax.axvline(test_start, color="navy", linewidth=1.0, linestyle="--", alpha=0.7,
               label=f"测试集起点 {test_start.date()}")
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_title("累计净值")
    ax.legend(fontsize=8)
    ax.set_ylabel("净值")

    # 2. 多空组合回撤
    ax = axes[1]
    drawdown = (cum_ls / cum_ls.cummax() - 1) * 100
    drawdown.plot(ax=ax, color="#c0392b", linewidth=1.0)
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="#c0392b")
    ax.set_title("多空组合回撤（%）")
    ax.set_ylabel("回撤 %")

    # 3. 滚动60日年化收益
    ax = axes[2]
    rolling_ann = ls_ret.rolling(60).mean() * ANNUAL_DAYS * 100
    rolling_ann.plot(ax=ax, color="#8e44ad", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("滚动60日年化收益（%）")
    ax.set_ylabel("年化收益 %")

    plt.tight_layout()
    out_path = os.path.join(_OUT_DIR, "backtest_result.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"\n图表已保存: {out_path}")


# ── 主流程 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 多空组合回测 ===")

    price_ret = load_price_returns()
    factor    = load_factors()

    long_stocks, short_stocks, price_ret = build_portfolio(factor, price_ret)

    ls_ret    = calc_portfolio_returns(long_stocks, short_stocks, price_ret)
    long_ret  = price_ret[long_stocks].mean(axis=1)
    short_ret = price_ret[short_stocks].mean(axis=1)

    # 测试集起点（最后20%）
    n = len(price_ret)
    test_start = price_ret.index[int(n * 0.8)]

    # 全期绩效
    perf_full = performance(ls_ret, label="多空组合（全期）")

    # 测试集绩效（样本外）
    ls_test = ls_ret[ls_ret.index >= test_start]
    perf_test = performance(ls_test, label=f"多空组合（测试集，样本外）")

    # 多头单独绩效
    performance(long_ret[long_ret.index >= test_start], label="多头组合（测试集）")

    # 保存绩效报告
    os.makedirs(_OUT_DIR, exist_ok=True)
    pd.DataFrame([perf_full, perf_test], index=["全期", "测试集"]).to_csv(
        os.path.join(_OUT_DIR, "perf_report.csv"), encoding="utf-8-sig"
    )

    plot_results(ls_ret, long_ret, short_ret, price_ret, test_start)
    print("\n=== 回测完成 ===")
