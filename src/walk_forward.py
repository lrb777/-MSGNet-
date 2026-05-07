# -*- coding: utf-8 -*-
"""
阶段六：Walk-Forward 滚动窗口回测

当前使用静态因子（训练后固定），Walk-Forward 的作用是：
  将全期收益流切成多个不重叠窗口，分别报告绩效，
  验证因子在不同市场环境（牛/熊/震荡）下的一致性。

注：静态因子的 Walk-Forward 是"市场环境压力测试"，
    不是真正的样本外泛化测试（那需要每个窗口重新训练模型）。
    升级到时间序列因子后，替换 load_factors() 即可复用本框架。

输出：facts/runs/<run_id>/walk_forward/
  wf_report.csv          ← 各窗口绩效明细
  wf_summary.csv         ← 汇总统计
  wf_result.png          ← 可视化
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
_OUT_DIR     = get_subdir("walk_forward")

USE_NEUTRALIZED = True   # 与 backtest.py 保持一致

WINDOW_DAYS  = 63        # 每个测试窗口长度（约3个月）
REBAL_FREQ   = 5
TOP_PCT      = 0.20
COST_BPS     = 10
ANNUAL_DAYS  = 252

IC_AMP = 0.2738
IC_PS  = 0.1525


# ── 数据加载（复用 backtest.py 逻辑）────────────────────────────────

def load_price_returns() -> pd.DataFrame:
    price = pd.read_csv(os.path.join(_DATA_DIR, "price.csv"),
                        parse_dates=["date"], index_col="date")
    return price.pct_change().dropna()


def load_factors() -> pd.Series:
    if USE_NEUTRALIZED:
        path = os.path.join(_FACTOR_DIR, "neutralized", "period_factors_neutralized.csv")
    else:
        path = os.path.join(_FACTOR_DIR, "period_factors.csv")

    pf  = pd.read_csv(path, index_col="ETF")
    amp = pf["amplitude"]
    ps  = pf["period_stability"]

    w_amp = IC_AMP / (IC_AMP + IC_PS)
    w_ps  = IC_PS  / (IC_AMP + IC_PS)
    composite = -amp * w_amp + ps * w_ps
    return composite.rank(pct=True)


def build_portfolio(factor: pd.Series, price_ret: pd.DataFrame):
    common       = factor.index.intersection(price_ret.columns)
    factor       = factor.reindex(common).dropna()
    price_ret    = price_ret[factor.index]
    top_n        = max(1, int(len(factor) * TOP_PCT))
    long_stocks  = factor.nlargest(top_n).index
    short_stocks = factor.nsmallest(top_n).index
    return long_stocks, short_stocks, price_ret


def calc_ls_returns(long_stocks, short_stocks, price_ret: pd.DataFrame) -> pd.Series:
    long_ret  = price_ret[long_stocks].mean(axis=1)
    short_ret = price_ret[short_stocks].mean(axis=1)
    ls_ret    = long_ret - short_ret
    cost_per_day = (COST_BPS * 2 / 10000) / REBAL_FREQ
    return ls_ret - cost_per_day


# ── 单窗口绩效 ────────────────────────────────────────────────────

def window_perf(ret: pd.Series) -> dict:
    if len(ret) == 0:
        return {}
    cum     = (1 + ret).cumprod()
    total   = cum.iloc[-1] - 1
    years   = len(ret) / ANNUAL_DAYS
    ann_ret = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = ret.std() * np.sqrt(ANNUAL_DAYS)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd  = (cum / cum.cummax() - 1).min()
    win_rt  = (ret > 0).mean()
    return dict(
        start    = ret.index[0].date(),
        end      = ret.index[-1].date(),
        ann_ret  = round(ann_ret * 100, 2),
        ann_vol  = round(ann_vol * 100, 2),
        sharpe   = round(sharpe, 3),
        max_dd   = round(max_dd * 100, 2),
        win_rate = round(win_rt * 100, 2),
    )


# ── Walk-Forward 主循环 ───────────────────────────────────────────

def run_walk_forward(ls_ret: pd.Series) -> pd.DataFrame:
    records = []
    n = len(ls_ret)
    starts = range(0, n - WINDOW_DAYS + 1, WINDOW_DAYS)

    for i, s in enumerate(starts):
        window = ls_ret.iloc[s: s + WINDOW_DAYS]
        perf   = window_perf(window)
        perf["window"] = i + 1
        records.append(perf)
        print(f"  窗口{i+1:02d}  {perf['start']} ~ {perf['end']}"
              f"  年化{perf['ann_ret']:+.1f}%  Sharpe{perf['sharpe']:+.3f}")

    return pd.DataFrame(records).set_index("window")


# ── 汇总统计 ─────────────────────────────────────────────────────

def summarize(report: pd.DataFrame) -> pd.DataFrame:
    pos_windows = (report["ann_ret"] > 0).sum()
    summary = pd.DataFrame({
        "均值":  report[["ann_ret","ann_vol","sharpe","max_dd","win_rate"]].mean(),
        "标准差": report[["ann_ret","ann_vol","sharpe","max_dd","win_rate"]].std(),
        "最大值": report[["ann_ret","ann_vol","sharpe","max_dd","win_rate"]].max(),
        "最小值": report[["ann_ret","ann_vol","sharpe","max_dd","win_rate"]].min(),
    }).T

    print(f"\n{'='*55}")
    print(f"  Walk-Forward 汇总（{len(report)} 个窗口，每窗口约3个月）")
    print(f"{'='*55}")
    print(f"  正收益窗口数  : {pos_windows} / {len(report)}"
          f"  ({pos_windows/len(report)*100:.0f}%)")
    print(f"  平均年化收益  : {report['ann_ret'].mean():+.2f}%")
    print(f"  平均 Sharpe   : {report['sharpe'].mean():+.3f}")
    print(f"  Sharpe 标准差 : {report['sharpe'].std():.3f}")
    print(f"  最差窗口年化  : {report['ann_ret'].min():+.2f}%")
    print(f"  最好窗口年化  : {report['ann_ret'].max():+.2f}%")
    return summary


# ── 可视化 ───────────────────────────────────────────────────────

def plot_results(ls_ret: pd.Series, report: pd.DataFrame):
    os.makedirs(_OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # 1. 累计净值 + 窗口分割线
    ax = axes[0]
    cum = (1 + ls_ret).cumprod()
    cum.plot(ax=ax, color="#e74c3c", linewidth=1.5, label="多空组合")
    ax.axhline(1.0, color="black", linewidth=0.5)
    for _, row in report.iterrows():
        ax.axvline(pd.Timestamp(row["start"]), color="gray",
                   linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("累计净值（灰色虚线为窗口边界）")
    ax.set_ylabel("净值")
    ax.legend(fontsize=9)

    # 2. 各窗口年化收益柱状图
    ax = axes[1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in report["ann_ret"]]
    ax.bar(report.index, report["ann_ret"], color=colors, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(report["ann_ret"].mean(), color="#e67e22",
               linewidth=1.2, linestyle="--", label=f"均值 {report['ann_ret'].mean():+.1f}%")
    ax.set_title("各窗口年化收益（%）")
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("年化收益 %")
    ax.legend(fontsize=9)

    # 3. 各窗口 Sharpe
    ax = axes[2]
    colors_s = ["#2ecc71" if v > 0 else "#e74c3c" for v in report["sharpe"]]
    ax.bar(report.index, report["sharpe"], color=colors_s, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(report["sharpe"].mean(), color="#e67e22",
               linewidth=1.2, linestyle="--", label=f"均值 {report['sharpe'].mean():+.3f}")
    ax.set_title("各窗口 Sharpe")
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(_OUT_DIR, "wf_result.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"\n图表已保存: {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Walk-Forward 滚动窗口回测 ===\n")
    print(f"  窗口长度: {WINDOW_DAYS} 交易日（约3个月）\n")

    price_ret = load_price_returns()
    factor    = load_factors()
    long_stocks, short_stocks, price_ret = build_portfolio(factor, price_ret)
    ls_ret    = calc_ls_returns(long_stocks, short_stocks, price_ret)

    print(f"多头: {list(long_stocks)}")
    print(f"空头: {list(short_stocks)}")
    print(f"\n总交易日: {len(ls_ret)}  →  {len(ls_ret)//WINDOW_DAYS} 个完整窗口\n")

    report  = run_walk_forward(ls_ret)
    summary = summarize(report)

    os.makedirs(_OUT_DIR, exist_ok=True)
    report.to_csv(os.path.join(_OUT_DIR, "wf_report.csv"), encoding="utf-8-sig")
    summary.to_csv(os.path.join(_OUT_DIR, "wf_summary.csv"), encoding="utf-8-sig")

    plot_results(ls_ret, report)
    print("\n=== 完成 ===")
