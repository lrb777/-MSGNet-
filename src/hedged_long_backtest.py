# -*- coding: utf-8 -*-
"""
ETF 多头组合 + 指数/市场对冲回测。

第一版默认使用 ETF20 等权市场作为对冲基准，不额外下载指数数据。
目标是验证 MSGNet 多头信号在剥离市场 Beta 后是否仍有收益质量。

输出：facts/runs/<run_id>/hedged_long/
  perf_report.csv          各对冲方案绩效
  hedge_report.csv         多头持仓、对冲参数和 Beta 摘要
  return_series.csv        各方案日收益
  hedged_long_result.png   净值、回撤和滚动 Beta 图
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
_DATA_DIR = os.path.join(_PROJECT_DIR, "data", "etf20")
from run_config import get_subdir

_FACTOR_DIR = get_subdir("factors")
_OUT_DIR = get_subdir("hedged_long")

USE_NEUTRALIZED = True
BENCHMARK_NAME = "ETF20等权市场"

TOP_PCT = 0.20
REBAL_FREQ = 5
COST_BPS = 10
ANNUAL_DAYS = 252
TEST_RATIO = 0.20

FIXED_HEDGE_RATIOS = [0.5, 0.8, 1.0]
ROLLING_WINDOWS = [60, 120]

IC_AMP = 0.2738
IC_PS = 0.1525


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
    amp = pf["amplitude"]
    ps = pf["period_stability"]

    w_amp = IC_AMP / (IC_AMP + IC_PS)
    w_ps = IC_PS / (IC_AMP + IC_PS)
    composite = -amp * w_amp + ps * w_ps
    return composite.rank(pct=True)


def build_long_portfolio(factor: pd.Series, price_ret: pd.DataFrame):
    common = factor.index.intersection(price_ret.columns)
    factor = factor.reindex(common).dropna()
    price_ret = price_ret[factor.index]
    top_n = max(1, int(len(factor) * TOP_PCT))
    long_stocks = factor.nlargest(top_n).index
    print(f"多头ETF数: {len(long_stocks)}")
    print(f"多头ETF: {list(long_stocks)}")
    return long_stocks, price_ret


def calc_rolling_beta(asset_ret: pd.Series, bench_ret: pd.Series,
                      window: int) -> pd.Series:
    cov = asset_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    beta = cov / var.replace(0, np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan)
    beta = beta.shift(1)
    return beta.fillna(0.0)


def calc_static_beta(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    var = bench_ret.var()
    if var <= 0 or pd.isna(var):
        return 0.0
    return float(asset_ret.cov(bench_ret) / var)


def build_strategy_returns(long_ret: pd.Series,
                           benchmark_ret: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = pd.DataFrame(index=long_ret.index)
    hedge_detail = pd.DataFrame(index=long_ret.index)

    series["多头原始"] = long_ret
    hedge_detail["固定_0.0"] = 0.0

    for ratio in FIXED_HEDGE_RATIOS:
        name = f"固定{ratio:.1f}对冲"
        series[name] = long_ret - ratio * benchmark_ret
        hedge_detail[f"固定_{ratio:.1f}"] = ratio

    for window in ROLLING_WINDOWS:
        beta = calc_rolling_beta(long_ret, benchmark_ret, window)
        beta = beta.clip(lower=-1.5, upper=1.5)
        name = f"滚动{window}日Beta对冲"
        series[name] = long_ret - beta * benchmark_ret
        hedge_detail[f"滚动Beta_{window}"] = beta

    return series, hedge_detail


def performance(ret: pd.Series, benchmark_ret: pd.Series, label: str,
                period_name: str) -> dict:
    ret = ret.dropna()
    bench = benchmark_ret.reindex(ret.index).dropna()
    ret = ret.reindex(bench.index)
    cum = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    years = len(ret) / ANNUAL_DAYS
    ann_ret = (1 + total) ** (1 / years) - 1
    ann_vol = ret.std() * np.sqrt(ANNUAL_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (ret > 0).mean()
    beta = calc_static_beta(ret, bench)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  区间          : {ret.index[0].date()} ~ {ret.index[-1].date()}")
    print(f"  总收益        : {total:+.2%}")
    print(f"  年化收益      : {ann_ret:+.2%}")
    print(f"  年化波动      : {ann_vol:.2%}")
    print(f"  Sharpe        : {sharpe:.3f}")
    print(f"  最大回撤      : {max_dd:.2%}")
    print(f"  Beta          : {beta:.3f}")

    return dict(
        区间类型=period_name,
        方案=label,
        区间=f"{ret.index[0].date()}~{ret.index[-1].date()}",
        总收益=f"{total:+.2%}",
        年化收益=f"{ann_ret:+.2%}",
        年化波动=f"{ann_vol:.2%}",
        Sharpe=f"{sharpe:.3f}",
        最大回撤=f"{max_dd:.2%}",
        Calmar=f"{calmar:.3f}",
        日胜率=f"{win_rate:.2%}",
        Beta=f"{beta:.3f}",
    )


def plot_results(strategy_ret: pd.DataFrame, benchmark_ret: pd.Series,
                 hedge_detail: pd.DataFrame):
    os.makedirs(_OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    ax = axes[0]
    cum = (1 + strategy_ret).cumprod()
    bench_cum = (1 + benchmark_ret.reindex(strategy_ret.index)).cumprod()
    cum.plot(ax=ax, linewidth=1.2)
    bench_cum.plot(ax=ax, color="#7f8c8d", linewidth=1.0,
                   linestyle=":", label=BENCHMARK_NAME)
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_title("多头组合与对冲方案累计净值")
    ax.set_ylabel("净值")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    drawdown = (cum / cum.cummax() - 1) * 100
    for col in ["多头原始", "固定1.0对冲", "滚动60日Beta对冲", "滚动120日Beta对冲"]:
        if col in drawdown:
            drawdown[col].plot(ax=ax, linewidth=1.0, label=col)
    ax.set_title("代表方案回撤（%）")
    ax.set_ylabel("回撤 %")
    ax.legend(fontsize=8)

    ax = axes[2]
    for col in hedge_detail.columns:
        if col.startswith("滚动Beta"):
            hedge_detail[col].plot(ax=ax, linewidth=1.0, label=col)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("滚动对冲 Beta")
    ax.set_ylabel("Beta")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(_OUT_DIR, "hedged_long_result.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"\n图表已保存: {out_path}")


if __name__ == "__main__":
    print("=== ETF 多头组合 + 指数/市场对冲回测 ===")
    print(f"  对冲基准: {BENCHMARK_NAME}")

    price_ret = load_price_returns()
    factor = load_factors()
    long_stocks, price_ret = build_long_portfolio(factor, price_ret)

    long_ret = price_ret[long_stocks].mean(axis=1)
    benchmark_ret = price_ret.mean(axis=1).reindex(long_ret.index)

    strategy_ret, hedge_detail = build_strategy_returns(long_ret, benchmark_ret)

    test_start = strategy_ret.index[int(len(strategy_ret) * (1 - TEST_RATIO))]
    reports = []
    for col in strategy_ret.columns:
        reports.append(performance(strategy_ret[col], benchmark_ret, col, "全期"))
        reports.append(performance(
            strategy_ret.loc[strategy_ret.index >= test_start, col],
            benchmark_ret.loc[benchmark_ret.index >= test_start],
            col,
            "测试集",
        ))

    os.makedirs(_OUT_DIR, exist_ok=True)
    pd.DataFrame(reports).to_csv(
        os.path.join(_OUT_DIR, "perf_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({
        "项目": ["对冲基准", "多头ETF", "TOP_PCT", "换仓频率", "交易成本bps", "测试集起点", "成本口径"],
        "取值": [
            BENCHMARK_NAME,
            ",".join(long_stocks),
            TOP_PCT,
            REBAL_FREQ,
            COST_BPS,
            str(test_start.date()),
            "静态因子持仓固定，第一版不做连续换仓成本摊销",
        ],
    }).to_csv(
        os.path.join(_OUT_DIR, "hedge_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [strategy_ret, benchmark_ret.rename(BENCHMARK_NAME), hedge_detail],
        axis=1,
    ).to_csv(
        os.path.join(_OUT_DIR, "return_series.csv"),
        encoding="utf-8-sig",
    )

    plot_results(strategy_ret, benchmark_ret, hedge_detail)
    print("\n=== 完成 ===")
