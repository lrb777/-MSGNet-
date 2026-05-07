# -*- coding: utf-8 -*-
"""
阶段七：滚动更新 Walk-Forward

每个 Fold：
  1. 用训练窗口数据重新训练 MSGNet
  2. 提取图结构因子和周期因子（静态截面）
  3. 用训练窗口计算 Beta，对 amplitude 做中性化
  4. 构建多空组合，在测试窗口评估绩效

参数：
  TRAIN_WINDOW = 500 日（约2年）
  TEST_WINDOW  = 63  日（约3个月）
  STEP         = 63  日（非重叠测试窗口）

输出：facts/runs/<run_id>/rolling_update/
  fold_XX/checkpoints/    ← 各 Fold 模型权重
  wf_report.csv           ← 各 Fold 绩效明细
  wf_summary.csv          ← 汇总统计
  wf_result.png           ← 可视化
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 路径配置 ─────────────────────────────────────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_MSGNET_DIR  = os.path.join(_PROJECT_DIR, "MSGNet-main")
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data", "etf20")
from run_config import get_subdir

_WF_DIR      = get_subdir("rolling_update")

# MSGNet 需要在其目录下运行
os.chdir(_MSGNET_DIR)
sys.path.insert(0, _MSGNET_DIR)
from exp.exp_main import Exp_Main        # noqa: E402
from models.MSGNet import Model          # noqa: E402
from data_provider.data_factory import data_provider  # noqa: E402

# ── 超参 ─────────────────────────────────────────────────────────
TRAIN_WINDOW = 500   # 训练窗口（交易日）
TEST_WINDOW  = 63    # 测试窗口（交易日）
STEP         = 63    # 滚动步长（与 TEST_WINDOW 相同 → 非重叠）

REBAL_FREQ   = 5
TOP_PCT      = 0.20
COST_BPS     = 10
ANNUAL_DAYS  = 252

IC_AMP = 0.2738
IC_PS  = 0.1525

SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ════════════════════════════════════════════════════════════════
# Fold 数据管理
# ════════════════════════════════════════════════════════════════

def fold_dirs(fold: int):
    """返回该 Fold 的数据目录和 checkpoint 目录"""
    data_dir = os.path.join(_WF_DIR, f"fold_{fold:02d}", "data")
    ckpt_dir = os.path.join(_WF_DIR, f"fold_{fold:02d}", "checkpoints")
    return data_dir, ckpt_dir


def write_fold_data(returns: pd.DataFrame, price: pd.DataFrame,
                   train_idx: pd.Index, fold: int):
    """把训练窗口的 returns 和 price 写入 Fold 专属目录"""
    data_dir, _ = fold_dirs(fold)
    os.makedirs(data_dir, exist_ok=True)

    train_ret = returns.loc[train_idx]
    train_ret.reset_index().rename(columns={"index": "date"}).to_csv(
        os.path.join(data_dir, "returns.csv"), index=False
    )
    np.save(
        os.path.join(data_dir, "returns.npy"),
        train_ret.values.astype(np.float32)
    )
    # price 供 Beta 计算用（训练窗口）
    price.loc[train_idx].reset_index().rename(columns={"index": "date"}).to_csv(
        os.path.join(data_dir, "price.csv"), index=False
    )


# ════════════════════════════════════════════════════════════════
# 训练
# ════════════════════════════════════════════════════════════════

def make_args(data_dir: str, ckpt_dir: str, n_etf: int,
              etf_names: list, fold: int) -> argparse.Namespace:
    return argparse.Namespace(
        task_name="long_term_forecast", is_training=1,
        model_id=f"WF{fold:02d}", model="MSGNet",
        data="custom",
        root_path=data_dir,
        data_path="returns.csv",
        features="M", target=etf_names[-1], freq="b",
        checkpoints=ckpt_dir,
        seasonal_patterns="Monthly",
        seq_len=60, label_len=30, pred_len=5,
        top_k=3, num_kernels=6,
        num_nodes=n_etf, subgraph_size=5, tanhalpha=3.0,
        node_dim=10, gcn_depth=2, gcn_dropout=0.3,
        propalpha=0.3, conv_channel=32, skip_channel=32,
        enc_in=n_etf, dec_in=n_etf, c_out=n_etf,
        d_model=64, n_heads=4, e_layers=2, d_layers=1,
        d_ff=512, moving_avg=25, factor=1, distil=True,
        dropout=0.1, embed="timeF", embed_type=0,
        activation="gelu", output_attention=False,
        do_predict=False, individual=False,
        num_workers=0, itr=1,
        train_epochs=50, batch_size=32, patience=10,
        learning_rate=0.0001, des="etf_factor", loss="MSE",
        lradj="type1", use_amp=False,
        use_gpu=torch.cuda.is_available(), gpu=0,
        use_multi_gpu=False, devices="0", test_flop=False,
    )


def train_fold(args: argparse.Namespace) -> str:
    setting = (
        f"{args.model_id}_MSGNet_custom"
        f"_ftM_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}"
        f"_tk{args.top_k}_{args.des}_0"
    )
    exp = Exp_Main(args)
    exp.train(setting)
    return setting


# ════════════════════════════════════════════════════════════════
# 因子提取（图结构 + 周期，跳过 Embedding 加速）
# ════════════════════════════════════════════════════════════════

def get_adjacency(graph_block) -> np.ndarray:
    v1 = graph_block.nodevec1
    v2 = graph_block.nodevec2
    with torch.no_grad():
        adp = F.softmax(F.relu(torch.mm(v1, v2)), dim=1)
    return adp.cpu().numpy()


def pagerank(adj: np.ndarray, d: float = 0.85, max_iter: int = 100) -> np.ndarray:
    N = adj.shape[0]
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = adj / row_sum
    pr = np.ones(N) / N
    for _ in range(max_iter):
        pr = d * P.T @ pr + (1 - d) / N
    return pr


def extract_factors(args: argparse.Namespace, setting: str,
                    etf_names: list) -> dict:
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")

    model = Model(args).float().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ── 图结构因子 ───────────────────────────────────────────────
    graph_records = {n: {} for n in etf_names}
    all_adjs = []
    for layer_idx, scale_block in enumerate(model.model):
        layer_adjs = []
        for scale_idx, gconv in enumerate(scale_block.gconv):
            adj = get_adjacency(gconv)
            tag = f"L{layer_idx+1}_S{scale_idx+1}"
            in_deg = adj.sum(axis=0)
            pr     = pagerank(adj)
            for i, name in enumerate(etf_names):
                graph_records[name][f"in_degree_{tag}"] = in_deg[i]
                graph_records[name][f"pagerank_{tag}"]  = pr[i]
            layer_adjs.append(adj)
        all_adjs.append(layer_adjs)

    for layer_idx, layer_adjs in enumerate(all_adjs):
        scale_diff = np.abs(layer_adjs[0] - layer_adjs[-1]).mean(axis=1)
        for i, name in enumerate(etf_names):
            graph_records[name][f"scale_diff_L{layer_idx+1}"] = scale_diff[i]

    graph_df = pd.DataFrame.from_dict(graph_records, orient="index")
    graph_df.index.name = "ETF"

    # ── 周期因子（FFT）───────────────────────────────────────────
    returns = np.load(os.path.join(args.root_path, "returns.npy"))
    T, N = returns.shape
    period_records = {}
    for i, name in enumerate(etf_names):
        x  = returns[:, i]
        xf = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(T)
        power = np.abs(xf)
        power[0] = 0
        top3_idx     = np.argsort(power)[-3:][::-1]
        top3_periods = np.where(freqs[top3_idx] > 0, 1/freqs[top3_idx], T)
        period_records[name] = {
            "dominant_period":  top3_periods[0],
            "period_stability": 1 / (np.std(top3_periods) + 1e-8),
            "amplitude":        power[top3_idx[0]],
            "period_2nd":       top3_periods[1],
            "period_3rd":       top3_periods[2],
        }

    period_df = pd.DataFrame.from_dict(period_records, orient="index")
    period_df.index.name = "ETF"

    return {"graph": graph_df, "period": period_df}


# ════════════════════════════════════════════════════════════════
# Beta 中性化
# ════════════════════════════════════════════════════════════════

def calc_betas(price_ret: pd.DataFrame) -> pd.Series:
    mkt = price_ret.mean(axis=1)
    var_mkt = mkt.var()
    betas = {col: price_ret[col].cov(mkt) / var_mkt if var_mkt > 0 else 1.0
             for col in price_ret.columns}
    return pd.Series(betas, name="beta")


def neutralize(factor: pd.Series, risk: pd.Series) -> pd.Series:
    common = factor.index.intersection(risk.index)
    f = factor.reindex(common).values.reshape(-1, 1)
    r = risk.reindex(common).values.reshape(-1, 1)
    reg      = LinearRegression().fit(r, f)
    residual = f.flatten() - reg.predict(r).flatten()
    return pd.Series(residual, index=common, name=factor.name)


# ════════════════════════════════════════════════════════════════
# 组合构建与绩效
# ════════════════════════════════════════════════════════════════

def build_composite(period_df: pd.DataFrame) -> pd.Series:
    amp = period_df["amplitude"]
    ps  = period_df["period_stability"]
    w_amp = IC_AMP / (IC_AMP + IC_PS)
    w_ps  = IC_PS  / (IC_AMP + IC_PS)
    composite = -amp * w_amp + ps * w_ps
    return composite.rank(pct=True)


def window_perf(ret: pd.Series) -> dict:
    if len(ret) < 5:
        return {}
    cum     = (1 + ret).cumprod()
    total   = cum.iloc[-1] - 1
    years   = len(ret) / ANNUAL_DAYS
    ann_ret = (1 + total) ** (1/years) - 1 if years > 0 else 0
    ann_vol = ret.std() * np.sqrt(ANNUAL_DAYS)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd  = (cum / cum.cummax() - 1).min()
    win_rt  = (ret > 0).mean()
    return dict(
        start    = str(ret.index[0].date()),
        end      = str(ret.index[-1].date()),
        ann_ret  = round(ann_ret * 100, 2),
        ann_vol  = round(ann_vol * 100, 2),
        sharpe   = round(sharpe, 3),
        max_dd   = round(max_dd * 100, 2),
        win_rate = round(win_rt * 100, 2),
    )


# ════════════════════════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════════════════════════

def plot_results(all_ls_ret: pd.Series, report: pd.DataFrame):
    os.makedirs(_WF_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # 1. 拼接后的多空组合累计净值
    ax = axes[0]
    cum = (1 + all_ls_ret).cumprod()
    cum.plot(ax=ax, color="#e74c3c", linewidth=1.5)
    ax.axhline(1.0, color="black", linewidth=0.5)
    for _, row in report.iterrows():
        ax.axvline(pd.Timestamp(row["start"]), color="gray",
                   linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("滚动更新 — 多空组合累计净值")
    ax.set_ylabel("净值")

    # 2. 各 Fold 年化收益
    ax = axes[1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in report["ann_ret"]]
    ax.bar(report.index, report["ann_ret"], color=colors, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(report["ann_ret"].mean(), color="#e67e22", linewidth=1.2,
               linestyle="--", label=f"均值 {report['ann_ret'].mean():+.1f}%")
    ax.set_title("各 Fold 年化收益（%）")
    ax.set_xlabel("Fold")
    ax.set_ylabel("年化收益 %")
    ax.legend(fontsize=9)

    # 3. 各 Fold Sharpe
    ax = axes[2]
    colors_s = ["#2ecc71" if v > 0 else "#e74c3c" for v in report["sharpe"]]
    ax.bar(report.index, report["sharpe"], color=colors_s, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(report["sharpe"].mean(), color="#e67e22", linewidth=1.2,
               linestyle="--", label=f"均值 {report['sharpe'].mean():+.3f}")
    ax.set_title("各 Fold Sharpe")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(_WF_DIR, "wf_result.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"\n图表已保存: {out_path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 滚动更新 Walk-Forward ===")
    print(f"  训练窗口: {TRAIN_WINDOW}日  测试窗口: {TEST_WINDOW}日  步长: {STEP}日\n")

    # ── 加载全量数据 ─────────────────────────────────────────────
    returns_path = os.path.join(_DATA_DIR, "returns.csv")
    price_path   = os.path.join(_DATA_DIR, "price.csv")
    returns = pd.read_csv(returns_path, parse_dates=["date"], index_col="date")
    price   = pd.read_csv(price_path,   parse_dates=["date"], index_col="date")
    price_ret = price.pct_change().dropna()

    etf_names = [c for c in returns.columns if c != "date"]
    N_ETF     = len(etf_names)
    T         = len(returns)

    # ── 计算 Fold 数量 ───────────────────────────────────────────
    fold_starts = list(range(0, T - TRAIN_WINDOW - TEST_WINDOW + 1, STEP))
    n_folds     = len(fold_starts)
    print(f"  总交易日: {T}  →  {n_folds} 个 Fold\n")

    results    = []
    all_ls_ret = []

    for fold, start in enumerate(fold_starts):
        train_end  = start + TRAIN_WINDOW
        test_end   = train_end + TEST_WINDOW
        train_idx  = returns.index[start:train_end]
        test_idx   = returns.index[train_end:test_end]

        print(f"\n{'─'*55}")
        print(f"  Fold {fold+1:02d}/{n_folds}"
              f"  train [{train_idx[0].date()} ~ {train_idx[-1].date()}]"
              f"  test [{test_idx[0].date()} ~ {test_idx[-1].date()}]")
        print(f"{'─'*55}")

        # 1. 写入 Fold 数据
        write_fold_data(returns, price, train_idx, fold)
        data_dir, ckpt_dir = fold_dirs(fold)

        # 2. 训练
        args    = make_args(data_dir, ckpt_dir, N_ETF, etf_names, fold)
        setting = train_fold(args)

        # 3. 提取因子
        factors = extract_factors(args, setting, etf_names)

        # 4. Beta 中性化（用训练窗口的价格收益率计算 Beta）
        train_price_ret = price_ret.loc[
            price_ret.index.intersection(train_idx)
        ]
        betas       = calc_betas(train_price_ret)
        amp_neutral = neutralize(factors["period"]["amplitude"], betas)
        factors["period"]["amplitude"] = amp_neutral.reindex(
            factors["period"].index
        )

        # 5. 合成因子，构建组合
        composite    = build_composite(factors["period"])
        common       = composite.index.intersection(price_ret.columns)
        composite    = composite.reindex(common).dropna()
        top_n        = max(1, int(len(composite) * TOP_PCT))
        long_stocks  = composite.nlargest(top_n).index
        short_stocks = composite.nsmallest(top_n).index

        # 6. 测试窗口绩效
        test_ret  = price_ret.loc[price_ret.index.intersection(test_idx)]
        test_ret  = test_ret[composite.index]
        long_ret  = test_ret[long_stocks].mean(axis=1)
        short_ret = test_ret[short_stocks].mean(axis=1)
        ls_ret    = long_ret - short_ret
        cost_pd   = (COST_BPS * 2 / 10000) / REBAL_FREQ
        ls_ret    = ls_ret - cost_pd

        perf = window_perf(ls_ret)
        perf["fold"] = fold + 1
        results.append(perf)
        all_ls_ret.append(ls_ret)

        print(f"  多头: {list(long_stocks)}")
        print(f"  空头: {list(short_stocks)}")
        print(f"  年化: {perf.get('ann_ret', 'N/A'):+}%"
              f"  Sharpe: {perf.get('sharpe', 'N/A')}")

    # ── 汇总 ────────────────────────────────────────────────────
    report      = pd.DataFrame(results).set_index("fold")
    all_ls_ret  = pd.concat(all_ls_ret).sort_index()
    pos_folds   = (report["ann_ret"] > 0).sum()

    print(f"\n{'='*55}")
    print(f"  滚动更新 Walk-Forward 汇总（{n_folds} 个 Fold）")
    print(f"{'='*55}")
    print(f"  正收益 Fold 数: {pos_folds} / {n_folds}"
          f"  ({pos_folds/n_folds*100:.0f}%)")
    print(f"  平均年化收益  : {report['ann_ret'].mean():+.2f}%")
    print(f"  平均 Sharpe   : {report['sharpe'].mean():+.3f}")
    print(f"  Sharpe 标准差 : {report['sharpe'].std():.3f}")

    # ── 保存 ────────────────────────────────────────────────────
    os.makedirs(_WF_DIR, exist_ok=True)
    report.to_csv(os.path.join(_WF_DIR, "wf_report.csv"), encoding="utf-8-sig")

    summary_rows = {
        "正收益Fold占比(%)": pos_folds / n_folds * 100,
        "平均年化收益(%)":   report["ann_ret"].mean(),
        "平均Sharpe":        report["sharpe"].mean(),
        "Sharpe标准差":      report["sharpe"].std(),
        "最差Fold年化(%)":   report["ann_ret"].min(),
        "最好Fold年化(%)":   report["ann_ret"].max(),
    }
    pd.Series(summary_rows).to_csv(
        os.path.join(_WF_DIR, "wf_summary.csv"), encoding="utf-8-sig"
    )

    plot_results(all_ls_ret, report)
    print("\n=== 完成 ===")
