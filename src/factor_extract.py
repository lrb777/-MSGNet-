# -*- coding: utf-8 -*-
"""
阶段三：因子提取

从训练好的 MSGNet 中提取三类因子：
  1. 图结构因子：邻接矩阵 → 入度/出度/PageRank/尺度差异
  2. Embedding 因子：最后一层隐层表示 → PCA 降维
  3. 周期因子：FFT 识别的主周期、稳定性、振幅

输出：facts/runs/<run_id>/factors/
  graph_factors.csv     ← 图结构因子（每只ETF一行）
  embedding_factors.csv ← PCA embedding 因子
  period_factors.csv    ← 周期因子
  adjacency/            ← 原始邻接矩阵 npy 文件
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── 路径配置 ────────────────────────────────────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_MSGNET_DIR  = os.path.join(_PROJECT_DIR, "MSGNet-main")
DATA_VARIANT = "etf20"
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data", DATA_VARIANT)
from run_config import get_run_dir, get_subdir

_RUN_DIR     = get_run_dir(create=True)
_CKPT_DIR    = get_subdir("checkpoints")
_FACTOR_DIR  = get_subdir("factors")
_ADJ_DIR     = os.path.join(_FACTOR_DIR, "adjacency")

os.chdir(_MSGNET_DIR)
sys.path.insert(0, _MSGNET_DIR)

from models.MSGNet import Model
from data_provider.data_factory import data_provider

# 动态读取股票名称和数量
_cols = pd.read_csv(os.path.join(_DATA_DIR, "returns.csv"), nrows=0).columns.tolist()
ETF_NAMES = [c for c in _cols if c != "date"]
N_ETF  = len(ETF_NAMES)
N_COMP = 10   # PCA 保留维度

SETTING = "ETF20_MSGNet_custom_ftM_sl60_ll30_pl5_dm64_nh4_el2_tk3_etf_factor_0"
CKPT_PATH = os.path.join(_CKPT_DIR, SETTING, "checkpoint.pth")

# ── 模型配置（与 train.py 保持一致）──────────────────────────────
args = argparse.Namespace(
    task_name="long_term_forecast",
    model_id="ETF20", model="MSGNet",
    data="custom",
    root_path=_DATA_DIR,
    data_path="returns.csv",
    features="M", target=ETF_NAMES[-1], freq="b",
    seasonal_patterns="Monthly",
    seq_len=60, label_len=30, pred_len=5,
    top_k=3, num_kernels=6,
    num_nodes=N_ETF, subgraph_size=5, tanhalpha=3.0,
    node_dim=10, gcn_depth=2, gcn_dropout=0.3,
    propalpha=0.3, conv_channel=32, skip_channel=32,
    enc_in=N_ETF, dec_in=N_ETF, c_out=N_ETF,
    d_model=64, n_heads=4, e_layers=2, d_layers=1,
    d_ff=512, moving_avg=25, factor=1, distil=True,
    dropout=0.1, embed="timeF", embed_type=0,
    activation="gelu", output_attention=False,
    do_predict=False, individual=False,
    num_workers=0, batch_size=32,
    use_gpu=torch.cuda.is_available(), gpu=0,
)

DEVICE = torch.device("cuda:0" if args.use_gpu else "cpu")


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def load_model() -> Model:
    model = Model(args).float().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[OK] 模型加载: {CKPT_PATH}")
    return model


def get_adjacency(graph_block) -> np.ndarray:
    """从 GraphBlock 提取邻接矩阵 [N, N]"""
    v1 = graph_block.nodevec1   # [N, node_dim]
    v2 = graph_block.nodevec2   # [node_dim, N]
    with torch.no_grad():
        adp = F.softmax(F.relu(torch.mm(v1, v2)), dim=1)
    return adp.cpu().numpy()


def pagerank(adj: np.ndarray, d: float = 0.85, max_iter: int = 100) -> np.ndarray:
    """简单 PageRank 迭代，adj 已归一化"""
    N = adj.shape[0]
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = adj / row_sum
    pr = np.ones(N) / N
    for _ in range(max_iter):
        pr = d * P.T @ pr + (1 - d) / N
    return pr


# ════════════════════════════════════════════════════════════════
# 因子一：图结构因子
# ════════════════════════════════════════════════════════════════

def extract_graph_factors(model: Model) -> pd.DataFrame:
    print("\n[1/3] 提取图结构因子...")
    os.makedirs(_ADJ_DIR, exist_ok=True)

    records = {name: {} for name in ETF_NAMES}
    all_adjs = []  # 收集所有邻接矩阵用于计算尺度差异

    for layer_idx, scale_block in enumerate(model.model):
        layer_adjs = []
        for scale_idx, gconv in enumerate(scale_block.gconv):
            adj = get_adjacency(gconv)
            tag = f"L{layer_idx+1}_S{scale_idx+1}"

            # 保存原始邻接矩阵
            np.save(os.path.join(_ADJ_DIR, f"adj_{tag}.npy"), adj)

            # 入度/出度/PageRank
            in_deg  = adj.sum(axis=0)         # 列求和 → 被影响程度
            out_deg = adj.sum(axis=1)          # 行求和 → 影响他人程度
            pr      = pagerank(adj)

            for i, name in enumerate(ETF_NAMES):
                records[name][f"in_degree_{tag}"]  = in_deg[i]
                records[name][f"out_degree_{tag}"] = out_deg[i]
                records[name][f"pagerank_{tag}"]   = pr[i]

            layer_adjs.append(adj)
        all_adjs.append(layer_adjs)

    # 尺度差异因子：同一层内长短周期邻接矩阵的差异
    for layer_idx, layer_adjs in enumerate(all_adjs):
        adj_s1 = layer_adjs[0]   # 短周期（FFT top1）
        adj_s3 = layer_adjs[-1]  # 长周期（FFT top3）
        scale_diff = np.abs(adj_s1 - adj_s3).mean(axis=1)
        for i, name in enumerate(ETF_NAMES):
            records[name][f"scale_diff_L{layer_idx+1}"] = scale_diff[i]

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "ETF"
    print(f"  图结构因子维度: {df.shape}  (邻接矩阵已保存至 {_ADJ_DIR})")
    return df


# ════════════════════════════════════════════════════════════════
# 因子二：Embedding 因子（滚动前向传播 → PCA）
# ════════════════════════════════════════════════════════════════

def extract_embedding_factors(model: Model) -> pd.DataFrame:
    print("\n[2/3] 提取 Embedding 因子...")
    from sklearn.decomposition import PCA

    # 用全量数据集（不分 train/val/test）跑一遍 forward
    # 通过 train+val+test 拼在一起，用 pred 模式拿到滚动 embedding
    _, loader = data_provider(args, "train")

    all_embeddings = []
    all_dates = []

    # 读 CSV 获取日期索引
    df_raw = pd.read_csv(os.path.join(_DATA_DIR, "returns.csv"),
                         parse_dates=["date"])
    dates = df_raw["date"].values

    # 钩子：捕获最后一层 ScaleGraphBlock 的输出
    captured = {}
    def hook_fn(module, input, output):
        captured["enc_out"] = output.detach().cpu()

    handle = model.model[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x      = batch_x.float().to(DEVICE)
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y      = batch_y.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(DEVICE)
            _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # enc_out: [B, T, d_model] → 取最后一个时间步 → [B, d_model]
            emb = captured["enc_out"][:, -1, :]    # [B, d_model]
            all_embeddings.append(emb.numpy())

    handle.remove()

    embeddings = np.concatenate(all_embeddings, axis=0)  # [samples, d_model]
    print(f"  原始 embedding 形状: {embeddings.shape}")

    # PCA 降维到 N_COMP 维
    pca = PCA(n_components=N_COMP, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA {N_COMP} 维累计方差解释率: {explained:.1%}")

    cols = [f"EMB_{i+1}" for i in range(N_COMP)]
    df = pd.DataFrame(emb_pca, columns=cols)
    # embedding 是 batch 级别（每个样本对应一个时间窗口末尾），不是 ETF 级别
    # 此处保存为时序因子（每个交易日一行）
    n_samples = len(df)
    sample_dates = dates[args.seq_len: args.seq_len + n_samples]
    df.insert(0, "date", sample_dates)
    print(f"  Embedding 因子形状: {df.shape}")
    return df


# ════════════════════════════════════════════════════════════════
# 因子三：周期因子（FFT）
# ════════════════════════════════════════════════════════════════

def extract_period_factors() -> pd.DataFrame:
    print("\n[3/3] 提取周期因子...")
    returns = np.load(os.path.join(_DATA_DIR, "returns.npy"))  # [T, N]
    T, N = returns.shape

    records = {}
    for i, name in enumerate(ETF_NAMES):
        x = returns[:, i]
        xf = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(T)
        power = np.abs(xf)
        power[0] = 0   # 去直流分量

        top3_idx    = np.argsort(power)[-3:][::-1]
        top3_periods = np.where(freqs[top3_idx] > 0, 1 / freqs[top3_idx], T)

        dominant_period   = top3_periods[0]
        period_stability  = 1 / (np.std(top3_periods) + 1e-8)
        amplitude         = power[top3_idx[0]]

        records[name] = {
            "dominant_period":  dominant_period,
            "period_stability": period_stability,
            "amplitude":        amplitude,
            "period_2nd":       top3_periods[1],
            "period_3rd":       top3_periods[2],
        }

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "ETF"
    print(f"  周期因子维度: {df.shape}")
    return df


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(_FACTOR_DIR, exist_ok=True)

    model = load_model()

    graph_df   = extract_graph_factors(model)
    embed_df   = extract_embedding_factors(model)
    period_df  = extract_period_factors()

    # 保存
    graph_df.to_csv(os.path.join(_FACTOR_DIR, "graph_factors.csv"))
    embed_df.to_csv(os.path.join(_FACTOR_DIR, "embedding_factors.csv"), index=False)
    period_df.to_csv(os.path.join(_FACTOR_DIR, "period_factors.csv"))

    print("\n=== 因子提取完成 ===")
    print(f"  图结构因子   : {graph_df.shape}  → {_FACTOR_DIR}/graph_factors.csv")
    print(f"  Embedding 因子: {embed_df.shape}  → {_FACTOR_DIR}/embedding_factors.csv")
    print(f"  周期因子     : {period_df.shape}  → {_FACTOR_DIR}/period_factors.csv")
    print(f"  邻接矩阵     : {_ADJ_DIR}/adj_L*_S*.npy")
