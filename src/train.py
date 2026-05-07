# -*- coding: utf-8 -*-
"""
MSGNet 行业ETF训练入口

数据：data/etf20/returns.csv（N只ETF对数收益率，已截面标准化）
输出：facts/runs/<run_id>/checkpoints/<setting>/checkpoint.pth

超参说明：
  seq_len=60   → 回看60个交易日（约3个月）
  pred_len=5   → 预测未来5日（周频信号）
  top_k=3      → 识别3个时间尺度（FFT自动选取）
  d_model=64   → GraphBlock kernel=(64-N+1,1)，N≤20 时感受野充裕
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch

# ── 路径配置 ───────────────────────────────────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_MSGNET_DIR  = os.path.join(_PROJECT_DIR, "MSGNet-main")
DATA_VARIANT = "etf20"
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data", DATA_VARIANT)
from run_config import get_run_dir, get_subdir

_RUN_DIR     = get_run_dir(create=True, new=True)
_CKPT_DIR    = get_subdir("checkpoints")

# MSGNet 内部使用相对 import，必须切换到其目录
os.chdir(_MSGNET_DIR)
sys.path.insert(0, _MSGNET_DIR)

from exp.exp_main import Exp_Main   # noqa: E402（切换目录后才能 import）

# ── 随机种子 ────────────────────────────────────────────────────
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 动态读取节点数（列数 = 股票数，减去 date 列）───────────────────
_price_csv = os.path.join(_DATA_DIR, "returns.csv")
_cols = pd.read_csv(_price_csv, nrows=0).columns.tolist()
_stock_cols = [c for c in _cols if c != "date"]
N_ETF = len(_stock_cols)
_TARGET_COL = _stock_cols[-1]   # data_loader 要求 target 为非 date 的真实列名
print(f"  动态节点数: {N_ETF} 只股票")

args = argparse.Namespace(
    # 任务
    task_name   = "long_term_forecast",
    is_training = 1,
    model_id    = "ETF20",
    model       = "MSGNet",

    # 数据
    data        = "custom",
    root_path   = _DATA_DIR,
    data_path   = "returns.csv",
    features    = "M",          # 多变量预测多变量
    target      = _TARGET_COL,  # data_loader 会 remove(target) 再 remove('date')，不能设为 'date'
    freq        = "b",          # business day
    checkpoints = _CKPT_DIR,
    seasonal_patterns = "Monthly",

    # 序列长度
    seq_len     = 60,           # 回看窗口
    label_len   = 30,           # decoder start token（seq_len 的一半）
    pred_len    = 5,            # 预测未来5日

    # MSGNet 图结构
    top_k         = 3,          # FFT 识别的时间尺度数
    num_kernels   = 6,
    num_nodes     = N_ETF,      # 图节点数 = 股票数（动态）
    subgraph_size = 5,          # 每个节点的邻居数（20只ETF取5，约25%密度）
    tanhalpha     = 3.0,

    # GCN
    node_dim     = 10,
    gcn_depth    = 2,
    gcn_dropout  = 0.3,
    propalpha    = 0.3,
    conv_channel = 32,
    skip_channel = 32,

    # Transformer 结构
    enc_in       = N_ETF,
    dec_in       = N_ETF,
    c_out        = N_ETF,
    d_model      = 64,    # GraphBlock 要求 d_model >= c_out=N_ETF；64 >> 20，kernel=(45,1) 感受野充裕
    n_heads      = 4,     # 64 / 4 = 16
    e_layers     = 2,
    d_layers     = 1,
    d_ff         = 512,
    moving_avg   = 25,
    factor       = 1,
    distil       = True,
    dropout      = 0.1,
    embed        = "timeF",
    embed_type   = 0,
    activation   = "gelu",
    output_attention = False,
    do_predict   = False,
    individual   = False,

    # 训练
    num_workers  = 0,           # Windows 下 DataLoader 多进程有坑，设0
    itr          = 1,
    train_epochs = 50,
    batch_size   = 32,          # d_model=64 × 20节点显存压力小，可用32
    patience     = 10,
    learning_rate = 0.0001,
    des          = "etf_factor",
    loss         = "MSE",
    lradj        = "type1",
    use_amp      = False,

    # GPU
    use_gpu      = torch.cuda.is_available(),
    gpu          = 0,
    use_multi_gpu = False,
    devices      = "0",
    test_flop    = False,
)

# ── 打印配置摘要 ────────────────────────────────────────────────
print("=" * 55)
print("MSGNet 行业ETF训练")
print("=" * 55)
print(f"  数据文件  : {os.path.join(_DATA_DIR, 'returns.csv')}")
print(f"  ETF 数量  : {N_ETF}")
print(f"  序列长度  : seq={args.seq_len}  label={args.label_len}  pred={args.pred_len}")
print(f"  时间尺度  : top_k={args.top_k}")
print(f"  模型维度  : d_model={args.d_model}  heads={args.n_heads}  layers={args.e_layers}")
print(f"  GPU       : {'可用 ✓' if args.use_gpu else 'CPU 模式'}")
print(f"  结果目录  : {_RUN_DIR}")
print(f"  Checkpoint: {_CKPT_DIR}")
print("=" * 55)

# ── 训练 ────────────────────────────────────────────────────────
setting = (
    f"{args.model_id}_{args.model}_{args.data}"
    f"_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
    f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}"
    f"_tk{args.top_k}_{args.des}_0"
)

exp = Exp_Main(args)
print(f"\n>>> 开始训练: {setting}\n")
exp.train(setting)

print(f"\n>>> 测试集评估\n")
exp.test(setting)

print("\n训练完成。")
print(f"模型已保存至: {os.path.join(_CKPT_DIR, setting, 'checkpoint.pth')}")
