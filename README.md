# MSGNet 因子挖掘

基于 MSGNet 的 ETF20 关系结构因子挖掘与多因子回测项目。

项目目标是利用 MSGNet 的多尺度图学习能力，从多变量 ETF 时序数据中自动提取传统线性方法难以发现的关系类因子，并接入多因子选股/择 ETF 框架验证有效性。

## 当前状态

全流程已跑通，核心结论已验证：

- MSGNet 可以提取 ETF 之间的关系结构因子。
- 低振幅、较高周期稳定性一类信号对多头 ETF 选择有效。
- ETF20 多空组合结构不可行，主要问题来自空头侧高 Beta 暴露。
- 静态 Walk-Forward 与滚动重训练 Walk-Forward 的结论一致。

## 研究流程

### 1. 数据准备

标的池采用 20 只行业/宽基 ETF。选择 ETF20 的原因：

- 噪声低于个股，ETF 之间关系更稳定。
- MSGNet GraphBlock 要求 `d_model >= N`。ETF20 使用 `d_model=64` 时感受野充足。
- CSI300 个股池虽然截面更大，但 `d_model` 约束会明显增加训练压力。

数据处理流程：

- 使用 `adata.fund.market.get_market_etf` 获取 ETF 行情。
- 通过 `change_pct` 链式重建前复权价格。
- 自动检测所有 ETF 共有有效起点 `EFFECTIVE_START`。
- 生成真实价格表 `data/etf20/price.csv`。
- 将价格转为截面标准化收益率，生成 MSGNet 输入：
  - `data/etf20/returns.csv`
  - `data/etf20/returns.npy`

### 2. MSGNet 训练

核心配置：

```yaml
seq_len: 60
label_len: 30
pred_len: 5
d_model: 64
n_heads: 4
e_layers: 2
top_k: 3
subgraph_size: 5
batch_size: 32
epochs: 50
patience: 10
num_workers: 0
```

训练目标不是单纯追求预测精度，而是让 MSGNet 学到具有经济含义的多尺度邻接矩阵。

### 3. 因子提取

从训练好的 MSGNet 中提取三类因子：

| 类型 | 因子 | 含义 |
|------|------|------|
| 图结构因子 | `in_degree` | 被其他 ETF 影响的程度 |
| 图结构因子 | `pagerank` | 在 ETF 关系网络中的重要程度 |
| 图结构因子 | `scale_diff` | 长短周期关系结构差异 |
| 周期因子 | `dominant_period` | 主周期长度 |
| 周期因子 | `period_stability` | 周期稳定性 |
| 周期因子 | `amplitude` | 主周期振幅 |
| Embedding 因子 | `EMB_1~10` | 隐层表示 PCA 降维 |

`out_degree` 已排除，因为 softmax 行归一化后该值恒约等于 1，缺少截面信息量。

### 4. 因子有效性验证

验证包含两套口径。

第一套是 IC 检验：

```text
|IC均值| > 0.03
|IC_IR| > 0.5
```

已通过的代表性因子：

| 因子 | 最佳预测期 | IC均值 | IC_IR | 方向 |
|------|-----------|--------|-------|------|
| `amplitude` | h=20 | -0.280 | -1.405 | 反向 |
| `in_degree_L2_S3` | h=10 | -0.213 | -1.065 | 反向 |
| `pagerank_L2_S3` | h=10 | -0.198 | -0.970 | 反向 |
| `period_stability` | h=20 | +0.180 | +0.897 | 正向 |
| `scale_diff_L1` | h=20 | +0.155 | +0.676 | 正向 |

第二套是 Beta 控制单因子收益显著性检验：

```text
future_ret = 常数项 + Beta暴露 + 因子残差暴露 + 残差
```

显著性门槛：

```text
|Newey-West t值| >= 2 且 |IR| >= 0.5
```

代表性结果：

| 因子 | 最佳预测期 | 年化因子收益 | 因子IR | NW_t值 | 方向 |
|------|-----------|--------------|--------|--------|------|
| `dominant_period` | h=20 | -4.20% | -5.635 | -3.635 | 反向 |
| `amplitude` | h=20 | +3.31% | +4.536 | +2.838 | 正向 |
| `pagerank_L1_S2` | h=20 | +4.04% | +4.404 | +2.799 | 正向 |
| `in_degree_L1_S2` | h=20 | +4.01% | +3.965 | +2.552 | 正向 |
| `scale_diff_L2` | h=20 | -3.45% | -3.865 | -2.381 | 反向 |

### 5. 中性化与回测

`amplitude` 的 IC 方向稳定且容易含 Beta 暴露，因此对其做 Beta 中性化：

```text
amplitude = a + b * beta + residual
```

合成因子：

```text
composite = -amplitude_neutralized * w_amp + period_stability * w_ps
```

回测结果：

| 指标 | 全期多空 | 测试集多空 | 测试集多头 |
|------|----------|------------|------------|
| 年化收益 | -14.63% | -17.05% | +26.39% |
| Sharpe | -1.512 | -2.102 | 1.623 |
| 最大回撤 | -58.43% | -20.15% | -9.47% |
| 日胜率 | 47.20% | 45.91% | 55.64% |

结论：多头信号有效，但多空组合失败。

### 6. Walk-Forward 验证

静态 Walk-Forward：

- 20 个非重叠窗口
- 每个窗口 63 个交易日
- 正收益窗口：2 / 20
- 平均年化收益：约 -13.84%
- 平均 Sharpe：约 -1.44

滚动重训练 Walk-Forward：

- 训练窗口 500 日
- 测试窗口 63 日
- 12 个 Fold
- 正收益 Fold：2 / 12
- 平均年化收益：-12.04%
- 平均 Sharpe：-1.189

滚动重训练没有改变结论：问题在 ETF20 多空结构，而不是单纯模型或参数问题。

## 核心结论

1. MSGNet 能提取 ETF 关系结构，图结构因子和周期因子均有可验证信息量。
2. 低振幅、高周期稳定性一类信号适合做多头 ETF 选择。
3. ETF20 多空组合不适合直接交易，空头侧天然集中在高 Beta ETF。
4. 后续更合理方向是多头 ETF 组合 + 指数对冲，或扩展到 CSI300 个股池做更稳定的截面中性化。

## 运行方法

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

标准主流程：

```powershell
python .\src\data_pipeline.py
python .\src\preprocess.py
python .\src\train.py
python .\src\factor_extract.py
python .\src\factor_validate.py
python .\src\factor_return_test.py
python .\src\factor_neutralize.py
python .\src\backtest.py
python .\src\walk_forward.py
python .\src\report.py
```

滚动重训练验证：

```powershell
python .\src\rolling_update.py
python .\src\report.py
```

报告生成：

- `report.py` 会读取最新 `facts/runs/<run_id>/`，自动生成 HTML 和 PDF 分析报告。
- 报告包含研究结论文本、因子文字说明、少量公式、IC 检验、Beta 控制单因子收益检验、静态回测和 Walk-Forward 结果。
- 如本次 run 没有 `rolling_update/` 产物，报告会保留该章节并注明缺失。
- PDF 导出调用本机 Chrome/Edge headless；在受限沙箱中运行时，可能需要允许 Chrome 子进程执行。

结果默认保存到：

```text
facts/runs/<run_id>/
```

如果跳过训练但想新开一轮结果目录：

```powershell
python .\src\start_run.py
```

也可以手动指定运行 ID：

```powershell
$env:MSGNET_RUN_ID="manual_001"
```

## 目录结构

```text
MSGNet因子挖掘/
├── MSGNet-main/                # 原始 MSGNet 模型代码
├── src/
│   ├── data_pipeline.py        # ETF 数据获取与前复权价格重建
│   ├── preprocess.py           # 收益率和截面标准化
│   ├── train.py                # MSGNet 训练
│   ├── factor_extract.py       # 因子提取
│   ├── factor_validate.py      # IC 检验与分组分析
│   ├── factor_return_test.py   # Beta 控制单因子收益检验
│   ├── factor_neutralize.py    # Beta 中性化
│   ├── backtest.py             # 静态组合回测
│   ├── walk_forward.py         # 静态 Walk-Forward
│   ├── rolling_update.py       # 滚动重训练 Walk-Forward
│   ├── report.py               # 自动生成 HTML/PDF 分析报告
│   ├── run_config.py           # 统一结果目录管理
│   └── start_run.py            # 手动创建新运行目录
├── facts/runs/<run_id>/        # 运行结果，不纳入 Git
├── data/etf20/                 # 行情和预处理数据，不纳入 Git
├── MSGNet因子挖掘研究流程.md
├── 项目进度.md
├── 技术要点.txt
└── requirements.txt
```

## 说明

运行数据、模型权重、回测结果、缓存文件均通过 `.gitignore` 排除，不上传到 GitHub。
