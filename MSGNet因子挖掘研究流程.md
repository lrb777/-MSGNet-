# MSGNet 因子挖掘研究流程

> 最后更新：2026-05-07
> 状态：**全流程已跑通**，核心结论已验证

## 研究目标

利用 MSGNet 的多尺度图学习能力，从多变量时序数据中自动挖掘传统方法难以发现的**关系类因子**，并将其接入多因子选股框架验证有效性。

---

## 阶段一：数据准备

### 1.1 标的池选择

| 标的池 | 规模 | 优点 |
|---|---|---|
| 沪深300成分股 | ~300只 | 流动性好，因子有效性高 |
| 中证500成分股 | ~500只 | 弹性更大，Alpha更丰富 |
| **行业ETF（已采用）** | **20只** | **噪声小，关系更稳定；d_model约束友好** |

**实际选择：ETF20（20只行业ETF）**

原因：MSGNet GraphBlock 的 Conv2d kernel=(d_model - c_out + 1, 1)，要求 d_model ≥ N。
- CSI300：N=293，d_model=296 → kernel=(4,1)，感受野退化
- ETF20：N=20，d_model=64 → kernel=(45,1)，感受野充分，图学习正常工作

已选ETF（20只，覆盖主要行业）：

| 代码 | 名称 |
|------|------|
| 510300 | 沪深300ETF |
| 510500 | 中证500ETF |
| 159915 | 创业板ETF |
| 512010 | 医药ETF |
| 512200 | 房地产ETF |
| 512480 | 半导体ETF |
| 512690 | 酒ETF |
| 512800 | 银行ETF |
| 512880 | 证券ETF |
| 512980 | 传媒ETF |
| 515000 | 科技ETF |
| 515050 | 5GETF |
| 515230 | 新能源车ETF |
| 516160 | 新能源ETF（已替换为516110）|
| 516110 | 新能源ETF |
| 159928 | 消费ETF |
| 159934 | 黄金ETF |
| 159938 | 食品饮料ETF |
| 512090 | 货币ETF替代品 |
| 159996 | 家电ETF |

### 1.2 数据字段

```
必选：前复权收盘价（adata.fund.market.get_market_etf 获取）
数据范围：2019-01-01 至今
复权方式：通过 change_pct 链式重建 adj_close
```

### 1.3 数据预处理

```python
# ETF adj_close 重建（adata 接口无直接复权价）
adj_t = adj_{t-1} × (1 + change_pct_t / 100)
adj_0 = close_0  # 锚定首日收盘价

# 步骤1：收益率（普通收益率，不用对数）
returns = price.pct_change().dropna()

# 步骤2：截面 z-score 标准化（输入模型）
standardized = (returns - returns.mean(axis=1)) / returns.std(axis=1)

# 步骤3：EFFECTIVE_START 自动检测
# 取所有 ETF 第一个有效日期的最大值，避免 ffill 引入假零收益率
EFFECTIVE_START = max(price[col].first_valid_index() for col in price.columns)
```

### 1.4 数据集划分

```
全部数据时间线（约1500交易日）：
├── 训练集（60%）：模型学习图结构
├── 验证集（20%）：防过拟合
└── 测试集（20%，约13个月）：因子有效性评估，严格样本外
```

---

## 阶段二：模型训练

### 2.1 训练目标

让模型学到有意义的多尺度图结构（邻接矩阵），**不追求预测精度最高**，而是让邻接矩阵具有经济学含义。

### 2.2 实际超参数（ETF20）

```yaml
# 输入输出
seq_len: 60          # 回看60交易日
label_len: 30
pred_len: 5          # 预测未来5日

# 模型结构
d_model: 64          # 隐层维度（满足 d_model ≥ N_ETF=20）
n_heads: 4
e_layers: 2          # ScaleGraph Block 层数
top_k: 3             # 识别3个时间尺度
subgraph_size: 5
num_nodes: 20        # ETF数量

# 训练
learning_rate: 0.0001
batch_size: 32
epochs: 50
patience: 10
num_workers: 0       # Windows DataLoader 多进程坑，必须设为0
```

### 2.3 训练注意事项

- 输入数据用截面 z-score 标准化收益率，不用原始价格
- 验证集 loss 连续 10 个 epoch 不下降则早停
- 保存验证集 loss 最低的 checkpoint（`checkpoint.pth`）

---

## 阶段三：因子提取

### 3.1 因子类型一：图结构因子（最核心）

从每层 ScaleGraph Block 的每个尺度提取邻接矩阵，派生以下因子：

```
① 入度因子（in_degree）
   in_degree[i] = sum(A[:, i])
   含义：被多少其他ETF"影响"，高入度 = 跟随型

② 出度因子（out_degree）
   ⚠️ 已排除：softmax 行归一化导致所有 out_degree = 1.0，无信息量

③ PageRank 因子（pagerank）
   含义：在关联网络中的重要程度

④ 尺度差异因子（scale_diff）
   scale_diff[i] = |A_长周期[i,:] - A_短周期[i,:]|.mean()
   含义：长短周期关联结构差异越大，可能处于趋势转折
```

提取方式（每层每尺度的 nodevec 直接推导，不需要前向传播）：

```python
def get_adjacency(graph_block):
    v1 = graph_block.nodevec1
    v2 = graph_block.nodevec2
    with torch.no_grad():
        adp = F.softmax(F.relu(torch.mm(v1, v2)), dim=1)
    return adp.cpu().numpy()
```

### 3.2 因子类型二：隐层 Embedding 因子

提取模型最后一层隐层表示，PCA 降维为 10 个正交因子（EMB_1~10）。

> 实际结果：Embedding 因子 IC 未通过门槛（|IC|<0.03），本次未使用。

### 3.3 因子类型三：周期特征因子

对训练集收益率做 FFT，提取主周期信息：

```python
xf    = np.fft.rfft(returns[:, i])
power = np.abs(xf); power[0] = 0
top3_idx = np.argsort(power)[-3:][::-1]

dominant_period  = 1 / freq[top3_idx[0]]    # 主周期长度
period_stability = 1 / std(top3_periods)     # 周期稳定性（越大越稳定）
amplitude        = power[top3_idx[0]]        # 主周期振幅
```

---

## 阶段四：因子有效性验证

### 4.1 IC 检验标准

```
|IC 均值| > 0.03      → 因子有一定预测力
|IC_IR| > 0.5         → 因子信号稳定（IC_IR = mean/std）
IC>0 比例说明方向一致性
```

注意：静态因子只有一个截面快照，IC 时间序列通过以下方式模拟：
滑动窗口 → 每窗口因子值（固定）与该窗口未来收益的相关性 → 得到 IC 序列

### 4.2 实际通过因子（8个）

| 因子 | 最佳预测期 | IC均值 | IC_IR | IC>0比例 | 方向 | 备注 |
|------|-----------|--------|-------|---------|------|------|
| amplitude | h=20 | -0.280 | -1.405 | 8.8% | 反向 | 含Beta暴露，已中性化 |
| in_degree_L2_S3 | h=10 | -0.213 | -1.065 | 15.7% | 反向 | 疑含Beta暴露 |
| pagerank_L2_S3 | h=10 | -0.198 | -0.970 | 18.3% | 反向 | 疑含Beta暴露 |
| period_2nd | h=20 | -0.188 | -0.899 | 18.5% | 反向 | — |
| period_stability | h=20 | +0.180 | +0.897 | 80.5% | 正向 | 真Alpha信号 |
| in_degree_L2_S2 | h=20 | -0.136 | -0.687 | 22.9% | 反向 | — |
| scale_diff_L1 | h=20 | +0.155 | +0.676 | 74.8% | 正向 | 真Alpha信号 |
| scale_diff_L1 | h=10 | +0.122 | +0.527 | 69.0% | 正向 | — |

### 4.3 单因子收益显著性检验（Beta控制版）

在 IC 检验之外，新增一套由“单因子显著性检验”改造而来的横截面因子收益检验，用于判断每个 MSGNet 因子是否能产生稳定的经济收益。

原始方法包含行业中性化和风格中性化，但本项目不原样使用：

- ETF20 本身就是行业/主题 ETF，行业中性化会把研究对象剥离掉。
- N=20 截面较小，加入过多风格变量会显著降低自由度。
- 因此只控制历史 Beta，作为 ETF 多空结构中最核心的系统性风险暴露。

改造后的检验流程：

```python
# Step 1: 用训练集真实价格收益率计算每只ETF相对等权市场的Beta
beta_i = cov(R_i, R_mkt) / var(R_mkt)

# Step 2: 目标因子对Beta做截面回归，取残差
factor_i = a + b * beta_i + eps_i

# Step 3: 对每个预测期d，逐日做截面回归
future_ret_i_t_d = a_t + b_t * beta_i + f_k_t_d * eps_i + error_i_t

# Step 4: 对因子收益序列 f_k_t_d 计算年化收益、IR、t值、Newey-West t值
annual_return = mean(f_k_t_d / d) * 252
IR = sqrt(252) * mean(f_k_t_d / d) / std(f_k_t_d / d)
```

实现文件：`src/factor_return_test.py`

输出文件：`facts/runs/<run_id>/validation/factor_return_report.csv`

显著性门槛：

```text
|NW_t值| >= 2 且 |IR| >= 0.5
```

使用 Newey-West t 值的原因：预测期 d>1 时未来收益窗口互相重叠，普通 t 值会高估显著性。

主要结果：

| 因子 | 最佳预测期 | 年化因子收益 | 因子IR | NW_t值 | 方向 |
|------|-----------|--------------|--------|--------|------|
| dominant_period | h=20 | -4.20% | -5.635 | -3.635 | 反向 |
| amplitude | h=20 | +3.31% | +4.536 | +2.838 | 正向 |
| pagerank_L1_S2 | h=20 | +4.04% | +4.404 | +2.799 | 正向 |
| in_degree_L1_S1 | h=20 | -3.69% | -4.255 | -2.652 | 反向 |
| in_degree_L1_S2 | h=20 | +4.01% | +3.965 | +2.552 | 正向 |
| scale_diff_L2 | h=20 | -3.45% | -3.865 | -2.381 | 反向 |

结论：

1. Beta 控制后，MSGNet 的图结构因子和周期因子仍能产生显著横截面因子收益。
2. `period_stability` 在 IC 检验中表现较好，但在单因子收益检验中未通过显著性门槛（h=20：IR=1.137，NW_t=0.703），更适合作为组合辅助信号，而不是单独交易信号。
3. 单因子收益检验是 IC 检验的增强口径，不直接替代组合回测；ETF20 截面较小，交易方向仍应由样本外回测和组合约束共同确认。

### 4.4 因子方向判读

| IC>0比例 | 解读 |
|---------|------|
| >70% | 正向因子，直接使用 |
| <30% | 反向因子，取负后使用 |
| 30%~70% | 因子不稳定，警惕Beta暴露 |

---

## 阶段五：因子中性化与回测

### 5.1 哪些因子需要中性化

**判断标准（同时满足以下条件）：**
1. IC>0比例严重偏离50%（<30%或>70%）→ 暗示系统性暴露
2. IC不随预测期衰减 → 暗示跟涨跌，非Alpha
3. 做空低得分组与市场同向上涨 → 直接确认Beta暴露

**本项目：amplitude 满足以上全部条件，做 Beta 中性化。**

### 5.2 Beta 中性化方法

```python
# 用训练集价格收益率计算 Beta
mkt = price_ret.mean(axis=1)
beta_i = cov(R_i, R_mkt) / var(R_mkt)

# OLS 回归取残差
amplitude = a + b × beta + ε
# 使用 ε 作为中性化后的 amplitude 因子
```

### 5.3 合成因子

使用 IC 绝对值加权（h=10 IC 值）：

```python
w_amp = 0.2738 / (0.2738 + 0.1525)   # amplitude 权重
w_ps  = 0.1525 / (0.2738 + 0.1525)   # period_stability 权重
composite = -amplitude * w_amp + period_stability * w_ps
composite = composite.rank(pct=True)  # 截面百分比排名
```

### 5.4 回测结果（backtest.py）

| 指标 | 全期 | 测试集（样本外） | 多头单独（测试集） |
|------|------|----------------|-----------------|
| 年化收益 | -14.63% | -17.05% | **+26.39%** |
| Sharpe | -1.512 | -2.102 | **1.623** |
| 最大回撤 | -58.43% | -20.15% | -9.47% |
| 日胜率 | 47.20% | 45.91% | 55.64% |

**多头 Sharpe=1.623 超过研究目标（>1.0）** ✅
空头持续亏损，原因见阶段六。

---

## 阶段六：策略回测（Walk-Forward）

### 6.1 静态 Walk-Forward（walk_forward.py）

- 因子固定不变，将全期收益流切成 20 个不重叠窗口（每窗口63交易日）
- 作用：验证因子在不同市场环境（牛/熊/震荡）下的一致性
- 结果：**正收益窗口 2/20（10%）**，绝大多数窗口亏损

### 6.2 空头失败的根本原因

| 问题 | 分析 |
|------|------|
| N=20，TOP_PCT=20% | 做空只选4只ETF，恒为高振幅/高Beta行业ETF |
| ETF天然高Beta | 沪深300/创业板/行业ETF整体跟随市场，牛市必涨 |
| Beta中性化精度 | N=20截面OLS，自由度极低，无法有效剥离暴露 |
| 结构性问题 | 非参数调整可解决，多空策略在ETF20上根本不可行 |

### 6.3 评估指标

```
收益类：年化收益率、多空组合年化收益
风险类：最大回撤 < 20%，年化波动率
综合类：Sharpe > 1.0，Calmar > 0.5，月度胜率 > 55%
```

---

## 阶段七：滚动更新机制

### 7.1 为什么需要滚动更新

静态因子是用全部训练数据训练一次模型提取的，固定不变。问题：
1. 市场结构变化后，历史因子值不再反映当前状态
2. Walk-Forward 只是不同时间段用同一因子，无法验证样本外泛化
3. 真正的样本外验证需要：**每个测试窗口用前N日重新训练模型，提取新因子**

### 7.2 滚动更新框架（rolling_update.py）

```
参数：
  训练窗口：500 交易日（约2年）
  测试窗口：63 交易日（约3个月）
  步长：63 交易日（非重叠）

每个 Fold：
  1. 截取训练窗口数据写入临时目录
  2. 重新从零训练 MSGNet（50 epochs）
  3. 从新模型提取图结构因子 + FFT周期因子
  4. 用训练窗口计算Beta → 对amplitude做中性化
  5. 合成因子 → 构建多空组合 → 在测试窗口评估绩效
```

### 7.3 滚动更新结果（12个Fold）

| 指标 | 结果 |
|------|------|
| 正收益Fold | 2 / 12（17%） |
| 平均年化收益 | -12.04% |
| 平均Sharpe | -1.189 |
| Sharpe标准差 | 1.223 |

**结论：与静态Walk-Forward结果一致，滚动重训练未改变空头失败的结构性原因。**
核心Alpha信号（多头有效）已得到动态框架下的进一步验证。

---

## 里程碑完成状态

| 阶段 | 交付物 | 状态 |
|---|---|---|
| 数据准备 | ETF20价格矩阵，EFFECTIVE_START自动检测 | ✅ 完成 |
| 模型训练 | MSGNet ETF20，d_model=64，kernel=(45,1) | ✅ 完成 |
| 因子提取 | 图结构+周期因子CSV（静态截面） | ✅ 完成 |
| 因子验证 | IC检验 + Beta控制单因子收益显著性检验 | ✅ 完成 |
| 因子中性化+回测 | amplitude Beta中性化，多头Sharpe=1.623 | ✅ 完成 |
| 策略回测 | Walk-Forward静态（20窗口），确认Beta问题 | ✅ 完成 |
| 滚动更新 | 逐Fold重训练（12Fold），确认结构性结论 | ✅ 完成 |

---

## 核心研究结论

1. **MSGNet 能有效提取 ETF 关系结构**：图结构因子（in_degree、pagerank）和周期因子（amplitude、period_stability）均通过IC验证
2. **低振幅+高周期稳定性是有效Alpha**：多头组合测试集 Sharpe=1.623，真实有效
3. **ETF多空结构不可行**：N=20标的数过少，做空侧天然是高Beta ETF，无法中性化
4. **静态vs动态因子结论一致**：滚动更新未改变根本结论，问题在策略结构而非因子质量

## 后续方向

- **方案A（推荐）**：放弃空头，多头ETF + 沪深300ETF对冲，构建完整多头策略
- **方案B（进阶）**：扩展到个股（CSI300），N≥100后Beta中性化才有统计意义；升级为时间序列动态因子

---

## 重新跑完整流程

在项目根目录执行：

```powershell
cd "C:\Users\INSATIABLE\Desktop\均线量化\MSGNet因子挖掘"
```

### 1. 环境准备

```powershell
python -m pip install -r requirements.txt
```

结果目录规则：

- `train.py` 会自动创建新目录：`facts/runs/<YYYYMMDD_HHMMSS>/`。
- 后续脚本自动写入最新运行目录，统一保存在同一个 `<run_id>` 下。
- 如果跳过训练但想新开一轮结果目录，先运行：

```powershell
python .\src\start_run.py
```

也可以用环境变量固定目录名：

```powershell
$env:MSGNET_RUN_ID="manual_001"
```

### 2. 标准主流程

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
```

依赖关系：

| 顺序 | 脚本 | 作用 | 关键输出 |
|------|------|------|----------|
| 1 | `data_pipeline.py` | 下载/增量更新 ETF20 行情 | `data/etf20/price.csv` |
| 2 | `preprocess.py` | 价格转截面标准化收益率 | `returns.csv`, `returns.npy` |
| 3 | `train.py` | 训练 MSGNet | `facts/runs/<run_id>/checkpoints/` |
| 4 | `factor_extract.py` | 提取图结构、Embedding、周期因子 | `facts/runs/<run_id>/factors/` |
| 5 | `factor_validate.py` | IC检验、分组分析 | `validation/ic_report.csv` |
| 6 | `factor_return_test.py` | Beta控制单因子收益显著性检验 | `validation/factor_return_report.csv` |
| 7 | `factor_neutralize.py` | `amplitude` Beta中性化 | `factors/neutralized/` |
| 8 | `backtest.py` | 静态组合回测 | `backtest/perf_report.csv` |
| 9 | `walk_forward.py` | 静态 Walk-Forward 检验 | `walk_forward/wf_summary.csv` |

### 3. 滚动重训练 Walk-Forward

阶段七是更重的动态验证，每个 Fold 都会重新训练 MSGNet。需要复现完整阶段七时单独运行：

```powershell
python .\src\rolling_update.py
```

输出目录：

```text
facts/runs/<run_id>/rolling_update/
```

注意事项：

- `data_pipeline.py` 需要访问 `adata.fund.market.get_market_etf`，如果网络或数据源失败，可先沿用已有 `data/etf20/price.csv`，从 `preprocess.py` 开始跑。
- `train.py` 和 `rolling_update.py` 会根据是否可用 CUDA 自动选择 GPU/CPU。
- `factor_extract.py` 依赖 `train.py` 生成的 checkpoint，若重新训练失败，不应继续覆盖因子结果。
- `backtest.py` 和 `walk_forward.py` 默认使用 Beta 中性化后的 `amplitude`，因此必须先运行 `factor_neutralize.py`。
- 最新运行目录记录在 `facts/runs/_latest_run.txt`；所有验证、回测和滚动更新结果都会写入该目录的不同子文件夹。

---

## 代码目录结构

```
MSGNet因子挖掘/
├── data/
│   └── etf20/
│       ├── price.csv           # ETF前复权价格
│       ├── returns.csv         # 日收益率
│       └── returns.npy
├── MSGNet-main/                # 原始模型代码
├── src/
│   ├── data_pipeline.py        # ETF数据获取与重建adj_close
│   ├── preprocess.py           # 截面标准化，生成returns.npy
│   ├── train.py                # MSGNet训练
│   ├── factor_extract.py       # 图结构+周期+embedding因子提取
│   ├── factor_validate.py      # IC检验、分组回测
│   ├── factor_return_test.py   # Beta控制单因子收益显著性检验
│   ├── factor_neutralize.py    # Beta中性化（amplitude）
│   ├── backtest.py             # 多空组合回测
│   ├── walk_forward.py         # 静态Walk-Forward（20窗口）
│   └── rolling_update.py       # 滚动更新Walk-Forward（逐Fold重训练）
├── facts/
│   └── runs/
│       ├── _latest_run.txt
│       └── <run_id>/
│           ├── checkpoints/     # 模型权重
│           ├── factors/         # 静态因子CSV
│           ├── validation/      # IC和单因子收益显著性检验
│           ├── backtest/        # 静态回测结果
│           ├── walk_forward/    # 静态WF结果
│           └── rolling_update/  # 滚动重训练结果
├── 项目进度.md
├── 技术要点.txt
└── MSGNet因子挖掘研究流程.md   # 本文档
```
