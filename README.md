# MSGNet 因子挖掘

基于 MSGNet 的 ETF20 关系结构因子挖掘与多因子回测项目。

## 主要内容

- ETF20 行情下载与预处理
- MSGNet 模型训练
- 图结构因子、周期因子和 embedding 因子提取
- IC 检验、分组分析、Beta 控制单因子收益显著性检验
- Beta 中性化、多空回测、Walk-Forward 和滚动重训练验证

## 运行

```powershell
python -m pip install -r requirements.txt

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

滚动重训练验证：

```powershell
python .\src\rolling_update.py
```

结果默认保存到 `facts/runs/<run_id>/`。运行数据和结果目录不纳入 Git。

详细流程见 [MSGNet因子挖掘研究流程.md](./MSGNet因子挖掘研究流程.md) 和 [项目进度.md](./项目进度.md)。
