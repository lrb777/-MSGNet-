import pandas as pd

gf = pd.read_csv('facts/csi300_1/factors/graph_factors.csv', index_col='ETF')
pf = pd.read_csv('facts/csi300_1/factors/period_factors.csv', index_col='ETF')
all_f = pd.concat([gf[[c for c in gf.columns if 'out_degree' not in c]], pf], axis=1)

corr = all_f.corr(method='spearman').round(2)
print(corr[['amplitude', 'scale_diff_L2', 'pagerank_L1_S1']].to_string())
