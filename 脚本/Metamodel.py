import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls
from sklearn.model_selection import KFold

# --------------------------------------------------
# 1. 读取 3 个模型打分
# --------------------------------------------------
files = {
    'VenusREM': 'VenusREM.csv',
    'ProSST_score': 'Protein_Mutated_with_ProSST.csv',
    'score': 'vespag_predictions.csv'
}
dfs = []
for col_name, file_name in files.items():
    df = pd.read_csv(file_name)
    df = df[['mutation', col_name]]
    dfs.append(df)

# 按 mutation 合并
df_scores = dfs[0]
for d in dfs[1:]:
    df_scores = df_scores.merge(d, on='mutant', how='inner')

# --------------------------------------------------
# 2. 读取标准评分并合并
# --------------------------------------------------
df_std = pd.read_csv('standard.csv')  # mutation,standard_score
df = df_scores.merge(df_std, on='mutation', how='inner')

# --------------------------------------------------
# 3. 准备特征和标签
# --------------------------------------------------
X = df[['VenusREM', 'ProSST_score', 'score']].values
y = df['standard_score'].values

# --------------------------------------------------
# 4. 5 折交叉验证 + NNLS 求非负权重
# --------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
weights = []
rmses = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    w, _ = nnls(X_tr, y_tr)
    w = w / w.sum()  # 归一化

    y_pred = X_val @ w
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    weights.append(w)
    rmses.append(rmse)

# --------------------------------------------------
# 5. 输出结果
# --------------------------------------------------
avg_weight = np.mean(weights, axis=0)
print('平均最优权重 (VenusREM, ProSST_score, score):',
      np.round(avg_weight, 4))
print('5 折交叉验证平均 RMSE :', np.round(np.mean(rmses), 4))

# --------------------------------------------------
# 6. 保存合并结果（含加权得分）
# --------------------------------------------------
df['Weighted'] = X @ avg_weight
df.to_csv('predictions_with_weights.csv', index=False)
print('已保存 predictions_with_weights.csv')