import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

os.makedirs('./output', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


def build_equation(intercept, coef, feature_names):
	terms = [f'{intercept:.6f}']
	for w, name in zip(coef, feature_names):
		sign = '+' if w >= 0 else '-'
		terms.append(f' {sign} {abs(w):.6f}·{name}_std')
	return 'y_hat = ' + ''.join(terms)


# 读取数据
diabetes = pd.read_csv('./数据集/diabetes.csv')
feature_names = [c for c in diabetes.columns if c != 'Outcome']
X = diabetes[feature_names].values
y = diabetes['Outcome'].values
noise = np.random.normal(0, 0.1, size=y.shape) 
y = y + noise
# 划分训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.3,
	random_state=42
)

# StandardScaler 标准化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ------------------ 基线模型：LinearRegression ------------------
lin_model = LinearRegression()
lin_model.fit(X_train_std, y_train)

# ------------------ Ridge：可视化 + RidgeCV ------------------
alphas_ridge = np.logspace(-4, 2, 60)
ridge_cv_mse = []
for a in alphas_ridge:
	scores = -cross_val_score(
		Ridge(alpha=a),
		X_train_std,
		y_train,
		cv=5,
		scoring='neg_mean_squared_error'
	)
	ridge_cv_mse.append(scores.mean())

ridge_best_alpha_visual = alphas_ridge[int(np.argmin(ridge_cv_mse))]

ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_std, y_train)
ridge_best_alpha = ridge_cv.alpha_

ridge_model = Ridge(alpha=ridge_best_alpha)
ridge_model.fit(X_train_std, y_train)

# ------------------ Lasso：可视化 + LassoCV ------------------
alphas_lasso = np.logspace(-4, 1, 60)
lasso_cv_mse = []
for a in alphas_lasso:
	scores = -cross_val_score(
		Lasso(alpha=a, max_iter=50000),
		X_train_std,
		y_train,
		cv=5,
		scoring='neg_mean_squared_error'
	)
	lasso_cv_mse.append(scores.mean())

lasso_best_alpha_visual = alphas_lasso[int(np.argmin(lasso_cv_mse))]

lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, random_state=42, max_iter=50000)
lasso_cv.fit(X_train_std, y_train)
lasso_best_alpha = lasso_cv.alpha_

lasso_model = Lasso(alpha=lasso_best_alpha, max_iter=50000)
lasso_model.fit(X_train_std, y_train)

# ------------------ 三模型测试集预测与评估 ------------------
predictions = {
	'LinearRegression': lin_model.predict(X_test_std),
	'Ridge': ridge_model.predict(X_test_std),
	'Lasso': lasso_model.predict(X_test_std)
}

metrics = []
for model_name, y_pred in predictions.items():
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	metrics.append({'Model': model_name, 'MSE': mse, 'R2': r2})

metrics_df = pd.DataFrame(metrics)
print('\n=== 三模型测试集性能对比 ===')
print(metrics_df.to_string(index=False))
metrics_df.to_csv('./output/2024150040邓瑞霖model_metrics.csv', index=False)

print('\n=== Alpha 选择结果 ===')
print(f'Ridge 可视化最优 alpha: {ridge_best_alpha_visual:.6f}')
print(f'RidgeCV 最优 alpha: {ridge_best_alpha:.6f}')
print(f'Lasso 可视化最优 alpha: {lasso_best_alpha_visual:.6f}')
print(f'LassoCV 最优 alpha: {lasso_best_alpha:.6f}')

# ------------------ Lasso系数与特征剔除分析 ------------------
lasso_coef = pd.Series(lasso_model.coef_, index=feature_names)
removed_features = lasso_coef[np.isclose(lasso_coef.values, 0.0, atol=1e-6)].index.tolist()

print('\n=== Lasso 模型系数 ===')
print(lasso_coef)
print('被Lasso剔除（系数≈0）的特征:', removed_features if removed_features else '无')

# ------------------ 模型方程输出（标准化特征空间） ------------------
print('\n=== 三模型最终方程（基于标准化后的特征）===')
print('[LinearRegression] ', build_equation(lin_model.intercept_, lin_model.coef_, feature_names))
print('[Ridge]            ', build_equation(ridge_model.intercept_, ridge_model.coef_, feature_names))
print('[Lasso]            ', build_equation(lasso_model.intercept_, lasso_model.coef_, feature_names))

# 单独输出 Ridge 的 alpha-MSE 曲线图
plt.figure(figsize=(8, 5))
plt.semilogx(alphas_ridge, ridge_cv_mse, marker='o', markersize=4, label='CV MSE')
plt.axvline(ridge_best_alpha, color='red', linestyle='--', label=f'RidgeCV最优alpha={ridge_best_alpha:.4g}')
plt.title('Ridge: alpha 与 CV-MSE')
plt.xlabel('alpha')
plt.ylabel('CV MSE')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖ridge_alpha_vs_mse.png')

# 单独输出 Lasso 的 alpha-MSE 曲线图
plt.figure(figsize=(8, 5))
plt.semilogx(alphas_lasso, lasso_cv_mse, marker='o', markersize=4, label='CV MSE')
plt.axvline(lasso_best_alpha, color='red', linestyle='--', label=f'LassoCV最优alpha={lasso_best_alpha:.4g}')
plt.title('Lasso: alpha 与 CV-MSE')
plt.xlabel('alpha')
plt.ylabel('CV MSE')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖lasso_alpha_vs_mse.png')

# ------------------ 图2：真实值-预测值折线对比 ------------------
idx = np.arange(len(y_test))
plt.figure(figsize=(12, 5))
plt.plot(idx, y_test, label='真实值', linewidth=2, color='black')
plt.plot(idx, predictions['LinearRegression'], label='LinearRegression预测', alpha=0.8)
plt.plot(idx, predictions['Ridge'], label='Ridge预测', alpha=0.8)
plt.plot(idx, predictions['Lasso'], label='Lasso预测', alpha=0.8)
plt.xlabel('测试样本索引')
plt.ylabel('Outcome')
plt.title('真实值与三模型预测值折线对比')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖真实值vs预测值折线图.png')

# ------------------ 图3：真实值-预测值散点对比 ------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
for ax, (name, y_pred) in zip(axes, predictions.items()):
	ax.scatter(y_test, y_pred, alpha=0.7)
	min_v = min(y_test.min(), y_pred.min())
	max_v = max(y_test.max(), y_pred.max())
	ax.plot([min_v, max_v], [min_v, max_v], 'r--')
	ax.set_title(f'{name} 真实值-预测值')
	ax.set_xlabel('真实值')
	ax.set_ylabel('预测值')
	ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖真是值vs预测值散点图.png')

# plt.show()