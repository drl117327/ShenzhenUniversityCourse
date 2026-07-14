import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False

# 1) 加载数据与标准化
wine = load_wine()
X = wine.data          # 13 维特征
y = wine.target        # 3 类标签
target_names = wine.target_names

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 2) PCA 降维（无监督）
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)

# 3) LDA 降维（有监督）
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_std, y)

# 4) 可视化对比：1x2 子图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：PCA 2D
for cls in range(len(target_names)):
	axes[0].scatter(
		X_pca[y == cls, 0],
		X_pca[y == cls, 1],
		label=target_names[cls],
		alpha=0.8,
		s=35
	)
axes[0].set_title('PCA 降维结果（2D）')
axes[0].set_xlabel('主成分 1')
axes[0].set_ylabel('主成分 2')
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].legend()

# 右图：LDA 2D
for cls in range(len(target_names)):
	axes[1].scatter(
		X_lda[y == cls, 0],
		X_lda[y == cls, 1],
		label=target_names[cls],
		alpha=0.8,
		s=35
	)
axes[1].set_title('LDA 降维结果（2D）')
axes[1].set_xlabel('判别轴 1')
axes[1].set_ylabel('判别轴 2')
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].legend()

plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖PCAvsLDA.png', dpi=300)
plt.show()