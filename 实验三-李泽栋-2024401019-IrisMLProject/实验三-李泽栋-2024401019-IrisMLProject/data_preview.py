import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 中文字体配置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 数据加载与预处理
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]
df_clean = df.dropna()

X_2d = iris.data[:, 2:]
X_train_2d, X_test_2d, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.3, random_state=42
)

X_3d = iris.data[:, :3]

# 任务0：数据预览可视化
print("===== 数据预览 =====")
print(df_clean.sample(5))

# 特征箱线图
plt.figure(figsize=(15, 12))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=df_clean)
    plt.title(f'{feature} 按花种类别的分布')
    plt.xlabel('花种类别')
    plt.ylabel(feature)
plt.tight_layout()
plt.savefig('data_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 交互式特征散点图矩阵
fig = px.scatter_matrix(
    df_clean,
    dimensions=feature_names,
    color='species',
    title='特征散点图矩阵',
    labels={col: col for col in feature_names},
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig.update_layout(width=1000, height=800)
fig.show()

# 任务1：不同分类器结果对比
print("\n===== 任务1：不同分类器结果对比 =====")
classifiers = {
    "逻辑回归": LogisticRegression(max_iter=200),
    "支持向量机": SVC(probability=True),
    "决策树": DecisionTreeClassifier()
}

xx, yy = np.meshgrid(
    np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 200),
    np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 200)
)

cmap_light = mcolors.ListedColormap(['#FFFFCC', '#CCFFCC', '#CCE5FF'])
cmap_bold = mcolors.ListedColormap(['#FFCC00', '#00CC66', '#0099CC'])

plt.figure(figsize=(18, 5))
for i, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train_2d, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.subplot(1, 3, i + 1)
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_bold,
        edgecolor='k', s=50
    )
    plt.title(f'{name} 分类结果', fontsize=14)
    plt.xlabel(feature_names[2], fontsize=12)
    if i == 0:
        plt.ylabel(feature_names[3], fontsize=12)
    plt.grid(alpha=0.3)

handles, _ = scatter.legend_elements()
plt.figlegend(handles, target_names, title="花种类别", loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('classifier_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 任务2：3D决策边界可视化
print("\n===== 任务2：3D决策边界可视化 =====")
model_3d = LogisticRegression(max_iter=200)
model_3d.fit(X_3d, y)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
    c=y, cmap=cmap_bold, s=80, edgecolor='k', alpha=0.8
)

coef = model_3d.coef_
intercept = model_3d.intercept_

xx, yy = np.meshgrid(
    np.linspace(X_3d[:, 0].min(), X_3d[:, 0].max(), 50),
    np.linspace(X_3d[:, 1].min(), X_3d[:, 1].max(), 50)
)

for i in range(3):
    zz = (-coef[i, 0] * xx - coef[i, 1] * yy - intercept[i]) / coef[i, 2]
    ax.plot_surface(xx, yy, zz, alpha=0.2, color=cmap_bold(i))

ax.set_xlabel(feature_names[0], fontsize=12)
ax.set_ylabel(feature_names[1], fontsize=12)
ax.set_zlabel(feature_names[2], fontsize=12)
ax.set_title('三维特征空间中的逻辑回归决策边界', fontsize=14)

ax.legend(handles, target_names, title="花种类别", loc='upper left')
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.savefig('3d_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

# 任务3：3D概率图可视化
print("\n===== 任务3：3D概率图可视化 =====")
probs = model_3d.predict_proba(X_3d)
max_probs = probs.max(axis=1)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
    c=max_probs, cmap='viridis', s=80, edgecolor='k', alpha=0.8
)

cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('最大类别概率', fontsize=12)

ax.set_xlabel(feature_names[0], fontsize=12)
ax.set_ylabel(feature_names[1], fontsize=12)
ax.set_zlabel(feature_names[2], fontsize=12)
ax.set_title('三维特征空间中的分类概率分布', fontsize=14)

for i, name in enumerate(target_names):
    ax.scatter([], [], [], c=[cmap_bold(i)], label=name, s=50)
ax.legend(title="花种类别", loc='upper left')
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.savefig('3d_probability_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 任务4：特征重要性可视化
print("\n===== 任务4：特征重要性分析 =====")
model = LogisticRegression(max_iter=200)
model.fit(X, y)

feature_importance = np.mean(np.abs(model.coef_), axis=0)
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = feature_importance[sorted_idx]

plt.figure(figsize=(10, 6))
bars = plt.bar(
    range(len(sorted_importance)),
    sorted_importance,
    color=plt.cm.Set3(range(len(sorted_importance))),
    alpha=0.7
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2., height + 0.01,
        f'{height:.3f}', ha='center', va='bottom', fontsize=10
    )

plt.title('特征重要性（逻辑回归系数绝对值的平均值）', fontsize=14)
plt.xticks(range(len(sorted_importance)), sorted_features, rotation=45, ha='right', fontsize=11)
plt.ylabel('重要性分数', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n所有可视化完成！生成的图片已保存为PNG文件。")