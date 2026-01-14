# create_weights.py
import numpy as np

# 创建模拟的投影矩阵
# 输入: 224*224*3 = 150528 维
# 输出: 768 维（DINOv2 base的特征维度）
input_dim = 224 * 224 * 3  # 150528
output_dim = 768

# 使用随机正交初始化
np.random.seed(42)
proj = np.random.randn(input_dim, output_dim).astype(np.float32)
proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)

np.savez("vit-dinov2-base.npz", proj=proj)
print("权重文件创建完成！")