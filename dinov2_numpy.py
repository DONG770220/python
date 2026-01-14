import numpy as np


class Dinov2Numpy:
    """
    简化版DINOv2模型，使用numpy实现特征提取
    """

    def __init__(self, weights):
        """
        初始化模型权重
        weights: 包含投影矩阵的numpy文件
        """
        self.w = weights["proj"]

    def __call__(self, x):
        """
        提取图像特征
        x: 预处理后的图像 (batch_size, channels, height, width)
        return: 特征向量 (batch_size, feature_dim)
        """
        # 将图像展平
        b = x.reshape(x.shape[0], -1)
        # 投影到特征空间
        features = b @ self.w
        # L2归一化
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norm + 1e-8)
        return features