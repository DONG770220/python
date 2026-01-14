import os
import faiss
import numpy as np

FEATURES_DIR = "features"


def build_faiss_index():
    """
    构建Faiss索引用于快速相似搜索
    """
    # 加载特征
    features_path = os.path.join(FEATURES_DIR, "features.npy")
    print("正在加载特征...")
    features = np.load(features_path).astype("float32")
    print(f"特征形状: {features.shape}")

    # 归一化特征（用于余弦相似度）
    faiss.normalize_L2(features)

    # 创建索引
    print("正在构建索引...")
    dimension = features.shape[1]

    # 使用内积索引（归一化后等同于余弦相似度）
    index = faiss.IndexFlatIP(dimension)
    index.add(features)

    # 保存索引
    index_path = os.path.join(FEATURES_DIR, "index.faiss")
    faiss.write_index(index, index_path)

    print(f"索引构建完成！")
    print(f"索引中的向量数量: {index.ntotal}")
    print(f"索引保存至: {index_path}")


if __name__ == "__main__":
    build_faiss_index()