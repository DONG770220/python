import os
import numpy as np
from tqdm import tqdm
from preprocess_image import center_crop
from dinov2_numpy import Dinov2Numpy

# 配置路径
IMG_DIR = "assignments/1"
FEATURES_DIR = "features"
WEIGHTS_FILE = "vit-dinov2-base.npz"


def extract_all_features():
    """
    提取所有图片的特征
    """
    # 创建特征目录
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 加载模型权重
    print("正在加载模型权重...")
    weights = np.load(WEIGHTS_FILE, allow_pickle=True)
    vit = Dinov2Numpy(weights)
    print("模型加载完成！")

    feats = []
    paths = []
    failed = []

    # 获取所有图片文件
    img_files = [f for f in os.listdir(IMG_DIR) if
                 f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]
    print(f"共找到 {len(img_files)} 张图片")

    # 提取特征
    for img in tqdm(img_files, desc="提取特征"):
        try:
            img_path = os.path.join(IMG_DIR, img)
            # 预处理并提取特征
            preprocessed = center_crop(img_path)
            feature = vit(preprocessed)[0]
            feats.append(feature)
            paths.append(img)
        except Exception as e:
            failed.append((img, str(e)))

    # 保存特征和路径
    feats_array = np.array(feats)
    paths_array = np.array(paths)

    np.save(os.path.join(FEATURES_DIR, "features.npy"), feats_array)
    np.save(os.path.join(FEATURES_DIR, "paths.npy"), paths_array)

    print(f"\n特征提取完成！")
    print(f"成功: {len(feats)} 张")
    print(f"失败: {len(failed)} 张")
    print(f"特征维度: {feats_array.shape}")

    if failed:
        print("\n失败的图片:")
        for img, error in failed[:10]:
            print(f"  - {img}: {error}")
        if len(failed) > 10:
            print(f"  ... 还有 {len(failed) - 10} 张")


if __name__ == "__main__":
    extract_all_features()