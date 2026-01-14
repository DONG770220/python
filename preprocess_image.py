import numpy as np
from PIL import Image


def center_crop(img_path, crop_size=224):
    """
    对图片进行中心裁剪和标准化处理
    """
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # 如果图片小于裁剪尺寸，先进行缩放
    if w < crop_size or h < crop_size:
        scale = max(crop_size / w, crop_size / h)
        new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
        image = image.resize((new_w, new_h), Image.LANCZOS)
        w, h = image.size

    # 中心裁剪
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))

    # 转换为numpy数组并归一化
    image = np.array(image).astype(np.float32) / 255.0

    # ImageNet标准化参数
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # 转换为CHW格式
    image = image.transpose(2, 0, 1)
    return image[None]  # 添加batch维度


def preprocess_for_search(img_path, crop_size=224):
    """
    用于搜索时的图片预处理
    """
    return center_crop(img_path, crop_size)