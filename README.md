# AI 图搜图应用开发
## 目录
项目总体说明
如何运行（快速开始）
核心功能一览
详细功能与文件说明
虚拟环境与依赖
核心模块说明
关键配置与异常处理
项目总体说明
本项目实现轻量级 AI 图搜图功能，基于 Numpy 模拟视觉特征提取逻辑，结合 Faiss 构建检索索引，完成图片相似性检索。无需深度学习框架，仅依赖基础 Python 库，适配小规模图片集检索场景，适合教学演示或技术验证。
如何运行（快速开始）
搭建并激活虚拟环境（Python 3.8+）：
powershell
cd 项目路径/AI-Image-Search
# Windows
python -m venv venv && .\venv\Scripts\activate
安装依赖（国内镜像加速）：
```powershell
pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy Pillow faiss-cpu tqdm --trusted-host mirrors.aliyun.com
```
核心运行步骤：
powershell
# 创建图片目录并放入待检索图片
mkdir assignments/1
# 生成模拟权重
python create_weights.py
# 批量提取图片特征
python extract_features.py
# 构建Faiss检索索引
python build_index.py
输出结果：
features/features.npy：图片特征向量文件
features/index.faiss：检索索引文件，可用于相似图搜索
核心功能一览
模块	核心能力
权重生成	模拟视觉模型投影矩阵（768 维特征）
图片预处理	裁剪、标准化，适配特征提取
特征提取	Numpy 实现 768 维特征向量生成
检索索引	Faiss 构建索引，快速搜相似图
异常处理	捕获图片处理错误，保证流程稳定
详细功能与文件说明
虚拟环境与依赖
核心依赖：Numpy（数值计算）、Pillow（图片处理）、Faiss（相似检索）、tqdm（进度展示）
兼容 Python 版本：3.8~3.11
## 核心模块说明
文件名称	功能
create_weights.py	生成模拟视觉模型投影权重
preprocess_image.py	图片裁剪、标准化等预处理
dinov2_numpy.py	Numpy 实现 768 维图片特征提取
extract_features.py	批量提取图片特征并保存
build_index.py	构建 Faiss 索引，支持相似图检索
## 关键配置与异常处理
路径配置：图片默认存assignments/1，特征 / 索引输出至features目录；
异常处理：提取特征时捕获图片解码 / 格式错误，记录失败案例，不中断批量处理流程。
