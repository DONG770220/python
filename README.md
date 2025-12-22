# Iris ML Project
本项目展示鸢尾花数据集的可视化与分类模型。主要使用的是机器学习课本里的第四章决策树的内容。
## 运行方式
> ```bash
> python data_preview.py
> ```
## 项目说明
基于 Python 的鸢尾花数据集分析项目，整合数据处理、可视化（2D/3D 图表）和分类模型（逻辑回归、SVM、决策树）训练，适合入门参考。
## 环境准备
先装 Python 3.9 以上版本，建议用 Anaconda 建个独立环境（名字叫 Iris 就行），避免依赖冲突
打开终端，跑这行命令装所有需要的库：
> ```bash
> pip install pandas numpy matplotlib seaborn plotly scikit-learn
> ```
##  核心文件
data_preview.py：数据加载、预处理及可视化（特征分布、3D 决策边界等）
classifier2d.py：2D 特征空间的决策边界与概率分布可视化
用法
## 用法
把三个核心文件放同一文件夹：data_preview.py（入口）、data_preview.py（数据和画图）、classifier2d.py（模型训练）
终端进入这个文件夹，输入命令启动：
> ```bash
> python project_main.py
> ```

