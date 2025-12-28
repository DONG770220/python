# Django 博客项目说明
## 目录
- 项目总体说明
- 如何运行（快速开始）
- 新增的核心功能一览（摘要）
- 详细功能与文件说明
  - 虚拟环境与依赖
  - pip 镜像配置
  - Django 应用与模型
  - 管理后台（Admin）
  - 视图（Views）与权限控制
  - 路由（URLs）
  - 模板（Templates）与继承结构
  - 静态资源（Static files：CSS / JS）
  - 实用脚本（创建演示数据）
  - 配置调整（settings.py 的变更）

---

## 项目总体说明

该仓库在原始 Django 项目基础上添加了一个名为 `blogs` 的应用，提供了：

- 博文模型 `BlogPost`（包含标题、正文、作者、添加日期等字段）。
- 完整的 CRUD 用户流程：主页浏览、发表新博文（仅对登录用户）、编辑（仅限作者）、详情页阅读（公开）。
- 前端友好的模板与响应式样式、移动端汉堡导航、简易注册/登录流程与管理员支持。

目标是一个小型但功能完善的博客示例，适合教学或演示，也可在做适当安全加固后商用。

## 如何运行（快速开始）

1. 确保已安装 Python 3.10+（项目使用 3.10），并在项目目录运行虚拟环境：

```powershell
# 进入项目根目录
cd 你的项目路径/Blog
# Windows 激活 blong 虚拟环境
.\blong\Scripts\activate
# macOS/Linux 激活
source blong/bin/activate
```

2. 安装依赖（国内加速）

```powershell
pip install -i https://mirrors.aliyun.com/pypi/simple/ django==5.2.9 --trusted-host mirrors.aliyun.com
```

3. 初始化数据库
```powershell
# 执行迁移（创建表结构）
python manage.py migrate
# （可选）创建超级管理员（后台管理用）
python manage.py createsuperuser
```
4.生成演示数据（可选）
```powershell
# 自动创建超级用户（admin/adminpass）和示例博文（仅测试用）
python .\scripts\create_demo.py
```
6. 访问地址
页面	访问地址	说明
博客主页	http://127.0.0.1:8000/	公开访问
管理后台	http://127.0.0.1:8000/admin/	需超级用户登录
登录 / 注册	http://127.0.0.1:8000/accounts/login/	普通用户认证
发布博文	http://127.0.0.1:8000/post/new/	需登录后访问

## 核心功能
模块	核心能力
用户认证	注册 / 登录 / 登出，未登录禁止发布 / 编辑文章
博文管理	发布 / 编辑（仅作者）/ 查看详情 / 首页列表展示
权限控制	非作者编辑文章返回 403 错误，保证数据安全
---

