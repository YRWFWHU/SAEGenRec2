# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAEGenRec 是一个基于 Cookiecutter Data Science 模板的 Python 数据科学/机器学习项目。项目名称暗示与稀疏自编码器（SAE）和生成式推荐（GenRec）相关。

## Environment Setup

```bash
# 创建 conda 环境
make create_environment
conda activate SAEGenRec

# 安装依赖
make requirements
```

需要 Python 3.12。使用 `python-dotenv` 加载 `.env` 中的环境变量（不提交到版本控制）。

## Common Commands

```bash
make lint      # 用 ruff 检查代码格式和风格
make format    # 用 ruff 自动格式化代码
make test      # 运行测试 (python -m pytest tests)
make clean     # 删除编译的 Python 文件
```

运行单个测试：
```bash
python -m pytest tests/test_data.py::test_function_name
```

## Code Style

使用 `ruff` 进行 lint 和格式化，行长限制 99 字符，并强制 import 排序（isort 规则）。`SAEGenRec` 包被视为第一方模块。

## Project Structure

- `SAEGenRec/` — 主源码包（目前仅有 `__init__.py`，待填充）
  - `config.py` — 全局变量和配置（按 README 规划）
  - `dataset.py` — 数据下载/生成脚本
  - `features.py` — 特征工程代码
  - `modeling/` — 模型训练 (`train.py`) 和推理 (`predict.py`)
  - `plots.py` — 可视化代码
- `data/` — 分层数据目录（raw → interim → processed）
- `notebooks/` — Jupyter 笔记本，命名约定：`序号-作者缩写-描述`
- `models/` — 训练好的模型文件
- `reports/figures/` — 生成的图表
- `tests/` — pytest 测试

## Build System

使用 `flit` 作为构建后端（`flit_core >=3.2,<4`），通过 `pip install -e .` 以可编辑模式安装。
