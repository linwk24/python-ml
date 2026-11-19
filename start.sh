#!/bin/bash
# LSTM 预测服务启动脚本

cd "$(dirname "$0")/.."

echo "安装 Python 依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "创建模型目录..."
mkdir -p models

echo "启动 LSTM 预测服务..."
echo "API 文档: http://localhost:8000/docs"
echo "预测接口: POST /predict"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
