#!/bin/bash


if [ ! -d "venv" ]; then
    echo "未找到虚拟环境。正在创建..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "创建虚拟环境失败。请确保已安装 Python 3。"
        exit 1
    fi
fi


echo "激活虚拟环境..."
source venv/bin/activate


echo "升级 pip..."
pip install --upgrade pip
echo


echo "从 requirements.txt 安装依赖..."
pip install -r requirements.txt


echo "检查并下载 Hugging Face 模型..."
if [ ! -d "./models/bge-small-zh" ]; then
    echo "正在下载 BAAI/bge-small-zh 模型..."
    HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-small-zh --local-dir ./models/bge-small-zh 
else
    echo "BAAI/bge-small-zh 模型已存在，跳过下载。"
fi

if [ ! -d "./models/all-MiniLM-L6-v2" ]; then
    echo "正在下载 sentence-transformers/all-MiniLM-L6-v2 模型..."
    HF_ENDPOINT=https://hf-mirror.com hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2 
else
    echo "sentence-transformers/all-MiniLM-L6-v2 模型已存在，跳过下载。"
fi


echo "安装 Playwright 浏览器..."
playwright install


# echo "Installing main package in development mode..."
# pip install -v -e .
# echo


# deactivate