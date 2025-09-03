#!/bin/bash


if [ ! -d "venv" ]; then
    echo "未找到虚拟环境。正在创建..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "创建虚拟环境失败。请确保已安装 Python 3。"
        exit 1
    fi
else
    echo "虚拟环境已存在，跳过创建。"
fi


echo "激活虚拟环境..."
source venv/bin/activate


echo "升级 pip..."
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
echo


echo "从 requirements.txt 安装依赖..."
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
echo


echo "prefect 当前生效的配置"
prefect config view --show-sources
echo


echo "安装 Playwright 浏览器..."
pip show playwright > /dev/null 2>&1
if [ $? -ne 0 ]; then
    pip install playwright -i https://mirrors.aliyun.com/pypi/simple/
fi
if ! python -m playwright show-trace 2>/dev/null | grep -q "chromium"; then
    python -m playwright install-deps
    python -m playwright install chromium
else
    echo "Playwright 浏览器已安装，跳过安装。"
fi


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


echo "检查 Docker 是否已安装..."
if ! command -v docker &> /dev/null
then
    echo "Docker 未安装。请先安装 Docker。"
    exit 1
else
    echo "Docker 已安装。"
fi


echo "下载所需的 Docker 镜像..."
docker images | grep -q "memgraph/memgraph-mage" || docker pull memgraph/memgraph-mage:latest
docker images | grep -q "memgraph/lab" || docker pull memgraph/lab:latest
docker images | grep -q "qdrant/qdrant" || docker pull qdrant/qdrant:latest
docker images | grep -q "valkey/valkey" || docker pull valkey/valkey:8-alpine
docker images | grep -q "searxng/searxng" || docker pull searxng/searxng:latest


echo "创建 Docker volumes..."
docker volume ls | grep -q "memgraph-data" || docker volume create memgraph-data
docker volume ls | grep -q "qdrant_storage" || docker volume create qdrant_storage
docker volume ls | grep -q "valkey-data2" || docker volume create valkey-data2
docker volume ls | grep -q "searxng-data" || docker volume create searxng-data


# echo "Installing main package in development mode..."
# pip install -v -e .
# echo


# deactivate





