#!/bin/bash

echo "检查 Docker 是否已安装..."
if ! command -v docker &> /dev/null
then
    echo "Docker 未安装。请先安装 Docker。"
    exit 1
else
    echo "Docker 已安装。"
fi

echo "下载所需的 Docker 镜像..."
docker pull memgraph/memgraph-mage:latest
docker pull memgraph/lab:latest
docker pull qdrant/qdrant:latest
docker pull valkey/valkey:8-alpine
docker pull searxng/searxng:latest
docker pull prefecthq/prefect:3-latest

