#!/bin/bash

source venv/bin/activate

mkdir -p ./models

# 多语言模型
HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-m3 --local-dir ./models/bge-m3
HF_ENDPOINT=https://hf-mirror.com hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2

# 中文模型
HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-large-zh-v1.5 --local-dir ./models/bge-large-zh-v1.5
HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5
HF_ENDPOINT=https://hf-mirror.com hf download moka-ai/m3e-base --local-dir ./models/m3e-base

# 英文模型
HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-large-en-v1.5 --local-dir ./models/bge-large-en-v1.5
HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-small-en-v1.5 --local-dir ./models/bge-small-en-v1.5
HF_ENDPOINT=https://hf-mirror.com hf download sentence-transformers/all-mpnet-base-v2 --local-dir ./models/all-mpnet-base-v2

