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
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

echo "从 requirements.txt 安装依赖..."
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 访问 PyTorch 官网 (https://pytorch.org/) 获取最适合您 CUDA 版本的安装命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "安装 Playwright 浏览器..."
PLAYWRIGHT_CHROMIUM_PATH="$HOME/.cache/ms-playwright/chromium-"
python -m playwright install-deps
python -m playwright install chromium


