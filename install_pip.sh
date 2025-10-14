#!/bin/bash

if [ ! -d "venv" ]; then
    echo "未找到虚拟环境。正在创建..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "创建虚拟环境失败。请确保已安装 Python 3。"
        exit 1
    fi
fi


source venv/bin/activate

pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

# echo "📦 正在安装/更新 pip-tools..."
# pip install --upgrade pip-tools -i https://mirrors.aliyun.com/pypi/simple/

# if [ ! -f "requirements.in" ]; then
#     echo "⚠️ 未找到 requirements.in 文件。正在根据 requirements.txt 创建一个..."
#     echo "# 这是自动生成的顶级依赖文件。请在此处管理你的直接依赖项。" > requirements.in
#     grep -vE '^\s*#|^\s*$' requirements.txt >> requirements.in
#     echo "✅ 已创建 requirements.in 文件。请检查并手动管理此文件。"
# fi

# echo "🔄 正在升级依赖并重新生成 requirements.txt..."
# pip-compile --upgrade --output-file=requirements.txt requirements.in -i https://mirrors.aliyun.com/pypi/simple/
# if [ $? -ne 0 ]; then
#     echo "❌ 错误: 升级依赖失败。"
#     exit 1
# fi



pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

python -m playwright install-deps
python -m playwright install chromium


# 访问 PyTorch 官网 (https://pytorch.org/) 获取最适合您 CUDA 版本的安装命令
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
