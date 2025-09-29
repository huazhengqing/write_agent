#!/bin/bash


echo "🐳 启动 Docker 服务..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "❌ 错误: 启动 Docker 服务失败。"
    exit 1
fi


echo "🐍 激活虚拟环境..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ 错误: 激活虚拟环境失败。"
    exit 1
fi

TASKS_FILE="./tasks.json"
if [ ! -f "$TASKS_FILE" ]; then
    echo "❌ 错误: 找不到任务文件 '$TASKS_FILE'。"
    kill $PREFECT_PID 2>/dev/null
    exit 1
fi

echo -e "\n▶️  正在运行主程序 (main.py)..."
python3 main.py "$TASKS_FILE" > main.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 错误: 主程序执行失败。请检查 write.log 获取详细信息。"
    kill $PREFECT_PID 2>/dev/null
    exit 1
fi

