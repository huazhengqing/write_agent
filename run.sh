#!/bin/bash

if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker。请先安装 Docker。"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: 未找到 docker-compose。请先安装 Docker Compose。"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "❌ 错误：找不到 'venv' 目录。请先运行 ./start.sh 来创建环境。"
    exit 1
fi

echo "🐳 启动 Docker 服务..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "❌ 错误: 启动 Docker 服务失败。"
    exit 1
fi

echo "⏳ 等待服务启动..."
sleep 10

echo "🐍 激活虚拟环境..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ 错误: 激活虚拟环境失败。"
    exit 1
fi

if ! command -v prefect &> /dev/null; then
    echo "❌ 错误: Prefect 未安装。请先运行 ./start.sh 来安装依赖。"
    exit 1
fi

echo "🚀 正在后台启动 Prefect server..."
prefect server start &
PREFECT_PID=$!
if [ $? -ne 0 ]; then
    echo "❌ 错误: 启动 Prefect server 失败。"
    exit 1
fi

echo "⏳ 正在等待 Prefect server 响应... (PID: $PREFECT_PID)"
max_attempts=30
attempt=0
while ! curl -s -f http://127.0.0.1:4200/api/health > /dev/null; do
    attempt=$((attempt+1))
    if [ $attempt -ge $max_attempts ]; then
        echo -e "\n❌ 错误: Prefect server 启动超时。"
        kill $PREFECT_PID 2>/dev/null
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo -e "\n✅ Prefect server 已就绪！UI 地址: http://127.0.0.1:4200"

TASKS_FILE="tasks.json"
if [ ! -f "$TASKS_FILE" ]; then
    echo "❌ 错误: 找不到任务文件 '$TASKS_FILE'。"
    kill $PREFECT_PID 2>/dev/null
    exit 1
fi

echo -e "\n▶️  正在运行主程序 (main.py)..."
python3 main.py "$TASKS_FILE" >> run.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 错误: 主程序执行失败。请检查 run.log 获取详细信息。"
    kill $PREFECT_PID 2>/dev/null
    exit 1
fi

echo -e "\n✅ 主程序执行完毕。"
echo "================================================="
echo "📝 日志文件: run.log"
echo "📊 Prefect UI 地址: http://127.0.0.1:4200"
echo "🛑 Prefect server 仍在后台运行 (PID: $PREFECT_PID)。"
echo "   要停止它, 请运行以下命令:"
echo "   kill $PREFECT_PID"
echo "================================================="