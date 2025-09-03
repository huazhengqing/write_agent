#!/bin/bash


# docker-compose stop
# docker-compose down -v
docker-compose up -d
sleep 10


if [ ! -d "venv" ]; then
    echo "❌ 错误：找不到 'venv' 目录。请先运行 ./start.sh 来创建环境。"
    exit 1
fi
source venv/bin/activate


echo "🚀 正在后台启动 Prefect server..."
prefect server start &
PREFECT_PID=$!


echo "⏳ 正在等待 Prefect server 响应... (PID: $PREFECT_PID)"
while ! curl -s -f http://127.0.0.1:4200/api/health > /dev/null; do
    echo -n "."
    sleep 1
done
echo -e "\n✅ Prefect server 已就绪！UI 地址: http://127.0.0.1:4200"


echo -e "\n▶️  正在运行主程序 (main.py)..."
python3 main.py "tasks.json" >> run.log 2>&1


echo -e "\n🏁 主程序执行完毕。"
echo "================================================="
echo "🛑 Prefect server 仍在后台运行 (PID: $PREFECT_PID)。"
echo "   要停止它, 请运行以下命令:"
echo "   kill $PREFECT_PID"
echo "================================================="
