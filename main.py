import sys
import json
import asyncio
import hashlib
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from util.models import Task
from memory import memory
from flow import agent_flow


"""


分析、审查当前文件的代码，找出bug并改正， 指出可以优化的地方。


根据以上分析，改进建议， 请直接修改 文件，并提供diff。



"""


###############################################################################


load_dotenv()


###############################################################################


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.remove()
logger.add(
    log_dir / "story_agent_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True,
    backtrace=True,
    diagnose=True
)


###############################################################################


async def run_all_tasks(tasks_data: list, max_steps: int):
    coroutines = []
    for task_info in tasks_data:
        task_params = {
            "id": "1",
            "root_name": task_info["name"], 
            "goal": task_info["goal"],
            "task_type": "write",
            "length": task_info.get("length", "根据任务要求确定"),
            "category": task_info.get("category"),
            "language": task_info.get("language"),
        }

        task_args = {k: v for k, v in task_params.items() if v is not None}
        root_task = Task(**task_args)
        stable_unique_id = hashlib.sha256(root_task.goal.encode('utf-8')).hexdigest()[:16]
        root_task.run_id = f"{task_info.get('category', '')}_{stable_unique_id}"
        logger.info(f"{root_task}")

        coro = agent_flow(current_task=root_task, max_steps=max_steps)
        coroutines.append(coro)

    await asyncio.gather(*coroutines, return_exceptions=True)
    logger.info("所有任务已完成。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        type=str, 
    )
    args = parser.parse_args()
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks_data = data.get("tasks")
    asyncio.run(run_all_tasks(tasks_data, 100))


###############################################################################


if __name__ == "__main__":
    main()

