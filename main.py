import re
import sys
import json
import asyncio
import hashlib
import argparse
from pathlib import Path
from loguru import logger
from prefect import flow, task
from dotenv import load_dotenv
from util.models import Task
from flow import flow_write

load_dotenv()


###############################################################################


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

def setup_global_logger():
    logger.remove()
    logger.add(
        log_dir / "main.log",
        filter=lambda record: record["extra"].get("run_id") == "", 
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.configure(extra={"run_id": ""})

def setup_task_logger(run_id: str) -> int:
    task_logger_id = logger.add(
        log_dir / f"{run_id}.log",
        filter=lambda record: record["extra"].get("run_id") == run_id, 
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.configure(extra={"run_id": run_id})
    return task_logger_id

def sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s[:100]


###############################################################################

@task
async def create_and_execute_task(task_info):
    if not task_info:
        raise ValueError("任务信息为空")

    category = task_info.get('category') or 'error'
    language = task_info.get('language') or 'error'
    root_name = task_info["name"]
    sanitized_category = sanitize_filename(category)
    sanitized_name = sanitize_filename(root_name)
    sanitized_language = sanitize_filename(language)
    goal = task_info["goal"]
    stable_unique_id = hashlib.sha256(goal.encode('utf-8')).hexdigest()[:16]
    run_id = f"{sanitized_category}_{sanitized_name}_{sanitized_language}_{stable_unique_id}"
    
    task_logger_id = setup_task_logger(run_id)

    task_params = {
        "id": "1",
        "parent_id": "",
        "task_type": "write",
        "goal": goal,
        "length": task_info.get("length", "根据任务要求确定"),
        "category": category,
        "language": language,
        "root_name": root_name,
        "run_id": run_id
    }
    task_args = {k: v for k, v in task_params.items() if v is not None}
    root_task = Task(**task_args)
    logger.info(f"{root_task}")
    
    try:
        await flow_write(current_task=root_task)
    except Exception as e:
        logger.error(f"任务执行失败: {str(e)}")
        logger.exception(f"任务异常详情:")
        return e
    finally:
        logger.remove(task_logger_id)

@flow
async def run_all_tasks_concurrently(tasks_data: list):
    return create_and_execute_task.map(tasks_data)

async def run_all_tasks(tasks_data: list):
    return await run_all_tasks_concurrently(tasks_data)

def main():
    setup_global_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        type=str, 
    )
    args = parser.parse_args()
    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(1)
    except json.JSONDecodeError as e:
        sys.exit(1)

    tasks_data = data.get("tasks")
    if not tasks_data:
        return
        
    asyncio.run(run_all_tasks(tasks_data))

if __name__ == "__main__":
    main()