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
from flow import flow_write, ensure_task_logger


load_dotenv()


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

def setup_global_logger():
    logger.remove()
    logger.add(
        log_dir / "main.log",
        filter=lambda record: not record["extra"].get("run_id"), 
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

def sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s[:100]


@task(
    task_run_name="task_init: {task_info.name}",
)
async def task_init(task_info):
    ensure_task_logger(run_id)
    with logger.contextualize(run_id=run_id):
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
        
        logger.info(f"创建根任务:\n{root_task.model_dump_json(indent=2, exclude_none=True)}")
        
        try:
            await flow_write(current_task=root_task)
        except Exception as e:
            logger.error(f"任务执行失败: {str(e)}")
            logger.exception(f"任务异常详情:")
            return e

@flow
async def flow_write_all(tasks_data: list):
    return task_init.map(tasks_data)

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
        
    asyncio.run(flow_write_all(tasks_data))

if __name__ == "__main__":
    main()