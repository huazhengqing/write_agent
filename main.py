import re
import sys
import json
import asyncio
import hashlib
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from util.models import Task
from flow import agent_flow


load_dotenv()


###############################################################################


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.remove()

logger.add(
    log_dir / "{extra[run_id]}.log",
    filter=lambda record: "run_id" in record["extra"],
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

logger.add(
    log_dir / "main.log",
    filter=lambda record: "run_id" not in record["extra"],
    rotation="00:00",
    level="INFO",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

def sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s[:100]

async def _run_task_in_context(coro, run_id: str):
    with logger.contextualize(run_id=run_id):
        try:
            result = await coro
            return result
        except Exception:
            raise


###############################################################################


async def run_all_tasks(tasks_data: list, max_steps: int):
    coroutines = []
    for task_info in tasks_data:
        task_params = {
            "id": "1",
            "parent_id": "",
            "task_type": "write",
            "goal": task_info["goal"],
            "length": task_info.get("length", "根据任务要求确定"),
            "category": task_info.get("category"),
            "language": task_info.get("language"),
            "root_name": task_info["name"], 
        }

        task_args = {k: v for k, v in task_params.items() if v is not None}
        root_task = Task(**task_args)
        stable_unique_id = hashlib.sha256(root_task.goal.encode('utf-8')).hexdigest()[:16]
        
        category = task_info.get('category') or 'uncategorized'
        language = root_task.language or 'unknown'
        sanitized_category = sanitize_filename(category)
        sanitized_name = sanitize_filename(root_task.root_name)
        sanitized_language = sanitize_filename(language)
        run_id = f"{sanitized_category}_{sanitized_name}_{sanitized_language}_{stable_unique_id}"
        root_task.run_id = run_id

        logger.info(f"{root_task} {max_steps}")
        agent_coro = agent_flow(current_task=root_task, max_steps=max_steps)
        wrapped_coro = _run_task_in_context(agent_coro, run_id)
        coroutines.append(wrapped_coro)

    results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    successful_tasks = 0
    failed_tasks = 0
    for res in results:
        if isinstance(res, Exception):
            failed_tasks += 1
        else:
            successful_tasks += 1
    
    logger.info(f"{successful_tasks} {failed_tasks}")


def main():
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
    except json.JSONDecodeError:
        sys.exit(1)

    tasks_data = data.get("tasks")
    if not tasks_data:
        return
        
    asyncio.run(run_all_tasks(tasks_data, 100))


if __name__ == "__main__":
    main()
