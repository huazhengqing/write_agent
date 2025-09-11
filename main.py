import re
import sys
import json
import asyncio
import hashlib
import logging
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from utils.models import Task
from flow import flow_write


load_dotenv()

def init_logger():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            # 找到调用栈的正确深度
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("llama_index").setLevel(logging.DEBUG)
    logger.add(
        log_dir / "main.log",
        filter=lambda record: not record["extra"].get("run_id"), 
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        level="DEBUG", 
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


init_logger()


def sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s[:100]


def flow_write_all(tasks_data: list):
    logger.info(f"接收到 {len(tasks_data)} 个任务, 准备并行处理...")
    flow_runs = []
    for task_info in tasks_data:
        if not task_info or not task_info.get('category') or not task_info.get('language'):
            logger.error(f"任务信息不完整, 跳过: {task_info}")
            continue

        category = task_info.get('category', 'error')
        language = task_info.get('language', 'error')
        root_name = task_info.get("name", "untitled")
        goal = task_info.get("goal", "")
        sanitized_category = sanitize_filename(category)
        sanitized_name = sanitize_filename(root_name)
        sanitized_language = sanitize_filename(language)
        stable_unique_id = hashlib.sha256(goal.encode('utf-8')).hexdigest()[:16]
        run_id = f"{sanitized_category}_{sanitized_name}_{sanitized_language}_{stable_unique_id}"
        task_params = {
            "id": "1",
            "parent_id": "",
            "task_type": "write",
            "hierarchical_position": "全书",
            "goal": goal,
            "length": task_info.get("length", "根据任务要求确定"),
            "category": category,
            "language": language,
            "root_name": root_name,
            "run_id": run_id
        }
        task_args = {k: v for k, v in task_params.items() if v is not None}
        root_task = Task(**task_args)
        flow_runs.append(flow_write(current_task=root_task))

    logger.info(f"即将并行启动 {len(flow_runs)} 个流程...")
    if flow_runs:
        async def run_concurrently():
            return await asyncio.gather(*flow_runs, return_exceptions=True)
        results = asyncio.run(run_concurrently())
        successful_runs = 0
        failed_runs = 0
        for res in results:
            if isinstance(res, Exception):
                failed_runs += 1
                logger.error(f"一个子流程执行失败: {res}")
            else:
                successful_runs += 1
        logger.info(f"所有并行的子流程已执行完毕。成功: {successful_runs}, 失败: {failed_runs}。")
    else:
        logger.info("没有需要执行的流程。")


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
    except json.JSONDecodeError as e:
        sys.exit(1)
    tasks_data = data.get("tasks")
    if not tasks_data:
        return
    flow_write_all(tasks_data)


if __name__ == "__main__":
    main()