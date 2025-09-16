import sys
import json
import hashlib
import concurrent.futures
import argparse
from loguru import logger
from utils.file import sanitize_filename
from utils.log import init_logger_by_runid
from utils.models import Task
from story.story_write import flow_story_write


init_logger_by_runid("write")


def write_all(tasks_data: list):
    logger.info(f"接收到 {len(tasks_data)} 个任务, 准备并行处理...")
    root_tasks = []
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
            "run_id": run_id,
            "day_wordcount_goal": task_info.get("day_wordcount_goal", 0)
        }
        task_args = {k: v for k, v in task_params.items() if v is not None}
        root_task = Task(**task_args)
        if root_task.category == "story":
            root_tasks.append(root_task)

    logger.info(f"即将并行启动 {len(root_tasks)} 个流程...")
    if root_tasks:
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(flow_story_write, task): task for task in root_tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.error(f"任务 {task.run_id} 在执行期间产生异常: {exc}")
                    results.append(exc)

        successful_runs = 0
        failed_runs = 0
        for res in results:
            if isinstance(res, Exception):
                failed_runs += 1
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
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks_data = data.get("tasks")
    if not tasks_data:
        return
    write_all(tasks_data)


if __name__ == "__main__":
    main()