import sys
import json
import hashlib
import asyncio
import argparse
from loguru import logger
from utils.file import sanitize_filename
from utils.log import init_logger_by_runid
from utils.sqlite_meta import get_meta_db
from utils.models import Task
from story.story_write import flow_story_write


init_logger_by_runid("write")


async def write_all(tasks_data: list):
    logger.info(f"接收到 {len(tasks_data)} 个任务, 准备并行处理...")

    tasks_by_category = {
        "story": [], 
        "book": [], 
        "report": []
    }

    for task_info in tasks_data:
        if not task_info or not task_info.get('category') or not task_info.get('language'):
            logger.error(f"任务信息不完整, 跳过: {task_info}")
            continue

        category = task_info.get('category')
        if category not in tasks_by_category:
            logger.warning(f"未知的任务类别 '{category}', 已跳过。当前支持的类别: {list(tasks_by_category.keys())}")
            continue

        language = task_info.get('language', 'error')
        root_name = task_info.get("name", "untitled")
        goal = task_info.get("goal", "")
        instructions_str = task_info.get("instructions", "")
        input_brief_str = task_info.get("input_brief", "")
        constraints_str = task_info.get("constraints", "")
        acceptance_criteria_str = task_info.get("acceptance_criteria", "")

        # 生成唯一的运行ID
        sanitized_category = sanitize_filename(category)
        sanitized_name = sanitize_filename(root_name)
        sanitized_language = sanitize_filename(language)
        stable_unique_id = hashlib.sha256(goal.encode('utf-8')).hexdigest()[:16]
        run_id = f"{sanitized_category}_{sanitized_name}_{sanitized_language}_{stable_unique_id}"

        # 构建根任务对象
        task_params = {
            "id": "1",
            "parent_id": "",
            "task_type": "write",
            "hierarchical_position": "全书" if category in ["story", "book"] else "报告",
            "goal": goal,
            "instructions": [line.strip() for line in instructions_str.split('。') if line.strip()],
            "input_brief": [line.strip() for line in input_brief_str.split('。') if line.strip()],
            "constraints": [line.strip() for line in constraints_str.split('。') if line.strip()],
            "acceptance_criteria": [line.strip() for line in acceptance_criteria_str.split('。') if line.strip()],
            "length": task_info.get("length", "根据任务要求确定"),
            "category": category,
            "language": language,
            "root_name": root_name,
            "run_id": run_id,
            "day_wordcount_goal": task_info.get("day_wordcount_goal", 0)
        }
        root_task = Task(**{k: v for k, v in task_params.items() if v is not None})

        # 将书籍元信息写入数据库
        book_meta_db = get_meta_db()
        book_meta_db.add_or_update_book_meta(task=root_task)
        logger.info(f"已为任务 {root_task.run_id} 添加/更新书籍元信息。")

        tasks_by_category[category].append(root_task)

    tasks_to_run = []
    launched_tasks = []

    # 为 'story' (小说) 类别准备任务
    if story_tasks := tasks_by_category["story"]:
        logger.info(f"准备为 {len(story_tasks)} 个 'story' 任务启动 'flow_story_write' 流程...")
        tasks_to_run.extend([flow_story_write(task) for task in story_tasks])
        launched_tasks.extend(story_tasks)

    # 为 'book' (工具书) 类别准备任务 (占位)
    if book_tasks := tasks_by_category["book"]:
        logger.warning(f"接收到 {len(book_tasks)} 个 'book' 任务, 但其处理流程尚未实现, 将被忽略。")
        # TODO: 当 flow_book_write 实现后, 在此添加其逻辑

    # 为 'report' (报告) 类别准备任务 (占位)
    if report_tasks := tasks_by_category["report"]:
        logger.warning(f"接收到 {len(report_tasks)} 个 'report' 任务, 但其处理流程尚未实现, 将被忽略。")
        # TODO: 当 flow_report_write 实现后, 在此添加其逻辑

    if tasks_to_run:
        logger.info(f"即将并行启动 {len(tasks_to_run)} 个流程...")
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        successful_runs = 0
        failed_runs = 0
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                failed_runs += 1
                logger.error(f"任务 {launched_tasks[i].run_id} 在执行期间产生异常: {res}")
            else:
                successful_runs += 1
        logger.info(f"所有并行的子流程已执行完毕。成功: {successful_runs}, 失败: {failed_runs}。")
    else:
        logger.info("没有可执行的流程。")


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
    asyncio.run(write_all(tasks_data))


if __name__ == "__main__":
    main()