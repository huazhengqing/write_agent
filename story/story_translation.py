import nest_asyncio
nest_asyncio.apply()
import json
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.log import init_logger_by_runid, ensure_task_logger
init_logger_by_runid("story_translation")
from loguru import logger
from utils.models import Task
from utils.sqlite_task import get_task_db
from agents.translation import translation_proposer, translation_critic, translation_refine
from story.story_rag import get_story_rag


###############################################################################


def save_translation_data(task: Task) -> bool:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        if not task.id or not task.goal:
            raise ValueError(f"传递给 save_translation_data 的任务信息不完整, 缺少ID或目标: {task}")
        get_story_rag().save_data(task, "task_translation")
        return True


###############################################################################


async def run_translation_for_task(task: Task):
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        logger.info(f"开始翻译任务: {task.id} - {task.goal}")
        task_result = await translation_proposer(task)
        task_result = await translation_critic(task_result)
        task_result = await translation_refine(task_result)
        save_translation_data(task_result)
        logger.success(f"任务 {task.id} 翻译完成并已保存。")


###############################################################################


async def run_all_translations(root_task: Task):
    """
    循环获取并处理指定 run_id 下需要翻译的任务。
    如果暂时没有任务, 会等待一段时间后重试。
    """
    ensure_task_logger(root_task.run_id)
    with logger.contextualize(run_id=root_task.run_id):
        logger.info(f"启动为 run_id='{root_task.run_id}' 的持续翻译流程...")        
        task_db = get_task_db(root_task.run_id)
        if not root_task:
            logger.error(f"根任务信息不完整, 无法继续翻译。")
            return

        while True:
            task_row = task_db.get_oldest_task_to_translate()
            if task_row:
                logger.info(f"获取到待翻译任务: {task_row['id']}")
                task_data = dict(task_row)
                task_data.update(root_task.model_dump(include={'category', 'language', 'root_name', 'run_id', 'day_wordcount_goal'}))
                for field in ['instructions', 'input_brief', 'constraints', 'acceptance_criteria', 'dependency']:
                    if task_data.get(field) and isinstance(task_data[field], str):
                        try:
                            task_data[field] = json.loads(task_data[field])
                        except json.JSONDecodeError:
                            logger.warning(f"任务 {task_data['id']} 的字段 {field} JSON解析失败, 将使用原始字符串。")
                
                current_task = Task(**task_data)
                await run_translation_for_task(current_task)
            else:
                await asyncio.sleep(60)


###############################################################################


async def start_translation_services(tasks_data: list):
    """
    根据任务数据, 为每个任务启动一个独立的、持续运行的翻译服务。
    """
    if not tasks_data:
        logger.warning("任务列表为空, 未启动任何翻译服务。")
        return

    logger.info(f"接收到 {len(tasks_data)} 个任务配置, 准备并行启动翻译服务...")

    tasks_to_run = []
    for task_info in tasks_data:
        # 仿照 main.py 的逻辑
        category = task_info.get('category')
        language = task_info.get('language', 'error')
        root_name = task_info.get("name", "untitled")
        goal = task_info.get("goal", "")
        instructions_str = task_info.get("instructions", "")
        input_brief_str = task_info.get("input_brief", "")
        constraints_str = task_info.get("constraints", "")
        acceptance_criteria_str = task_info.get("acceptance_criteria", "")

        from utils.file import sanitize_filename
        import hashlib
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
        tasks_to_run.append(run_all_translations(root_task))

    if tasks_to_run:
        await asyncio.gather(*tasks_to_run)


###############################################################################


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="启动一个或多个书籍的持续翻译服务。")
    parser.add_argument(
        "json_file",
        type=str,
        help="包含任务配置的 JSON 文件路径 (例如: tasks.json)",
    )
    args = parser.parse_args()
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    await start_translation_services(data.get("tasks", []))


###############################################################################


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
