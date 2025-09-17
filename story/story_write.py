from pathlib import Path
from typing import Any, Dict, Callable, Literal
from loguru import logger
from utils.log import ensure_task_logger
from utils.prefect_utils import get_cache_key, local_storage, readable_json_serializer
from utils.models import Task
from utils.sqlite_task import get_task_db
from agents.atom import atom
from agents.plan import plan, plan_reflection
from agents.design import design, design_aggregate, design_reflection
from agents.search import search, search_aggregate
from agents.write import write, write_before_reflection, write_reflection
from agents.summary import summary, summary_aggregate
from agents.hierarchy import hierarchy, hierarchy_reflection
from agents.review import review_design, review_write
from agents.route import route
from story.story_rag import get_story_rag
from prefect import flow, task


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/atom.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_atom",
)
async def task_atom(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await atom(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_plan",
)
async def task_plan(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_reflection.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_plan_reflection",
)
async def task_plan_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_reflection(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/route.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_route",
)
async def task_route(task: Task) -> str:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await route(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/design_{parameters[category]}.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_design_{category}",
)
async def task_design(task: Task, category: str) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design(task, category)


@task(
    persist_result=True, 
    result_storage=local_storage,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/design_reflection.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_design_reflection",
)
async def task_design_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design_reflection(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/hierarchy.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_hierarchy",
)
async def task_hierarchy(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await hierarchy(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/hierarchy_reflection.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_hierarchy_reflection",
)
async def task_hierarchy_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await hierarchy_reflection(task)
    

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/search.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_search",
)
async def task_search(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await search(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/write_before_reflection.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_write_before_reflection",
)
async def task_write_before_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await write_before_reflection(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/write.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_write",
)
async def task_write(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        ret = await write(task)
        return ret


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/write_reflection.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_write_reflection",
)
async def task_write_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        ret = await write_reflection(task)
        return ret


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/summary.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_summary",
)
async def task_summary(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        ret = await summary(task)
        return ret


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/aggregate_design.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_aggregate_design",
)
async def task_aggregate_design(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design_aggregate(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/aggregate_search.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_aggregate_search",
)
async def task_aggregate_search(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await search_aggregate(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/aggregate_summary.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_aggregate_summary",
)
async def task_aggregate_summary(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await summary_aggregate(task)


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/review_design.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_review_design",
)
async def task_review_design(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await review_design(task)


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/review_write.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_review_write",
)
async def task_review_write(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await review_write(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/store_{parameters[operation_name]}.json",
    result_serializer=readable_json_serializer,
    retries=1,
    task_run_name="{task.run_id}_{task.id}_store_{operation_name}",
)
def task_save_data(task: Task, operation_name: str) -> bool:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        if not task.id or not task.goal:
            raise ValueError(f"传递给 task_save_data 的任务信息不完整, 缺少ID或目标: {task}")
        get_story_rag().save_data(task, operation_name)
        return True


@flow(
    persist_result=False, 
    result_storage=local_storage,
    name="flow_story_write", 
    flow_run_name="{current_task.run_id}_flow_story_write_{current_task.id}",
)
async def flow_story_write(current_task: Task):
    ensure_task_logger(current_task.run_id)
    with logger.contextualize(run_id=current_task.run_id):
        logger.info(f"开始处理任务: {current_task.run_id} {current_task.id} {current_task.task_type} {current_task.goal}")
        
        if not current_task.id or not current_task.goal:
            raise ValueError("任务ID和目标不能为空。")
        if current_task.task_type == "write":
            if not current_task.length:
                raise ValueError("写作任务没有长度要求")

        day_wordcount_goal = getattr(current_task, 'day_wordcount_goal', 0)
        if day_wordcount_goal > 0:
            db = get_task_db(run_id=current_task.run_id)
            word_count_24h = db.get_word_count_last_24h()
            if word_count_24h >= day_wordcount_goal:
                logger.info(f"已达到最近24小时字数目标 ({word_count_24h}字), 暂停任务: {current_task.run_id}")
                return

        logger.info(f"判断原子任务='{current_task.id}' ")
        task_result = await task_atom(current_task)
        task_save_data(task_result, "task_atom")
        if task_result.results.get("atom_result") == "atom": 
            if task_result.task_type == "design":
                logger.info(f"执行: design 任务='{current_task.id}'")
                category = await task_route(task_result)
                logger.info(f"执行: design {category} 任务='{current_task.id}'")
                task_result = await task_design(task_result, category=category)
                task_save_data(task_result, f"task_design_{category}")
            elif task_result.task_type == "search":
                logger.info(f"执行: search 任务='{current_task.id}'")
                task_result = await task_search(task_result)
                task_save_data(task_result, "task_search")
            elif task_result.task_type == "write":
                logger.info(f"执行: 设计反思: write 任务='{current_task.id}'")
                task_result = await task_write_before_reflection(task_result)
                task_save_data(task_result, "task_write_before_reflection")

                logger.info(f"执行: 写作初稿: write 任务='{current_task.id}'")
                task_result = await task_write(task_result)
                task_save_data(task_result, "task_write")

                logger.info(f"执行: 反思初稿: write 任务='{current_task.id}'")
                task_result = await task_write_reflection(task_result)
                task_save_data(task_result, "task_write_reflection")

                logger.info(f"执行: 正文摘要: write 任务='{current_task.id}'")
                task_result = await task_summary(task_result)
                task_save_data(task_result, "task_summary")
                
                keywords_to_skip_review = ["全书", "卷", "幕", "段落", "场景"]
                position = task_result.hierarchical_position
                if position and "章" in position and not any(keyword in position for keyword in keywords_to_skip_review):
                    logger.info(f"执行: 审查正文: write 任务='{task_result.id}' ")
                    task_result = await task_review_write(task_result)
                    task_save_data(task_result, "task_review_write")
            else:
                raise ValueError(f"未知的原子任务类型: {task_result.task_type}")
        else:
            logger.info(f"任务 '{current_task.id}' 不是原子任务, 进行分解。")
            complex_reasons = task_result.results.get("complex_reasons", [])
            if (task_result.task_type == "write" and 'design_insufficient' not in complex_reasons):
                logger.info(f"审查设计方案: write 任务='{task_result.id}' ")
                task_result = await task_review_design(task_result)
                task_save_data(task_result, "task_review_design")

                logger.info(f"划分结构: write 任务='{task_result.id}' ")
                task_result = await task_hierarchy(task_result)
                task_save_data(task_result, "task_hierarchy")

                logger.info(f"反思划分结构结果: write 任务='{task_result.id}' ")
                task_result = await task_hierarchy_reflection(task_result)
                task_save_data(task_result, "task_hierarchy_reflection")
            
            logger.info(f"分解: write 任务='{task_result.id}' ")
            task_result = await task_plan(task_result)
            task_save_data(task_result, "task_plan")

            if task_result.task_type in ["write", "design"]:
                logger.info(f"反思分解结果: write 任务='{task_result.id}' ")
                task_result = await task_plan_reflection(task_result)
                task_save_data(task_result, "task_plan_reflection")

            if task_result.sub_tasks:
                logger.info(f"任务 '{current_task.id}' 分解为 {len(task_result.sub_tasks)} 个子任务, 开始递归处理。")
                for sub_task in task_result.sub_tasks:

                    if day_wordcount_goal > 0:
                        db = get_task_db(run_id=current_task.run_id)
                        word_count_24h = db.get_word_count_last_24h()
                        if word_count_24h >= day_wordcount_goal:
                            logger.info(f"已达到最近24小时字数目标 ({word_count_24h}字), 暂停处理后续子任务: {sub_task.run_id}")
                            return

                    await flow_story_write(sub_task)
                
                if task_result.task_type == "write":
                    keywords_to_skip_review = ["全书", "卷", "幕", "段落", "场景"]
                    position = task_result.hierarchical_position
                    if position and "章" in position and not any(keyword in position for keyword in keywords_to_skip_review):
                        logger.info(f"审查正文: write 任务='{task_result.id}' ")
                        task_result = await task_review_write(task_result)
                        task_save_data(task_result, "task_review_write")

                if task_result.task_type == "design":
                    logger.info(f"聚合: design 任务='{task_result.id}' ")
                    task_result = await task_aggregate_design(task_result)
                    task_save_data(task_result, "task_aggregate_design")
                elif task_result.task_type == "search":
                    logger.info(f"聚合: search 任务='{task_result.id}' ")
                    task_result = await task_aggregate_search(task_result)
                    task_save_data(task_result, "task_aggregate_search")
                elif task_result.task_type == "write":
                    logger.info(f"聚合摘要: write 任务='{task_result.id}' ")
                    task_result = await task_aggregate_summary(task_result)
                    task_save_data(task_result, "task_aggregate_summary")
                else:
                    raise ValueError(f"未知的聚合任务类型: {task_result.task_type}")
            else:
                logger.error(f"规划失败, 任务 '{task_result.id}' 没有产生任何子任务。")
                raise Exception(f"任务 '{task_result.id}' 规划失败, 没有子任务。")
