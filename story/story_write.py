from loguru import logger

from utils.log import ensure_task_logger
from utils.prefect import get_cache_key, local_storage, readable_json_serializer
from utils.models import Task
from utils.sqlite_task import get_task_db
from agents.atom import atom
from agents.plan import (
    plan_write_proposer, plan_write_critic, plan_write_synthesizer,
    plan_design_proposer, plan_design_critic, plan_design_synthesizer,
    plan_search_planner, plan_search_synthesizer
)
from agents.design import design_guideline, design_execute, design_aggregate
from agents.search import search, search_aggregate
from agents.write import write_plan, write_draft, write_critic, write_refine, write_review
from agents.summary import summary, summary_aggregate
from agents.hierarchy import hierarchy_proposer, hierarchy_critic, hierarchy_synthesizer
from agents.refine import refine
from story.story_rag import get_story_rag
from prefect import flow, task



@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/refine.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_refine",
)
async def task_refine(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await refine(task)


###############################################################################


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/atom.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_atom",
)
async def task_atom(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await atom(task)


###############################################################################


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_write_proposer.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_write_proposer",
)
async def task_plan_write_proposer(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_write_proposer(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_write_critic.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_write_critic",
)
async def task_plan_write_critic(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_write_critic(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_write_synthesizer.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_write_synthesizer",
)
async def task_plan_write_synthesizer(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_write_synthesizer(task)


###############################################################################


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_design_proposer.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_design_proposer",
)
async def task_plan_design_proposer(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_design_proposer(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_design_critic.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_design_critic",
)
async def task_plan_design_critic(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_design_critic(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_design_synthesizer.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_design_synthesizer",
)
async def task_plan_design_synthesizer(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_design_synthesizer(task)


###############################################################################


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_search_planner.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_search_planner",
)
async def task_plan_search_planner(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_search_planner(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_search_synthesizer.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_plan_search_synthesizer",
)
async def task_plan_search_synthesizer(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_search_synthesizer(task)


###############################################################################


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/design_guideline.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_design_guideline",
)
async def task_design_guideline(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design_guideline(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/design_execute.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_design_execute",
)
async def task_design_execute(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design_execute(task)


###############################################################################


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/search.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_search",
)
async def task_search(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await search(task)


###############################################################################


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/hierarchy_proposer.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_hierarchy_proposer",
)
async def task_hierarchy_proposer(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await hierarchy_proposer(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/hierarchy_critic.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_hierarchy_critic",
)
async def task_hierarchy_critic(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await hierarchy_critic(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/hierarchy_synthesizer.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_hierarchy_synthesizer",
)
async def task_hierarchy_synthesizer(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await hierarchy_synthesizer(task)


###############################################################################


@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/write_plan.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_write_plan",
)
async def task_write_plan(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await write_plan(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/write_draft.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_write_draft",
)
async def task_write_draft(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await write_draft(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/write_critic.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_write_critic",
)
async def task_write_critic(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await write_critic(task)

@task(
    persist_result=True,
    result_storage=local_storage,
    cache_key_fn=get_cache_key,
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/write_refine.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_write_refine",
)
async def task_write_refine(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await write_refine(task)


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/summary.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
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
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/review_write.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_review_write",
)
async def task_write_review(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await write_review(task)


###############################################################################


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/aggregate_design.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=5,
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
    retry_delay_seconds=5,
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
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_aggregate_summary",
)
async def task_aggregate_summary(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await summary_aggregate(task)


###############################################################################


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/store_{parameters[operation_name]}.json",
    result_serializer=readable_json_serializer,
    retries=10,
    retry_delay_seconds=5,
    task_run_name="{task.run_id}_{task.id}_store_{operation_name}",
)
def task_save_data(task: Task, operation_name: str) -> bool:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        if not task.id or not task.goal:
            raise ValueError(f"传递给 task_save_data 的任务信息不完整, 缺少ID或目标: {task}")
        get_story_rag().save_data(task, operation_name)
        return True


###############################################################################


async def run_review_flow(task: Task):
    """封装审查和保存的流程"""
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        logger.info(f"执行: 审查正文: write 任务='{task.id}'")
        review_task_result = await task_write_review(task)
        task_save_data(review_task_result, "task_write_review")


###############################################################################


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

        logger.info(f"改进任务信息='{current_task.id}' ")
        task_result = await task_refine(current_task)
        task_save_data(task_result, "task_refine")
        
        logger.info(f"判断原子任务='{task_result.id}' ")
        task_result.results["atom_result"] = ""
        if current_task.task_type == "write":
            db = get_task_db(run_id=current_task.run_id)
            if not db.has_preceding_sibling_design_tasks(current_task):
                task_result.results["atom_result"] = "complex"
                task_result.results["complex_reasons"] = ["design_insufficient"]
                task_result.results["has_preceding_sibling_design_tasks"] = "false"
        if task_result.results["atom_result"] != "complex":
            task_result = await task_atom(current_task)
        task_save_data(task_result, "task_atom")

        if task_result.results.get("atom_result") == "atom": 
            if task_result.task_type == "design":
                logger.info(f"执行: design 任务='{current_task.id}'")
                task_result = await task_design_guideline(task_result)
                task_result = await task_design_execute(task_result)
                task_save_data(task_result, "task_design")
            elif task_result.task_type == "search":
                logger.info(f"执行: search 任务='{current_task.id}'")
                task_result = await task_search(task_result)
                task_save_data(task_result, "task_search")
            elif task_result.task_type == "write":
                logger.info(f"执行: 写作: write 任务='{current_task.id}'")
                task_result = await task_write_plan(task_result)
                task_result = await task_write_draft(task_result)
                task_result = await task_write_critic(task_result)
                task_result = await task_write_refine(task_result)
                task_save_data(task_result, "task_write")

                logger.info(f"执行: 正文摘要: write 任务='{current_task.id}'")
                task_result = await task_summary(task_result)
                task_save_data(task_result, "task_summary")

                keywords_to_skip_review = ["场景", "节拍", "段落"]
                position = task_result.hierarchical_position
                if position and "章" in position and not any(keyword in position for keyword in keywords_to_skip_review):
                    logger.info(f"执行: 审查正文: write 任务='{task_result.id}' ")
                    task_result = await task_write_review(task_result)
                    task_save_data(task_result, "task_write_review")
            else:
                raise ValueError(f"未知的原子任务类型: {task_result.task_type}")
        else:
            logger.info(f"任务 '{current_task.id}' 不是原子任务, 进行分解。")
            if task_result.task_type == "write":
                # complex_reasons = task_result.results.get("complex_reasons", [])
                has_preceding_sibling_design_tasks = task_result.results.get("has_preceding_sibling_design_tasks", "")
                if has_preceding_sibling_design_tasks == "false":
                    logger.info(f"缺少设计方案，分解写作任务='{task_result.id}' ")
                    task_result = await task_plan_write_proposer(task_result)
                    task_result = await task_plan_write_critic(task_result)
                    task_result = await task_plan_write_synthesizer(task_result)
                    task_save_data(task_result, "task_plan")
                else:
                    logger.info(f"划分层级结构，分解写作任务='{task_result.id}' ")
                    task_result = await task_hierarchy_proposer(task_result)
                    task_result = await task_hierarchy_critic(task_result)
                    task_result = await task_hierarchy_synthesizer(task_result)
                    task_save_data(task_result, "task_hierarchy")
            elif task_result.task_type == "design":
                logger.info(f"分解设计任务='{task_result.id}' ")
                task_result = await task_plan_design_proposer(task_result)
                task_result = await task_plan_design_critic(task_result)
                task_result = await task_plan_design_synthesizer(task_result)
                task_save_data(task_result, "task_plan")
            elif task_result.task_type == "search":
                logger.info(f"分解搜索任务='{task_result.id}' ")
                task_result = await task_plan_search_planner(task_result)
                task_result = await task_plan_search_synthesizer(task_result)
                task_save_data(task_result, "task_plan")
            else:
                raise ValueError(f"未知的任务类型: {task_result.task_type}")

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

                    keywords_to_skip_review = ["场景", "节拍", "段落"]
                    position = task_result.hierarchical_position
                    if position and "章" in position and not any(keyword in position for keyword in keywords_to_skip_review):
                        logger.info(f"审查正文: write 任务='{task_result.id}' ")
                        task_result = await task_write_review(task_result)
                        task_save_data(task_result, "task_write_review")
                else:
                    raise ValueError(f"未知的聚合任务类型: {task_result.task_type}")
            else:
                logger.error(f"规划失败, 任务 '{task_result.id}' 没有产生任何子任务。")
                raise Exception(f"任务 '{task_result.id}' 规划失败, 没有子任务。")
