import asyncio
from pathlib import Path
from agents import route
from loguru import logger
from typing import Any, Dict, Callable, Literal
from prefect import flow, task, get_run_logger
from prefect.filesystems import LocalFileSystem
from prefect.exceptions import ObjectNotFound
from prefect.serializers import JSONSerializer
from prefect.context import TaskRunContext
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from utils.models import Task, get_preceding_sibling_ids
from agents.atom import atom
from agents.plan import plan, plan_reflection
from agents.design import design, design_aggregate, design_reflection
from agents.search import search, search_aggregate
from agents.write import write, write_before_reflection, write_reflection
from agents.summary import summary, summary_aggregate
from agents.hierarchy import hierarchy, hierarchy_reflection
from agents.review import review_design, review_write
from utils.rag import get_rag
from utils.db import get_db


def setup_prefect_storage() -> LocalFileSystem:
    block_name = "write-storage"
    project_dir = Path(__file__).parent.resolve()
    storage_path = project_dir / ".prefect" / "storage"
    try:
        storage_block = LocalFileSystem.load(block_name)
        logger.info(f"成功加载已存在的 Prefect 存储块 '{block_name}'。")
        return storage_block
    except (ObjectNotFound, ValueError):
        logger.info(f"Prefect 存储块 '{block_name}' 不存在, 正在创建...")
        storage_path.mkdir(parents=True, exist_ok=True)
        storage_block = LocalFileSystem(basepath=str(storage_path))
        storage_block.save(name=block_name, overwrite=True)
        logger.success(f"成功创建并保存了 Prefect 存储块 '{block_name}'。")
        return storage_block

local_storage = setup_prefect_storage()


readable_json_serializer = JSONSerializer(dumps_kwargs={"indent": 2, "ensure_ascii": False})


_SINK_IDS = {}
def ensure_task_logger(run_id: str):
    if run_id in _SINK_IDS:
        return
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    sink_id = logger.add(
        log_dir / f"{run_id}.log",
        filter=lambda record: record["extra"].get("run_id") == run_id,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    _SINK_IDS[run_id] = sink_id


def get_cache_key(context: TaskRunContext, parameters: Dict[str, Any]) -> str:
    task: Task = parameters["task"]
    task_name = context.task.name.removeprefix("task_")
    extra_params = {k: v for k, v in parameters.items() if k != 'task'}
    extra_key = "_".join(str(v) for k, v in sorted(extra_params.items()))
    base_key = f"{task.run_id}_{task.id}_{task_name}"
    return f"{base_key}_{extra_key}" if extra_key else base_key

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
async def task_store(task: Task, operation_name: str) -> bool:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        if not task.id or not task.goal:
            raise ValueError(f"传递给 task_store 的任务信息不完整, 缺少ID或目标: {task}")
        await get_rag().add(task, operation_name)
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
            if not task_result.length:
                raise ValueError("写作任务没有长度要求")

        day_wordcount_goal = getattr(current_task, 'day_wordcount_goal', 0)
        if day_wordcount_goal > 0:
            db = get_db(run_id=current_task.run_id, category=current_task.category)
            word_count_24h = await asyncio.to_thread(db.get_word_count_last_24h)
            if word_count_24h >= day_wordcount_goal:
                logger.info(f"已达到最近24小时字数目标 ({word_count_24h}字), 暂停任务: {current_task.run_id}")
                return

        logger.info(f"判断原子任务='{task_result.id}' ")
        task_result = await task_atom(current_task)
        await task_store(task_result, "task_atom")
        if task_result.results.get("atom_result") == "atom": 
            if task_result.task_type == "design":
                logger.info(f"执行: design 任务='{current_task.id}'")
                category = await task_route(task_result)
                logger.info(f"执行: design {category} 任务='{current_task.id}'")
                task_result = await task_design(task_result, category=category)
                await task_store(task_result, f"task_design_{category}")
            elif task_result.task_type == "search":
                logger.info(f"执行: search 任务='{current_task.id}'")
                task_result = await task_search(task_result)
                await task_store(task_result, "task_search")
            elif task_result.task_type == "write":
                logger.info(f"执行: 设计反思: write 任务='{current_task.id}'")
                task_result = await task_write_before_reflection(task_result)
                await task_store(task_result, "task_write_before_reflection")

                logger.info(f"执行: 写作初稿: write 任务='{current_task.id}'")
                task_result = await task_write(task_result)
                await task_store(task_result, "task_write")

                logger.info(f"执行: 反思初稿: write 任务='{current_task.id}'")
                task_result = await task_write_reflection(task_result)
                await task_store(task_result, "task_write_reflection")

                logger.info(f"执行: 正文摘要: write 任务='{current_task.id}'")
                task_result = await task_summary(task_result)
                await task_store(task_result, "task_summary")
                
                keywords_to_skip_review = ["全书", "卷", "幕", "段落", "场景"]
                position = task_result.hierarchical_position
                if position and "章" in position and not any(keyword in position for keyword in keywords_to_skip_review):
                    logger.info(f"执行: 审查正文: write 任务='{task_result.id}' ")
                    task_result = await task_review_write(task_result)
                    await task_store(task_result, "task_review_write")
            else:
                raise ValueError(f"未知的原子任务类型: {task_result.task_type}")
        else:
            logger.info(f"任务 '{current_task.id}' 不是原子任务, 进行分解。")
            complex_reasons = task_result.results.get("complex_reasons", [])
            if (task_result.task_type == "write" and 'design_insufficient' not in complex_reasons):
                logger.info(f"审查设计方案: write 任务='{task_result.id}' ")
                task_result = await task_review_design(task_result)
                await task_store(task_result, "task_review_design")

                logger.info(f"划分结构: write 任务='{task_result.id}' ")
                task_result = await task_hierarchy(task_result)
                await task_store(task_result, "task_hierarchy")

                logger.info(f"反思划分结构结果: write 任务='{task_result.id}' ")
                task_result = await task_hierarchy_reflection(task_result)
                await task_store(task_result, "task_hierarchy_reflection")
            
            logger.info(f"分解: write 任务='{task_result.id}' ")
            task_result = await task_plan(task_result)
            await task_store(task_result, "task_plan")

            if task_result.task_type in ["write", "design"]:
                logger.info(f"反思分解结果: write 任务='{task_result.id}' ")
                task_result = await task_plan_reflection(task_result)
                await task_store(task_result, "task_plan_reflection")

            if task_result.sub_tasks:
                logger.info(f"任务 '{current_task.id}' 分解为 {len(task_result.sub_tasks)} 个子任务, 开始递归处理。")
                for sub_task in task_result.sub_tasks:

                    if day_wordcount_goal > 0:
                        db = get_db(run_id=current_task.run_id, category=current_task.category)
                        word_count_24h = await asyncio.to_thread(db.get_word_count_last_24h)
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
                        await task_store(task_result, "task_review_write")

                if task_result.task_type == "design":
                    logger.info(f"聚合: design 任务='{task_result.id}' ")
                    task_result = await task_aggregate_design(task_result)
                    await task_store(task_result, "task_aggregate_design")
                elif task_result.task_type == "search":
                    logger.info(f"聚合: search 任务='{task_result.id}' ")
                    task_result = await task_aggregate_search(task_result)
                    await task_store(task_result, "task_aggregate_search")
                elif task_result.task_type == "write":
                    logger.info(f"聚合摘要: write 任务='{task_result.id}' ")
                    task_result = await task_aggregate_summary(task_result)
                    await task_store(task_result, "task_aggregate_summary")
                else:
                    raise ValueError(f"未知的聚合任务类型: {task_result.task_type}")
            else:
                logger.error(f"规划失败, 任务 '{task_result.id}' 没有产生任何子任务。")
                raise Exception(f"任务 '{task_result.id}' 规划失败, 没有子任务。")
