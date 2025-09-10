import asyncio
from pathlib import Path
from loguru import logger
from typing import Any, Dict
from prefect import flow, task
from prefect.cache_policies import INPUTS
from prefect.filesystems import LocalFileSystem
from prefect.exceptions import ObjectNotFound
from prefect.context import TaskRunContext
from utils.models import Task


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

day_wordcount_goal = 10000

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
    operation_name: str = parameters.get("operation_name", "")
    base_key = f"{task.run_id}_{task.id}_{context.task.name}"
    return f"{base_key}_{operation_name}" if operation_name else base_key


@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/atom.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_atom",
)
async def task_atom(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.atom import atom
        return await atom(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_before_reflection.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_plan_before_reflection",
)
async def task_plan_before_reflection(task: Task) -> Task:
    if task.id == "1":
        return task
    if task.task_type != "write":
        return task
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.plan_before_reflection import plan_before_reflection
        return await plan_before_reflection(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_plan",
)
async def task_plan(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.plan import plan
        return await plan(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/plan_reflection.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_plan_reflection",
)
async def task_plan_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.plan_reflection import plan_reflection
        return await plan_reflection(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/execute_design.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_execute_design",
)
async def task_execute_design(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.design import design
        return await design(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/execute_design_reflection.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_execute_design_reflection",
)
async def task_execute_design_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.design_reflection import design_reflection
        return await design_reflection(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/execute_search.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_execute_search",
)
async def task_execute_search(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.search import search
        return await search(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/execute_write_before_reflection.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_execute_write_before_reflection",
)
async def task_execute_write_before_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.write_before_reflection import write_before_reflection
        return await write_before_reflection(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/execute_write.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_execute_write",
)
async def task_execute_write(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.write import write
        ret = await write(task)
        return ret

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/execute_write_reflection.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_execute_write_reflection",
)
async def task_execute_write_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.write_reflection import write_reflection
        ret = await write_reflection(task)
        return ret

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/execute_write_summary.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_execute_write_summary",
)
async def task_execute_summary(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.summary import summary
        ret = await summary(task)
        return ret

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/aggregate_design.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_aggregate_design",
)
async def task_aggregate_design(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.design_aggregate import design_aggregate
        return await design_aggregate(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/aggregate_search.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_aggregate_search",
)
async def task_aggregate_search(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.search_aggregate import search_aggregate
        return await search_aggregate(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/aggregate_summary.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_aggregate_summary",
)
async def task_aggregate_summary(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        from agents.summary_aggregate import summary_aggregate
        return await summary_aggregate(task)

@task(
    persist_result=True, 
    result_storage=local_storage,
    cache_key_fn=get_cache_key, 
    result_storage_key="{parameters[task].run_id}/{parameters[task].id}/store_{parameters[operation_name]}.pickle", 
    retries=1,
    task_run_name="{task.run_id}_{task.id}_store_{operation_name}",
)
async def task_store(task: Task, operation_name: str) -> bool:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        if not task.id or not task.goal:
            raise ValueError(f"传递给 task_store 的任务信息不完整, 缺少ID或目标: {task}")
        from utils.rag import get_rag
        await get_rag().add(task, operation_name)
        return True

@flow(
    persist_result=False, 
    result_storage=local_storage,
    name="flow_write", 
    flow_run_name="{current_task.run_id}_flow_write_{current_task.id}",
)
async def flow_write(current_task: Task):
    ensure_task_logger(current_task.run_id)
    with logger.contextualize(run_id=current_task.run_id):
        logger.info(f"开始处理任务: {current_task.run_id} {current_task.id} {current_task.task_type} {current_task.goal}")
        
        if not current_task.id or not current_task.goal:
            raise ValueError("任务ID和目标不能为空。")

        from utils.db import get_db
        db = get_db(run_id=current_task.run_id, category=current_task.category)
        word_count_24h = await asyncio.to_thread(db.get_word_count_last_24h)
        if word_count_24h >= day_wordcount_goal:
            logger.info(f"已达到最近24小时字数目标 ({word_count_24h}字), 暂停任务: {current_task.run_id}")
            return

        # 判断任务是否为原子任务
        task_result = await task_atom(current_task)
        await task_store(task_result, "task_atom")
        if task_result.results.get("atom_result") == "atom": 
            logger.info(f"任务 '{current_task.id}' 是原子任务, 直接执行。")
            
            if task_result.task_type == "design":
                task_result = await task_execute_design(task_result)
                await task_store(task_result, "task_execute_design")
                
                task_result = await task_execute_design_reflection(task_result)
                await task_store(task_result, "task_execute_design_reflection")
            elif task_result.task_type == "search":
                task_result = await task_execute_search(task_result)
                await task_store(task_result, "task_execute_search")
            elif task_result.task_type == "write":
                # 执行完整的写作流程: 设计反思 -> 写作 -> 写作反思 -> 总结
                if not task_result.length:
                    raise ValueError("写作任务没有长度要求")
                
                task_result = await task_execute_write_before_reflection(task_result)
                await task_store(task_result, "task_execute_write_before_reflection")

                task_result = await task_execute_write(task_result)
                await task_store(task_result, "task_execute_write")

                task_result = await task_execute_write_reflection(task_result)
                await task_store(task_result, "task_execute_write_reflection")

                task_result = await task_execute_summary(task_result)
                await task_store(task_result, "task_execute_summary")
            else:
                raise ValueError(f"未知的原子任务类型: {task_result.task_type}")
        else:
            logger.info(f"任务 '{current_task.id}' 不是原子任务, 进行分解。")

            task_result = await task_plan_before_reflection(task_result)
            await task_store(task_result, "task_plan_before_reflection")

            task_result = await task_plan(task_result)
            await task_store(task_result, "task_plan")

            task_result = await task_plan_reflection(task_result)
            await task_store(task_result, "task_plan_reflection")

            if task_result.sub_tasks:
                logger.info(f"任务 '{current_task.id}' 分解为 {len(task_result.sub_tasks)} 个子任务, 开始递归处理。")
                for sub_task in task_result.sub_tasks:
                    word_count_24h = await asyncio.to_thread(db.get_word_count_last_24h)
                    if word_count_24h >= day_wordcount_goal:
                        logger.info(f"已达到最近24小时字数目标 ({word_count_24h}字), 暂停处理后续子任务: {sub_task.run_id}")
                        return
                    
                    await flow_write(sub_task)
                
                logger.info(f"所有子任务处理完毕, 开始聚合任务 '{current_task.id}' 的结果。")
                if task_result.task_type == "design":
                    task_result = await task_aggregate_design(task_result)
                    await task_store(task_result, "task_aggregate_design")
                elif task_result.task_type == "search":
                    task_result = await task_aggregate_search(task_result)
                    await task_store(task_result, "task_aggregate_search")
                elif task_result.task_type == "write":
                    task_result = await task_aggregate_summary(task_result)
                    await task_store(task_result, "task_aggregate_summary")
                else:
                    raise ValueError(f"未知的聚合任务类型: {task_result.task_type}")
            else:
                logger.error(f"规划失败, 任务 '{task_result.id}' 没有产生任何子任务。")
                raise Exception(f"任务 '{task_result.id}' 规划失败, 没有子任务。")
