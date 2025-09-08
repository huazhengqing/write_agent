import os
from pathlib import Path
from loguru import logger
from diskcache import Cache
from typing import Any, Dict
from prefect import flow, task
from prefect.cache_policies import INPUTS
from datetime import date
from utils.models import Task
from utils.rag import get_rag
from agents.atom import atom
from agents.plan import plan
from agents.plan_reflection import plan_reflection
from agents.design import design
from agents.design_reflection import design_reflection
from agents.design_aggregate import design_aggregate
from agents.search import search
from agents.search_aggregate import search_aggregate
from agents.write import write
from agents.write_reflection import write_reflection
from agents.summary import summary
from agents.summary_aggregate import summary_aggregate


day_wordcount_goal = 10000
wordcount_cache = Cache(os.path.join(".cache", "wordcount"), size_limit=int(32 * (1024**2)))


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


@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="atom: {task.run_id} - {task.id}",
)
async def task_atom(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await atom(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="plan: {task.run_id} - {task.id}",
)
async def task_plan(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="plan_reflection: {task.run_id} - {task.id}",
)
async def task_plan_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await plan_reflection(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="execute_design: {task.run_id} - {task.id}",
)
async def task_execute_design(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="execute_design_reflection: {task.run_id} - {task.id}",
)
async def task_execute_design_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design_reflection(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="execute_search: {task.run_id} - {task.id}",
)
async def task_execute_search(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await search(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="execute_write: {task.run_id} - {task.id}",
)
async def task_execute_write(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        ret = await write(task)
        return ret

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="execute_write_reflection: {task.run_id} - {task.id}",
)
async def task_execute_write_reflection(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        ret = await write_reflection(task)
        with wordcount_cache.transact():
            today_key = f"{task.run_id}:{date.today().isoformat()}"
            count = wordcount_cache.get(today_key, 0)
            wordcount_cache.set(today_key, count + len(ret.results["write_reflection"]))
        return ret

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="execute_write_summary: {task.run_id} - {task.id}",
)
async def task_execute_summary(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        ret = await summary(task)
        return ret

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="aggregate_design: {task.run_id} - {task.id}",
)
async def task_aggregate_design(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await design_aggregate(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="aggregate_search: {task.run_id} - {task.id}",
)
async def task_aggregate_search(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await search_aggregate(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="aggregate_summary: {task.run_id} - {task.id}",
)
async def task_aggregate_summary(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        return await summary_aggregate(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=1,
    task_run_name="store: {task.run_id} - {task.id} - {operation_name}",
)
async def task_store(task: Task, operation_name: str):
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        if not task.id or not task.goal:
            raise ValueError(f"任务信息中未找到任务ID {task.id} \n 任务信息: {task}")
        if task.task_type not in ["design", "search", "write"]:
            raise ValueError(f"未知的任务类型: {task.task_type}")
        await get_rag().add(task, operation_name)
        return True

@flow(
    name="flow_write", 
    flow_run_name="flow_write: {current_task.run_id} - {current_task.id}"
)
async def flow_write(current_task: Task):
    """
    主工作流, 用于递归处理写作任务。

    该流程首先判断任务是否为“原子”任务(即不可再分)。
    - 如果是原子任务, 则直接执行(设计、搜索或写作)。
    - 如果不是原子任务, 则将其分解为子任务, 对每个子任务递归调用本流程, 
      最后将所有子任务的结果聚合起来。

    Args:
        current_task: 当前需要处理的任务对象。
    """
    logger.info(f"开始处理任务: {current_task.run_id} {current_task.id} {current_task.task_type} {current_task.goal}")
    
    # 验证任务基本信息
    if not current_task.id or not current_task.goal:
        raise ValueError("任务ID和目标不能为空。")

    # 检查是否完成当日字数目标
    today_key = f"{current_task.run_id}:{date.today().isoformat()}"
    day_wordcount = wordcount_cache.get(today_key, 0)
    if day_wordcount > day_wordcount_goal:
        logger.info(f"已达到当日字数目标, 完成任务: {current_task.run_id} ({day_wordcount}字)")
        return

    # 步骤1: 判断任务是否为原子任务
    # 原子任务是足够小、可以被单个Agent直接处理的任务。
    ret_atom = await task_atom(current_task)
    await task_store(ret_atom, "task_atom")

    is_atom = ret_atom.results.get("atom_result") == "atom"
    
    # 步骤2: 根据是否为原子任务, 选择不同执行路径
    if is_atom:
        # --- 原子任务执行路径 ---
        logger.info(f"任务 '{current_task.id}' 是原子任务, 直接执行。")
        
        if ret_atom.task_type == "design":
            # 执行设计任务
            ret_excute = await task_execute_design(ret_atom)
            await task_store(ret_excute, "task_execute_design")
        elif ret_atom.task_type == "search":
            # 执行搜索任务
            ret_excute = await task_execute_search(ret_atom)
            await task_store(ret_excute, "task_execute_search")
        elif ret_atom.task_type == "write":
            # 执行完整的写作流程: 设计反思 -> 写作 -> 写作反思 -> 总结
            if not ret_atom.length:
                raise ValueError("写作任务没有长度要求")
            
            # 设计反思
            ret_design_reflection = await task_execute_design_reflection(ret_atom)
            await task_store(ret_design_reflection, "task_execute_design_reflection")

            # 核心写作
            ret_write = await task_execute_write(ret_design_reflection)
            await task_store(ret_write, "task_execute_write")

            # 写作反思
            ret_write_reflection = await task_execute_write_reflection(ret_write)
            await task_store(ret_write_reflection, "task_execute_write_reflection")

            # 生成总结
            ret_write_summary = await task_execute_summary(ret_write_reflection)
            await task_store(ret_write_summary, "task_execute_summary")
        else:
            # 未知任务类型, 抛出异常
            raise ValueError(f"未知的原子任务类型: {ret_atom.task_type}")
    else:
        # --- 任务分解与递归路径 ---
        logger.info(f"任务 '{current_task.id}' 不是原子任务, 进行规划和分解。")

        # 步骤 2.1: 规划子任务
        ret_plan = await task_plan(ret_atom)
        await task_store(ret_plan, "task_plan")

        # 步骤 2.2: 对规划进行反思和调整
        ret_plan_reflection = await task_plan_reflection(ret_plan)
        await task_store(ret_plan_reflection, "task_plan_reflection")

        if ret_plan_reflection.sub_tasks:
            # 步骤 2.3: 递归处理所有子任务
            logger.info(f"任务 '{current_task.id}' 分解为 {len(ret_plan_reflection.sub_tasks)} 个子任务, 开始递归处理。")
            for sub_task in ret_plan_reflection.sub_tasks:
                # 每次递归前都检查字数
                today_key = f"{sub_task.run_id}:{date.today().isoformat()}"
                day_wordcount = wordcount_cache.get(today_key, 0)
                if day_wordcount > day_wordcount_goal:
                    logger.info(f"已完成当日字数目标, 暂停处理后续子任务: {sub_task.run_id}")
                    return
                
                # 递归调用自身来处理子任务
                await flow_write(sub_task)
            
            # 步骤 2.4: 聚合子任务结果
            logger.info(f"所有子任务处理完毕, 开始聚合任务 '{current_task.id}' 的结果。")
            if ret_plan_reflection.task_type == "design":
                ret_aggregate = await task_aggregate_design(ret_plan_reflection)
                await task_store(ret_aggregate, "task_aggregate_design")
            elif ret_plan_reflection.task_type == "search":
                ret_aggregate = await task_aggregate_search(ret_plan_reflection)
                await task_store(ret_aggregate, "task_aggregate_search")
            elif ret_plan_reflection.task_type == "write":
                ret_aggregate = await task_aggregate_summary(ret_plan_reflection)
                await task_store(ret_aggregate, "task_aggregate_summary")
            else:
                raise ValueError(f"未知的聚合任务类型: {ret_plan_reflection.task_type}")
        else:
            # 如果规划后没有产生子任务, 说明规划失败
            logger.error(f"规划失败, 任务 '{ret_plan_reflection.id}' 没有产生任何子任务。")
            raise Exception(f"任务 '{ret_plan_reflection.id}' 规划失败, 没有子任务。")
