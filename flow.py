import os
from pathlib import Path
from loguru import logger
from diskcache import Cache
from typing import Any, Dict
from prefect import flow, task
from prefect.cache_policies import INPUTS
from datetime import date
from util.models import Task
from util.rag import get_rag
from agents.atom import atom
from agents.design import design
from agents.plan import plan
from agents.write import write
from agents.search import search
from agents.design_aggregate import design_aggregate
from agents.search_aggregate import search_aggregate
from agents.write_aggregate import write_aggregate


day_wordcount_limit = 10000
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
    retries=100, 
    task_run_name="atom: {task.id}",
)
async def task_atom(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        # logger.info(f"{task.run_id} {task.id} {task.task_type} {task.goal}")
        return await atom(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=100, 
    task_run_name="plan: {task.id}",
)
async def task_plan(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        # logger.info(f"{task.run_id} {task.id} {task.task_type} {task.goal}")
        return await plan(task)

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=100, 
    task_run_name="execute: {task.id} - {task.task_type}",
)
async def task_execute(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        # logger.info(f"{task.run_id} {task.id} {task.task_type} {task.goal}")
        if task.task_type == "design":
            return await design(task)
        elif task.task_type == "write":
            ret = await write(task)
            with wordcount_cache.transact():
                today_key = f"{task.run_id}:{date.today().isoformat()}"
                count = wordcount_cache.get(today_key, 0)
                wordcount_cache.set(today_key, count + len(ret.results["result"]))
            return ret
        elif task.task_type == "search":
            return await search(task)
        else:
            raise ValueError(f"{task}")

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=100, 
    task_run_name="aggregate: {task.id} - {task.task_type}",
)
async def task_aggregate(task: Task) -> Task:
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        logger.info(f"{task.run_id} {task.id} {task.task_type} {task.goal}")
        if task.task_type == "design":
            return await design_aggregate(task)
        elif task.task_type == "search":
            return await search_aggregate(task)
        elif task.task_type == "write":
            return await write_aggregate(task)
        else:
            raise ValueError(f"{task}")

@task(
    persist_result=True, 
    cache_policy=INPUTS,
    retries=100,
    task_run_name="rag_store: {task.id} - {operation_name}",
)
async def task_rag_store(task: Task, operation_name: str):
    ensure_task_logger(task.run_id)
    with logger.contextualize(run_id=task.run_id):
        # logger.info(f"{task.run_id} {task.id} {task.task_type} {task.goal}")
        if not task.id:
            raise ValueError(f"任务信息中未找到任务ID {task.id} \n 任务信息: {task}")
        await get_rag().add(task, operation_name)
        return True

@flow(
    name="flow_write", 
    flow_run_name="flow_write: {current_task.run_id} - {current_task.id}"
)
async def flow_write(current_task: Task):
    logger.info(f"{current_task.run_id} {current_task.id} {current_task.task_type} {current_task.goal}")

    today_key = f"{current_task.run_id}:{date.today().isoformat()}"
    day_wordcount = wordcount_cache.get(today_key, 0)
    if day_wordcount > day_wordcount_limit:
        logger.info(f"已完成当天任务 {current_task.run_id} {day_wordcount}")
        return

    ret_atom = await task_atom(current_task)
    await task_rag_store(ret_atom, "task_atom")

    is_atom = ret_atom.results.get("atom_result") == "atom"
    if is_atom:
        ret_excute = await task_execute(ret_atom)
        await task_rag_store(ret_excute, "task_execute")
    else:
        ret_plan = await task_plan(ret_atom)
        await task_rag_store(ret_plan, "task_plan")

        if ret_plan.sub_tasks:
            for sub_task in ret_plan.sub_tasks:
                today_key = f"{sub_task.run_id}:{date.today().isoformat()}"
                day_wordcount = wordcount_cache.get(today_key, 0)
                if day_wordcount > day_wordcount_limit:
                    logger.info(f"已完成当天任务 {sub_task.run_id} {day_wordcount}")
                    return
                
                await flow_write(sub_task)

            ret_aggregate = await task_aggregate(ret_plan)
            await task_rag_store(ret_aggregate, "task_aggregate")
        else:
            logger.error(f"规划失败 {ret_plan}")
            raise Exception(f"任务 '{ret_plan.id}' 规划失败, 没有子任务。")
