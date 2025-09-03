import asyncio
from typing import Any, Dict
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.context import TaskRunContext
from util.models import Task
from memory import memory
from agents.atom import atom
from agents.design import design
from agents.design_aggregate import design_aggregate
from agents.plan import plan
from agents.write import write
from agents.search import search
from agents.search_aggregate import search_aggregate


def generate_task_cache_key(context: TaskRunContext, parameters: Dict[str, Any]) -> str:
    flow_run_name = context.flow_run.name
    task_id = parameters['task'].id
    return task_input_hash(context, {"flow_run": flow_run_name, "task_id": task_id})


@task(
    name="判断任务原子性",
    cache_key_fn=generate_task_cache_key,
    retries=2, 
    retry_delay_seconds=10 
)
async def task_atom(task: Task) -> Task:
    logger = get_run_logger()
    logger.debug(f"判断任务 '{task.id}' 的原子性...")
    updated_task = await atom(task)
    return updated_task


@task(
    name="分解复杂任务",
    cache_key_fn=generate_task_cache_key,
    retries=2,
    retry_delay_seconds=10
)
async def task_plan(task: Task) -> Task:
    logger = get_run_logger()
    logger.debug(f"为任务 '{task.id}' 进行规划和分解...")
    pland_task = await plan(task)
    return pland_task


@task(
    name="执行原子动作",
    cache_key_fn=generate_task_cache_key,
    retries=2,
    retry_delay_seconds=10
)
async def task_execute(task: Task) -> Task:
    logger = get_run_logger()
    logger.debug(f"执行任务 '{task.id}' (类型: {task.task_type})...")
    if task.task_type == "design":
        return await design(task)
    elif task.task_type == "write":
        return await write(task)
    elif task.task_type == "search":
        return await search(task)
    else:
        raise ValueError(f"未知的任务类型: {task.task_type}")


@task(
    name="整合子任务结果",
    cache_key_fn=generate_task_cache_key,
    retries=2,
    retry_delay_seconds=10
)
async def task_aggregate(task: Task) -> Task:
    logger = get_run_logger()
    logger.debug(f"整合任务 '{task.id}' 的子任务结果...")
    if task.task_type == "design":
        return await design_aggregate(task)
    elif task.task_type == "search":
        return await search_aggregate(task)
    return task


@task(name="存储结果到记忆库", cache_key_fn=generate_task_cache_key)
async def store_memory(task: Task, operation_name: str):
    logger = get_run_logger()
    logger.debug(f"将任务 '{task.id}' 在 '{operation_name}' 步骤的结果存入记忆库...")
    await memory.add(task, operation_name)
    return True


###############################################################################


@flow(name="Agent Sequential Flow", log_prints=True, retries=0)
async def agent_flow(current_task: Task, max_steps: int = 50, current_step: int = 0) -> int:
    logger = get_run_logger()
    if current_step >= max_steps:
        logger.warning(f"已达到最大步骤数 {max_steps}, 流程终止。")
        return current_step

    current_step += 1
    logger.debug(f"--- [步骤 {current_step}/{max_steps}] --- 开始处理任务: {current_task.id} ---")

    ret_atom = await task_atom(current_task)
    await store_memory(ret_atom, "task_atom")

    is_atom = ret_atom.results.get("atom_result") == "atom"
    if is_atom:
        logger.debug(f"任务 '{current_task.id}' 是原子任务, 准备执行。")
        ret_excute = await task_execute(ret_atom)
        await store_memory(ret_excute, "task_execute")
    else:
        logger.debug(f"任务 '{current_task.id}' 是复杂任务, 准备规划子任务。")
        ret_plan = await task_plan(ret_atom)
        await store_memory(ret_plan, "task_plan")

        if ret_plan.sub_tasks:
            for sub_task in ret_plan.sub_tasks:
                if current_step >= max_steps:
                    logger.debug("在处理子任务时达到最大步骤数, 中断执行。")
                    break
                current_step = await agent_flow(sub_task, max_steps, current_step)

            if ret_plan.task_type in ["design", "search"]:
                ret_aggregate = await task_aggregate(ret_plan)
                await store_memory(ret_aggregate, "task_aggregate")
        else:
            logger.error(f"任务 '{ret_plan.id}' 被判断为复杂任务, 但规划后没有产生任何子任务。")
            raise Exception(f"任务 '{ret_plan.id}' 规划失败, 没有子任务。")

    logger.debug(f"--- [步骤 {current_step}] --- 完成处理任务: {current_task.id} ---")
    return current_step

