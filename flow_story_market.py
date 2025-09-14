import asyncio
from pathlib import Path
from loguru import logger
from typing import Any, Dict
from prefect import flow, task, get_run_logger
from prefect.filesystems import LocalFileSystem
from prefect.exceptions import ObjectNotFound
from prefect.serializers import JSONSerializer
from prefect.context import TaskRunContext

from utils.models import Task
from agents.design import design
from agents.search import search
from utils.rag import get_rag


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
    name="flow_story_market", 
    flow_run_name="{current_task.run_id}_flow_story_market",
)
async def flow_story_market(current_task: Task):
    ensure_task_logger(current_task.run_id)
    with logger.contextualize(run_id=current_task.run_id):
        logger.info(f"启动市场洞察工作流: {current_task.goal}")
        
        if not current_task.id or not current_task.goal:
            raise ValueError("任务ID和目标不能为空。")

        # 1. 定义子任务
        search_sub_task = Task(
            id=f"{current_task.id}.1",
            parent_id=current_task.id,
            task_type="search",
            goal=f"为分析市场趋势，抓取主流平台（如番茄、七猫）的公开榜单信息和热门作品评论。",
            # 继承父任务的元数据
            category=current_task.category,
            language=current_task.language,
            root_name=current_task.root_name,
            run_id=current_task.run_id,
        )

        design_sub_task = Task(
            id=f"{current_task.id}.2",
            parent_id=current_task.id,
            task_type="design",
            goal="基于收集的市场数据，提炼热门标签、核心爽点和新兴题材，生成结构化的市场趋势报告。",
            dependency=[search_sub_task.id],
            # 继承父任务的元数据
            category=current_task.category,
            language=current_task.language,
            root_name=current_task.root_name,
            run_id=current_task.run_id,
        )
        
        # 将子任务规划存入父任务，以便RAG上下文检索
        current_task.sub_tasks = [search_sub_task, design_sub_task]
        await task_store(current_task, "market_flow_plan")

        # 2. 执行 Search 任务
        logger.info(f"执行 [Search] 任务: {search_sub_task.goal}")
        search_result_task = await task_search(search_sub_task)
        await task_store(search_result_task, "task_search")
        logger.info(f"完成 [Search] 任务，结果已存储。")

        # 3. 执行 Design 任务
        logger.info(f"执行 [Design] 任务: {design_sub_task.goal}")
        # 使用 'market_trends' 类别来调用 design_market_trends_cn.py 提示词
        design_result_task = await task_design(design_sub_task, category="market_trends")
        await task_store(design_result_task, "task_design_market_trends")
        logger.info(f"完成 [Design] 任务，生成市场趋势报告。")

        # 4. 将最终报告作为高优先级文档存入向量数据库
        final_report_content = design_result_task.results.get("design")
        if not final_report_content:
            logger.error("市场趋势分析任务未能生成报告内容，工作流终止。")
            raise ValueError("Design task did not produce a report.")

        final_report_task = Task(
            id=f"{current_task.id}.final_report",
            parent_id=current_task.id,
            task_type="design", # 标记为 design 类型，但内容是最终报告
            goal="市场趋势与题材机会报告 (周期性产出)",
            results={"design": final_report_content},
            category=current_task.category,
            language=current_task.language,
            root_name="题材库", # 使用特定名称“题材库”，便于后续检索
            run_id=current_task.run_id,
        )
        
        await task_store(final_report_task, "market_trends_report")
        logger.success(f"市场洞察工作流完成，【题材库】已成功更新。")
