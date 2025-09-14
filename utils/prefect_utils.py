import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from utils.models import Task
load_dotenv()
prefect_home_path = os.getenv("PREFECT_HOME")
if prefect_home_path:
    Path(prefect_home_path).mkdir(parents=True, exist_ok=True)
from prefect import flow, task
from prefect.context import TaskRunContext
from prefect.filesystems import LocalFileSystem
from prefect.exceptions import ObjectNotFound
from prefect.serializers import JSONSerializer
from loguru import logger


def setup_prefect_storage() -> LocalFileSystem:
    block_name = "write-storage"
    storage_path = Path(__file__).parent.parent.resolve() / ".prefect" / "storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    try:
        storage_block = LocalFileSystem.load(block_name)
        if Path(storage_block.basepath).resolve() != storage_path.resolve():
            logger.warning(f"存储块 '{block_name}' 的路径已过时。正在更新为: {storage_path}")
            storage_block.basepath = str(storage_path)
            storage_block.save(name=block_name, overwrite=True)
        return storage_block
    except (ObjectNotFound, ValueError):
        logger.info(f"Prefect 存储块 '{block_name}' 不存在, 正在创建...")
        storage_block = LocalFileSystem(basepath=str(storage_path))
        storage_block.save(name=block_name, overwrite=True)
        logger.success(f"成功创建并保存了 Prefect 存储块 '{block_name}'。")
        return storage_block

local_storage = setup_prefect_storage()


readable_json_serializer = JSONSerializer(dumps_kwargs={"indent": 2, "ensure_ascii": False})


def get_cache_key(context: TaskRunContext, parameters: Dict[str, Any]) -> str:
    task: Task = parameters["task"]
    task_name = context.task.name.removeprefix("task_")
    extra_params = {k: v for k, v in parameters.items() if k != 'task'}
    extra_key = "_".join(str(v) for k, v in sorted(extra_params.items()))
    base_key = f"{task.run_id}_{task.id}_{task_name}"
    return f"{base_key}_{extra_key}" if extra_key else base_key





