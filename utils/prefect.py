from pathlib import Path
from typing import Any, Dict
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

from prefect.context import TaskRunContext
from prefect.filesystems import LocalFileSystem
from prefect.serializers import JSONSerializer


def setup_prefect_storage() -> LocalFileSystem:
    block_name = "write-storage"
    from prefect.exceptions import ObjectNotFound
    from utils.file import prefect_storage_path
    try:
        storage_block = LocalFileSystem.load(block_name)
        if Path(storage_block.basepath).resolve() != prefect_storage_path.resolve():
            logger.warning(f"存储块 '{block_name}' 的路径已过时。正在更新为: {prefect_storage_path}")
            storage_block.basepath = str(prefect_storage_path)
            storage_block.save(name=block_name, overwrite=True)
        return storage_block
    except (ObjectNotFound, ValueError):
        logger.info(f"Prefect 存储块 '{block_name}' 不存在, 正在创建...")
        storage_block = LocalFileSystem(basepath=str(prefect_storage_path))
        storage_block.save(name=block_name, overwrite=True)
        logger.success(f"成功创建并保存了 Prefect 存储块 '{block_name}'。")
        return storage_block

local_storage = setup_prefect_storage()


readable_json_serializer = JSONSerializer(dumps_kwargs={"indent": 2, "ensure_ascii": False})


def get_cache_key(context: TaskRunContext, parameters: Dict[str, Any]) -> str:
    from utils.models import Task
    task: Task = parameters["task"]
    task_name = context.task.name.removeprefix("task_")
    extra_params = {k: v for k, v in parameters.items() if k != 'task'}
    extra_key = "_".join(str(v) for k, v in sorted(extra_params.items()))
    base_key = f"{task.run_id}_{task.id}_{task_name}"
    return f"{base_key}_{extra_key}" if extra_key else base_key
