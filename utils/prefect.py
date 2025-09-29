from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict
from async_lru import alru_cache
from loguru import logger


from utils.file import prefect_storage_path


from prefect.context import TaskRunContext
from prefect.filesystems import LocalFileSystem
from prefect.serializers import JSONSerializer



local_storage: LocalFileSystem = None

@alru_cache(maxsize=1)
async def setup_prefect_storage() -> None:
    block_name = "write-storage"
    from prefect.exceptions import ObjectNotFound
    try:
        storage_block = await LocalFileSystem.load(block_name)
        if Path(storage_block.basepath).resolve() != prefect_storage_path.resolve():
            storage_block.basepath = str(prefect_storage_path)
            await storage_block.save(name=block_name, overwrite=True)
    except (ObjectNotFound, ValueError):
        storage_block = LocalFileSystem(basepath=str(prefect_storage_path))
        await storage_block.save(name=block_name, overwrite=True)

    global local_storage
    local_storage = storage_block



readable_json_serializer = JSONSerializer(dumps_kwargs={"indent": 2, "ensure_ascii": False})



@lru_cache(maxsize=30)
def get_cache_key(context: TaskRunContext, parameters: Dict[str, Any]) -> str:
    from utils.models import Task
    task: Task = parameters["task"]
    task_name = context.task.name.removeprefix("task_")
    extra_params = {k: v for k, v in parameters.items() if k != 'task'}
    extra_key = "_".join(str(v) for k, v in sorted(extra_params.items()))
    base_key = f"{task.run_id}_{task.id}_{task_name}"
    return f"{base_key}_{extra_key}" if extra_key else base_key


