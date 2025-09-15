import os
from pathlib import Path
import hashlib
import json
from typing import Any, Dict
from dotenv import load_dotenv
from utils.models import Task
load_dotenv()
prefect_home_path = os.getenv("PREFECT_HOME")
if prefect_home_path:
    Path(prefect_home_path).mkdir(parents=True, exist_ok=True)
from prefect.context import TaskRunContext
from prefect.filesystems import LocalFileSystem
from prefect.exceptions import ObjectNotFound
from prefect.serializers import JSONSerializer
from loguru import logger
from prefect import flow, task


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


def generate_readable_cache_key(context: TaskRunContext, parameters: Dict[str, Any]) -> str:
    """
    为 Prefect 任务生成一个人类可读的、基于其所有参数的缓存键。

    这个函数会创建一个如下格式的键：
    `{task_name}/{simple_param_values}_{hash_of_complex_params}.json`

    - 'task_name': 任务的函数名，作为子目录。
    - 'simple_param_values': 由简短、适合做文件名的参数值（如字符串、数字）组成。
    - 'hash_of_complex_params': 复杂或过长的参数（如字典、列表、长文本）的MD5哈希值。

    这确保了：
    1. 缓存键对于相同的输入是唯一的。
    2. 生成的文件名部分可读，便于调试和检查。
    3. 文件名对于文件系统是安全的。
    """
    task_name = context.task.fn.__name__

    simple_parts = []
    complex_params = {}

    # 1. 按键排序以保证顺序，然后将参数分为简单和复杂两类
    for key, value in sorted(parameters.items()):
        is_simple = isinstance(value, (str, int, float, bool)) and len(str(value)) < 50 and '/' not in str(value) and '\\' not in str(value)
        if is_simple:
            # 将空格替换为下划线，确保文件名安全
            simple_parts.append(str(value).replace(" ", "_"))
        else:
            complex_params[key] = value

    # 2. 从复杂参数构建哈希值
    hash_part = ""
    if complex_params:
        # 使用 json.dumps 序列化复杂参数，确保一致性
        complex_bytes = json.dumps(complex_params, sort_keys=True, default=str).encode('utf-8')
        hash_part = hashlib.md5(complex_bytes).hexdigest()[:12]

    # 3. 组合文件名
    filename_parts = simple_parts
    if hash_part:
        filename_parts.append(hash_part)

    # 如果没有任何参数，或者所有参数都太复杂，则生成一个总哈希作为文件名
    if not filename_parts:
        all_params_bytes = json.dumps(parameters, sort_keys=True, default=str).encode('utf-8')
        filename = hashlib.md5(all_params_bytes).hexdigest()[:12]
    else:
        filename = "_".join(filename_parts)

    # 最终的键格式为 "task_name/generated_filename.json"
    return f"{task_name}/{filename}.json"


def get_cache_key(context: TaskRunContext, parameters: Dict[str, Any]) -> str:
    task: Task = parameters["task"]
    task_name = context.task.name.removeprefix("task_")
    extra_params = {k: v for k, v in parameters.items() if k != 'task'}
    extra_key = "_".join(str(v) for k, v in sorted(extra_params.items()))
    base_key = f"{task.run_id}_{task.id}_{task_name}"
    return f"{base_key}_{extra_key}" if extra_key else base_key
