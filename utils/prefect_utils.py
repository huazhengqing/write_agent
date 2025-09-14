from pathlib import Path
from loguru import logger
from prefect.filesystems import LocalFileSystem
from prefect.exceptions import ObjectNotFound
from prefect.serializers import JSONSerializer


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


readable_json_serializer = JSONSerializer(dumps_kwargs={"indent": 2, "ensure_ascii": False})


