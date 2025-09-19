import os
import sys
import json
import pytest
from loguru import logger
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vector import get_vector_store, vector_add, vector_add_from_dir
from tests.test_data import VECTOR_TEST_SIMPLE_CN, VECTOR_TEST_SIMPLE_JSON


@pytest.fixture(scope="function")
def vector_store(test_dirs):
    """为每个入库测试函数提供一个干净的向量存储。"""
    return get_vector_store(db_path=test_dirs["db_path"], collection_name="ingestion_test")


@pytest.mark.asyncio
async def test_vector_add_single(vector_store):
    """测试通过 vector_add 添加入库单个内容。"""
    logger.info("--- 测试 vector_add (单个内容) ---")

    # 测试添加普通文本
    added_text = vector_add(vector_store, VECTOR_TEST_SIMPLE_CN, {"type": "text", "source": "manual_add"}, doc_id="manual_text_1")
    assert added_text

    # 测试添加 JSON 内容
    added_json = vector_add(vector_store, content=VECTOR_TEST_SIMPLE_JSON, metadata={"type": "json", "source": "manual_json"}, content_format="json", doc_id="manual_json_1")
    assert added_json


@pytest.mark.asyncio
async def test_vector_add_edge_cases(vector_store):
    """测试 vector_add 的边缘场景。"""
    logger.info("--- 测试 vector_add (边缘场景) ---")

    # 测试添加空内容
    added_empty = vector_add(vector_store, content="  ", metadata={"type": "empty"}, doc_id="empty_content")
    assert not added_empty

    # 测试添加包含错误关键字的内容
    added_error = vector_add(vector_store, content="生成报告时出错。", metadata={"type": "error"}, doc_id="error_content")
    assert not added_error

    # 测试添加无法解析出节点的内容
    added_no_nodes = vector_add(vector_store, content="---\n---\n", metadata={"type": "no_nodes"}, doc_id="no_nodes_content")
    assert not added_no_nodes


@pytest.mark.asyncio
async def test_vector_add_from_dir(ingested_store):
    """测试从目录添加入库并验证结果。"""
    logger.info("--- 测试 vector_add_from_dir (常规) ---")
    # `ingested_store` fixture 已经执行了添加入库操作，我们只需验证结果。
    client = ingested_store.client
    collection = client.get_collection("test_collection")
    count = collection.count()
    logger.info(f"集合中共有 {count} 个节点。")
    assert count > 0
    logger.success("--- 从目录添加入库测试通过 ---")


@pytest.mark.asyncio
async def test_vector_add_from_empty_dir(vector_store, test_dirs):
    """测试从空目录或仅包含无效文件的目录添加入库。"""
    logger.info("--- 测试 vector_add_from_dir (空目录) ---")
    empty_input_dir = Path(test_dirs["input_path"]) / "empty_dir"
    empty_input_dir.mkdir(exist_ok=True)
    (empty_input_dir / "unsupported.log").write_text("log data", encoding='utf-8')
    (empty_input_dir / "empty.txt").write_text("   ", encoding='utf-8')

    added_from_empty = vector_add_from_dir(vector_store, str(empty_input_dir))
    assert not added_from_empty
    logger.success("--- 从空目录或无效文件目录添加入库测试通过 ---")