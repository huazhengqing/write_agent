import os
import sys
import json
import pytest
import chromadb
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
async def test_vector_add_similarity_check(vector_store, test_dirs):
    """测试 vector_add 的相似度检查和去重功能。"""
    logger.info("--- 测试 vector_add (相似度检查) ---")

    original_content = "这是一个用于测试相似度功能的原始文档，内容独特。"
    # 相似内容，与原始文档的嵌入向量会非常接近
    similar_content = "这是一个用于测试相似度功能的原始文档，内容独特。"
    # 更新后的内容，与原始文档也相似，但用于测试更新逻辑
    updated_content = "这是一个用于测试相似度功能的原始文档，内容独特，但现在它被更新了。"

    # 1. 首次添加原始文档
    added_original = vector_add(
        vector_store,
        content=original_content,
        metadata={"type": "similarity_test", "version": 1},
        doc_id="sim_test_1"
    )
    assert added_original
    logger.info("步骤1: 首次添加原始文档成功。")

    # 2. 尝试添加一个高度相似但不同ID的文档，并开启相似度检查
    # 预期：添加失败 (被去重)
    added_similar = vector_add(
        vector_store,
        content=similar_content,
        metadata={"type": "similarity_test", "version": "similar"},
        doc_id="sim_test_2",
        check_similarity=True,
        similarity_threshold=0.99 # 相似度检查阈值设得很高
    )
    assert not added_similar
    logger.info("步骤2: 添加高度相似的文档被成功阻止。")

    # 3. 尝试更新原始文档（相同ID），内容也高度相似，并开启相似度检查
    # 预期：添加成功 (因为doc_id相同，应被视为更新操作，绕过相似度检查)
    added_update = vector_add(
        vector_store,
        content=updated_content,
        metadata={"type": "similarity_test", "version": 2},
        doc_id="sim_test_1",
        check_similarity=True,
        similarity_threshold=0.99
    )
    assert added_update
    logger.info("步骤3: 使用相似内容更新原始文档成功。")

    # 4. 验证数据库的最终状态
    client = chromadb.PersistentClient(path=test_dirs["db_path"])
    collection = client.get_collection("ingestion_test")
    
    # 由于 sim_test_2 被去重，sim_test_1 被更新，最终集合中应该只包含 sim_test_1 的更新后节点。
    # 假设每个短文本只生成一个节点。
    count = collection.count()
    assert count == 1
    logger.info(f"步骤4: 最终集合中的节点数量为 {count}，符合预期。")

    # 进一步验证留下的节点确实是更新后的版本
    retrieved_docs = collection.get(where={"doc_id": "sim_test_1"})
    assert len(retrieved_docs['ids']) == 1
    assert retrieved_docs['documents'][0] == updated_content
    assert retrieved_docs['metadatas'][0]['version'] == 2
    logger.info("步骤4: 验证集合中的文档内容和元数据均为更新后的版本。")

    logger.success("--- 相似度检查测试通过 ---")


@pytest.mark.asyncio
async def test_vector_add_from_dir(ingested_store, test_dirs):
    """测试从目录添加入库并验证结果。"""
    logger.info("--- 测试 vector_add_from_dir (常规) ---")
    # `ingested_store` fixture 已经执行了添加入库操作，我们只需验证结果。
    # 由于 llama-index 更新，不再直接暴露 chroma_collection 属性。
    # 我们通过重新连接到数据库来验证。
    client = chromadb.PersistentClient(path=test_dirs["db_path"])
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