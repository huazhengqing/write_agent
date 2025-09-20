import os
import sys
import hashlib
import pytest
from loguru import logger
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.kg import get_kg_store, kg_add
from utils.vector import get_vector_store
from tests.test_data import VECTOR_TEST_DATASET


@pytest.fixture(scope="function")
def basic_kg_stores(tmp_path):
    """为每个基础测试提供干净、隔离的KG和向量存储。"""
    kg_db_path = tmp_path / "kuzu_db_basic"
    vector_db_path = tmp_path / "chroma_for_kg_basic"
    kg_store = get_kg_store(db_path=str(kg_db_path))
    vector_store = get_vector_store(db_path=str(vector_db_path), collection_name="kg_basic_hybrid")
    logger.info(f"基础测试的临时数据库已在 {tmp_path} 创建。")
    yield kg_store, vector_store


@patch('llama_index.core.indices.knowledge_graph.base.KnowledgeGraphIndex.from_documents')
def test_kg_add_happy_path(mock_from_documents, basic_kg_stores):
    """
    测试 kg_add 的基本添加功能。只要函数能被成功调用且不抛出异常，就视为通过。
    """
    kg_store, vector_store = basic_kg_stores
    content = "龙傲天是青云宗的弟子。"
    doc_id = "test_doc_1"

    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content,
        metadata={"source": "test_source"},
        doc_id=doc_id,
    )

    mock_from_documents.assert_called_once()
    logger.info("kg_add 基本调用测试通过。")


@patch('llama_index.core.indices.knowledge_graph.base.KnowledgeGraphIndex.from_documents')
def test_kg_add_no_triplets(mock_from_documents, basic_kg_stores):
    """测试当内容不包含三元组时，函数依然能正常处理而不报错。"""
    kg_store, vector_store = basic_kg_stores
    content = "这是一段没有实体关系的普通描述性文字。"

    kg_add(
        kg_store,
        vector_store,
        content,
        metadata={"source": "test_doc_no_rel"},
        doc_id="test_doc_no_rel"
    )

    mock_from_documents.assert_called_once()
    logger.info("无三元组内容添加测试成功。")


@patch('llama_index.core.indices.knowledge_graph.base.KnowledgeGraphIndex.from_documents')
def test_kg_add_content_unchanged_skips_processing(mock_from_documents, basic_kg_stores):
    """测试当内容未改变时，kg_add 应跳过核心处理流程。"""
    kg_store, vector_store = basic_kg_stores
    content = "龙傲天是青云宗的弟子。"
    doc_id = "test_doc_1"

    # 第一次调用，内容会实际写入数据库，包括其哈希值
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content,
        metadata={"source": "test_source"},
        doc_id=doc_id,
    )
    mock_from_documents.assert_called_once()

    # 第二次调用，由于内容和doc_id相同，哈希检查会通过，应跳过处理
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content,
        metadata={"source": "test_source"},
        doc_id=doc_id,
    )
    # 验证 from_documents 没有被再次调用，证明跳过逻辑生效
    mock_from_documents.assert_called_once()
    logger.info("内容未变时跳过处理的测试通过。")


@patch('llama_index.core.indices.knowledge_graph.base.KnowledgeGraphIndex.from_documents')
def test_kg_add_content_updated_reprocesses(mock_from_documents, basic_kg_stores):
    """测试当内容更新时，kg_add 应重新处理。"""
    kg_store, vector_store = basic_kg_stores
    doc_id = "test_doc_update"
    content_v1 = "龙傲天是青云宗的弟子。"
    content_v2 = "龙傲天加入了魔教。"

    # 第一次调用
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content_v1,
        metadata={"source": "test_source", "version": 1},
        doc_id=doc_id,
    )

    # 第二次调用，内容更新
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content_v2,
        metadata={"source": "test_source", "version": 2},
        doc_id=doc_id,
    )
    # 验证 from_documents 被调用了两次，证明重新处理逻辑生效
    assert mock_from_documents.call_count == 2
    logger.info("内容更新时重新处理的测试通过。")


@pytest.mark.parametrize("content", VECTOR_TEST_DATASET)
@patch('llama_index.core.indices.knowledge_graph.base.KnowledgeGraphIndex.from_documents')
def test_kg_add_with_diverse_data(mock_from_documents, basic_kg_stores, content):
    """
    测试 kg_add 函数处理来自 test_data.py 的各种格式和复杂度的内容。
    """
    kg_store, vector_store = basic_kg_stores
    
    # 为每个内容生成唯一的 doc_id 以避免冲突
    doc_id = f"diverse_test_{hashlib.sha1(content.encode('utf-8')).hexdigest()}"
    
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content,
        metadata={"source": "diverse_test"},
        doc_id=doc_id
    )

    # 注意：根据 kg.py 的当前实现，它不会跳过空内容，而是会继续处理。
    # 因此，测试逻辑验证 from_documents 总是被调用。
    # 由于 parametrize 为每个用例提供了干净的数据库，哈希检查不会导致跳过。
    mock_from_documents.assert_called_once()

    if not content or not content.strip():
        logger.info(f"kg_add 正确地将空内容传递给了处理流程。")
    else:
        logger.info(f"kg_add 成功处理了内容 (doc_id: {doc_id})。")