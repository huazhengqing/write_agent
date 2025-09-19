import os
import sys
import asyncio
import pytest
from loguru import logger

from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vector import (
    clear_vector_index_cache,
    get_vector_query_engine,
    get_vector_store,
    vector_add,
)



@pytest.fixture(scope="function")
def vector_store(test_dirs):
    """为每个管理测试函数提供一个干净的向量存储。"""
    # 使用不同的集合名称以避免测试间冲突
    return get_vector_store(db_path=test_dirs["db_path"], collection_name="management_test")


@pytest.mark.asyncio
async def test_node_deletion(vector_store):
    """测试节点的显式删除功能。"""
    logger.info("--- 测试节点删除 ---")
    doc_id_to_delete = "doc_to_be_deleted"
    content_to_delete = "这是一个即将被删除的独特内容XYZ。"
    vector_add(vector_store, content_to_delete, {"source": "delete_test"}, doc_id=doc_id_to_delete)

    await asyncio.sleep(2)  # 等待 ChromaDB 完成后台索引

    # 验证节点存在
    filters = MetadataFilters(filters=[MetadataFilter(key="ref_doc_id", value=doc_id_to_delete)])
    query_engine_before = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=1)
    response_before = await query_engine_before.aquery("any")
    assert response_before.source_nodes and content_to_delete in response_before.source_nodes[0].get_content()

    # 删除节点
    vector_store.delete(ref_doc_id=doc_id_to_delete)
    clear_vector_index_cache(vector_store)
    await asyncio.sleep(2)

    # 验证节点已被删除
    query_engine_after = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=1)
    response_after = await query_engine_after.aquery("any")
    assert not response_after.source_nodes
    logger.success("--- 节点删除测试通过 ---")


@pytest.mark.asyncio
async def test_node_update(vector_store):
    """测试通过覆盖 doc_id 来更新节点。"""
    logger.info("--- 测试节点更新 ---")
    doc_id = "doc_to_be_updated"
    vector_add(vector_store, "文档版本 V1", {"version": 1}, doc_id=doc_id)
    await asyncio.sleep(2)

    # 使用相同的 doc_id 添加 V2，实现更新
    vector_add(vector_store, "文档版本 V2", {"version": 2}, doc_id=doc_id)
    await asyncio.sleep(2)

    # 验证 V2 已存在且 V1 已被覆盖
    filters = MetadataFilters(filters=[MetadataFilter(key="ref_doc_id", value=doc_id)])
    query_engine_v2 = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=1)
    response_v2 = await query_engine_v2.aquery("any")
    assert response_v2.source_nodes and "V2" in response_v2.source_nodes[0].get_content() and "V1" not in response_v2.source_nodes[0].get_content()
    logger.success("--- 节点更新测试通过 ---")