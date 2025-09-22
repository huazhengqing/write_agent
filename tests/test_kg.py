import pytest
import os
import hashlib
import asyncio
from loguru import logger

from llama_index.core.graph_stores.types import LabelledNode
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.graph_stores.types import (
    ChunkNode, EntityNode, LabelledNode
)
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.indices.property_graph.base import PropertyGraphIndex
from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore
from llama_index.llms.litellm import LiteLLM 
from llama_index.core.node_parser import SentenceSplitter, NodeParser, MarkdownElementNodeParser, \
    SimpleNodeParser
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.postprocessor.siliconflow_rerank.base import SiliconFlowRerank


from utils.kg import (
    get_kg_store,
    kg_add, 
    get_kg_query_engine,
)
from utils.vector import index_query
from tests import test_data


@pytest.fixture(scope="function")
def kg_store(tmp_path) -> "KuzuPropertyGraphStore":
    db_path = tmp_path / ".kuzu_db"
    logger.info(f"为测试函数创建全新的 Kuzu 数据库于: {db_path}")
    store = get_kg_store(str(db_path))
    yield store
    logger.info(f"测试函数结束，临时数据库 {db_path} 将被自动删除。")


def test_kg_add_new_content(kg_store: KuzuPropertyGraphStore):
    """测试向知识图谱添加新内容。"""
    logger.info("开始测试: test_kg_add_new_content")
    doc_id = "new_content_doc"
    content = test_data.VECTOR_TEST_CHARACTER_INFO
    metadata = {"source": "test_novel", "type": "character"}
    
    # 添加新内容
    logger.info(f"向知识图谱添加新内容, doc_id: {doc_id}")
    kg_add(kg_store, content, metadata, doc_id, content_format="md")

    # 1. 验证文档哈希是否已存储
    logger.info("验证文档哈希...")
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    query_result = kg_store.structured_query(
        "MATCH (d:__Document__ {doc_id: $doc_id}) RETURN d.content_hash AS hash",
        param_map={"doc_id": doc_id}
    )
    assert len(query_result) == 1
    assert query_result[0]['hash'] == content_hash
    logger.info("文档哈希验证成功。")

    # 2. 验证三元组是否已添加到图中
    # 由于LLM输出不确定，只检查是否生成了节点和关系
    logger.info("验证三元组是否已添加...")
    nodes = kg_store.get_llama_nodes()
    relations = kg_store.get_llama_relations()
    logger.info(f"图中找到 {len(nodes)} 个节点和 {len(relations)} 个关系。")
    assert len(nodes) > 0
    assert len(relations) > 0
    logger.info("三元组验证成功。")

    # 3. 验证知识图谱中是否已存在与文档关联的 Chunk 节点
    logger.info("验证知识图谱中是否存在与文档关联的 Chunk 节点...")
    chunk_nodes_query = kg_store.structured_query(
        "MATCH (c:Chunk {ref_doc_id: $doc_id}) RETURN c.id",
        param_map={"doc_id": doc_id}
    )
    logger.info(f"知识图谱中为 doc_id '{doc_id}' 找到 {len(chunk_nodes_query)} 个 Chunk 节点。")
    assert len(chunk_nodes_query) > 0
    logger.info("Chunk 节点验证成功。")


def test_kg_add_duplicate_content(kg_store: KuzuPropertyGraphStore):
    """测试添加重复内容时应跳过处理。"""
    logger.info("开始测试: test_kg_add_duplicate_content")
    doc_id = "duplicate_content_doc"
    content = test_data.VECTOR_TEST_CHARACTER_INFO
    metadata = {"source": "test_novel", "type": "character"}

    # 1. 首次添加内容
    logger.info(f"首次添加内容, doc_id: {doc_id}")
    kg_add(kg_store, content, metadata, doc_id, content_format="md")
    
    nodes_before_add = kg_store.get_llama_nodes()
    relations_before_add = kg_store.get_llama_relations()
    chunk_nodes_before_add_query = kg_store.structured_query(
        "MATCH (c:Chunk {ref_doc_id: $doc_id}) RETURN c.id",
        param_map={"doc_id": doc_id}
    )
    chunk_nodes_before_add = [r['c.id'] for r in chunk_nodes_before_add_query]
    logger.info(f"首次添加后: {len(nodes_before_add)} 个节点, {len(relations_before_add)} 个关系, {len(chunk_nodes_before_add)} 个 Chunk 节点。")

    # 2. 再次添加相同内容
    logger.info(f"再次添加相同内容, doc_id: {doc_id}")
    kg_add(kg_store, content, metadata, doc_id, content_format="md")

    # 3. 验证图和向量存储内容没有变化
    logger.info("验证图和向量存储内容没有变化...")
    nodes_after_add = kg_store.get_llama_nodes()
    relations_after_add = kg_store.get_llama_relations()
    chunk_nodes_after_add_query = kg_store.structured_query(
        "MATCH (c:Chunk {ref_doc_id: $doc_id}) RETURN c.id",
        param_map={"doc_id": doc_id}
    )
    chunk_nodes_after_add = [r['c.id'] for r in chunk_nodes_after_add_query]
    logger.info(f"再次添加后: {len(nodes_after_add)} 个节点, {len(relations_after_add)} 个关系, {len(chunk_nodes_after_add)} 个 Chunk 节点。")

    assert len(nodes_after_add) == len(nodes_before_add)
    assert len(relations_after_add) == len(relations_before_add)
    assert set(chunk_nodes_after_add) == set(chunk_nodes_before_add)
    logger.info("验证成功，重复内容未导致变化。")


def test_kg_add_updated_content(kg_store: KuzuPropertyGraphStore):
    """测试更新现有 doc_id 的内容。"""
    logger.info("开始测试: test_kg_add_updated_content")
    doc_id = "updated_content_doc"
    metadata = {"source": "test_novel", "type": "character"}
    content_1 = test_data.VECTOR_TEST_CHARACTER_INFO
    content_2 = test_data.VECTOR_TEST_TABLE_DATA

    # 1. 添加初始内容
    logger.info(f"添加初始内容, doc_id: {doc_id}")
    kg_add(kg_store, content_1, metadata, doc_id, content_format="md")
    
    # 记录初始状态
    initial_chunk_nodes_query = kg_store.structured_query(
        "MATCH (c:Chunk {ref_doc_id: $doc_id}) RETURN c.id",
        param_map={"doc_id": doc_id}
    )
    initial_chunk_node_ids = [r['c.id'] for r in initial_chunk_nodes_query]
    logger.info(f"初始内容添加后，知识图谱中有 {len(initial_chunk_node_ids)} 个 Chunk 节点。")
    assert len(initial_chunk_node_ids) > 0

    # 2. 添加更新内容
    logger.info(f"添加更新内容, doc_id: {doc_id}")
    kg_add(kg_store, content_2, metadata, doc_id, content_format="md")

    # 3. 验证文档哈希是否已更新
    logger.info("验证文档哈希是否已更新...")
    content_hash_2 = hashlib.sha256(content_2.encode('utf-8')).hexdigest()
    query_result = kg_store.structured_query(
        "MATCH (d:__Document__ {doc_id: $doc_id}) RETURN d.content_hash AS hash",
        param_map={"doc_id": doc_id}
    )
    assert len(query_result) == 1
    assert query_result[0]['hash'] == content_hash_2
    logger.info("文档哈希更新验证成功。")

    # 4. 验证知识图谱中的旧 Chunk 节点已被删除，新节点已添加
    logger.info("验证知识图谱中的 Chunk 节点是否已更新...")
    updated_chunk_nodes_query = kg_store.structured_query(
        "MATCH (c:Chunk {ref_doc_id: $doc_id}) RETURN c.id",
        param_map={"doc_id": doc_id}
    )
    updated_chunk_node_ids = [r['c.id'] for r in updated_chunk_nodes_query]
    logger.info(f"更新内容后，知识图谱中有 {len(updated_chunk_node_ids)} 个 Chunk 节点。")
    assert len(updated_chunk_node_ids) > 0
    # 确保节点 ID 已更改，表明旧节点被删除
    assert set(initial_chunk_node_ids) != set(updated_chunk_node_ids)
    logger.info("知识图谱 Chunk 节点更新验证成功。")

    # 5. 验证图内容已更新（新节点/关系存在）
    logger.info("验证图内容是否已更新...")
    nodes_after_update = kg_store.get_llama_nodes()
    node_names_after_update = {n.name for n in nodes_after_update if n.label != "Chunk"}
    logger.info(f"更新后图中的实体: {node_names_after_update}")
    # 确认新内容中的实体已添加
    assert "叶凡" in node_names_after_update or "萧炎" in node_names_after_update or "林动" in node_names_after_update
    logger.info("图内容更新验证成功。")


@pytest.mark.asyncio
async def test_kg_query(kg_store: KuzuPropertyGraphStore):
    """测试对知识图谱进行端到端的查询。"""
    logger.info("开始测试: test_kg_query")
    doc_id = "query_test_doc"
    content = test_data.VECTOR_TEST_CHARACTER_INFO
    metadata = {"source": "test_novel", "type": "character"}

    # 1. 添加数据到知识图谱 (kg_add 会处理重复/更新)
    logger.info(f"为查询测试添加数据, doc_id: {doc_id}")
    kg_add(kg_store, content, metadata, doc_id, content_format="md")

    # 2. 构建查询引擎
    logger.info("构建知识图谱查询引擎...")
    query_engine = get_kg_query_engine(
        kg_store=kg_store,
        kg_rerank_top_n=0, # 在测试中禁用 reranker 以节省 API 调用
    )

    # 3. 执行查询
    question = "陆离有什么特点？"
    logger.info(f"执行查询: '{question}'")
    answer = await index_query(query_engine, question)
    logger.info(f"收到回答: '{answer}'")

    # 4. 验证结果
    # 由于 LLM 输出不确定，只检查关键信息是否存在
    logger.info("验证查询结果...")
    assert answer is not None
    assert "陆离" in answer
    assert ("剑" in answer or "修士" in answer or "神秘" in answer)
    logger.info("查询结果验证成功。")