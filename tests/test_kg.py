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
def kg_store(tmp_path) -> KuzuPropertyGraphStore:
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

    # 1. 验证三元组是否已添加到图中
    # 由于LLM输出不确定，我们只检查是否生成了关系（三元组）。
    # 如果关系存在，那么其关联的节点也必然存在。
    logger.info("验证三元组是否已添加...")
    relations = kg_store.get_triplets()
    logger.info(f"图中找到 {len(relations)} 个关系。")
    assert len(relations) > 0
    logger.info("三元组验证成功。")

    # 2. 验证知识图谱中是否已存在与文档关联的 Chunk 节点
    logger.info("验证知识图谱中是否存在与文档关联的 Chunk 节点...")
    chunk_nodes_query = kg_store.structured_query(
        "MATCH (c:Chunk {ref_doc_id: $doc_id}) RETURN c.id AS id",
        param_map={"doc_id": doc_id}
    )
    logger.info(f"知识图谱中为 doc_id '{doc_id}' 找到 {len(chunk_nodes_query)} 个 Chunk 节点。")
    assert len(chunk_nodes_query) > 0
    logger.info("Chunk 节点验证成功。")


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