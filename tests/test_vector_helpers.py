"""
对 utils/vector.py 中重构出的辅助函数进行单元测试。
这些测试旨在验证每个小函数的独立功能，确保其正确性和鲁棒性。
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 被测试的函数
from utils.vector import (
    _load_and_filter_documents,
    _parse_docs_to_nodes_by_format,
    _is_content_too_similar,
    _parse_content_to_nodes,
    _create_reranker,
    _create_auto_retriever_engine,
    _create_standard_query_engine,
    file_metadata_default,
    get_vector_store_info_default,
)
from tests.test_data import VECTOR_TEST_SIMPLE_MD, VECTOR_TEST_SIMPLE_JSON, VECTOR_TEST_SIMPLE_TXT

# --- 测试数据加载与解析 ---

def test_load_and_filter_documents(tmp_path):
    """测试从目录加载文档并正确过滤空文件和不支持的文件类型。"""
    # 准备：创建测试文件
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "valid.md").write_text("This is a valid markdown file.", encoding="utf-8")
    (input_dir / "valid.txt").write_text("This is a valid text file.", encoding="utf-8")
    (input_dir / "empty.md").write_text("", encoding="utf-8")
    (input_dir / "whitespace.txt").write_text("   \n\t ", encoding="utf-8")
    (input_dir / "unsupported.log").write_text("log data", encoding="utf-8")

    # 执行
    documents = _load_and_filter_documents(str(input_dir), file_metadata_default)

    # 断言
    assert len(documents) == 2, "应只加载2个有效文件"
    doc_names = sorted([Path(doc.metadata["file_path"]).name for doc in documents])
    assert doc_names == ["valid.md", "valid.txt"], "加载的文件名应正确"

def test_parse_docs_to_nodes_by_format():
    """测试根据文件格式（元数据）将文档列表解析为节点列表。"""
    # 准备：创建不同格式的模拟文档
    docs = [
        Document(text=VECTOR_TEST_SIMPLE_MD, metadata={"file_path": "test.md"}),
        Document(text=VECTOR_TEST_SIMPLE_TXT, metadata={"file_path": "test.txt"}),
        Document(text=VECTOR_TEST_SIMPLE_JSON, metadata={"file_path": "test.json"}),
        # 这个文档解析后应产生无效节点（仅包含非词汇字符），并被过滤掉
        Document(text="---\n---\n", metadata={"file_path": "invalid.md"}),
    ]

    # 执行
    nodes = _parse_docs_to_nodes_by_format(docs)

    # 断言
    assert len(nodes) > 0, "应从有效文档中解析出节点"
    # 验证是否从每种有效文档中都解析出了内容
    node_texts = " ".join([node.text for node in nodes])
    assert "基础测试文档" in node_texts, "应包含Markdown文档的内容"
    assert "简单文本文件" in node_texts, "应包含纯文本文档的内容"
    assert "简单JSON格式测试数据" in node_texts, "应包含JSON文档的内容"

def test_parse_content_to_nodes():
    """测试将单个内容字符串解析为节点，并自动添加日期元数据。"""
    # 准备
    content = "## 标题\n一些Markdown内容。"
    metadata = {"source": "test_source"}
    
    # 执行
    nodes = _parse_content_to_nodes(content, metadata, content_format="md", doc_id="test_doc_1")

    # 断言
    assert len(nodes) > 0, "应能解析出至少一个节点"
    node = nodes[0]
    assert "标题" in node.text, "节点内容应正确"
    assert node.metadata["source"] == "test_source", "原始元数据应保留"
    assert "date" in node.metadata, "应自动添加日期元数据"
    assert node.ref_doc_id == "test_doc_1", "doc_id应被正确设置"

# --- 测试相似度检查 ---

@pytest.mark.asyncio
async def test_is_content_too_similar():
    """全面测试内容相似度检查逻辑，包括相似、不相似和更新场景。"""
    # 准备：模拟一个 VectorStore
    mock_vector_store = MagicMock()
    
    # 场景1: 发现高度相似的文档，且doc_id不同
    similar_node = NodeWithScore(node=MagicMock(ref_doc_id="existing_doc_1"), score=0.9999)
    mock_vector_store.query.return_value = VectorStoreQueryResult(nodes=[similar_node], similarities=[0.9999])
    
    is_similar = _is_content_too_similar(mock_vector_store, "very similar content", threshold=0.99, doc_id="new_doc_1")
    assert is_similar is True, "对于不同ID的相似内容，应返回True"
    mock_vector_store.query.assert_called_once()

    # 场景2: 未发现相似文档（相似度低于阈值）
    mock_vector_store.reset_mock()
    dissimilar_node = NodeWithScore(node=MagicMock(ref_doc_id="existing_doc_2"), score=0.8)
    mock_vector_store.query.return_value = VectorStoreQueryResult(nodes=[dissimilar_node], similarities=[0.8])

    is_similar = _is_content_too_similar(mock_vector_store, "different content", threshold=0.99, doc_id="new_doc_2")
    assert is_similar is False, "对于不相似的内容，应返回False"
    mock_vector_store.query.assert_called_once()

    # 场景3: 发现高度相似的文档，但doc_id相同（视为更新操作）
    mock_vector_store.reset_mock()
    same_id_node = NodeWithScore(node=MagicMock(ref_doc_id="doc_to_update"), score=0.9999)
    mock_vector_store.query.return_value = VectorStoreQueryResult(nodes=[same_id_node], similarities=[0.9999])

    is_similar = _is_content_too_similar(mock_vector_store, "updated content", threshold=0.99, doc_id="doc_to_update")
    assert is_similar is False, "对于相同ID的更新操作，应返回False"
    mock_vector_store.query.assert_called_once()

    # 场景4: 向量库中没有任何匹配项
    mock_vector_store.reset_mock()
    mock_vector_store.query.return_value = VectorStoreQueryResult(nodes=[], similarities=[])

    is_similar = _is_content_too_similar(mock_vector_store, "any content", threshold=0.99, doc_id="new_doc_3")
    assert is_similar is False, "当向量库无返回时，应返回False"
    mock_vector_store.query.assert_called_once()

# --- 测试查询引擎组件创建 ---

def test_create_reranker():
    """测试 Reranker 后处理器的创建逻辑。"""
    # 场景1: top_n > 0, 应该成功创建一个 reranker 实例
    reranker = _create_reranker(rerank_top_n=5)
    assert isinstance(reranker, SiliconFlowRerank)
    assert reranker.top_n == 5

    # 场景2: top_n = 0, 应该返回 None
    reranker = _create_reranker(rerank_top_n=0)
    assert reranker is None

    # 场景3: top_n < 0, 应该返回 None
    reranker = _create_reranker(rerank_top_n=-1)
    assert reranker is None

@patch('utils.vector.RetrieverQueryEngine')
@patch('utils.vector.VectorIndexAutoRetriever')
def test_create_auto_retriever_engine(mock_auto_retriever, mock_retriever_q_engine):
    """测试自动检索查询引擎的创建流程。"""
    # 准备
    mock_index = MagicMock(spec=VectorStoreIndex)
    vector_store_info = get_vector_store_info_default()
    
    # 执行
    engine = _create_auto_retriever_engine(
        index=mock_index,
        vector_store_info=vector_store_info,
        similarity_top_k=10,
        similarity_cutoff=0.5,
        postprocessors=[]
    )

    # 断言
    mock_auto_retriever.assert_called_once()
    mock_retriever_q_engine.assert_called_once()
    assert engine == mock_retriever_q_engine.return_value, "应返回查询引擎实例"

@patch('llama_index.core.VectorStoreIndex.as_query_engine')
def test_create_standard_query_engine(mock_as_query_engine):
    """测试标准查询引擎的创建流程。"""
    # 准备
    mock_index = MagicMock(spec=VectorStoreIndex)
    # 将 mock 方法附加到实例上
    mock_index.as_query_engine = mock_as_query_engine
    
    # 执行
    engine = _create_standard_query_engine(
        index=mock_index,
        filters=None,
        similarity_top_k=20,
        similarity_cutoff=0.6,
        postprocessors=[]
    )

    # 断言
    mock_as_query_engine.assert_called_once()
    # 验证关键参数是否正确传递
    _, kwargs = mock_as_query_engine.call_args
    assert kwargs['similarity_top_k'] == 20
    assert kwargs['similarity_cutoff'] == 0.6
    assert engine == mock_as_query_engine.return_value, "应返回查询引擎实例"