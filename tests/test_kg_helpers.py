"""
对 utils/kg.py 中重构出的辅助函数进行单元测试。
这些测试旨在验证每个小函数的独立功能，确保其正确性和鲁棒性。
"""

import os
import sys
import pytest
import hashlib
from unittest.mock import MagicMock, patch, call

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import Document
from llama_index.core.schema import TextNode

# 被测试的函数
from utils.kg import (
    _check_content_unchanged,
    _parse_and_update_vector_store,
    _extract_and_normalize_triplets,
    _cleanup_old_graph_data,
    _write_new_graph_data,
    _update_document_hash,
)

# --- 测试 _check_content_unchanged ---

def test_check_content_unchanged_is_unchanged():
    """测试当内容哈希值与存储中的哈希值匹配时的情况。"""
    mock_kg_store = MagicMock()
    content = "some content"
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    mock_kg_store.query.return_value = [{'old_hash': content_hash}]

    is_unchanged, new_hash = _check_content_unchanged(mock_kg_store, "doc1", content)

    assert is_unchanged is True, "内容未变时应返回 True"
    assert new_hash == content_hash, "应返回新内容的哈希值"
    mock_kg_store.query.assert_called_once_with(
        "MATCH (d:__Document__ {doc_id: $doc_id}) RETURN d.content_hash AS old_hash",
        param_map={"doc_id": "doc1"}
    )

def test_check_content_unchanged_is_changed():
    """测试当内容哈希值与存储中的哈希值不匹配时的情况。"""
    mock_kg_store = MagicMock()
    content = "new content"
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    mock_kg_store.query.return_value = [{'old_hash': 'old_hash_value'}]

    is_unchanged, new_hash = _check_content_unchanged(mock_kg_store, "doc1", content)

    assert is_unchanged is False, "内容已变时应返回 False"
    assert new_hash == content_hash, "应返回新内容的哈希值"

def test_check_content_unchanged_is_new():
    """测试当文档为新文档（存储中无哈希值）时的情况。"""
    mock_kg_store = MagicMock()
    content = "new content"
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    mock_kg_store.query.return_value = []  # 模拟未找到文档

    is_unchanged, new_hash = _check_content_unchanged(mock_kg_store, "doc1", content)

    assert is_unchanged is False, "新文档应返回 False"
    assert new_hash == content_hash, "应返回新内容的哈希值"

# --- 测试 _parse_and_update_vector_store ---

@patch('utils.kg.VectorStoreIndex')
def test_parse_and_update_vector_store(mock_vector_store_index):
    """测试解析内容并更新向量库的流程。"""
    mock_vector_store = MagicMock()
    mock_index_instance = MagicMock()
    mock_vector_store_index.from_vector_store.return_value = mock_index_instance

    doc_id = "test_doc"
    content = "## Markdown 标题\n一些文本内容。"
    metadata = {"source": "test"}

    nodes = _parse_and_update_vector_store(mock_vector_store, doc_id, content, metadata, "md")

    assert len(nodes) > 0, "应能解析出节点"
    assert isinstance(nodes[0], TextNode), "解析出的应为 TextNode 实例"
    mock_vector_store_index.from_vector_store.assert_called_once_with(mock_vector_store)
    mock_index_instance.delete_ref_doc.assert_called_once_with(doc_id, delete_from_docstore=True)
    mock_index_instance.insert_nodes.assert_called_once_with(nodes)

# --- 测试 _extract_and_normalize_triplets ---

@patch('utils.kg.KnowledgeGraphIndex')
def test_extract_and_normalize_triplets(mock_kg_index):
    """测试三元组的提取、规范化和去重逻辑。"""
    # 准备需要被“提取”的脏数据三元组
    raw_triplets = [
        ("  主角A ", "  关系是  ", " 实体B  "),  # 需要修剪空格
        ("主角A", "关系是", "实体B"),             # 上一条的重复项
        ("实体C", "属于", "组织D"),             # 干净的三元组
        ("实体E", "", "实体F"),                   # 关系为空，应被过滤
    ]

    # 模拟临时 KnowledgeGraphIndex 内部的 graph_store
    mock_graph_store = MagicMock()
    mock_graph_store.get_rel_map.return_value = {
        "  主角A ": [], "主角A": [], "实体C": [], "实体E": []
    }
    def get_side_effect(subj):
        return [(rel, obj) for s, rel, obj in raw_triplets if s == subj]
    mock_graph_store.get.side_effect = get_side_effect

    # 模拟 KnowledgeGraphIndex 实例以使用我们的模拟 graph_store
    mock_index_instance = MagicMock()
    mock_index_instance.graph_store = mock_graph_store
    mock_kg_index.return_value = mock_index_instance

    nodes = [Document(text="任意内容")]  # 输入的节点内容是虚拟的
    
    triplets = _extract_and_normalize_triplets(nodes, 10, "任意提示词")

    assert len(triplets) == 2, "最终应只保留2条有效且唯一的三元组"
    assert ("主角A", "关系是", "实体B") in triplets, "应包含规范化后的三元组"
    assert ("实体C", "属于", "组织D") in triplets, "应包含干净的三元组"

# --- 测试图数据库操作 ---

def test_cleanup_old_graph_data():
    """测试是否执行了正确的图数据清理查询。"""
    mock_kg_store = MagicMock()
    doc_id = "doc_to_clean"

    _cleanup_old_graph_data(mock_kg_store, doc_id)

    mock_kg_store.query.assert_called_once()
    args, kwargs = mock_kg_store.query.call_args
    query_str = ' '.join(args[0].split()) # 规范化查询字符串以忽略空格差异
    assert "MATCH (n:__Entity__) WHERE $doc_id IN n.doc_ids" in query_str
    assert "SET n.doc_ids = new_doc_ids, n.status = CASE WHEN size(new_doc_ids) = 0 THEN 'inactive' ELSE n.status END" in query_str
    assert kwargs['param_map'] == {"doc_id": doc_id}

def test_write_new_graph_data():
    """测试是否为写入节点和关系执行了正确的查询。"""
    mock_kg_store = MagicMock()
    triplets = [
        ("实体A", "关系1", "实体B"),
        ("实体C", "关系1", "实体D"),  # 相同关系类型
        ("实体A", "关系2", "实体C"),  # 不同关系类型
    ]
    doc_id = "new_doc"

    _write_new_graph_data(mock_kg_store, triplets, doc_id)

    # 预期调用3次：1次用于所有节点，2次用于两种唯一的关系类型
    assert mock_kg_store.query.call_count == 3

    # 检查节点创建调用
    node_call = mock_kg_store.query.call_args_list[0]
    node_query_str = ' '.join(node_call.args[0].split())
    node_params = node_call.kwargs['param_map']
    assert "UNWIND $entities AS entity MERGE (n:__Entity__ {name: entity.name})" in node_query_str
    assert len(node_params['entities']) == 4, "应为4个唯一实体 (A, B, C, D) 创建节点"
    assert node_params['doc_id'] == doc_id

    # 检查关系创建调用（顺序可能不同）
    rel_calls = mock_kg_store.query.call_args_list[1:]
    
    rel1_call = next((c for c in rel_calls if "MERGE (s)-[:`关系1`]->(o)" in c.args[0]), None)
    assert rel1_call is not None, "应有为 '关系1' 创建关系的调用"
    assert len(rel1_call.kwargs['param_map']['pairs']) == 2, "'关系1' 应包含2对实体"

    rel2_call = next((c for c in rel_calls if "MERGE (s)-[:`关系2`]->(o)" in c.args[0]), None)
    assert rel2_call is not None, "应有为 '关系2' 创建关系的调用"
    assert len(rel2_call.kwargs['param_map']['pairs']) == 1, "'关系2' 应包含1对实体"

def test_update_document_hash():
    """测试是否执行了正确的文档哈希值更新查询。"""
    mock_kg_store = MagicMock()
    doc_id = "doc1"
    content_hash = "some_hash_value"

    _update_document_hash(mock_kg_store, doc_id, content_hash)

    expected_query = """
    MERGE (d:__Document__ {doc_id: $doc_id})
    SET d.content_hash = $content_hash
    """
    mock_kg_store.query.assert_called_once_with(
        expected_query,
        param_map={"doc_id": doc_id, "content_hash": content_hash}
    )