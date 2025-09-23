import sys
import pytest
import os
import asyncio
from pathlib import Path
from loguru import logger

from llama_index.core import Document, Settings
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.schema import TextNode
from llama_index.core.base.base_query_engine import BaseQueryEngine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.vector import (
    get_vector_store,
    file_metadata_default,
    _load_and_filter_documents,
    _get_node_parser,
    filter_invalid_nodes,
    vector_add,
    vector_add_from_dir,
    get_vector_query_engine,
    index_query,
    index_query_batch,
)
from tests import test_data


@pytest.fixture(scope="function")
def vector_store(tmp_path) -> VectorStore:
    """提供一个临时的、空的 ChromaDB 向量存储。"""
    db_path = tmp_path / ".chroma_db_test"
    collection_name = "test_collection"
    logger.info(f"为测试函数创建全新的 ChromaDB 于: {db_path}")
    store = get_vector_store(str(db_path), collection_name)
    yield store
    logger.info(f"测试函数结束, 临时数据库 {db_path} 将被自动删除。")


@pytest.fixture(scope="module")
def populated_vector_store(test_dirs, input_dir_with_test_files) -> VectorStore:
    """提供一个包含测试数据的 ChromaDB 向量存储, 模块级别共享。"""
    db_path = test_dirs["vector_db_path"]
    collection_name = "populated_test_collection"
    logger.info(f"为测试模块创建包含数据的 ChromaDB 于: {db_path}")
    store = get_vector_store(db_path, collection_name)
    
    client = store.client
    collection = client.get_or_create_collection(name=collection_name)
    if collection.count() == 0:
        logger.info("向量库为空, 开始添加测试数据...")
        vector_add(
            vector_store=store,
            content=test_data.VECTOR_TEST_CHARACTER_INFO,
            metadata={"source": "character_info", "type": "character"},
            content_format="md",
            doc_id="char_info_doc"
        )
        vector_add_from_dir(store, input_dir_with_test_files)
        logger.info("测试数据添加完毕。")
    else:
        logger.info("向量库已包含数据, 跳过添加。")
        
    return store


def test_get_vector_store(tmp_path):
    """测试 get_vector_store 能否成功创建并返回一个 ChromaVectorStore 实例。"""
    logger.info("开始测试: test_get_vector_store")
    db_path = tmp_path / "test_db"
    collection_name = "my_test_collection"
    
    store = get_vector_store(str(db_path), collection_name)
    
    assert store is not None
    assert db_path.exists()
    assert hasattr(store, 'cache')
    
    client = store.client
    collections = client.list_collections()
    assert any(c.name == collection_name for c in collections)
    logger.info("test_get_vector_store 测试通过。")


def test_file_metadata_default(tmp_path):
    """测试 file_metadata_default 能否为文件生成正确的元数据。"""
    logger.info("开始测试: test_file_metadata_default")
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    
    metadata = file_metadata_default(str(test_file))
    
    assert metadata["file_name"] == "test.txt"
    assert metadata["file_path"] == str(test_file)
    assert "creation_date" in metadata
    assert "modification_date" in metadata
    logger.info("test_file_metadata_default 测试通过。")


def test_load_and_filter_documents(input_dir_with_test_files):
    """测试 _load_and_filter_documents 能否正确加载、过滤文件。"""
    logger.info("开始测试: test_load_and_filter_documents")
    input_dir = Path(input_dir_with_test_files)
    (input_dir / "empty.md").write_text("")
    (input_dir / "whitespace.txt").write_text("   \n\t ")
    (input_dir / "ignored.log").write_text("Log content.")
    
    docs = _load_and_filter_documents(str(input_dir), file_metadata_default)
    
    assert len(docs) > 10
    
    filenames = {doc.metadata["file_name"] for doc in docs}
    assert "simple.md" in filenames
    assert "empty.md" not in filenames
    assert "whitespace.txt" not in filenames
    assert "ignored.log" not in filenames
    logger.info("test_load_and_filter_documents 测试通过。")


@pytest.mark.parametrize(
    "content_format, expected_parser_type",
    [
        ("md", "CustomMarkdownNodeParser"),
        ("txt", "SentenceSplitter"),
        ("json", "JSONNodeParser"),
    ]
)
def test_get_node_parser(content_format, expected_parser_type):
    """测试 _get_node_parser 能否根据格式返回正确的解析器类型。"""
    logger.info(f"开始测试: test_get_node_parser (format: {content_format})")
    parser = _get_node_parser(content_format, 1000)
    assert parser.__class__.__name__ == expected_parser_type
    logger.info(f"test_get_node_parser (format: {content_format}) 测试通过。")


def test_filter_invalid_nodes():
    """测试 filter_invalid_nodes 能否过滤掉无效节点。"""
    logger.info("开始测试: test_filter_invalid_nodes")
    nodes = [
        TextNode(text="Valid node."),
        TextNode(text=""),
        TextNode(text="   "),
        TextNode(text="\n\t"),
        TextNode(text="Another valid node."),
    ]
    
    valid_nodes = filter_invalid_nodes(nodes)
    
    assert len(valid_nodes) == 2
    assert valid_nodes[0].text == "Valid node."
    assert valid_nodes[1].text == "Another valid node."
    logger.info("test_filter_invalid_nodes 测试通过。")


def test_vector_add_success(vector_store: VectorStore):
    """测试 vector_add 成功添加内容。"""
    logger.info("开始测试: test_vector_add_success")
    content = test_data.VECTOR_TEST_SIMPLE_MD
    metadata = {"source": "test_doc", "type": "simple_md"}
    
    success = vector_add(vector_store, content, metadata, doc_id="test_doc_1")
    
    assert success is True
    client = vector_store.client
    collection = client.get_collection(name=vector_store.collection_name)
    assert collection.count() > 0
    logger.info("test_vector_add_success 测试通过。")


def test_vector_add_empty_content(vector_store: VectorStore):
    """测试 vector_add 传入空内容时应失败。"""
    logger.info("开始测试: test_vector_add_empty_content")
    success = vector_add(vector_store, "", {"source": "empty"})
    assert success is False
    
    success = vector_add(vector_store, "   \n\t ", {"source": "whitespace"})
    assert success is False
    
    client = vector_store.client
    collection = client.get_collection(name=vector_store.collection_name)
    assert collection.count() == 0
    logger.info("test_vector_add_empty_content 测试通过。")


def test_vector_add_from_dir_success(vector_store: VectorStore, input_dir_with_test_files: str):
    """测试 vector_add_from_dir 成功从目录添加文件。"""
    logger.info("开始测试: test_vector_add_from_dir_success")
    
    success = vector_add_from_dir(vector_store, input_dir_with_test_files)
    
    assert success is True
    client = vector_store.client
    collection = client.get_collection(name=vector_store.collection_name)
    assert collection.count() > 0
    logger.info("test_vector_add_from_dir_success 测试通过。")


def test_vector_add_from_dir_empty(vector_store: VectorStore, tmp_path):
    """测试 vector_add_from_dir 从空目录运行时应失败。"""
    logger.info("开始测试: test_vector_add_from_dir_empty")
    input_dir = tmp_path / "empty_dir"
    input_dir.mkdir()
    
    success = vector_add_from_dir(vector_store, str(input_dir))
    
    assert success is False
    client = vector_store.client
    collection = client.get_collection(name=vector_store.collection_name)
    assert collection.count() == 0
    logger.info("test_vector_add_from_dir_empty 测试通过。")


@pytest.mark.asyncio
async def test_index_query_e2e(populated_vector_store: VectorStore):
    """端到端测试: 添加数据并进行查询。"""
    logger.info("开始测试: test_index_query_e2e")
    
    query_engine = get_vector_query_engine(
        vector_store=populated_vector_store,
        similarity_top_k=5,
        top_n=3
    )
    
    assert isinstance(query_engine, BaseQueryEngine)
    
    question = "龙傲天是谁?"
    answer = await index_query(query_engine, question)
    
    logger.info(f"对 '{question}' 的回答: {answer}")
    assert answer is not None
    assert len(answer) > 0
    assert "龙傲天" in answer
    assert ("穿越者" in answer or "神秘" in answer or "天赋" in answer)
    logger.info("test_index_query_e2e 测试通过。")


@pytest.mark.asyncio
async def test_index_query_no_results(populated_vector_store: VectorStore):
    """测试查询不相关问题时, 应返回空结果。"""
    logger.info("开始测试: test_index_query_no_results")
    query_engine = get_vector_query_engine(populated_vector_store)
    
    question = "太阳的质量是多少?"
    answer = await index_query(query_engine, question)
    
    assert answer == ""
    logger.info("test_index_query_no_results 测试通过。")


@pytest.mark.asyncio
async def test_index_query_batch_e2e(populated_vector_store: VectorStore):
    """端到端测试批量查询功能。"""
    logger.info("开始测试: test_index_query_batch_e2e")
    query_engine = get_vector_query_engine(populated_vector_store)
    
    questions = [
        "龙傲天有什么特点?",
        "介绍一下世界树体系。",
        "今天天气怎么样?",
    ]
    
    answers = await index_query_batch(query_engine, questions)
    
    assert len(answers) == 3
    assert "龙傲天" in answers[0]
    assert "世界树" in answers[1]
    assert answers[2] == ""
    logger.info("test_index_query_batch_e2e 测试通过。")