import sys
import pytest
import os
import asyncio
from loguru import logger

from llama_index.core import Document, Settings
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
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
from tests.conftest import get_all_test_data_params
from tests import test_data 
from tests.conftest import data_map_for_test_files


@pytest.fixture(scope="function")
def vector_store(tmp_path) -> VectorStore:
    """提供一个临时的、空的 ChromaDB 向量存储。"""
    db_path = tmp_path / ".chroma_db_test"
    collection_name = "test_collection"
    logger.info(f"为测试函数创建全新的 ChromaDB 于: {db_path}")
    store = get_vector_store(str(db_path), collection_name)
    yield store
    logger.info(f"测试函数结束, 临时数据库 {db_path} 将被自动删除。")


def test_get_vector_store(tmp_path):
    """测试 get_vector_store 能否成功创建并返回一个 ChromaVectorStore 实例。"""
    logger.info("开始测试: test_get_vector_store")
    db_path = tmp_path / "test_db"
    collection_name = "my_test_collection"
    
    store = get_vector_store(str(db_path), collection_name)
    
    assert store is not None
    assert db_path.exists()
    
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
    from pathlib import Path
    logger.info("开始测试: test_load_and_filter_documents")    
    input_dir = Path(input_dir_with_test_files)
    (input_dir / "empty.md").write_text("")
    (input_dir / "whitespace.txt").write_text("   \n\t ")
    (input_dir / "ignored.log").write_text("Log content.")
    
    docs = _load_and_filter_documents(str(input_dir), file_metadata_default)
    
    # 根据 conftest._write_test_data_to_files 中写入的非空文件数量进行断言
    expected_file_count = len([c for c in data_map_for_test_files.values() if c and c.strip()])
    assert len(docs) == expected_file_count, f"预期加载 {expected_file_count} 个文档, 实际加载了 {len(docs)} 个"
    
    filenames = {doc.metadata["file_name"] for doc in docs}
    assert "simple.md" in filenames
    assert "empty.md" not in filenames
    assert "whitespace.txt" not in filenames
    assert "ignored.log" not in filenames
    logger.info("test_load_and_filter_documents 测试通过。")


@pytest.mark.parametrize(
    "content_format, expected_parser_type",
    [("md", "CustomMarkdownNodeParser"), ("txt", "SentenceSplitter"), ("json", "JSONNodeParser")],
    ids=["markdown", "text", "json"]
)
def test_get_node_parser(content_format, expected_parser_type):
    """测试 _get_node_parser 能否根据格式返回正确的解析器类型。"""
    logger.info(f"开始测试: test_get_node_parser (format: {content_format})")
    parser = _get_node_parser(content_format, 1000)
    assert parser.__class__.__name__ == expected_parser_type

    # 增加对 SentenceSplitter chunk_size 的测试
    if content_format == "txt":
        # 短文本
        parser_small = _get_node_parser(content_format, 1000)
        assert parser_small.chunk_size == 256
        # 中等文本
        parser_medium = _get_node_parser(content_format, 6000)
        assert parser_medium.chunk_size == 512
        # 长文本
        parser_large = _get_node_parser(content_format, 30000)
        assert parser_large.chunk_size == 1024
        logger.info("SentenceSplitter 的动态 chunk_size 测试通过。")


    logger.info(f"test_get_node_parser (format: {content_format}) 测试通过。")


def test_filter_invalid_nodes():
    """测试 filter_invalid_nodes 能否过滤掉无效节点。"""
    logger.info("开始测试: test_filter_invalid_nodes")
    nodes = [
        TextNode(text="Valid node."),
        TextNode(text=""),
        TextNode(text="   "),
        TextNode(text="\n\t"),
        TextNode(text="---\n"), # 仅包含分隔符的节点也应被视为空
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
    
    # 测试重复添加
    success_again = vector_add(vector_store, content, metadata, doc_id="test_doc_1")
    assert success_again is True, "重复添加已存在内容应该返回 True"
    assert collection.count() > 0, "重复添加不应改变节点数量"
    logger.info("重复添加内容的测试通过。")


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


test_data_params = get_all_test_data_params()


@pytest.mark.parametrize(
    "test_id, content, content_format, expect_nodes",
    test_data_params["params"],
    ids=test_data_params["ids"]
)
def test_vector_add_data_coverage(vector_store: VectorStore, test_id: str, content: str, content_format: str, expect_nodes: bool):
    """
    测试向向量库中添加各种类型的数据, 确保数据能够被正确解析和存储。
    此测试覆盖了 test_data.py 中的所有数据样本。
    """
    logger.info(f"开始覆盖率测试: test_vector_add_data_coverage (数据: {test_id})")
    doc_id = f"coverage_doc_{test_id}"
    metadata = {"source": "test_data_coverage", "type": test_id}

    success = vector_add(vector_store, content, metadata, doc_id=doc_id, content_format=content_format)

    if not content.strip():
        assert success is False, f"预期空内容 {test_id} 添加失败, 但实际成功"
    else:
        assert success is True, f"预期内容 {test_id} 添加成功, 但实际失败"

    client = vector_store.client
    collection = client.get_collection(name=vector_store.collection_name)
    count = collection.count()

    if expect_nodes:
        assert count > 0, f"预期为 {test_id} 生成节点, 但实际为 0"
        logger.info(f"为 {test_id} 成功生成 {count} 个节点。")
    else:
        assert count == 0, f"预期为 {test_id} 不生成节点, 但实际生成了 {count} 个"
        logger.info(f"为 {test_id} 成功跳过节点生成。")


def _validate_answer_keywords(answer: str, expected_keywords: list, test_id: str):
    """辅助函数, 用于验证答案是否包含预期的关键词。"""
    assert answer is not None and answer.strip() != "", f"场景 '{test_id}' 的回答不应为空"
    for keyword_group in expected_keywords:
        if isinstance(keyword_group, str):
            assert keyword_group in answer, f"场景 '{test_id}' 的答案中未找到必须的关键词: '{keyword_group}'"
        elif isinstance(keyword_group, (list, tuple)):
            assert any(kw in answer for kw in keyword_group), f"场景 '{test_id}' 的答案中未找到任一关键词组: {keyword_group}"


query_scenarios = [
    pytest.param(
        "character_info",
        test_data.VECTOR_TEST_CHARACTER_INFO,
        "md",
        {"source": "test_novel", "type": "character"},
        "龙傲天有什么特点?",
        ["龙傲天", ("穿越者", "血脉", "天赋")],
        id="query_character_info"
    ),
    pytest.param(
        "table_data",
        test_data.VECTOR_TEST_TABLE_DATA,
        "md",
        {"source": "test_novel_tables"},
        "萧炎属于哪个门派?",
        ["萧炎", "炎盟"],
        id="query_table_data"
    ),
    pytest.param(
        "large_table_data",
        test_data.VECTOR_TEST_LARGE_TABLE_DATA,
        "md",
        {"source": "test_novel_large_tables"},
        "叶良辰的阵营和核心身份是什么?",
        ["叶良辰", "敌对阵营", "北冥魔殿少主"],
        id="query_large_table_data"
    ),
    pytest.param(
        "structured_json",
        test_data.VECTOR_TEST_STRUCTURED_JSON,
        "json",
        {"source": "test_character_json"},
        "药尘的职业是什么?",
        ["药尘", "炼药师"],
        id="query_structured_json"
    ),
    pytest.param(
        "complex_markdown",
        test_data.VECTOR_TEST_COMPLEX_MARKDOWN,
        "md",
        {"source": "test_world_wiki"},
        "九霄大陆的中心区域是哪里? 有哪些主要势力?",
        ["中央神州", "青云宗", "万象宗"],
        id="query_complex_markdown"
    ),
    pytest.param(
        "diagram_content",
        test_data.VECTOR_TEST_DIAGRAM_CONTENT,
        "md",
        {"source": "test_diagram"},
        "龙傲天和叶良辰是什么关系?",
        ["龙傲天", "叶良辰", "宿敌"],
        id="query_diagram_content"
    ),
    pytest.param(
        "novel_full_outline",
        test_data.VECTOR_TEST_NOVEL_FULL_OUTLINE,
        "md",
        {"source": "test_novel_outline"},
        "小说《代码之魂: 奇点》的第一卷结局是什么?",
        ["林奇", "数字意识", "沉睡", "苏菲", "逃离"],
        id="query_novel_full_outline"
    ),
    pytest.param(
        "composite_structure",
        test_data.VECTOR_TEST_COMPOSITE_STRUCTURE,
        "md",
        {"source": "test_composite"},
        "叶凡的职位是什么? 龙傲天和赵日天是什么关系?",
        ["叶凡", "天帝", "龙傲天", "赵日天", "挚友"],
        id="query_composite_structure"
    ),
    pytest.param(
        "complex_mermaid_diagram",
        test_data.VECTOR_TEST_COMPLEX_MERMAID_DIAGRAM,
        "md",
        {"source": "test_complex_diagram"},
        "议长德雷克派谁去追捕凯尔？",
        ["议长德雷克", "暗影", "追捕", "凯尔"],
        id="query_complex_mermaid_diagram"
    ),
]

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, content, content_format, metadata, question, expected_keywords",
    query_scenarios
)
async def test_vector_query_scenarios(vector_store: VectorStore, test_id: str, content: str, content_format: str, metadata: dict, question: str, expected_keywords: list):
    """测试不同数据类型的端到端向量库查询。"""
    logger.info(f"开始查询场景测试: {test_id}")
    doc_id = f"query_scenario_{test_id}_doc"

    vector_add(vector_store, content, metadata, doc_id, content_format=content_format)
    
    # 使用标准检索器以保证可预测性
    query_engine = get_vector_query_engine(vector_store, use_auto_retriever=False)

    logger.info(f"执行查询: '{question}'")
    answer = await index_query(query_engine, question)
    logger.info(f"收到回答: '{answer}'")

    _validate_answer_keywords(answer, expected_keywords, test_id)
    logger.info(f"查询场景 '{test_id}' 验证成功。")


@pytest.fixture(scope="function")
def populated_vector_store_for_query(vector_store: VectorStore) -> VectorStore:
    """用 query_scenarios 中的数据填充向量存储。"""
    logger.info("用查询场景数据填充向量存储...")
    for param in query_scenarios:
        test_id, content, content_format, metadata, _, _ = param.values
        doc_id = f"batch_query_scenario_{test_id}_doc"
        vector_add(vector_store, content, metadata, doc_id, content_format=content_format)
    return vector_store


@pytest.mark.asyncio
async def test_vector_query_batch_scenarios(populated_vector_store_for_query: VectorStore):
    """测试对多种数据类型进行端到端的批量查询。"""
    logger.info("开始批量查询场景测试")
    query_engine = get_vector_query_engine(populated_vector_store_for_query, use_auto_retriever=False)
    questions = [param.values[4] for param in query_scenarios]
    questions.append("一个完全不相关的问题?") # 添加一个预期无结果的查询
    logger.info(f"执行批量查询: {questions}")

    # 4. 执行批量查询
    answers = await index_query_batch(query_engine, questions)
    logger.info(f"收到批量回答: {answers}")

    # 5. 验证结果
    assert len(answers) == len(questions)
    
    # 验证每个场景的回答
    for i, param in enumerate(query_scenarios):
        expected_keywords = param.values[5]
        answer = answers[i]
        _validate_answer_keywords(answer, expected_keywords, param.id)

    # 验证不相关问题的回答
    assert answers[-1] == "", "不相关问题的回答应该为空"

    logger.info("批量查询场景测试验证成功。")


@pytest.mark.asyncio
async def test_vector_query_with_filters(vector_store: VectorStore):
    """测试 get_vector_query_engine 的元数据过滤功能。"""
    logger.info("开始测试: test_vector_query_with_filters")

    # 1. 添加带有不同元数据的文档
    content_a = "龙傲天是主角, 他来自地球。"
    metadata_a = {"source": "test_novel", "task_id": "1.1"}
    vector_add(vector_store, content_a, metadata_a, doc_id="doc_a")

    content_b = "叶良辰是反派, 他来自本地。"
    metadata_b = {"source": "test_novel", "task_id": "1.2"}
    vector_add(vector_store, content_b, metadata_b, doc_id="doc_b")

    # 2. 测试 '==' 过滤器, 只应匹配到 '龙傲天'
    logger.info("开始测试 '==' 过滤器...")
    filters_eq = MetadataFilters(
        filters=[MetadataFilter(key="task_id", value="1.1", operator="==")]
    )
    query_engine_eq = get_vector_query_engine(vector_store, filters=filters_eq)
    answer_eq = await index_query(query_engine_eq, "介绍一下书中的角色")
    
    assert "龙傲天" in answer_eq
    assert "叶良辰" not in answer_eq
    logger.info("使用 '==' 过滤器的测试通过。")

    # 3. 测试 'nin' (not in) 过滤器, 只应匹配到 '叶良辰'
    logger.info("开始测试 'nin' 过滤器...")
    filters_nin = MetadataFilters(
        filters=[MetadataFilter(key="task_id", value=["1.1"], operator="nin")]
    )
    query_engine_nin = get_vector_query_engine(vector_store, filters=filters_nin)
    answer_nin = await index_query(query_engine_nin, "介绍一下书中的角色")

    assert "叶良辰" in answer_nin
    assert "龙傲天" not in answer_nin
    logger.info("使用 'nin' 过滤器的测试通过。")