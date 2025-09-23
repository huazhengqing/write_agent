import sys
import pytest
import os
from loguru import logger

from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.vector import (
    get_vector_store,
    vector_add,
    get_vector_query_engine,
    index_query,
    index_query_batch,
)
from tests.conftest import get_all_test_data_params
from tests import test_data


@pytest.fixture(scope="function")
def vector_store(tmp_path) -> ChromaVectorStore:
    """提供一个临时的、空的 ChromaDB 向量存储。"""
    db_path = tmp_path / ".chroma_db_test"
    collection_name = "test_collection"
    logger.info(f"为测试函数创建全新的 ChromaDB 于: {db_path}")
    store = get_vector_store(str(db_path), collection_name)
    assert isinstance(store, ChromaVectorStore), "get_vector_store 应该返回 ChromaVectorStore 实例"
    yield store
    logger.info(f"测试函数结束, 临时数据库 {db_path} 将被自动删除。")


all_data_params = get_all_test_data_params()


@pytest.mark.parametrize(
    "test_id, content, content_format, expect_nodes",
    all_data_params["params"],
    ids=all_data_params["ids"]
)
def test_vector_add_data_coverage(vector_store: ChromaVectorStore, test_id: str, content: str, content_format: str, expect_nodes: bool):
    """
    测试向向量库中添加各种类型的数据, 确保数据能够被正确解析和存储。
    此测试覆盖了 test_data.py 中的所有数据样本。
    """
    logger.info(f"开始覆盖率测试: test_vector_add_data_coverage (数据: {test_id})")
    doc_id = f"coverage_doc_{test_id}"
    metadata = {"source": "test_data_coverage", "type": test_id}

    success = vector_add(vector_store, content, metadata, content_format=content_format, doc_id=doc_id)

    if not content.strip():
        assert success is False, f"预期空内容 {test_id} 添加失败, 但实际成功"
    else:
        assert success is True, f"预期内容 {test_id} 添加成功, 但实际失败"
    
    count = vector_store.client.get_collection("test_collection").count()

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
        ["中央神州", ["青云宗", "万象宗"]],
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
async def test_vector_query_scenarios(vector_store: ChromaVectorStore, test_id: str, content: str, content_format: str, metadata: dict, question: str, expected_keywords: list):
    """测试不同数据类型的端到端向量库查询。"""
    logger.info(f"开始查询场景测试: {test_id}")
    doc_id = f"query_scenario_{test_id}_doc"

    vector_add(vector_store, content, metadata, content_format=content_format, doc_id=doc_id)
    
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
        vector_add(vector_store, content, metadata, content_format=content_format, doc_id=doc_id)
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