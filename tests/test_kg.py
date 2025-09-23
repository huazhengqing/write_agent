import sys
import pytest
import os
from loguru import logger

from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.kg import (
    get_kg_store,
    kg_add,
    get_kg_query_engine,
)
from utils.vector import index_query, index_query_batch
from tests import test_data


@pytest.fixture(scope="function")
def kg_store(tmp_path) -> KuzuPropertyGraphStore:
    db_path = tmp_path / ".kuzu_db"
    logger.info(f"为测试函数创建全新的 Kuzu 数据库于: {db_path}")
    store = get_kg_store(str(db_path))
    yield store
    logger.info(f"测试函数结束, 临时数据库 {db_path} 将被自动删除。")


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
]

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, content, content_format, metadata, question, expected_keywords",
    query_scenarios
)
async def test_kg_query_scenarios(kg_store: KuzuPropertyGraphStore, test_id: str, content: str, content_format: str, metadata: dict, question: str, expected_keywords: list):
    """测试不同数据类型的端到端知识图谱查询。"""
    logger.info(f"开始查询场景测试: {test_id}")
    doc_id = f"query_scenario_{test_id}_doc"

    kg_add(kg_store, content, metadata, doc_id, content_format=content_format)
    query_engine = get_kg_query_engine(kg_store)

    logger.info(f"执行查询: '{question}'")
    answer = await index_query(query_engine, question)
    logger.info(f"收到回答: '{answer}'")

    assert answer is not None
    for keyword_group in expected_keywords:
        if isinstance(keyword_group, str):
            assert keyword_group in answer, f"答案中未找到必须的关键词: '{keyword_group}'"
        elif isinstance(keyword_group, (list, tuple)):
            assert any(kw in answer for kw in keyword_group), f"答案中未找到任一关键词组: {keyword_group}"
    logger.info(f"查询场景 '{test_id}' 验证成功。")


@pytest.mark.asyncio
async def test_kg_query_batch(kg_store: KuzuPropertyGraphStore):
    """测试对知识图谱进行端到端的批量查询。"""
    logger.info("开始测试: test_kg_query_batch")
    doc_id = "batch_query_test_doc"
    content = test_data.VECTOR_TEST_CHARACTER_INFO
    metadata = {"source": "test_novel", "type": "character"}

    # 1. 添加数据
    logger.info(f"为批量查询测试添加数据, doc_id: {doc_id}")
    kg_add(kg_store, content, metadata, doc_id, content_format="md")

    # 2. 构建查询引擎
    logger.info("构建知识图谱查询引擎...")
    query_engine = get_kg_query_engine(kg_store)

    # 3. 定义批量查询问题
    questions = [
        "龙傲天有什么特点?",
        "龙傲天来自哪里?",
        "这个故事里有谁?"
    ]
    logger.info(f"执行批量查询: {questions}")

    # 4. 执行批量查询
    answers = await index_query_batch(query_engine, questions)
    logger.info(f"收到批量回答: {answers}")

    # 5. 验证结果
    logger.info("验证批量查询结果...")
    assert isinstance(answers, list)
    assert len(answers) == len(questions)

    # 验证每个答案都包含一些基本信息, 因为LLM输出不稳定
    assert "龙傲天" in answers[0] and ("穿越者" in answers[0] or "血脉" in answers[0] or "天赋" in answers[0])
    assert "龙傲天" in answers[1] and ("异世界" in answers[1] or "穿越" in answers[1])
    assert "龙傲天" in answers[2]

    logger.info("批量查询结果验证成功。")


def get_test_data_params():
    """返回用于参数化测试的参数列表, 覆盖所有测试数据。"""
    # (test_id, content, content_format, expect_triplets)
    params = [
        # 1. 基础与边缘用例
        ("empty", test_data.VECTOR_TEST_EMPTY, "text", False),
        ("simple_txt", test_data.VECTOR_TEST_SIMPLE_TXT, "text", True),
        ("simple_cn", test_data.VECTOR_TEST_SIMPLE_CN, "text", True),
        ("simple_md", test_data.VECTOR_TEST_SIMPLE_MD, "md", True),
        ("simple_json", test_data.VECTOR_TEST_SIMPLE_JSON, "json", True),
        ("mixed_lang", test_data.VECTOR_TEST_MIXED_LANG, "md", True),
        # 2. 结构化内容
        ("table_data", test_data.VECTOR_TEST_TABLE_DATA, "md", True),
        ("large_table_data", test_data.VECTOR_TEST_LARGE_TABLE_DATA, "md", True),
        ("nested_list", test_data.VECTOR_TEST_NESTED_LIST, "md", True),
        ("structured_json", test_data.VECTOR_TEST_STRUCTURED_JSON, "json", True),
        ("deep_hierarchy_json", test_data.VECTOR_TEST_DEEP_HIERARCHY_JSON, "json", True),
        ("multi_paragraph", test_data.VECTOR_TEST_MULTI_PARAGRAPH, "md", True),
        ("complex_markdown", test_data.VECTOR_TEST_COMPLEX_MARKDOWN, "md", True),
        ("novel_structured_info", test_data.VECTOR_TEST_NOVEL_STRUCTURED_INFO, "md", True),
        ("conversational_log", test_data.VECTOR_TEST_CONVERSATIONAL_LOG, "md", True),
        ("philosophical_text", test_data.VECTOR_TEST_PHILOSOPHICAL_TEXT, "md", True),
        ("composite_structure", test_data.VECTOR_TEST_COMPOSITE_STRUCTURE, "md", True),
        # 3. 特殊格式与代码块
        ("diagram_content", test_data.VECTOR_TEST_DIAGRAM_CONTENT, "md", True),
        ("complex_mermaid_diagram", test_data.VECTOR_TEST_COMPLEX_MERMAID_DIAGRAM, "md", True),
        ("special_chars", test_data.VECTOR_TEST_SPECIAL_CHARS, "md", True),
        ("md_with_code_block", test_data.VECTOR_TEST_MD_WITH_CODE_BLOCK, "md", True),
        ("json_with_code_block", test_data.VECTOR_TEST_JSON_WITH_CODE_BLOCK, "json", True),
        ("md_with_complex_json_code_block", test_data.VECTOR_TEST_MD_WITH_COMPLEX_JSON_CODE_BLOCK, "md", True),
        # 4. 领域场景: 小说创作
        ("character_info", test_data.VECTOR_TEST_CHARACTER_INFO, "md", True),
        ("worldview", test_data.VECTOR_TEST_WORLDVIEW, "md", True),
        ("novel_worldview", test_data.VECTOR_TEST_NOVEL_WORLDVIEW, "md", True),
        ("novel_characters", test_data.VECTOR_TEST_NOVEL_CHARACTERS, "json", True),
        ("novel_plot_arc", test_data.VECTOR_TEST_NOVEL_PLOT_ARC, "md", True),
        ("novel_magic_system", test_data.VECTOR_TEST_NOVEL_MAGIC_SYSTEM, "md", True),
        ("novel_factions", test_data.VECTOR_TEST_NOVEL_FACTIONS, "md", True),
        ("novel_chapter", test_data.VECTOR_TEST_NOVEL_CHAPTER, "md", True),
        ("novel_summary", test_data.VECTOR_TEST_NOVEL_SUMMARY, "md", True),
        ("novel_full_outline", test_data.VECTOR_TEST_NOVEL_FULL_OUTLINE, "md", True),
        # 5. 领域场景: 报告撰写
        ("report_outline", test_data.VECTOR_TEST_REPORT_OUTLINE, "md", True),
        ("detailed_report_outline", test_data.VECTOR_TEST_DETAILED_REPORT_OUTLINE, "md", True),
        ("report_market_data", test_data.VECTOR_TEST_REPORT_MARKET_DATA, "json", True),
        ("report_tech_trends", test_data.VECTOR_TEST_REPORT_TECH_TRENDS, "md", True),
        ("report_case_study", test_data.VECTOR_TEST_REPORT_CASE_STUDY, "md", True),
        # 6. 领域场景: 技术文档
        ("technical_book_chapter", test_data.VECTOR_TEST_TECHNICAL_BOOK_CHAPTER, "md", True),
    ]
    ids = [p[0] for p in params]
    return {"params": params, "ids": ids}


test_data_params = get_test_data_params()


@pytest.mark.parametrize(
    "test_id, content, content_format, expect_triplets",
    test_data_params["params"],
    ids=test_data_params["ids"]
)
def test_kg_add_data_coverage(kg_store: KuzuPropertyGraphStore, test_id: str, content: str, content_format: str, expect_triplets: bool):
    """
    测试向知识图谱中添加各种类型的数据, 确保数据能够被处理并生成图结构。
    此测试覆盖了 test_data.py 中的所有数据样本。
    """
    logger.info(f"开始覆盖率测试: test_kg_add_data_coverage (数据: {test_id})")
    doc_id = f"coverage_doc_{test_id}"
    metadata = {"source": "test_data_coverage", "type": test_id}

    kg_add(kg_store, content, metadata, doc_id, content_format=content_format)

    relations = kg_store.get_triplets()
    if expect_triplets:
        assert len(relations) > 0, f"预期为 {test_id} 生成三元组, 但实际为 0"
        logger.info(f"为 {test_id} 成功生成 {len(relations)} 个三元组。")
    else:
        assert len(relations) == 0, f"预期为 {test_id} 不生成三元组, 但实际生成了 {len(relations)} 个"
        logger.info(f"为 {test_id} 成功跳过三元组生成。")

    chunk_nodes_query = kg_store.structured_query(
        "MATCH (c:Chunk {ref_doc_id: $doc_id}) RETURN c.id AS id",
        param_map={"doc_id": doc_id}
    )
    if content:
        assert len(chunk_nodes_query) > 0, f"预期为 {test_id} 生成 Chunk 节点, 但实际为 0"
        logger.info(f"为 {test_id} 成功找到 {len(chunk_nodes_query)} 个 Chunk 节点。")
    else:
        assert len(chunk_nodes_query) == 0, f"预期为空内容 {test_id} 不生成 Chunk 节点, 但实际生成了"
        logger.info(f"为空内容 {test_id} 成功跳过 Chunk 节点生成。")