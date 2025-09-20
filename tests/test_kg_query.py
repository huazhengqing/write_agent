import os
import sys
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.kg import get_kg_store, kg_add, get_kg_query_engine
from utils.vector import get_vector_store, index_query, index_query_batch
from tests.test_data import (
    VECTOR_TEST_NOVEL_WORLDVIEW,
    VECTOR_TEST_NOVEL_CHARACTERS,
    VECTOR_TEST_NOVEL_PLOT_ARC,
    VECTOR_TEST_NOVEL_FACTIONS,
    VECTOR_TEST_NOVEL_CHAPTER,
)


@pytest.fixture(scope="function")
def basic_kg_stores(tmp_path):
    """为每个基础测试提供干净、隔离的KG和向量存储。"""
    kg_db_path = tmp_path / "kuzu_db_basic"
    vector_db_path = tmp_path / "chroma_for_kg_basic"
    kg_store = get_kg_store(db_path=str(kg_db_path))
    vector_store = get_vector_store(db_path=str(vector_db_path), collection_name="kg_basic_hybrid")
    logger.info(f"基础测试的临时数据库已在 {tmp_path} 创建。")
    yield kg_store, vector_store


@pytest.fixture(scope="module")
def realistic_kg_stores(tmp_path_factory):
    """
    为真实场景测试提供已灌入数据的KG和向量存储。
    此 Fixture 在模块级别仅执行一次，以提高效率。
    """
    module_tmp_path = tmp_path_factory.mktemp("kg_realistic")
    kg_db_path = module_tmp_path / "kuzu_db_realistic"
    vector_db_path = module_tmp_path / "chroma_for_kg_realistic"

    kg_store = get_kg_store(db_path=str(kg_db_path))
    vector_store = get_vector_store(db_path=str(vector_db_path), collection_name="kg_realistic_hybrid")

    # 直接从内存灌入数据，不再创建和读取文件
    logger.info("--- (Fixture) 开始将真实场景数据添加入知识图谱 ---")
    
    realistic_data = [
        ("1.1_design_worldview", VECTOR_TEST_NOVEL_WORLDVIEW, "md"),
        ("1.2_design_characters", VECTOR_TEST_NOVEL_CHARACTERS, "json"),
        ("1.3_design_plot_arc1", VECTOR_TEST_NOVEL_PLOT_ARC, "md"),
        ("1.4_design_factions", VECTOR_TEST_NOVEL_FACTIONS, "md"),
        ("3.1.1_write_chapter1", VECTOR_TEST_NOVEL_CHAPTER, "md"),
    ]

    for doc_id, content, content_format in realistic_data:
        # 模拟文件名作为 source
        source_name = f"{doc_id}.{'json' if content_format == 'json' else 'md'}"
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content,
            metadata={"source": source_name, "doc_id": doc_id},
            doc_id=doc_id,
            content_format=content_format,
            chars_per_triplet=120,
        )

    logger.success("--- (Fixture) 真实场景数据全部添加完毕 ---")
    yield kg_store, vector_store


@pytest.mark.asyncio
async def test_kg_query_single(basic_kg_stores):
    """测试基础的知识图谱查询功能 (单个查询)。"""
    kg_store, vector_store = basic_kg_stores
    # 灌入用于查询的数据
    kg_add(kg_store, vector_store, "龙傲天是青云宗的弟子。", metadata={"source": "doc1"}, doc_id="doc1", content_format="txt")
    kg_add(kg_store, vector_store, "龙傲天加入了合欢派。", metadata={"source": "doc1"}, doc_id="doc1", content_format="txt")  # 更新
    kg_add(kg_store, vector_store, "赵日天与龙傲天在苍梧山之巅有过一次对决。", metadata={"source": "doc2"}, doc_id="doc2", content_format="txt")

    kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)

    # 查询更新后的数据
    question1 = "龙傲天现在属于哪个门派？"
    response1 = await index_query(kg_query_engine, question1)
    logger.info(f"Q: {question1}\nA: {response1}")
    # LlamaIndex的KG实现会添加新事实，而不是删除旧事实。
    # 最终答案取决于LLM的综合能力，因此我们只断言新信息存在。
    assert "合欢派" in response1

    # 查询其他数据
    question2 = "赵日天和龙傲天在哪里对决过？"
    response2 = await index_query(kg_query_engine, question2)
    logger.info(f"Q: {question2}\nA: {response2}")
    assert "苍梧山" in response2
    logger.info("知识图谱单次查询测试成功。")


@pytest.mark.asyncio
async def test_realistic_kg_queries_batch(realistic_kg_stores):
    """测试针对已灌入真实数据的图谱进行复杂的批量查询。"""
    kg_store, vector_store = realistic_kg_stores
    kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)

    questions = [
        "龙傲天和叶良辰是什么关系？他们之间发生了什么？",
        "叶良辰属于哪个势力？这个势力的特点是什么？",
        "龙傲天在海底遗迹中获得了什么？",
    ]
    
    results = await index_query_batch(kg_query_engine, questions)
    
    assert len(results) == len(questions)

    # 验证每个问题的查询结果
    r1, r2, r3 = results

    logger.info(f"Q: {questions[0]}\nA: {r1}")
    assert "宿敌" in r1 and "黑风寨" in r1 and "三年之约" in r1

    logger.info(f"Q: {questions[1]}\nA: {r2}")
    assert "北冥魔殿" in r2 and ("魔道" in r2 or "诡秘" in r2 or "吞噬" in r2)

    logger.info(f"Q: {questions[2]}\nA: {r3}")
    assert "御水决" in r3

    logger.success("--- 真实场景批量查询测试通过 ---")