import os
import sys
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.kg import get_kg_store, kg_add, get_kg_query_engine
from utils.vector import get_vector_store
from tests.test_data import (
    VECTOR_TEST_NOVEL_WORLDVIEW,
    VECTOR_TEST_NOVEL_CHARACTERS,
    VECTOR_TEST_NOVEL_PLOT_ARC,
    VECTOR_TEST_NOVEL_FACTIONS,
    VECTOR_TEST_NOVEL_CHAPTER,
)


# --- Fixtures ---

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
            max_triplets_per_chunk=15
        )

    logger.success("--- (Fixture) 真实场景数据全部添加完毕 ---")
    yield kg_store, vector_store


# --- 查询测试 (Query Tests) ---

@pytest.mark.asyncio
async def test_kg_query(basic_kg_stores):
    """测试基础的知识图谱查询功能。"""
    kg_store, vector_store = basic_kg_stores
    # 灌入用于查询的数据
    kg_add(kg_store, vector_store, "龙傲天是青云宗的弟子。", doc_id="doc1")
    kg_add(kg_store, vector_store, "龙傲天加入了合欢派。", doc_id="doc1")  # 更新, 这会使"青云宗"变为inactive
    kg_add(kg_store, vector_store, "赵日天与龙傲天在苍梧山之巅有过一次对决。", doc_id="doc2")

    kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)

    # 查询更新后的数据
    response1 = await kg_query_engine.aquery("龙傲天现在属于哪个门派？")
    assert "合欢派" in str(response1) and "青云宗" not in str(response1)

    # 查询其他数据
    response2 = await kg_query_engine.aquery("赵日天和龙傲天在哪里对决过？")
    assert "苍梧山" in str(response2)
    logger.info("知识图谱查询测试成功。")


@pytest.mark.asyncio
async def test_realistic_kg_queries(realistic_kg_stores):
    """测试针对已灌入真实数据的图谱进行复杂查询。"""
    kg_store, vector_store = realistic_kg_stores
    kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)

    # 问题1: 关系查询
    question1 = "龙傲天和叶良辰是什么关系？他们之间发生了什么？"
    r1 = await kg_query_engine.aquery(question1)
    logger.info(f"Q: {question1}\nA: {r1}")
    assert "宿敌" in str(r1) and "黑风寨" in str(r1) and "三年之约" in str(r1)

    # 问题2: 实体属性查询
    question2 = "叶良辰属于哪个势力？这个势力的特点是什么？"
    r2 = await kg_query_engine.aquery(question2)
    logger.info(f"Q: {question2}\nA: {r2}")
    assert "北冥魔殿" in str(r2) and ("魔道" in str(r2) or "诡秘" in str(r2))

    # 问题3: 事件查询
    question3 = "龙傲天在海底遗迹中获得了什么？"
    r3 = await kg_query_engine.aquery(question3)
    logger.info(f"Q: {question3}\nA: {r3}")
    assert "御水决" in str(r3)

    logger.success("--- 真实场景查询测试通过 ---")