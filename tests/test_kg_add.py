import os
import sys
import json
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.kg import get_kg_store, kg_add
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
        ("1.1_design_worldview", VECTOR_TEST_NOVEL_WORLDVIEW, "markdown"),
        ("1.2_design_characters", VECTOR_TEST_NOVEL_CHARACTERS, "json"),
        ("1.3_design_plot_arc1", VECTOR_TEST_NOVEL_PLOT_ARC, "markdown"),
        ("1.4_design_factions", VECTOR_TEST_NOVEL_FACTIONS, "markdown"),
        ("3.1.1_write_chapter1", VECTOR_TEST_NOVEL_CHAPTER, "markdown"),
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


# --- 数据存入测试 (Ingestion Tests) ---

@pytest.mark.asyncio
async def test_kg_add_initial(basic_kg_stores):
    """测试首次向知识图谱添加内容。"""
    kg_store, vector_store = basic_kg_stores
    content = "龙傲天是青云宗的首席大弟子。龙傲天使用的武器是'赤霄剑'。"
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content,
        metadata={"source": "test_doc_1", "version": 1},
        doc_id="test_doc_1",
        max_triplets_per_chunk=10
    )
    res = kg_store.query("MATCH (n:__Entity__ {name: '龙傲天'}) RETURN n.status, n.doc_id")
    assert res[0] == ['active', 'test_doc_1']
    logger.info("首次添加验证成功。")


@pytest.mark.asyncio
async def test_kg_add_update(basic_kg_stores):
    """测试通过相同 doc_id 更新知识图谱内容。"""
    kg_store, vector_store = basic_kg_stores
    # 版本1
    kg_add(
        kg_store, vector_store, "龙傲天是青云宗的弟子。",
        metadata={"source": "test_doc_1", "version": 1}, doc_id="test_doc_1"
    )
    # 版本2 (更新)
    content_v2 = "龙傲天叛逃了青云宗，加入了合欢派。"
    kg_add(
        kg_store, vector_store, content_v2,
        metadata={"source": "test_doc_1", "version": 2}, doc_id="test_doc_1"
    )

    res_old = kg_store.query("MATCH (n:__Entity__ {name: '青云宗'}) RETURN n.status")
    assert res_old[0] == ['inactive']

    res_new = kg_store.query("MATCH (n:__Entity__ {name: '合欢派'}) RETURN n.status")
    assert res_new[0] == ['active']
    logger.info("更新文档验证成功，旧实体已标记为 inactive。")


@pytest.mark.asyncio
async def test_kg_add_complex_formats(basic_kg_stores):
    """测试从 Markdown 表格和 JSON 中提取三元组。"""
    kg_store, vector_store = basic_kg_stores
    # Markdown 表格
    content_md = """
    # 势力成员表
    | 姓名 | 门派 | 职位 |
    |---|---|---|
    | 赵日天 | 天机阁 | 阁主 |
    """
    kg_add(kg_store, vector_store, content_md, metadata={"source": "test_doc_md"}, doc_id="test_doc_md")
    res_md = kg_store.query("MATCH (:__Entity__ {name: '赵日天'})-[:属于]->(:__Entity__ {name: '天机阁'}) RETURN count(*)")
    assert res_md[0][0] > 0

    # JSON 内容
    content_json = json.dumps({"event": "苍梧山之巅对决", "location": "苍梧山之巅"}, ensure_ascii=False)
    kg_add(kg_store, vector_store, content_json, metadata={"source": "test_doc_json"}, doc_id="test_doc_json", content_format="json")
    res_json = kg_store.query("MATCH (n:__Entity__ {name: '苍梧山之巅对决'}) RETURN n.status")
    assert res_json[0] == ['active']
    logger.info("复杂格式(MD, JSON)添加测试成功。")


@pytest.mark.asyncio
async def test_kg_add_no_triplets(basic_kg_stores):
    """测试当内容不包含三元组时，不应向图谱添加任何实体。"""
    kg_store, vector_store = basic_kg_stores
    content = "这是一段没有实体关系的普通描述性文字。"
    kg_add(kg_store, vector_store, content, metadata={"source": "test_doc_no_rel"}, doc_id="test_doc_no_rel")
    res = kg_store.query("MATCH (n) WHERE n.doc_id = 'test_doc_no_rel' RETURN count(n)")
    assert res[0] == [0]
    logger.info("无三元组内容添加测试成功。")


@pytest.mark.asyncio
async def test_realistic_kg_update(realistic_kg_stores):
    """测试在已灌入真实数据的图谱上执行更新操作。"""
    kg_store, vector_store = realistic_kg_stores

    update_content = "# 角色动态更新\n龙傲天因为理念不合，离开了青云宗，现在是一名散修。"
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=update_content,
        metadata={"source": "update_doc"},
        doc_id="1.1_design_worldview"  # 覆盖原始世界观文档中的关系
    )

    res_old = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[r:属于]->(:__Entity__ {name: '青云宗'}) RETURN count(r)")
    assert res_old[0][0] == 0, "龙傲天'属于'青云宗的旧关系应已被删除"

    res_new = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[r:离开]->(:__Entity__ {name: '青云宗'}) RETURN count(r)")
    assert res_new[0][0] > 0, "龙傲天'离开'青云宗的新关系应已建立"
    logger.info("真实数据更新逻辑验证成功。")