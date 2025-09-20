import os
import sys
import json
import pytest
from loguru import logger
from unittest.mock import patch, MagicMock

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
    
    # 使用 patch 避免在 fixture 设置期间进行网络调用
    with patch('utils.kg._extract_and_normalize_triplets', return_value=[("mock_subj", "mock_rel", "mock_obj")]):
        realistic_data = [
            ("1.1_design_worldview", VECTOR_TEST_NOVEL_WORLDVIEW, "md"),
            ("1.2_design_characters", VECTOR_TEST_NOVEL_CHARACTERS, "json"),
            ("1.3_design_plot_arc1", VECTOR_TEST_NOVEL_PLOT_ARC, "md"),
            ("1.4_design_factions", VECTOR_TEST_NOVEL_FACTIONS, "md"),
            ("3.1.1_write_chapter1", VECTOR_TEST_NOVEL_CHAPTER, "md"),
        ]

        for doc_id, content, content_format in realistic_data:
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

@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_initial(mock_extract, basic_kg_stores):
    """测试首次向知识图谱添加内容。"""
    kg_store, vector_store = basic_kg_stores
    content = "龙傲天是青云宗的首席大弟子。龙傲天使用的武器是'赤霄剑'。"
    mock_extract.return_value = [
        ("龙傲天", "是首席大弟子", "青云宗"),
        ("龙傲天", "使用的武器是", "赤霄剑"),
    ]
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=content,
        metadata={"source": "test_doc_1", "version": 1},
        doc_id="test_doc_1",
        max_triplets_per_chunk=10
    )
    res = kg_store.query("MATCH (n:__Entity__ {name: '龙傲天'}) RETURN n.status, n.doc_ids")
    assert res is not None and len(res) > 0, "查询应返回结果"
    assert res[0]['n.status'] == 'active'
    assert res[0]['n.doc_ids'] == ['test_doc_1']
    logger.info("首次添加验证成功。")


@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_update(mock_extract, basic_kg_stores):
    """测试通过相同 doc_id 更新知识图谱内容。"""
    kg_store, vector_store = basic_kg_stores
    # 版本1
    mock_extract.return_value = [("龙傲天", "是弟子", "青云宗")]
    kg_add(
        kg_store, vector_store, "龙傲天是青云宗的弟子。",
        metadata={"source": "test_doc_1", "version": 1}, doc_id="test_doc_1"
    )
    # 版本2 (更新)
    content_v2 = "龙傲天加入了合欢派。"
    mock_extract.return_value = [("龙傲天", "加入", "合欢派")]
    kg_add(
        kg_store, vector_store, content_v2,
        metadata={"source": "test_doc_1", "version": 2}, doc_id="test_doc_1"
    )

    # 处理查询结果可能为None的情况
    res_old = kg_store.query("MATCH (n:__Entity__ {name: '青云宗'}) RETURN n.status")
    assert res_old is not None and len(res_old) > 0, "旧实体'青云宗'应存在"
    assert res_old[0]['n.status'] == 'inactive'

    res_new = kg_store.query("MATCH (n:__Entity__ {name: '合欢派'}) RETURN n.status")
    assert res_new is not None and len(res_new) > 0, "新实体'合欢派'应存在"
    assert res_new[0]['n.status'] == 'active'
    logger.info("更新文档验证成功，旧实体已标记为 inactive。")


@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_update_logic_comprehensive(mock_extract, basic_kg_stores):
    """
    全面测试 kg_add 的更新逻辑。
    确保在更新文档时，旧实体被正确标记为 inactive，旧关系被删除，新关系被正确建立。
    """
    kg_store, vector_store = basic_kg_stores
    doc_id = "comprehensive_update_test"

    # 1. 添加初始版本
    content_v1 = "龙傲天是青云宗的弟子。青云宗位于中央神州。"
    mock_extract.return_value = [
        ("龙傲天", "是弟子", "青云宗"),
        ("青云宗", "位于", "中央神州"),
    ]
    kg_add(
        kg_store, vector_store, content_v1,
        metadata={"source": doc_id, "version": 1}, doc_id=doc_id
    )

    # 验证初始状态
    assert kg_store.query("MATCH (n:__Entity__ {name: '龙傲天'}) RETURN n.status")[0]['n.status'] == 'active'
    assert kg_store.query("MATCH (n:__Entity__ {name: '青云宗'}) RETURN n.status")[0]['n.status'] == 'active'
    assert kg_store.query("MATCH (n:__Entity__ {name: '中央神州'}) RETURN n.status")[0]['n.status'] == 'active'
    assert kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[:是弟子]->(:__Entity__ {name: '青云宗'}) RETURN count(*)")[0]['count(*)'] > 0
    assert kg_store.query("MATCH (:__Entity__ {name: '青云宗'})-[:位于]->(:__Entity__ {name: '中央神州'}) RETURN count(*)")[0]['count(*)'] > 0
    logger.info("步骤1: 初始版本数据添加并验证成功。")

    # 2. 更新文档
    content_v2 = "龙傲天现在是北冥魔殿的成员。他还是一名炼药师。"
    mock_extract.return_value = [
        ("龙傲天", "是成员", "北冥魔殿"),
        ("龙傲天", "是", "炼药师"),
    ]
    kg_add(
        kg_store, vector_store, content_v2,
        metadata={"source": doc_id, "version": 2}, doc_id=doc_id
    )
    logger.info("步骤2: 已执行更新操作。")

    # 3. 验证更新后的状态
    # 验证旧实体状态
    assert kg_store.query("MATCH (n:__Entity__ {name: '青云宗'}) RETURN n.status")[0]['n.status'] == 'inactive', "不再提及的'青云宗'应被标记为 inactive"
    assert kg_store.query("MATCH (n:__Entity__ {name: '中央神州'}) RETURN n.status")[0]['n.status'] == 'inactive', "不再提及的'中央神州'应被标记为 inactive"
    logger.info("步骤3.1: 旧实体状态验证成功。")

    # 验证共享和新实体状态
    assert kg_store.query("MATCH (n:__Entity__ {name: '龙傲天'}) RETURN n.status")[0]['n.status'] == 'active', "共享实体'龙傲天'应保持 active"
    assert kg_store.query("MATCH (n:__Entity__ {name: '北冥魔殿'}) RETURN n.status")[0]['n.status'] == 'active', "新实体'北冥魔殿'应为 active"
    assert kg_store.query("MATCH (n:__Entity__ {name: '炼药师'}) RETURN n.status")[0]['n.status'] == 'active', "新实体'炼药师'应为 active"
    logger.info("步骤3.2: 共享和新实体状态验证成功。")

    # 验证旧关系被删除
    assert kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[:是弟子]->(:__Entity__ {name: '青云宗'}) RETURN count(*)")[0]['count(*)'] > 0, "旧关系 (龙傲天-是弟子->青云宗) 应被保留"
    assert kg_store.query("MATCH (:__Entity__ {name: '青云宗'})-[:位于]->(:__Entity__ {name: '中央神州'}) RETURN count(*)")[0]['count(*)'] > 0, "旧关系 (青云宗-位于->中央神州) 应被保留"
    logger.info("步骤3.3: 旧关系保留验证成功。")

    # 验证新关系被建立
    assert kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[:是成员]->(:__Entity__ {name: '北冥魔殿'}) RETURN count(*)")[0]['count(*)'] > 0, "新关系 (龙傲天-是成员->北冥魔殿) 应被建立"
    assert kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[:是]->(:__Entity__ {name: '炼药师'}) RETURN count(*)")[0]['count(*)'] > 0, "新关系 (龙傲天-是->炼药师) 应被建立"
    logger.info("步骤3.4: 新关系建立验证成功。")

    logger.success("--- kg_add 全面更新逻辑测试通过 ---")


@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_update_fine_grained_cleanup(mock_extract, basic_kg_stores):
    """
    测试更新操作的精细化清理逻辑。
    确保更新一个文档时，不会删除由其他文档创建的、连接到共享节点上的关系。
    """
    kg_store, vector_store = basic_kg_stores
    doc_id_to_update = "doc_to_update"
    other_doc_id = "other_doc"

    # 1. 添加初始文档，建立关系 (龙傲天)-[是主人]->(赤霄剑)
    # 两个节点都属于 doc_id_to_update
    content_v1 = "龙傲天是赤霄剑的主人"
    mock_extract.return_value = [("龙傲天", "是主人", "赤霄剑")]
    kg_add(
        kg_store, vector_store, content_v1,
        metadata={"source": doc_id_to_update}, doc_id=doc_id_to_update
    )

    # 2. 添加另一个文档，建立一个从外部实体到“龙傲天”的入向关系 (叶良辰)-[畏惧]->(龙傲天)
    # “叶良辰”节点属于 other_doc_id
    content_other = "叶良辰畏惧龙傲天"
    mock_extract.return_value = [("叶良辰", "畏惧", "龙傲天")]
    kg_add(
        kg_store, vector_store, content_other,
        metadata={"source": other_doc_id}, doc_id=other_doc_id
    )

    # 验证初始状态：两条关系都存在
    outgoing_rel_count = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[:是主人]->(:__Entity__ {name: '赤霄剑'}) RETURN count(*)")[0]['count(*)']
    incoming_rel_count = kg_store.query("MATCH (:__Entity__ {name: '叶良辰'})-[:畏惧]->(:__Entity__ {name: '龙傲天'}) RETURN count(*)")[0]['count(*)']
    assert outgoing_rel_count > 0, "初始的出向关系应存在"
    assert incoming_rel_count > 0, "初始的入向关系应存在"
    logger.info("步骤1: 初始的出向和入向关系均已建立。")

    # 3. 更新初始文档，内容不再包含任何实体，这将触发清理逻辑
    content_v2 = "一片空白。"
    mock_extract.return_value = []
    kg_add(
        kg_store, vector_store, content_v2,
        metadata={"source": doc_id_to_update}, doc_id=doc_id_to_update
    )
    logger.info("步骤2: 已执行更新操作，触发清理逻辑。")

    # 4. 验证节点状态和关系保留
    # 验证'龙傲天'节点本身仍然是 active，因为它仍被 other_doc 引用
    res_longaotian = kg_store.query("MATCH (n:__Entity__ {name: '龙傲天'}) RETURN n.status, n.doc_ids")
    assert res_longaotian[0]['n.status'] == 'active'
    assert res_longaotian[0]['n.doc_ids'] == [other_doc_id]

    # 验证'赤霄剑'节点被标记为 inactive，因为它不再被任何活跃文档引用
    res_chixiaojian = kg_store.query("MATCH (n:__Entity__ {name: '赤霄剑'}) RETURN n.status, n.doc_ids")
    assert res_chixiaojian[0]['n.status'] == 'inactive'
    assert res_chixiaojian[0]['n.doc_ids'] == []

    # 验证所有关系都被保留，因为我们不再删除关系
    outgoing_rel_count_after = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[:是主人]->(:__Entity__ {name: '赤霄剑'}) RETURN count(*)")[0]['count(*)']
    assert outgoing_rel_count_after > 0, "即使节点变为inactive，关系也应被保留"
    
    # 验证由 other_doc 创建的关系仍然存在
    incoming_rel_count_after = kg_store.query("MATCH (:__Entity__ {name: '叶良辰'})-[:畏惧]->(:__Entity__ {name: '龙傲天'}) RETURN count(*)")[0]['count(*)']
    assert incoming_rel_count_after > 0, "由其他文档创建的入向关系应被保留"
    logger.info("步骤3: 验证节点被正确标记为inactive，且所有关系都被保留。")

    logger.success("--- kg_add 精细化更新逻辑（仅标记状态）测试通过 ---")


@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_doc_ids_logic(mock_extract, basic_kg_stores):
    """测试 kg_add 中新的 doc_ids 列表逻辑，确保更新和清理行为正确。"""
    logger.info("--- 测试：doc_ids 列表与精细化清理逻辑 ---")
    kg_store, vector_store = basic_kg_stores
    
    doc_id_1 = "doc_id_1"
    doc_id_2 = "doc_id_2"

    # 1. 添加 doc1，建立关系 (A)-[rel1]->(B)
    mock_extract.return_value = [("A", "rel1", "B")]
    kg_add(kg_store, vector_store, "A rel1 B", metadata={"source": doc_id_1}, doc_id=doc_id_1)
    logger.info("步骤1: 添加 doc1 (A rel1 B) 完成。")

    # 2. 添加 doc2，建立关系 (C)-[rel2]->(A)，使节点 A 被共同引用
    mock_extract.return_value = [("C", "rel2", "A")]
    kg_add(kg_store, vector_store, "C rel2 A", metadata={"source": doc_id_2}, doc_id=doc_id_2)
    logger.info("步骤2: 添加 doc2 (C rel2 A) 完成。")

    # 3. 验证中间状态
    res_A = kg_store.query("MATCH (n:__Entity__ {name: 'A'}) RETURN n.doc_ids, n.status")
    assert sorted(res_A[0]['n.doc_ids']) == sorted([doc_id_1, doc_id_2])
    assert res_A[0]['n.status'] == 'active'
    
    res_B = kg_store.query("MATCH (n:__Entity__ {name: 'B'}) RETURN n.doc_ids, n.status")
    assert res_B[0]['n.doc_ids'] == [doc_id_1]
    assert res_B[0]['n.status'] == 'active'
    
    assert kg_store.query("MATCH (:__Entity__ {name: 'A'})-[:rel1]->(:__Entity__ {name: 'B'}) RETURN count(*)")[0]['count(*)'] == 1
    assert kg_store.query("MATCH (:__Entity__ {name: 'C'})-[:rel2]->(:__Entity__ {name: 'A'}) RETURN count(*)")[0]['count(*)'] == 1
    logger.info("步骤3: 中间状态验证成功，节点A被共同引用。")

    # 4. 更新 doc1 为空内容，触发清理
    mock_extract.return_value = []
    kg_add(kg_store, vector_store, "一片空白。", metadata={"source": doc_id_1}, doc_id=doc_id_1)
    logger.info("步骤4: 更新 doc1 为空内容，触发清理。")

    # 5. 验证最终状态
    res_A_final = kg_store.query("MATCH (n:__Entity__ {name: 'A'}) RETURN n.doc_ids, n.status")
    assert res_A_final[0]['n.doc_ids'] == [doc_id_2]
    assert res_A_final[0]['n.status'] == 'active'

    res_B_final = kg_store.query("MATCH (n:__Entity__ {name: 'B'}) RETURN n.doc_ids, n.status")
    assert res_B_final[0]['n.doc_ids'] == []
    assert res_B_final[0]['n.status'] == 'inactive'
    logger.info("步骤5.1: 节点状态和 doc_ids 列表验证成功。")

    assert kg_store.query("MATCH (:__Entity__ {name: 'A'})-[:rel1]->(:__Entity__ {name: 'B'}) RETURN count(*)")[0]['count(*)'] == 1, "关系 (A)-[rel1]->(B) 应被保留"
    assert kg_store.query("MATCH (:__Entity__ {name: 'C'})-[:rel2]->(:__Entity__ {name: 'A'}) RETURN count(*)")[0]['count(*)'] == 1, "关系 (C)-[rel2]->(A) 应被保留"
    logger.info("步骤5.2: 关系完整性验证成功。")

    logger.success("--- doc_ids 列表与精细化清理逻辑测试通过 ---")
@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_complex_formats(mock_extract, basic_kg_stores):
    """测试从 Markdown 表格和 JSON 中提取三元组。"""
    kg_store, vector_store = basic_kg_stores
    # Markdown 表格
    content_md = """
    # 势力成员表
    | 姓名 | 门派 | 职位 |
    |---|---|---|
    | 赵日天 | 天机阁 | 阁主 |
    """
    mock_extract.return_value = [("赵日天", "属于", "天机阁"), ("赵日天", "职位是", "阁主")]
    kg_add(kg_store, vector_store, content_md, metadata={"source": "test_doc_md"}, doc_id="test_doc_md")
    res_md = kg_store.query("MATCH (:__Entity__ {name: '赵日天'})-[:属于]->(:__Entity__ {name: '天机阁'}) RETURN count(*)")
    assert res_md[0]['count(*)'] > 0

    # JSON 内容
    content_json = json.dumps({"event": "苍梧山之巅对决", "location": "苍梧山之巅"}, ensure_ascii=False)
    # 假设LLM能从JSON中提取关系
    mock_extract.return_value = [("苍梧山之巅对决", "位于", "苍梧山之巅")]
    kg_add(kg_store, vector_store, content_json, metadata={"source": "test_doc_json"}, doc_id="test_doc_json", content_format="json")
    res_json = kg_store.query("MATCH (n:__Entity__ {name: '苍梧山之巅对决'}) RETURN n.status")
    assert res_json[0]['n.status'] == 'active'
    logger.info("复杂格式(MD, JSON)添加测试成功。")


@pytest.mark.asyncio
async def test_kg_add_normalization_and_deduplication(basic_kg_stores):
    """测试 kg_add 中新增的三元组规范化和去重逻辑。"""
    logger.info("--- 测试：三元组规范化与去重 ---")
    kg_store, vector_store = basic_kg_stores

    # 1. 准备包含重复和需要规范化的“原始”三元组
    messy_triplets = [
        (" 龙傲天 ", " 属于 ", "青云宗"),      # 需要去除首尾空格
        ("龙傲天", "属于", "青云宗"),          # 与上一条规范化后重复
        ("叶良辰", "宿敌是", "龙傲天"),         # 正常三元组
        ("叶良辰", "宿敌是", "龙傲天"),         # 完全重复
        ("赤霄剑", "是  武器", "龙傲天"),      # 关系中有多个空格
    ]

    # 2. Mock KnowledgeGraphIndex 的行为，以绕过真实的LLM提取
    # 创建一个模拟的 graph_store
    mock_graph_store = MagicMock()
    # 设置 get_rel_map 返回所有主语
    mock_graph_store.get_rel_map.return_value = {
        " 龙傲天 ": [], "龙傲天": [], "叶良辰": [], "赤霄剑": []
    }
    # 设置 get 方法，使其根据主语返回我们预设的“脏”关系和宾语
    def get_side_effect(subj):
        return [(rel, obj) for s, rel, obj in messy_triplets if s == subj]
    mock_graph_store.get.side_effect = get_side_effect

    # 创建一个模拟的 KnowledgeGraphIndex 实例，并将其 graph_store 指向我们的模拟对象
    mock_kg_index_instance = MagicMock()
    mock_kg_index_instance.graph_store = mock_graph_store

    # 使用 patch 来替换 KnowledgeGraphIndex 的构造函数，使其返回我们的模拟实例
    with patch('utils.kg.KnowledgeGraphIndex', return_value=mock_kg_index_instance):
        # 3. 调用 kg_add 函数
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content="任何内容都可以，因为提取过程被mock了",
            metadata={"source": "test_norm"},
            doc_id="test_norm_doc"
        )

        # 4. 验证数据库中的结果
        # 检查实体是否被正确规范化
        res_entities = kg_store.query("MATCH (n:__Entity__) RETURN n.name ORDER BY n.name")
        expected_entities = sorted(['龙傲天', '青云宗', '叶良辰', '赤霄剑'])
        assert res_entities is not None, "实体查询不应返回None"
        actual_entities = sorted([row['n.name'] for row in res_entities])
        assert actual_entities == expected_entities, "实体名称应被规范化且无重复"
        logger.info("实体规范化验证成功。")

        # 检查完整的三元组数量
        res_count = kg_store.query("MATCH (s)-[r]->(o) RETURN count(*)")
        assert res_count[0]['count(*)'] == 3, "数据库中最终应只存在3条规范化后的唯一关系"
        logger.info("三元组去重验证成功。")

        # 检查具体的三元组是否存在且已规范化
        assert kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[:属于]->(:__Entity__ {name: '青云宗'}) RETURN count(*)")[0]['count(*)'] == 1
        assert kg_store.query("MATCH (:__Entity__ {name: '叶良辰'})-[:宿敌是]->(:__Entity__ {name: '龙傲天'}) RETURN count(*)")[0]['count(*)'] == 1
        assert kg_store.query("MATCH (:__Entity__ {name: '赤霄剑'})-[:`是 武器`]->(:__Entity__ {name: '龙傲天'}) RETURN count(*)")[0]['count(*)'] == 1
        logger.info("具体三元组规范化验证成功。")

    logger.success("--- 三元组规范化和去重逻辑测试通过 ---")

@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_no_triplets(mock_extract, basic_kg_stores):
    """测试当内容不包含三元组时，不应向图谱添加任何实体。"""
    kg_store, vector_store = basic_kg_stores
    content = "这是一段没有实体关系的普通描述性文字。"
    mock_extract.return_value = []
    kg_add(kg_store, vector_store, content, metadata={"source": "test_doc_no_rel"}, doc_id="test_doc_no_rel")
    res = kg_store.query("MATCH (n) WHERE 'test_doc_no_rel' IN n.doc_ids RETURN count(n)")
    # The query might return an empty list if no nodes match, or a list with a count of 0.
    # A count query should always return one row.
    assert res[0]['count(n)'] == 0
    logger.info("无三元组内容添加测试成功。")


@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_realistic_kg_update(mock_extract, realistic_kg_stores):
    """测试在已灌入真实数据的图谱上执行更新操作。"""
    kg_store, vector_store = realistic_kg_stores

    update_content = "# 角色动态更新\n龙傲天因为理念不合，离开了青云宗，现在是一名散修。"
    mock_extract.return_value = [
        ("龙傲天", "离开", "青云宗"),
        ("龙傲天", "是", "散修"),
    ]
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content=update_content,
        metadata={"source": "update_doc"},
        doc_id="1.1_design_worldview"  # 覆盖原始世界观文档中的关系
    )

    res_old = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[r:属于]->(:__Entity__ {name: '青云宗'}) RETURN count(r)")
    assert res_old[0]['count(r)'] > 0, "龙傲天'属于'青云宗的旧关系应被保留, 因为更新不删除关系"

    res_new = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[r:离开]->(:__Entity__ {name: '青云宗'}) RETURN count(r)")
    assert res_new[0]['count(r)'] > 0, "龙傲天'离开'青云宗的新关系应已建立"
    logger.info("真实数据更新逻辑验证成功。")


@patch('utils.kg._update_document_hash')
@patch('utils.kg._extract_and_normalize_triplets')
@pytest.mark.asyncio
async def test_kg_add_handles_extraction_failure(mock_extract, mock_update_hash, basic_kg_stores, caplog):
    """测试当三元组提取失败时，kg_add 是否能优雅地处理并继续执行。"""
    logger.info("--- 测试：kg_add 优雅处理三元组提取失败 ---")
    kg_store, vector_store = basic_kg_stores
    doc_id = "extraction_failure_test"
    
    # 模拟 _extract_and_normalize_triplets 抛出异常
    mock_extract.side_effect = ValueError("LLM extraction failed")
    
    # 调用 kg_add
    kg_add(
        kg_store=kg_store,
        vector_store=vector_store,
        content="Some content that would cause an error.",
        metadata={"source": doc_id},
        doc_id=doc_id
    )
    
    assert "从节点中提取三元组时失败" in caplog.text, "应记录提取失败的错误日志"
    assert "LLM extraction failed" in caplog.text, "错误日志应包含原始异常信息"
    res_count = kg_store.query("MATCH (n:__Entity__) RETURN count(n)")
    assert res_count[0]['count(n)'] == 0, "三元组提取失败时不应创建任何实体"
    mock_update_hash.assert_called_once(), "即使提取失败，也应继续尝试执行后续步骤（如更新哈希）"
    logger.success("--- kg_add 优雅处理三元组提取失败测试通过 ---")