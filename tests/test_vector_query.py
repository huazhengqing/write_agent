import os
import sys
import pytest
import asyncio
from loguru import logger

from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vector import get_vector_query_engine, index_query, index_query_batch, vector_add
from tests.test_data import (
    VECTOR_TEST_NOVEL_WORLDVIEW,
    VECTOR_TEST_NOVEL_CHARACTERS,
    VECTOR_TEST_NOVEL_PLOT_ARC,
    VECTOR_TEST_NOVEL_STRUCTURED_INFO,
)


# 新增的包含结构化信息的测试数据
# VECTOR_TEST_NOVEL_STRUCTURED_INFO 已移至 test_data.py


@pytest.fixture(scope="module")
def realistic_store(ingested_store):
    """
    在现有 store 的基础上，添加带有 task_id 和 type 元数据的、更真实的测试数据。
    这模拟了 story_rag.py 在处理真实任务时的存储情况。
    """
    logger.info("--- (Fixture) 注入模拟真实场景的测试数据 ---")
    
    # 模拟来自不同任务的设计文档、章节内容和结构化信息
    realistic_data = {
        "1.1_design_worldview": {
            "content": VECTOR_TEST_NOVEL_WORLDVIEW,
            "metadata": {"task_id": "1.1", "type": "design", "status": "active"}
        },
        "1.2_design_characters": {
            "content": VECTOR_TEST_NOVEL_CHARACTERS,
            "metadata": {"task_id": "1.2", "type": "design", "status": "active"}
        },
        "1.3_design_plot": {
            "content": VECTOR_TEST_NOVEL_PLOT_ARC,
            "metadata": {"task_id": "1.3", "type": "design", "status": "active"}
        },
        # 添加一个独特的、用于 "nin" 过滤器测试的文档
        "1.3.1_write_scene": {
            "content": "在黑风寨，叶良辰夺走海图残卷后，冷冷地说：'我叫叶良辰，你，记住了。'",
            "metadata": {"task_id": "1.3.1", "type": "write", "status": "active"}
        },
        # 添加包含列表和表格的结构化数据
        "1.4_design_structured": {
            "content": VECTOR_TEST_NOVEL_STRUCTURED_INFO,
            "metadata": {"task_id": "1.4", "type": "design", "status": "active"}
        }
    }

    for doc_id, data in realistic_data.items():
        vector_add(ingested_store, data["content"], data["metadata"], doc_id=doc_id)
    
    # 在异步环境中，最好有更可靠的等待机制，但对于本地文件DB，短暂停顿通常足够
    # asyncio.run(asyncio.sleep(1)) # 在 fixture 中不能直接 run
    return ingested_store


@pytest.mark.asyncio
async def test_realistic_novel_queries(realistic_store):
    """测试针对复杂小说项目的真实查询场景，模拟续写和设计。"""
    logger.info("--- 测试：真实小说项目查询场景 ---")
    query_engine = get_vector_query_engine(realistic_store, similarity_top_k=10, rerank_top_n=5)

    questions = [
        # 场景1: 续写第二卷前，回顾第一卷的核心情节和人物关系
        "龙傲天和叶良辰的三年之约是如何结下的？涉及到哪些关键物品和人物？",
        # 场景2: 设计新角色时，需要参考现有世界观和势力设定
        "我想设计一个来自'北境魔域'的新角色，请提供该区域的背景信息，以及已知的相关势力（如北冥魔殿）。",
        # 场景3: 检查角色能力是否符合设定
        "龙傲天获得的'鸿蒙道体'有什么特殊之处？"
    ]
    
    results = await index_query_batch(query_engine, questions)
    r1, r2, r3 = results

    logger.info(f"Q1: {questions[0]}\nA1: {r1}")
    assert all(k in r1 for k in ["黑风寨", "海图残卷", "赵日天", "重伤"])

    logger.info(f"Q2: {questions[1]}\nA2: {r2}")
    assert all(k in r2 for k in ["北境魔域", "魔道修士", "北冥魔殿", "叶良辰"])

    logger.info(f"Q3: {questions[2]}\nA3: {r3}")
    assert all(k in r3 for k in ["极高的亲和力", "修炼无瓶颈"])
    logger.success("--- 真实小说项目查询场景测试通过 ---")


@pytest.mark.asyncio
async def test_query_structured_data(realistic_store):
    """测试针对包含列表、表格等结构化信息的查询。"""
    logger.info("--- 测试：结构化数据（列表、表格）查询 ---")
    query_engine = get_vector_query_engine(realistic_store, similarity_top_k=5, rerank_top_n=3)

    # 问题1: 查询列表信息
    question_list = "苍穹剑派有哪些核心成员？"
    result_list = await index_query(query_engine, question_list)
    logger.info(f"Q_list: {question_list}\nA_list: {result_list}")
    assert "风清扬" in result_list and "令狐冲" in result_list

    # 问题2: 查询表格信息
    question_table = "修炼等级'金丹'后面是哪个等级？"
    result_table = await index_query(query_engine, question_table)
    logger.info(f"Q_table: {question_table}\nA_table: {result_table}")
    assert "元婴" in result_table

    # 问题3: 跨结构查询
    question_cross = "叶良辰属于哪个势力？"
    result_cross = await index_query(query_engine, question_cross)
    logger.info(f"Q_cross: {question_cross}\nA_cross: {result_cross}")
    assert "北冥魔殿" in result_cross
    logger.success("--- 结构化数据查询测试通过 ---")


@pytest.mark.asyncio
async def test_complex_filtering_nin_operator(realistic_store):
    """测试 "nin" (not in) 元数据过滤器，模拟 story_rag.py 中排除先行任务的场景。"""
    logger.info("--- 测试：复杂过滤器 (nin) ---")
    await asyncio.sleep(1)  # 确保数据已入库

    # 模拟场景：正在处理后续任务，需要查询之前的设计，但要排除 task_id 为 '1.3.1' 的内容。
    filters = MetadataFilters(filters=[MetadataFilter(key="task_id", value=["1.3.1"], operator="nin")])
    query_engine = get_vector_query_engine(realistic_store, filters=filters, similarity_top_k=5, rerank_top_n=3)
    
    question_filtered = "叶良辰在黑风寨说了什么？"
    result_filtered = await index_query(query_engine, question_filtered)
    logger.info(f"'nin' 过滤器查询结果:\n{result_filtered}")
    assert "我叫叶良辰" not in result_filtered, "被 'nin' 过滤器排除的文档内容不应出现在结果中"

    # 作为对比，不带过滤器进行查询，确保数据本身是可查询的
    query_engine_unfiltered = get_vector_query_engine(realistic_store, similarity_top_k=5, rerank_top_n=3)
    result_unfiltered = await index_query(query_engine_unfiltered, question_filtered)
    logger.info(f"无过滤器查询结果:\n{result_unfiltered}")
    assert "我叫叶良辰" in result_unfiltered, "无过滤器时应能查到该内容"
    logger.success("--- 复杂过滤器 (nin) 测试通过 ---")

    
@pytest.mark.asyncio
async def test_complex_filtering_and_operator(realistic_store):
    """测试多个过滤条件的组合 (AND 逻辑)，模拟查询特定类型的活跃文档。"""
    logger.info("--- 测试：复杂过滤器 (AND) ---")
    await asyncio.sleep(1)
    # 查找所有类型为 'design' 且状态为 'active' 的文档
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="type", value="design"), 
            MetadataFilter(key="status", value="active")
        ], 
        condition="and"
    )
    query_engine = get_vector_query_engine(realistic_store, filters=filters, similarity_top_k=10, rerank_top_n=5)
    
    question = "列出所有活跃的设计文档内容，包括世界观、角色和情节。"
    result = await index_query(query_engine, question)
    logger.info(f"'AND' 过滤器查询结果:\n{result}")
    
    # 应该包含所有 design 文档的内容
    assert all(k in result for k in ["九霄大陆", "鸿蒙道体", "黑风寨之乱"])
    # 不应该包含 write 类型的文档内容
    assert "我叫叶良辰" not in result
    logger.success("--- 复杂过滤器 (AND) 测试通过 ---")