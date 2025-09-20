import os
import sys
import asyncio
import re
import pytest
import pytest_asyncio
import tempfile
import shutil
from typing import List, Literal, Optional, Tuple, Union
from loguru import logger

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_data import (
    VECTOR_TEST_NOVEL_WORLDVIEW,
    VECTOR_TEST_NOVEL_CHARACTERS,
    VECTOR_TEST_NOVEL_PLOT_ARC,
    VECTOR_TEST_NOVEL_MAGIC_SYSTEM,
    VECTOR_TEST_NOVEL_FACTIONS,
    VECTOR_TEST_NOVEL_CHAPTER
)

from utils.config import llm_temperatures, get_llm_params
from utils.llm import get_llm_messages, llm_completion
from utils.vector import index_query, get_vector_store, vector_add, get_vector_query_engine
from utils.kg import get_kg_store, kg_add, get_kg_query_engine
from utils.agent import call_react_agent
from utils.log import init_logger
from utils.hybrid_query import hybrid_query, hybrid_query_batch, hybrid_query_react


@pytest_asyncio.fixture(scope="module")
async def hybrid_query_test_data():
    """
    准备混合查询测试所需的数据和查询引擎。
    这个fixture在模块级别运行一次，以减少资源消耗。
    """
    logger.info("--- (Fixture) 准备混合查询测试数据 ---)")
    init_logger("hybrid_query_test")
    
    # 1. 初始化临时目录
    test_dir = tempfile.mkdtemp()
    vector_db_path = os.path.join(test_dir, "vector_db")
    kg_db_path = os.path.join(test_dir, "kg_db")
    vector_for_kg_db_path = os.path.join(test_dir, "vector_for_kg_db")
    logger.info(f"测试目录已创建: {test_dir}")
    
    # 2. 准备Vector Store和KG Store
    vector_store = get_vector_store(vector_db_path, "hybrid_test_v")
    kg_store = get_kg_store(kg_db_path)
    vector_store_for_kg = get_vector_store(vector_for_kg_db_path, "hybrid_test_kg_v")

    # 3. 添加数据
    # 向量库数据 (偏向描述性、非结构化文本)
    vector_add(vector_store, VECTOR_TEST_NOVEL_CHAPTER, {"doc_type": "chapter_content", "chapter": 1}, doc_id="chapter_1")
    vector_add(vector_store, VECTOR_TEST_NOVEL_MAGIC_SYSTEM, {"doc_type": "setting", "category": "magic_system"}, doc_id="magic_system")
    vector_add(vector_store, VECTOR_TEST_NOVEL_CHARACTERS, {"doc_type": "character_sheet", "format": "json"}, doc_id="character_sheet")
    logger.info("数据已添加到向量库。")

    # 知识图谱数据 (偏向事实、关系)
    # 使用包含明确关系和实体的文本
    kg_add(kg_store, vector_store_for_kg, VECTOR_TEST_NOVEL_WORLDVIEW, {"doc_type": "setting", "category": "worldview"}, doc_id="worldview")
    kg_add(kg_store, vector_store_for_kg, VECTOR_TEST_NOVEL_FACTIONS, {"doc_type": "setting", "category": "factions"}, doc_id="factions")
    kg_add(kg_store, vector_store_for_kg, VECTOR_TEST_NOVEL_PLOT_ARC, {"doc_type": "plot_arc", "arc": "东海风云"}, doc_id="plot_arc_1")
    logger.info("数据已添加到知识图谱。")

    # 4. 创建查询引擎
    vector_query_engine = get_vector_query_engine(vector_store)
    kg_query_engine = get_kg_query_engine(kg_store, vector_store_for_kg)
    auto_vector_query_engine = get_vector_query_engine(vector_store, use_auto_retriever=True)
    logger.info("向量和知识图谱查询引擎已创建。")
    
    # 提供测试数据和查询引擎
    yield {
        "vector_query_engine": vector_query_engine,
        "kg_query_engine": kg_query_engine,
        "auto_vector_query_engine": auto_vector_query_engine,
        "test_dir": test_dir
    }
    
    # 5. 清理
    shutil.rmtree(test_dir)
    logger.info(f"测试目录已删除: {test_dir}")


@pytest.mark.asyncio
async def test_hybrid_query_basic(hybrid_query_test_data):
    """测试基本的混合查询功能。"""
    logger.info("--- 测试 hybrid_query (基本功能) ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    question = "全面介绍一下龙傲天，包括他的背景、目标、朋友和敌人。"
    result = await hybrid_query(vector_query_engine, kg_query_engine, question)
    logger.info(f"hybrid_query 对 '{question}' 的回答:\n{result}")
    
    assert "地球" in result and "穿越" in result, "结果应包含向量库中的背景描述"
    assert "赵日天" in result and "叶良辰" in result, "结果应包含知识图谱中的关系信息"
    assert "青云宗" in result, "结果应包含知识图谱中的势力信息"
    logger.success("--- hybrid_query (基本功能) 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_react(hybrid_query_test_data):
    """测试基于ReAct的混合查询功能。"""
    logger.info("--- 测试 hybrid_query_react ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    question = "龙傲天在'东海风云'情节中，是如何从初到临海镇到最后击退北冥魔殿的？请描述关键事件、他遇到的关键人物以及他获得的能力提升。"
    result = await hybrid_query_react(vector_query_engine, kg_query_engine, question, "你是一个小说情节分析师。")
    logger.info(f"hybrid_query_react 对 '{question}' 的回答:\n{result}")
    
    assert "临海镇" in result and "赵日天" in result, "ReAct结果应包含初始情节和人物"
    assert "叶良辰" in result and "海图残卷" in result, "ReAct结果应包含核心冲突和关键物品"
    assert "御水决" in result, "ReAct结果应包含获得的新能力"
    assert "筑基期" in result, "ReAct结果应包含修为突破信息"
    logger.success("--- hybrid_query_react 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_react_multi_step_reasoning(hybrid_query_test_data):
    """
    测试 ReAct Agent 的多步推理能力。
    问题需要 Agent 先找到一个实体，再根据该实体进行第二次查询，最后综合答案。
    """
    logger.info("--- 测试 hybrid_query_react (多步推理) ---")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]

    # 这个问题需要多步才能回答:
    # 1. 龙傲天的宿敌是谁？ (KG Search -> 叶良辰)
    # 2. 叶良辰属于哪个势力？ (KG Search -> 北冥魔殿)
    # 3. 北冥魔殿的特点是什么？ (Vector or KG Search -> 魔道巨擘, 行事诡秘)
    question = "龙傲天的宿敌是谁？这个宿敌属于哪个势力，该势力的特点是什么？"
    
    result = await hybrid_query_react(vector_query_engine, kg_query_engine, question, "你是一个小说世界观分析师。")
    logger.info(f"hybrid_query_react (多步推理) 对 '{question}' 的回答:\n{result}")

    # 验证最终答案是否综合了多步查询的结果
    assert "叶良辰" in result, "答案应明确指出宿敌是'叶良辰'"
    assert "北冥魔殿" in result, "答案应包含宿敌所属的势力'北冥魔殿'"
    assert "魔道" in result or "诡秘" in result or "吞噬" in result, "答案应包含势力的特点描述"
        
    logger.success("--- hybrid_query_react (多步推理) 测试通过 ---")


@pytest.mark.asyncio
async def test_hybrid_query_react_tool_selection(hybrid_query_test_data):
    """
    测试 ReAct Agent 是否能根据问题类型正确选择工具。
    - 描述性问题 -> vector_search
    - 事实/关系问题 -> knowledge_graph_search
    """
    logger.info("--- 测试 hybrid_query_react (工具选择) ---")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]

    # 问题1: 适合向量搜索的描述性问题
    question_vector = "请详细描述一下'御水决'这个功法。"
    result_vector = await hybrid_query_react(vector_query_engine, kg_query_engine, question_vector)
    logger.info(f"对向量搜索问题的回答:\n{result_vector}")
    # '御水决' 的详细描述在 VECTOR_TEST_NOVEL_MAGIC_SYSTEM 中，该文件被加载到向量库
    assert "操控水元素" in result_vector and ("水中呼吸" in result_vector or "高速移动" in result_vector), "应从向量库中找到'御水决'的详细描述"

    # 问题2: 适合知识图谱搜索的关系问题
    question_kg = "青云宗和天剑阁是什么关系？"
    result_kg = await hybrid_query_react(vector_query_engine, kg_query_engine, question_kg)
    logger.info(f"对知识图谱问题的回答:\n{result_kg}")
    # 这个关系在 VECTOR_TEST_NOVEL_FACTIONS 中，该文件被加载到知识图谱
    assert "关系密切" in result_kg or "正道联盟" in result_kg, "应从知识图谱中找到两个势力的关系"
    
    logger.success("--- hybrid_query_react (工具选择) 测试通过 ---")


@pytest.mark.asyncio
async def test_hybrid_query_react_no_relevant_tool(hybrid_query_test_data):
    """
    测试当问题与所有可用工具都无关时，Agent 的行为。
    """
    logger.info("--- 测试 hybrid_query_react (无相关工具) ---")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]

    question = "请问今天北京的天气怎么样？"
    result = await hybrid_query_react(vector_query_engine, kg_query_engine, question)
    logger.info(f"对无关问题的回答:\n{result}")

    # Agent 应该表明它无法回答这个问题
    assert any(keyword in result.lower() for keyword in ["无法回答", "没有工具", "不知道", "cannot answer", "don't have a tool"]), \
        "Agent 应该表明它无法回答与工具无关的问题"
        
    logger.success("--- hybrid_query_react (无相关工具) 测试通过 ---")


@pytest.mark.asyncio
async def test_hybrid_query_with_auto_retriever(hybrid_query_test_data):
    """测试带有自动元数据过滤的混合查询。"""
    logger.info("--- 测试带有自动元数据过滤的混合查询 ---)")
    auto_vector_query_engine = hybrid_query_test_data["auto_vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 这个问题会触发元数据过滤, 因为它明确要求 "角色设定表" (character_sheet)
    question = "根据角色设定表(character_sheet)，叶良辰的背景是什么？"
    result = await hybrid_query(auto_vector_query_engine, kg_query_engine, question)
    logger.info(f"带有自动元数据过滤的 hybrid_query 对 '{question}' 的回答:\n{result}")
    
    assert "北冥魔殿" in result and "殿主之子" in result, "自动过滤查询应准确找到角色设定表中的背景"
    assert "鸿蒙道体" not in result, "结果不应包含其他角色的信息"
    logger.success("--- 带有自动元数据过滤的混合查询测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_partial_results(hybrid_query_test_data):
    """测试混合查询在部分结果为空时的表现。"""
    logger.info("--- 测试混合查询 (部分结果为空) ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 测试仅知识图谱有结果的情况
    logger.info("--- 测试仅知识图谱有结果 ---)")
    question1 = "天机阁是一个什么样的组织？"
    result1 = await hybrid_query(vector_query_engine, kg_query_engine, question1)
    logger.info(f"对 '{question1}' 的回答:\n{result1}")
    assert "中立" in result1 and "贩卖情报" in result1
    
    # 测试仅向量库有结果的情况
    logger.info("--- 测试仅向量库有结果 ---)")
    question2 = "鸿蒙道体有什么用？"
    result2 = await hybrid_query(vector_query_engine, kg_query_engine, question2)
    logger.info(f"对 '{question2}' 的回答:\n{result2}")
    assert "亲和力" in result2 and "修炼速度" in result2
    
    logger.success("--- 混合查询 (部分结果为空) 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_invalid_input(hybrid_query_test_data):
    """测试混合查询对无效输入的处理。"""
    logger.info("--- 测试混合查询 (无效输入) ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 测试空问题
    result1 = await hybrid_query(vector_query_engine, kg_query_engine, "")
    assert result1 == "", "空问题应返回空字符串"
    
    # 测试非字符串问题
    result2 = await hybrid_query(vector_query_engine, kg_query_engine, None)
    assert result2 == "", "非字符串问题应返回空字符串"
    
    logger.success("--- 混合查询 (无效输入) 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_batch_basic(hybrid_query_test_data):
    """测试基本的批量混合查询功能。"""
    logger.info("--- 测试 hybrid_query_batch (基本功能) ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 准备多个测试问题
    questions = [
        "龙傲天的宿敌是谁？",
        "九霄大陆的能量体系是怎样的？",
        "青云宗是一个什么样的门派？"
    ]
    
    # 执行批量查询
    results = await hybrid_query_batch(vector_query_engine, kg_query_engine, questions)
    
    # 验证结果数量与问题数量一致
    assert len(results) == len(questions), "批量查询结果数量应与问题数量一致"
    
    # 验证每个问题的查询结果
    logger.info(f"批量查询结果1: {results[0]}")
    assert "叶良辰" in results[0], "结果1应包含宿敌信息"
    
    logger.info(f"批量查询结果2: {results[1]}")
    assert "灵力" in results[1] and "炼气" in results[1], "结果2应包含能量体系信息"
    
    logger.info(f"批量查询结果3: {results[2]}")
    assert "正道" in results[2] and "剑道" in results[2], "结果3应包含青云宗的描述"
    
    logger.success("--- hybrid_query_batch (基本功能) 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_batch_empty_input(hybrid_query_test_data):
    """测试批量混合查询对空输入的处理。"""
    logger.info("--- 测试 hybrid_query_batch (空输入) ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 测试空列表
    results = await hybrid_query_batch(vector_query_engine, kg_query_engine, [])
    assert results == [], "空问题列表应返回空列表"
    
    logger.success("--- hybrid_query_batch (空输入) 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_batch_partial_results(hybrid_query_test_data):
    """测试批量混合查询在部分结果可能为空时的表现。"""
    logger.info("--- 测试 hybrid_query_batch (部分结果为空) ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 包含一个可能无法获取结果的问题
    questions = [
        "龙傲天在第一章里做了什么？",
        "孙悟空在九霄大陆的战力如何？",  # 这个问题没有相关数据
        "北冥魔殿的特点是什么？"
    ]
    
    results = await hybrid_query_batch(vector_query_engine, kg_query_engine, questions)
    
    # 验证结果数量与问题数量一致
    assert len(results) == len(questions), "即使部分查询无结果，也应返回相同数量的结果"
    
    # 验证已知的有效结果
    assert "临海镇" in results[0] and "赵日天" in results[0], "结果0应包含第一章的有效信息"
    assert "魔道" in results[2] and "吞噬" in results[2], "结果2应包含北冥魔殿的有效信息"
    
    # 第二个结果可能是空字符串或包含一些“不知道”的信息，但不应导致整个批量查询失败
    assert isinstance(results[1], str), "单个查询失败不应影响结果类型"
    
    logger.success("--- hybrid_query_batch (部分结果为空) 测试通过 ---)")
