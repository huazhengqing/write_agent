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
    # 向量库数据 (偏向描述性)
    vector_add(
        vector_store,
        "龙傲天是一位性格孤傲的剑客，他身着白衣，常年游走于江湖，寻找着能与自己匹敌的对手。他的剑法出神入化，被誉为'天下第一剑'。",
        {"type": "character_profile", "source": "test_profile_1"},
        doc_id="char_lat_profile"
    )
    # 知识图谱数据 (偏向事实和关系)
    kg_add(
        kg_store,
        vector_store_for_kg,
        "龙傲天是'青云剑派'的弟子。龙傲天的师父是'风清扬'。龙傲天有一个宿敌叫'叶良辰'。",
        {"type": "character_relation", "source": "test_relation_1"},
        doc_id="char_lat_relation",
    )
    logger.info("数据已添加到向量库和知识图谱。")
    # 添加仅存在于KG的数据
    kg_add(
        kg_store,
        vector_store_for_kg,
        "叶良辰居住在'北境之巅'。",
        {"type": "character_location", "source": "test_location_1"},
        doc_id="char_ylc_location",
    )
    # 添加仅存在于Vector Store的数据
    vector_add(
        vector_store,
        "风清扬是一位隐世高人，剑术超凡，但从不参与江湖纷争。",
        {"type": "character_profile", "source": "test_profile_2"},
        doc_id="char_fqy_profile"
    )
    logger.info("已添加部分缺失的测试数据。")
    
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
    
    question = "龙傲天是谁？他有什么关系？"
    result = await hybrid_query(vector_query_engine, kg_query_engine, question)
    logger.info(f"hybrid_query 对 '{question}' 的回答:\n{result}")
    
    assert "孤傲" in result and "剑客" in result, "结果应包含向量库中的性格描述"
    assert "青云剑派" in result and "风清扬" in result and "叶良辰" in result, "结果应包含知识图谱中的关系信息"
    logger.success("--- hybrid_query (基本功能) 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_react(hybrid_query_test_data):
    """测试基于ReAct的混合查询功能。"""
    logger.info("--- 测试 hybrid_query_react ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    question = "请全面介绍一下龙傲天这个角色，包括他的性格、师承和主要对手。"
    result = await hybrid_query_react(vector_query_engine, kg_query_engine, question, "你是一个角色档案分析师。")
    logger.info(f"hybrid_query_react 对 '{question}' 的回答:\n{result}")
    
    assert "性格" in result and "孤傲" in result, "ReAct结果应包含性格信息"
    assert "师承" in result and "风清扬" in result, "ReAct结果应包含师承信息"
    assert "对手" in result and "叶良辰" in result, "ReAct结果应包含对手信息"
    logger.success("--- hybrid_query_react 测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_with_auto_retriever(hybrid_query_test_data):
    """测试带有自动元数据过滤的混合查询。"""
    logger.info("--- 测试带有自动元数据过滤的混合查询 ---)")
    auto_vector_query_engine = hybrid_query_test_data["auto_vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 这个问题会触发元数据过滤, 因为它明确要求 "角色简介" (character_profile)
    question = "请根据龙傲天的角色简介，介绍一下他的性格。"
    result = await hybrid_query(auto_vector_query_engine, kg_query_engine, question)
    logger.info(f"带有自动元数据过滤的 hybrid_query 对 '{question}' 的回答:\n{result}")
    
    assert "孤傲" in result and "剑客" in result, "自动过滤查询应准确找到角色简介"
    assert "隐世高人" not in result, "结果不应包含其他角色的简介信息"
    logger.success("--- 带有自动元数据过滤的混合查询测试通过 ---)")


@pytest.mark.asyncio
async def test_hybrid_query_partial_results(hybrid_query_test_data):
    """测试混合查询在部分结果为空时的表现。"""
    logger.info("--- 测试混合查询 (部分结果为空) ---)")
    vector_query_engine = hybrid_query_test_data["vector_query_engine"]
    kg_query_engine = hybrid_query_test_data["kg_query_engine"]
    
    # 测试仅知识图谱有结果的情况
    logger.info("--- 测试仅知识图谱有结果 ---)")
    question1 = "叶良辰住在哪里？"
    result1 = await hybrid_query(vector_query_engine, kg_query_engine, question1)
    logger.info(f"对 '{question1}' 的回答:\n{result1}")
    assert "北境之巅" in result1
    
    # 测试仅向量库有结果的情况
    logger.info("--- 测试仅向量库有结果 ---)")
    question2 = "风清扬是个怎样的人？"
    result2 = await hybrid_query(vector_query_engine, kg_query_engine, question2)
    logger.info(f"对 '{question2}' 的回答:\n{result2}")
    assert "隐世高人" in result2 and "剑术超凡" in result2
    
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
