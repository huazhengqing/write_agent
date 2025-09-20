import os
import sys
import pytest
import re
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm import llm_completion, get_llm_params, get_llm_messages, llm_temperatures
from prompts.story.kg import kg_gen_cypher_prompt_design, kg_gen_cypher_prompt_write


# MODEL_GROUPS = ["fast", "summary", "reasoning"]
MODEL_GROUPS = ["summary"]


SCHEMA_DESIGN = """
Node properties: [{'properties': [('name', 'STRING'), ('doc_ids', 'STRING[]')], 'label': '__Entity__'}]
Relationship properties: [{'properties': [('label', 'STRING')], 'label': 'relationship'}]
Relationships: [('__Entity__', '定义了', '__Entity__'), ('__Entity__', '背景是', '__Entity__'), ('__Entity__', '核心能力是', '__Entity__'), ('__Entity__', '设计理念是', '__Entity__'), ('__Entity__', '分为', '__Entity__'), ('__Entity__', '包含', '__Entity__'), ('__Entity__', '发生在', '__Entity__'), ('__Entity__', '正向影响', '__Entity__'), ('__Entity__', '负向影响', '__Entity__')]
"""


DESIGN_QUERIES = [
    ("角色A的背景设定是什么?", 'MATCH (c:__Entity__ {name: "角色A"})-[:背景是]->(b:__Entity__)'),
    ("第一卷包含了哪些章节？", 'MATCH (v:__Entity__ {name: "第一卷"})-[:包含]->(c:__Entity__)'),
    ("角色A和角色B的相遇对谁产生了影响?", 'MATCH (e:__Entity__ {name: "角色A与角色B的相遇"})-[r:正向影响|负向影响]->(t:__Entity__)'),
    ("角色A的设计理念是什么?", 'MATCH (d:__Entity__)-[:定义了]->(c:__Entity__ {name: "角色A"})'),
    ("一个不存在的关系", "INVALID_QUERY"), # 测试无效查询的生成
]


SCHEMA_WRITE = """
Node properties: [{'properties': [('name', 'STRING'), ('doc_ids', 'STRING[]')], 'label': '__Entity__'}]
Relationship properties: [{'properties': [('label', 'STRING')], 'label': 'relationship'}]
Relationships: [('__Entity__', '拥有动机', '__Entity__'), ('__Entity__', '位于', '__Entity__'), ('__Entity__', '发现线索', '__Entity__'), ('__Entity__', '持有信念', '__Entity__'), ('__Entity__', '导致', '__Entity__'), ('__Entity__', '健康状态是', '__Entity__'), ('__Entity__', '状态变为', '__Entity__')]
"""


WRITE_QUERIES = [
    ("角色A的健康状态是什么?", 'MATCH (c:__Entity__ {name: "角色A"})-[:健康状态是]->(s:__Entity__)'),
    ("是什么事件导致了角色A状态变为重伤？", 'MATCH (e:__Entity__)-[:导致]->(r:__Entity__ {name: "角色A状态变为重伤"})'),
    ("角色A在地点X发现了什么线索?", 'MATCH (c:__Entity__ {name: "角色A"})-[:位于]->(l:__Entity__ {name: "地点X"})'),
    ("龙傲天的宿敌是谁？", "INVALID_QUERY"), # 测试当关系不存在于Schema中时的处理
]


def _validate_cypher_query(query_str: str, question: str, expected_pattern: str):
    """
    验证生成的Cypher查询字符串。
    """
    if expected_pattern == "INVALID_QUERY":
        assert query_str == "INVALID_QUERY", f"对于问题 '{question}', 预期返回 'INVALID_QUERY', 实际返回: {query_str}"
        return

    assert query_str != "INVALID_QUERY", f"对于问题 '{question}', 不应返回 'INVALID_QUERY'"
    assert "\n" not in query_str, "Cypher 查询必须是单行文本"

    # 检查核心模式是否存在
    assert expected_pattern in query_str, f"生成的查询 '{query_str}' 未包含预期模式 '{expected_pattern}'"


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_group", MODEL_GROUPS)
@pytest.mark.parametrize("question, expected_pattern", DESIGN_QUERIES)
async def test_kg_gen_cypher_design(llm_group, question, expected_pattern):
    """
    测试针对设计文档知识图谱的Cypher查询生成。
    """
    logger.info(f"--- 测试 Cypher 生成 (Design) | 模型组: {llm_group} | 问题: {question} ---")

    # 1. 准备LLM参数和消息
    llm_params = get_llm_params(llm_group=llm_group, temperature=llm_temperatures["classification"])
    
    context_dict = {
        "query_str": question,
        "schema": SCHEMA_DESIGN
    }
    # 注意：这里的提示词模板变量是 {query_str} 和 {schema}，它们应该直接在 system_prompt 中格式化
    messages = get_llm_messages(
        system_prompt=kg_gen_cypher_prompt_design,
        context_dict_system=context_dict
    )
    llm_params["messages"] = messages

    # 2. 调用LLM
    response = await llm_completion(llm_params)
    result_str = response.content.strip()

    logger.info(f"模型组 '{llm_group}' 生成的 Cypher 查询:\n{result_str}")

    # 3. 验证结果
    _validate_cypher_query(result_str, question, expected_pattern)
    
    logger.success(f"--- Cypher 生成 (Design) 测试通过 | 模型组: {llm_group} ---")


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_group", MODEL_GROUPS)
@pytest.mark.parametrize("question, expected_pattern", WRITE_QUERIES)
async def test_kg_gen_cypher_write(llm_group, question, expected_pattern):
    """
    测试针对小说正文知识图谱的Cypher查询生成。
    """
    logger.info(f"--- 测试 Cypher 生成 (Write) | 模型组: {llm_group} | 问题: {question} ---")

    # 1. 准备LLM参数和消息
    llm_params = get_llm_params(llm_group=llm_group, temperature=llm_temperatures["classification"])
    
    context_dict = {
        "query_str": question,
        "schema": SCHEMA_WRITE
    }
    messages = get_llm_messages(
        system_prompt=kg_gen_cypher_prompt_write,
        context_dict_system=context_dict
    )
    llm_params["messages"] = messages

    # 2. 调用LLM
    response = await llm_completion(llm_params)
    result_str = response.content.strip()

    logger.info(f"模型组 '{llm_group}' 生成的 Cypher 查询:\n{result_str}")

    # 3. 验证结果
    _validate_cypher_query(result_str, question, expected_pattern)
    
    logger.success(f"--- Cypher 生成 (Write) 测试通过 | 模型组: {llm_group} ---")