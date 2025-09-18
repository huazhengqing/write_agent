import os
import sys
import asyncio
import re
from typing import List, Literal, Optional, Tuple, Union
from loguru import logger
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures  # noqa
from utils.vector import index_query
from utils.agent import call_react_agent


synthesis_system_prompt_default = """
# 角色
信息整合分析师。

# 任务
根据`当前问题`, 整合`向量数据库检索结果`(提供上下文)和`知识图谱检索结果`(提供事实), 生成一个统一、精炼、且完全基于所提供信息的回答。

# 工作流程
1.  事实优先: 首先分析`知识图谱检索结果`, 提取核心事实与关系。
2.  上下文补充: 结合`向量数据库检索结果`, 为核心事实补充丰富的描述性上下文。
3.  综合回答: 整合两类信息, 形成一个逻辑连贯、直接回答`当前问题`的最终报告。

# 核心原则
- 忠于原文: 你的回答必须完全基于提供的信息, 禁止引入外部知识或主观推断。
- 冲突解决: 当信息冲突时, 优先采信`知识图谱检索结果`中的事实性信息。
- 空值处理: 如果两类信息均为空或与问题无关, 则不输出任何内容。
- 禁止罗列: 必须对信息进行整合提炼, 禁止直接复制粘贴原文。
- 禁止元对话: 禁止任何关于你自身或任务过程的描述 (例如, “根据您的要求...”, “整合信息后...”)。

# 输出要求
- 格式: 使用Markdown。通过列表、表格等元素清晰组织内容。
- 可视化: 对实体关系、时间线等复杂结构, 使用Mermaid图表呈现。
- 风格: 清晰、简洁、客观。
"""


synthesis_user_prompt_default = """
# 当前问题
{question}

# 从数据库中检索到的信息源

## 向量数据库检索结果 (语义与上下文)
{vector_str}

## 知识图谱检索结果 (事实与关系)
{kg_str}
"""


async def hybrid_query(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    question: str,
    synthesis_system_prompt: Optional[str] = None,
    synthesis_user_prompt: Optional[str] = None,
) -> str:
    if not question or not isinstance(question, str):
        logger.warning("查询问题为空或类型不正确, 无法执行混合查询。")
        return ""

    logger.info(f"开始对问题 '{question}' 执行混合查询...")

    system_prompt = synthesis_system_prompt or synthesis_system_prompt_default
    user_prompt = synthesis_user_prompt or synthesis_user_prompt_default

    logger.debug("正在并行执行向量查询和知识图谱查询...")
    vector_task = index_query(vector_query_engine, [question])
    kg_task = index_query(kg_query_engine, [question])
    vector_contents, kg_contents = await asyncio.gather(vector_task, kg_task)

    if not vector_contents and not kg_contents:
        logger.warning(f"对于问题 '{question}', 向量和知识图谱查询均未返回任何内容。")
        return ""

    formatted_vector_str = "\n\n---\n\n".join(vector_contents)
    formatted_kg_str = "\n\n---\n\n".join(kg_contents)

    logger.debug(f"向量查询结果 (片段数: {len(vector_contents)}):\n{formatted_vector_str[:500]}...")
    logger.debug(f"知识图谱查询结果 (片段数: {len(kg_contents)}):\n{formatted_kg_str[:500]}...")

    context_dict_user = {"vector_str": formatted_vector_str, "kg_str": formatted_kg_str, "question": question}
    messages = get_llm_messages(system_prompt, user_prompt, None, context_dict_user)

    logger.info("开始调用LLM进行信息整合...")
    final_llm_params = get_llm_params(llm_group='reasoning', messages=messages, temperature=llm_temperatures["synthesis"])
    final_message = await llm_completion(final_llm_params)

    logger.success(f"混合查询完成，生成回答长度: {len(final_message.content)}")
    logger.debug(f"最终回答:\n{final_message.content}")

    return final_message.content.strip()


async def hybrid_query_batch(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    questions: List[str],
    synthesis_system_prompt: Optional[str] = None,
    synthesis_user_prompt: Optional[str] = None,
) -> List[str]:
    if not questions:
        return []

    logger.info(f"开始执行 {len(questions)} 个问题的批量混合查询...")
    tasks = [
        hybrid_query(
            vector_query_engine,
            kg_query_engine,
            q,
            synthesis_system_prompt,
            synthesis_user_prompt,
        )
        for q in questions
    ]
    results = await asyncio.gather(*tasks)
    logger.success(f"批量混合查询完成。")
    return results


###############################################################################


async def hybrid_query_react(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    query_str: str,
    agent_system_prompt: Optional[str] = None,
) -> str:
    logger.info(f"开始对问题 '{query_str}' 执行基于ReAct的混合查询...")
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name="vector_search",
        description="用于查找设定、摘要等语义相似的内容 (例如: 角色背景, 世界观设定, 物品描述)。当问题比较复杂时, 你可以多次调用此工具来回答问题的不同部分, 然后综合答案。"
    )
    kg_tool = QueryEngineTool.from_defaults(
        query_engine=kg_query_engine,
        name="knowledge_graph_search",
        description="用于探索实体及其关系 (例如: 角色A和角色B是什么关系? 事件C导致了什么后果?)。当问题比较复杂时, 你可以多次调用此工具来回答问题的不同部分, 然后综合答案。"
    )
    result = await call_react_agent(
        system_prompt=agent_system_prompt,
        user_prompt=query_str,
        tools=[vector_tool, kg_tool],
        llm_group="reasoning",
        temperature=llm_temperatures["reasoning"]
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)

    logger.success(f"基于ReAct的混合查询完成。")
    return result.strip()


###############################################################################


if __name__ == '__main__':
    import tempfile
    import shutil
    from utils.log import init_logger
    from utils.vector import get_vector_store, vector_add, get_vector_query_engine
    from utils.kg import get_kg_store, kg_add, get_kg_query_engine

    init_logger("hybrid_query_test")

    # 1. 初始化临时目录
    test_dir = tempfile.mkdtemp()
    vector_db_path = os.path.join(test_dir, "vector_db")
    kg_db_path = os.path.join(test_dir, "kg_db")
    vector_for_kg_db_path = os.path.join(test_dir, "vector_for_kg_db")
    logger.info(f"测试目录已创建: {test_dir}")

    async def main():
        # 2. 准备 Vector Store 和 KG Store
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

        # 4. 创建查询引擎
        vector_query_engine = get_vector_query_engine(vector_store)
        kg_query_engine = get_kg_query_engine(kg_store, vector_store_for_kg)
        logger.info("向量和知识图谱查询引擎已创建。")

        # 5. 测试 hybrid_query
        logger.info("--- 测试 hybrid_query ---")
        question1 = "龙傲天是谁？他有什么关系？"
        result1 = await hybrid_query(vector_query_engine, kg_query_engine, question1)
        logger.info(f"hybrid_query 对 '{question1}' 的回答:\n{result1}")

        # 6. 测试 hybrid_query_react
        logger.info("--- 测试 hybrid_query_react ---")
        question2 = "请全面介绍一下龙傲天这个角色，包括他的性格、师承和主要对手。"
        result2 = await hybrid_query_react(vector_query_engine, kg_query_engine, question2, "你是一个角色档案分析师。")
        logger.info(f"hybrid_query_react 对 '{question2}' 的回答:\n{result2}")

        # 7. 测试带有自动元数据过滤的混合查询
        logger.info("--- 测试带有自动元数据过滤的混合查询 ---")
        # 创建一个启用了 auto_retriever 的向量查询引擎
        auto_vector_query_engine = get_vector_query_engine(vector_store, use_auto_retriever=True)
        # 这个问题会触发元数据过滤, 因为它明确要求 "角色简介" (character_profile)
        question3 = "请根据龙傲天的角色简介，介绍一下他的性格。"
        # 使用常规的 hybrid_query, 但传入的是启用了自动过滤的引擎
        result3 = await hybrid_query(auto_vector_query_engine, kg_query_engine, question3)
        logger.info(f"带有自动元数据过滤的 hybrid_query 对 '{question3}' 的回答:\n{result3}")

    try:
        asyncio.run(main())
    finally:
        # 8. 清理
        shutil.rmtree(test_dir)
        logger.info(f"测试目录已删除: {test_dir}")
