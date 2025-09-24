import asyncio
from typing import List
from loguru import logger

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool

from utils.llm_api import llm_temperatures, get_llm_params
from utils.llm import get_llm_messages, llm_completion
from utils.vector import index_query
from utils.react_agent import call_react_agent, react_system_prompt
from hybrid_prompts import synthesis_system_prompt, synthesis_user_prompt


async def hybrid_query(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    question: str,
    synthesis_system_prompt: str = synthesis_system_prompt,
    synthesis_user_prompt: str = synthesis_user_prompt,
) -> str:
    if not question or not isinstance(question, str):
        logger.warning("查询问题为空或类型不正确, 无法执行混合查询。")
        return ""

    logger.info(f"开始对问题 '{question}' 执行混合查询...")
    vector_task = index_query(vector_query_engine, question)
    kg_task = index_query(kg_query_engine, question)
    vector_content, kg_content = await asyncio.gather(vector_task, kg_task)

    if not vector_content and not kg_content:
        logger.warning(f"对于问题 '{question}', 向量和知识图谱查询均未返回任何内容。")
        return ""

    formatted_vector_str = vector_content or ""
    formatted_kg_str = kg_content or ""
    context_dict_user = {
        "vector_str": formatted_vector_str, 
        "kg_str": formatted_kg_str, 
        "question": question
    }
    messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)
    final_llm_params = get_llm_params(llm_group='summary', messages=messages, temperature=llm_temperatures["synthesis"])
    final_message = await llm_completion(final_llm_params)
    final_answer = final_message.content.strip()
    return final_answer


async def hybrid_query_batch(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    questions: List[str],
    synthesis_system_prompt: str = synthesis_system_prompt,
    synthesis_user_prompt: str = synthesis_user_prompt,
) -> List[str]:
    if not questions:
        return []

    logger.info(f"开始执行 {len(questions)} 个问题的批量混合查询...")

    sem = asyncio.Semaphore(3)

    async def safe_hybrid_query(question: str) -> str:
        async with sem:
            try:
                return await hybrid_query(
                    vector_query_engine,
                    kg_query_engine,
                    question,
                    synthesis_system_prompt,
                    synthesis_user_prompt,
                )
            except Exception as e:
                logger.error("批量混合查询中, 问题 '{}' 失败: {}", question, e, exc_info=True)
                return ""

    tasks = [safe_hybrid_query(q) for q in questions]
    results = await asyncio.gather(*tasks)
    logger.success(f"批量混合查询完成。")
    return results


###############################################################################


async def hybrid_query_react(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    query_str: str,
    react_system_prompt: str = react_system_prompt,
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
        user_prompt=query_str,
        system_prompt=react_system_prompt,
        tools=[vector_tool, kg_tool]
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)

    logger.success(f"基于ReAct的混合查询完成。")
    return result.strip()
