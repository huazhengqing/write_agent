import asyncio
from typing import List
from loguru import logger

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool

from utils.config import llm_temperatures, get_llm_params
from utils.llm import get_llm_messages, llm_completion
from utils.vector import index_query
from utils.agent import call_react_agent, react_system_prompt


synthesis_system_prompt = """
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


synthesis_user_prompt = """
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
    synthesis_system_prompt: str = synthesis_system_prompt,
    synthesis_user_prompt: str = synthesis_user_prompt,
) -> str:
    if not question or not isinstance(question, str):
        logger.warning("查询问题为空或类型不正确, 无法执行混合查询。")
        return ""

    logger.info(f"开始对问题 '{question}' 执行混合查询...")

    logger.debug("正在并行执行向量查询和知识图谱查询...")
    vector_task = index_query(vector_query_engine, question)
    kg_task = index_query(kg_query_engine, question)
    vector_content, kg_content = await asyncio.gather(vector_task, kg_task)

    if not vector_content and not kg_content:
        logger.warning(f"对于问题 '{question}', 向量和知识图谱查询均未返回任何内容。")
        return ""

    formatted_vector_str = vector_content or ""
    formatted_kg_str = kg_content or ""

    logger.debug(f"向量查询结果 (片段数: {1 if vector_content else 0}):\n{formatted_vector_str[:500]}...")
    logger.debug(f"知识图谱查询结果 (片段数: {1 if kg_content else 0}):\n{formatted_kg_str[:500]}...")

    context_dict_user = {
        "vector_str": formatted_vector_str, 
        "kg_str": formatted_kg_str, 
        "question": question
    }
    messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)
    final_llm_params = get_llm_params(llm_group='summary', messages=messages, temperature=llm_temperatures["synthesis"])
    final_message = await llm_completion(final_llm_params)

    final_answer = final_message.content.strip()
    logger.success(f"混合查询完成，生成回答长度: {len(final_message.content)}")
    logger.debug(f"最终回答:\n{final_message.content}")

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
                logger.error("批量混合查询中，问题 '{}' 失败: {}", question, e, exc_info=True)
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
