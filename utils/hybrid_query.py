import os
import sys
import asyncio
import re
from typing import List, Literal, Optional, Tuple, Union
from loguru import logger
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import call_react_agent, get_llm_messages, get_llm_params, llm_completion, llm_temperatures  # noqa
from utils.vector import index_query


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
        
    system_prompt = synthesis_system_prompt or synthesis_system_prompt_default
    user_prompt = synthesis_user_prompt or synthesis_user_prompt_default

    vector_task = index_query(vector_query_engine, [question])
    kg_task = index_query(kg_query_engine, [question])
    vector_contents, kg_contents = await asyncio.gather(vector_task, kg_task)

    if not vector_contents and not kg_contents:
        logger.warning(f"对于问题 '{question}', 向量和知识图谱查询均未返回任何内容。")
        return ""

    formatted_vector_str = "\n\n---\n\n".join(vector_contents)
    formatted_kg_str = "\n\n---\n\n".join(kg_contents)

    context_dict_user = {"vector_str": formatted_vector_str, "kg_str": formatted_kg_str, "question": question}
    messages = get_llm_messages(system_prompt, user_prompt, None, context_dict_user)
    final_llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=llm_temperatures["synthesis"])
    final_message = await llm_completion(final_llm_params)
    return final_message.content


async def hybrid_query_batch(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    questions: List[str],
    synthesis_system_prompt: Optional[str] = None,
    synthesis_user_prompt: Optional[str] = None,
) -> List[str]:
    if not questions:
        return []

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
    return results



async def hybrid_query_react(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    query_str: str,
    agent_system_prompt: Optional[str] = None,
) -> str:
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
        llm_type="reasoning",
        temperature=llm_temperatures["reasoning"]
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)
    return result
