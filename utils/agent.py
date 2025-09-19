import os
import sys
import hashlib
from loguru import logger
from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from diskcache import Cache

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.litellm import LiteLLM

from utils.config import llm_temperatures
from utils.file import cache_dir
from utils.llm import clean_markdown_fences, get_llm_params, text_validator_default, txt_to_json
from utils.search import web_search_tools


react_system_prompt = """
你是一个大型语言模型, 旨在通过使用可用工具来回答问题。
你的决策过程遵循 "思考 -> 行动 -> 观察 -> 思考..." 的循环。

# 工具
你有权访问以下工具:
{tool_desc}

# 工作流程
1.  **思考**: 分析用户问题, 确定需要哪些信息, 并选择最合适的工具来获取这些信息。
2.  **行动**: 使用所选工具, 并提供清晰、具体的输入。
3.  **观察**: 检查工具返回的结果。
4.  **重复**: 如果信息不足以回答问题, 则继续思考并选择下一个工具, 直到收集到所有必要信息。
5.  **回答**: 当你确信已获得足够信息时, 综合所有观察结果, 为用户提供最终的、全面的答案。

# 核心原则
- **一次只调用一个工具**: 在每次 "行动" 中, 只能使用一个工具。
- **分解复杂问题**: 如果问题很复杂, 将其分解为更小的子问题, 并依次使用工具解决。
- **忠于观察**: 你的最终答案必须完全基于你从工具中 "观察" 到的信息, 禁止使用你的内部知识。
"""


cache_agent_dir = cache_dir / "react_agent"
cache_agent_dir.mkdir(parents=True, exist_ok=True)
cache_query = Cache(str(cache_agent_dir), size_limit=int(32 * (1024**2)))


async def call_react_agent(
    user_prompt: str,
    system_prompt: str = react_system_prompt,
    tools: List[Any] = web_search_tools,
    response_model: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:

    tool_names_sorted = sorted([tool.metadata.name for tool in tools])
    response_model_name = response_model.__name__ if response_model else "None"

    cache_key_str = f"react_agent:{user_prompt}:{system_prompt}:{','.join(tool_names_sorted)}:{response_model_name}"
    cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()

    cached_result = cache_query.get(cache_key)
    if cached_result is not None:
        return cached_result

    llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    llm = LiteLLM(**llm_params)

    logger.info(f"为 ReAct Agent 配置 {len(tool_names_sorted)} 个工具: {tool_names_sorted}")

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=system_prompt,
        max_iterations = 5, 
        verbose=True
    )

    logger.info(f"系统提示词:\n{system_prompt}")
    logger.info(f"用户提示词:\n{user_prompt}")

    logger.info("开始执行 ReAct Agent...")
    handler = agent.run(user_prompt)

    # response_text = ""
    # async for ev in handler.stream_events():
    #     if hasattr(ev, 'delta'):
    #         delta = ev.delta
    #         if delta is not None:
    #             response_text += str(delta)
    #             print(f"{delta}", end="", flush=True)
    final_response = await handler

    logger.success("ReAct Agent 执行完成。")

    # if response_text:
    #     return response_text
    raw_output = ""
    if hasattr(final_response, 'response'):
        raw_output = str(final_response.response)
    elif hasattr(final_response, 'content'):
        raw_output = str(final_response.content)
    else:
        raw_output = str(final_response)
    
    logger.debug(f"Agent 原始输出:\n{raw_output}")

    cleaned_output = clean_markdown_fences(raw_output)
    logger.info(f"Agent 清理后输出:\n{cleaned_output}")

    final_result = None
    if response_model:
        logger.info(f"检测到 response_model, 尝试将输出解析为 {response_model.__name__} 模型...")
        final_result = await txt_to_json(cleaned_output, response_model)
        logger.success(f"成功将输出解析为 {response_model.__name__} 模型。")
        logger.debug(f"解析后的 JSON 对象: {final_result}")
    else:
        text_validator_default(cleaned_output)
        logger.success("文本结果校验通过。")
        final_result = cleaned_output

    if final_result is not None:
        cache_query.set(cache_key, final_result)

    return final_result
