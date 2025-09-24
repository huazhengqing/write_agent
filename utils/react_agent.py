import os
import sys
import hashlib
from loguru import logger
from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from diskcache import Cache

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.litellm import LiteLLM
from llama_index.core.workflow import Context

from utils.llm_api import llm_temperatures
from utils.file import cache_dir
from utils.llm import clean_markdown_fences, get_llm_params, text_validator_default, txt_to_json
from utils.search import web_search_tools
from utils.react_agent_prompt import react_system_header, react_system_prompt, state_prompt


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
        state_prompt=state_prompt,
        verbose=True
    )
    agent.update_prompts({"react_header": react_system_header})

    logger.info(f"提示词:\n{agent.get_prompts()}")
    logger.info(f"系统提示词:\n{system_prompt}")
    logger.info(f"用户提示词:\n{user_prompt}")

    logger.info("开始执行 ReAct Agent...")
    ctx = Context(agent)
    handler = agent.run(
        user_prompt, 
        ctx=ctx, 
        max_iterations=5
    )

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
