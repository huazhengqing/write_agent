from functools import lru_cache
from loguru import logger
from typing import List, Any, Optional, Type, Union
from pydantic import BaseModel
from diskcache import Cache
from utils.search import web_search_tools
from utils.file import cache_dir



cache_agent_dir = cache_dir / "react_agent"
cache_agent_dir.mkdir(parents=True, exist_ok=True)
cache_query = Cache(str(cache_agent_dir), size_limit=int(32 * (1024**2)))



async def call_react_agent(
    system_prompt: Optional[str] = None,
    user_prompt: str = "",
    tools: List[Any] = web_search_tools,
    response_model: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    tool_names_sorted = sorted([tool.metadata.name for tool in tools])
    response_model_name = response_model.__name__ if response_model else "None"

    cache_key_str = f"react_agent:{user_prompt}:{system_prompt}:{','.join(tool_names_sorted)}:{response_model_name}"
    import hashlib
    cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()
    cached_result = cache_query.get(cache_key)
    if cached_result is not None:
        return cached_result

    logger.info(f"system_prompt=\n{system_prompt}")
    logger.info(f"user_prompt=\n{user_prompt}")
    logger.info(f"tools=\n{','.join(tool_names_sorted)}")
    logger.info(f"response_model=\n{response_model_name}")

    from utils.llm_api import llm_temperatures, get_llm_params
    llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.llms.litellm import LiteLLM
    from utils.react_agent_prompt import state_prompt
    agent = ReActAgent(
        tools=tools,
        llm=LiteLLM(**llm_params),
        system_prompt=system_prompt,
        state_prompt=state_prompt,
        verbose=True
    )
    from utils.react_agent_prompt import react_system_header
    agent.update_prompts(
        {"react_header": react_system_header}
    )
    from llama_index.core.workflow import Context
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
    logger.info("执行完成。")

    # if response_text:
    #     return response_text
    raw_output = ""
    if hasattr(final_response, 'response'):
        raw_output = str(final_response.response)
    elif hasattr(final_response, 'content'):
        raw_output = str(final_response.content)
    else:
        raw_output = str(final_response)

    from utils.llm import clean_markdown_fences
    cleaned_output = clean_markdown_fences(raw_output)
    logger.info(f"输出:\n{cleaned_output}")

    final_result = None
    if response_model:
        from utils.llm import txt_to_json
        final_result = await txt_to_json(cleaned_output, response_model)
        logger.info(f"解析后的 JSON 对象: \n{final_result}")
    else:
        from utils.llm import text_validator_default
        text_validator_default(cleaned_output)
        final_result = cleaned_output

    if final_result is not None:
        cache_query.set(cache_key, final_result)

    return final_result
