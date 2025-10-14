from functools import lru_cache
from loguru import logger
from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from diskcache import Cache
from utils.search import web_search_tools
from utils.file import cache_dir



async def call_react_agent(
    llm_group: Literal['reasoning', 'fast', 'summary'] = 'reasoning',
    system_prompt: Optional[str] = None,
    user_prompt: str = "",
    tools: List[Any] = web_search_tools,
    response_model: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    tool_names_sorted = sorted([tool.metadata.name for tool in tools])
    response_model_name = response_model.__name__ if response_model else "None"

    logger.info(f"system_prompt=\n{system_prompt}")
    logger.info(f"user_prompt=\n{user_prompt}")
    logger.info(f"tools=\n{','.join(tool_names_sorted)}")
    logger.info(f"response_model=\n{response_model_name}")

    from utils.llm import get_llm_params
    llm_params = get_llm_params(llm_group=llm_group, temperature=0.1)

    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.llms.litellm import LiteLLM
    agent = ReActAgent(
        tools=tools,
        llm=LiteLLM(**llm_params),
        system_prompt=system_prompt,
    )

    from llama_index.core.workflow import Context
    ctx = Context(agent)

    handler = agent.run(
        user_prompt, 
        ctx=ctx,
    )

    response_text = ""
    async for ev in handler.stream_events():
        if hasattr(ev, 'delta'):
            delta = ev.delta
            if delta is not None:
                response_text += str(delta)
                print(f"{delta}", end="", flush=True)
    
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
        if cleaned_output.strip().lower() == "null":
            logger.info("LLM 返回 'null', 表示任务完成或无可用输出。")
            return None

        from utils.llm import txt_to_json
        final_result = await txt_to_json(cleaned_output, response_model)
        logger.info(f"解析后的 JSON 对象: \n{final_result}")
    else:
        from utils.llm import text_validator_default
        text_validator_default(cleaned_output)
        final_result = cleaned_output

    import asyncio
    await asyncio.sleep(0.1)

    return final_result
