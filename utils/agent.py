import os
import sys
from loguru import logger
from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.llm import clean_markdown_fences, llm_temperatures, get_llm_params, text_validator_default, txt_to_json
from utils.search import web_search_tools


async def call_react_agent(
    system_prompt: Optional[str],
    user_prompt: str,
    tools: List[Any] = web_search_tools,
    llm_group: Literal['reasoning', 'fast'] = 'reasoning',
    temperature: float = llm_temperatures["reasoning"],
    response_model: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    
    llm_params = get_llm_params(llm_group=llm_group, temperature=temperature)
    llm = LiteLLM(**llm_params)

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=system_prompt,
        max_iterations = 5, 
        verbose=True
    )

    logger.info(f"系统提示词:\n{system_prompt}")
    logger.info(f"用户提示词:\n{user_prompt}")

    handler = agent.run(user_prompt)
    # response_text = ""
    # async for ev in handler.stream_events():
    #     if hasattr(ev, 'delta'):
    #         delta = ev.delta
    #         if delta is not None:
    #             response_text += str(delta)
    #             print(f"{delta}", end="", flush=True)
    final_response = await handler
    # if response_text:
    #     return response_text
    raw_output = ""
    if hasattr(final_response, 'response'):
        raw_output = str(final_response.response)
    elif hasattr(final_response, 'content'):
        raw_output = str(final_response.content)
    else:
        raw_output = str(final_response)
    cleaned_output = clean_markdown_fences(raw_output)
    
    logger.info(f"llm 返回\n{cleaned_output}")

    if response_model:
        return await txt_to_json(cleaned_output, response_model)
    else:
        text_validator_default(cleaned_output)
        return cleaned_output


###############################################################################


