import litellm
from loguru import logger
from typing import List
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper
from ..util.models import Task
from ..util.llm import get_llm_params
from ..memory import get_llm_messages


async def search_aggregate(task: Task) -> Task:
    from ..prompts.story.search_aggregate_cn import SYSTEM_PROMPT, USER_PROMPT
    messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    
    llm_params = get_llm_params(messages, temperature=0.1)
    logger.info(f"{llm_params}")

    response = await litellm.acompletion(**llm_params)
    if not response.choices or not response.choices[0].message:
        raise ValueError("LLM API 调用失败, 没有返回任何 choices。")
    
    message = response.choices[0].message
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
    if not content:
        raise ValueError("LLM API 调用失败, 没有返回任何 content。")
    
    updated_task = task.model_copy(deep=True)
    updated_task.results = {
        "result": content,
        "reasoning": reason,
    }

    logger.info(f"{updated_task}")
    return updated_task


