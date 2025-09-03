from loguru import logger
from typing import List
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper
from ..util.models import Task
from ..util.llm import get_llm_params, llm_acompletion
from ..memory import get_llm_messages


async def search_aggregate(task: Task) -> Task:
    logger.info(f"{task}")

    if task.category == "story":
        from ..prompts.story.search_aggregate_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.category == "report":
        from ..prompts.report.search_aggregate_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.category == "book":
        from ..prompts.book.search_aggregate_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    else:
        raise ValueError(f"未知的 category: {task.category}")

    llm_params = get_llm_params(messages, temperature=0.1)
    message = await llm_acompletion(llm_params)
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
    
    updated_task = task.model_copy(deep=True)
    updated_task.results = {
        "result": content,
        "reasoning": reason,
    }

    logger.info(f"{updated_task}")
    return updated_task


