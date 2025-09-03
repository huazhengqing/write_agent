import json
from loguru import logger
from ..util.models import Task
from ..util.llm import get_llm_params, llm_acompletion
from ..memory import get_llm_messages


async def write(task: Task) -> Task:
    logger.info(f"{task}")

    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if task.task_type != "write":
        raise ValueError("Task type must be 'write'.")
    if not task.length:
        raise ValueError("Task length must be set.")
    
    if task.category == "story":
        from ..prompts.story.writing_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.category == "report":
        from ..prompts.report.writing_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.category == "book":
        from ..prompts.book.writing_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    else:
        raise ValueError(f"未知的 category: {task.category}")
    
    llm_params = get_llm_params(messages, temperature=0.75)
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




