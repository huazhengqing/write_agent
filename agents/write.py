import json
import litellm
from loguru import logger
from ..util.models import Task
from ..util.llm import get_llm_params
from ..memory import get_llm_messages


async def write(task: Task) -> Task:
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
    logger.info(f"{llm_params}")

    response = await litellm.acompletion(**llm_params)
    if not response.choices or not response.choices[0].message:
        raise ValueError("LLM API 调用失败, 没有返回任何 choices 或 message。")
    
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




