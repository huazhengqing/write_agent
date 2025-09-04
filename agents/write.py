import json
from loguru import logger
from util.models import Task
from util.llm import get_llm_params, llm_acompletion
from memory import get_llm_messages
from util.prompt_loader import load_prompts


async def write(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    if not task.id or not task.goal:
        raise ValueError(f"{task}")
    if task.task_type != "write":
        raise ValueError(f"{task}")
    if not task.length:
        raise ValueError(f"{task}")
    VALID_CATEGORIES = {"story", "report", "book"}
    if task.category not in VALID_CATEGORIES:
        raise ValueError(f"{task}")

    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "writing_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)

    llm_params = get_llm_params(messages, temperature=0.75)
    message = await llm_acompletion(llm_params)
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
    
    updated_task = task.model_copy(deep=True)
    updated_task.results = {
        "result": content,
        "reasoning": reason,
    }

    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task