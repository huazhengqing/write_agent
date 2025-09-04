from loguru import logger
from util.models import Task
from util.llm import get_llm_params, llm_acompletion
from memory import get_llm_messages
from util.prompt_loader import load_prompts


async def design(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if task.task_type != "design":
        raise ValueError("Task type must be 'design'.")
    
    VALID_CATEGORIES = {"story", "report", "book"}
    if task.category not in VALID_CATEGORIES:
        raise ValueError(f"未知的 category: {task.category}")

    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "design_cn", "SYSTEM_PROMPT", "USER_PROMPT")
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