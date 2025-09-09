from loguru import logger
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


async def design(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "design_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context = await get_rag().get_context_base(task)
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages, temperature=0.75)
    message = await llm_acompletion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = content
    updated_task.results["design_reasoning"] = reasoning

    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task