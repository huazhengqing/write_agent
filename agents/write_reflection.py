import collections
import os
from loguru import logger
from util.models import Task
from util.llm import get_llm_messages, get_llm_params, llm_acompletion
from util.rag import get_rag
from util.prompt_loader import load_prompts


async def write_reflection(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["result_reflection"] = task.results.get("result")
        updated_task.results["reasoning_reflection"] = ""
    else:
        SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "write_reflection_cn", "SYSTEM_PROMPT", "USER_PROMPT")

        context = await get_rag().get_context_base(task)
        context["to_reflection"] = task.results.get("result")
        
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)

        llm_params = get_llm_params(messages, temperature=0.75)
        message = await llm_acompletion(llm_params)
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        content = message.content
        
        updated_task.results["result_reflection"] = content
        updated_task.results["reasoning_reflection"] = reasoning

    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task