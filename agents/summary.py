from loguru import logger
import collections
import litellm
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


async def summary(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "summary_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context_dict_user = {
        "task": task.model_dump_json(
            indent=2,
            exclude_none=True,
            include={'goal'}
        ),
        "text": task.results.get("write_reflection")
    }
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context_dict_user)
    llm_params = get_llm_params(messages, temperature=LLM_TEMPERATURES["summarization"])
    message = await llm_acompletion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = content
    updated_task.results["summary_reasoning"] = reasoning

    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task
