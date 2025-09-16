import os
from utils.sqlite import get_db
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, LLM_TEMPERATURES, llm_completion
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


def review_design(task: Task) -> Task:
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "review_design_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context = get_rag().get_context(task)
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
    message = llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task.results["review_design"] = message.content
    updated_task.results["review_design_reasoning"] = reasoning
    return updated_task

def review_write(task: Task) -> Task:
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "review_write_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context = get_rag().get_context(task)
    context.update({
        "text": get_db(task.run_id, task.category).get_write_text(task)
    })
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
    message = llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task.results["review_write"] = message.content
    updated_task.results["review_write_reasoning"] = reasoning
    return updated_task
