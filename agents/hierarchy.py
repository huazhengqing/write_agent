import os
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, LLM_TEMPERATURES
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


def hierarchy(task: Task) -> Task:
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "hierarchy_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context = get_rag().get_context(task)
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["creative"])
    message = llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = content
    updated_task.results["design_reasoning"] = reasoning
    return updated_task

def hierarchy_reflection(task: Task) -> Task:
    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["design_reflection"] = task.results.get("design")
        updated_task.results["design_reflection_reasoning"] = ""
    else:
        SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "hierarchy_reflection_cn", "SYSTEM_PROMPT", "USER_PROMPT")
        context = get_rag().get_context(task)
        context["to_reflection"] = task.results.get("design")
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
        llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["creative"])
        message = llm_completion(llm_params)
        content = message.content
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        updated_task.results["design_reflection"] = content
        updated_task.results["design_reflection_reasoning"] = reasoning
    return updated_task
