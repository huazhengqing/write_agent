import os
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, LLM_TEMPERATURES
from story.story_rag import get_story_rag
from utils.prompt_loader import load_prompts


def write_before_reflection(task: Task) -> Task:
    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["design_reflection"] = ""
        updated_task.results["design_reflection_reasoning"] = ""
    else:
        system_prompt, user_prompt = load_prompts(task.category, "design_batch_reflection_cn", "system_prompt", "user_prompt")
        context = get_story_rag().get_context(task)
        messages = get_llm_messages(system_prompt, user_prompt, None, context)
        llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["creative"])
        message = llm_completion(llm_params)
        content = message.content
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        updated_task.results["design_reflection"] = content
        updated_task.results["design_reflection_reasoning"] = reasoning
    return updated_task


def write(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(task.category, "write_cn", "system_prompt", "user_prompt")
    context = get_story_rag().get_context(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["creative"])
    message = llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["write"] = content
    updated_task.results["write_reasoning"] = reasoning
    return updated_task


def write_reflection(task: Task) -> Task:
    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["write_reflection"] = task.results.get("write")
        updated_task.results["write_reflection_reasoning"] = ""
    else:
        system_prompt, user_prompt = load_prompts(task.category, "write_reflection_cn", "system_prompt", "user_prompt")
        context = get_story_rag().get_context(task)
        context["to_reflection"] = task.results.get("write")
        messages = get_llm_messages(system_prompt, user_prompt, None, context)
        llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["creative"])
        message = llm_completion(llm_params)
        content = message.content
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        updated_task.results["write_reflection"] = content
        updated_task.results["write_reflection_reasoning"] = reasoning
    return updated_task

