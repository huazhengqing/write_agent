import os
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from story.story_rag import get_story_rag
from utils.prompt_loader import load_prompts


async def hierarchy(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(task.category, "hierarchy_cn", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["creative"])
    message = await llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = content
    updated_task.results["design_reasoning"] = reasoning
    return updated_task


async def hierarchy_reflection(task: Task) -> Task:
    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["design_reflection"] = task.results.get("design")
        updated_task.results["design_reflection_reasoning"] = ""
    else:
        system_prompt, user_prompt = load_prompts(task.category, "hierarchy_reflection_cn", "system_prompt", "user_prompt")
        context = await get_story_rag().get_context(task)
        context["to_reflection"] = task.results.get("design")
        messages = get_llm_messages(system_prompt, user_prompt, None, context)
        llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["creative"])
        message = await llm_completion(llm_params)
        content = message.content
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        updated_task.results["design_reflection"] = content
        updated_task.results["design_reflection_reasoning"] = reasoning
    return updated_task
