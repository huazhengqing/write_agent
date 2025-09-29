import os
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.story_rag import get_story_rag
from utils.react_agent import call_react_agent


async def design_guideline(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.design.design_1_guideline", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["creative"])
    message = await llm_completion(llm_params)
    design_guideline_content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design_guideline"] = design_guideline_content
    updated_task.results["design_guideline_reasoning"] = reasoning
    return updated_task


async def design_execute(task: Task) -> Task:
    design_guideline = task.results.get("design_guideline")
    if not design_guideline:
        raise ValueError(f"任务 {task.id} 缺少 'design_guideline'，无法执行设计。")
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.design.design_2_execute", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    context["design_guideline"] = design_guideline
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["creative"])
    message = await llm_completion(llm_params)
    final_design = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = final_design
    updated_task.results["design_reasoning"] = reasoning
    return updated_task


###############################################################################


async def design_aggregate(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.design.design_aggregate", "system_prompt", "user_prompt")
    context = get_story_rag().get_aggregate_design(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["creative"])
    message = await llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = content
    updated_task.results["design_reasoning"] = reasoning
    return updated_task
