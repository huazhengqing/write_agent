import os
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from story.story_rag import get_story_rag
from utils.prompt_loader import load_prompts


async def design(task: Task, category: str) -> Task:
    prompt_file_map = {
        "market": "design_market_cn",
        "title": "design_title_cn",
        "style": "design_style_cn",
        "character": "design_cn",
        "system": "design_cn",
        "concept": "design_cn",
        "worldview": "design_cn",
        "plot": "design_cn",
        "scene_atmosphere": "design_scene_atmosphere_cn",
        "faction_culture": "design_faction_culture_cn",
        "power_system": "design_power_system_rules_cn",
        "narrative_pacing": "design_narrative_pacing_cn",
        "thematic_imagery": "design_thematic_imagery_cn",
        "hierarchy": "hierarchy_cn",
        "trend_integration": "design_trend_integration_cn",
        "general": "design_cn",
    }
    prompt_file = prompt_file_map.get(category, "design_cn")
    temperature = llm_temperatures["creative"]
    system_prompt, user_prompt = load_prompts(task.category, prompt_file, "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=temperature)
    message = await llm_completion(llm_params)
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = message.content
    updated_task.results["design_reasoning"] = reasoning
    return updated_task


async def design_reflection(task: Task) -> Task:
    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["design_reflection"] = task.results.get("design")
        updated_task.results["design_reflection_reasoning"] = ""
    else:
        system_prompt, user_prompt = load_prompts(task.category, "design_reflection_cn", "system_prompt", "user_prompt")
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


async def design_aggregate(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(task.category, "design_aggregate_cn", "system_prompt", "user_prompt")
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
