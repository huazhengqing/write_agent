from loguru import logger
from utils.models import Task
from utils.loader import load_prompts
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.react_agent import call_react_agent
from story.story_rag import get_story_rag


async def search(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.search.search", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    messages = get_llm_messages(system_prompt=system_prompt, user_prompt=user_prompt, context_dict_user=context)
    final_system_prompt = messages[0]["content"]
    final_user_prompt = messages[1]["content"]
    search_result = await call_react_agent(
        system_prompt=final_system_prompt,
        user_prompt=final_user_prompt
    )
    if search_result is None:
        raise ValueError(f"Agent在为任务 '{task.id}' 执行搜索时, 经过多次重试后仍然失败。")
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = search_result
    return updated_task


async def search_aggregate(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.search.search_aggregate", "system_prompt", "user_prompt")
    context = get_story_rag().get_aggregate_search(task)
    messages = get_llm_messages(system_prompt=system_prompt, user_prompt=user_prompt, context_dict_user=context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["reasoning"])
    message = await llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = content
    updated_task.results["search_reasoning"] = reasoning
    return updated_task
