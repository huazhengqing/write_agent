from collections import defaultdict
from loguru import logger
from utils.models import Task
from utils.prompt_loader import load_prompts
from utils.llm import get_llm_messages, get_llm_params, llm_completion, LLM_TEMPERATURES, call_ReActAgent
from story.story_rag import get_story_rag


def search(task: Task) -> Task:
    system_prompt = load_prompts(task.category, "search_cn", "system_prompt")[0]
    context = get_story_rag().get_context_base(task)
    context["goal"] = task.goal
    system_prompt = system_prompt.format_map(defaultdict(str, context))
    user_prompt = f"请根据系统提示中的目标 '{task.goal}' 开始你的研究。"
    logger.info(f"开始搜索任务: Agent 开始为目标 {task.id} - '{task.goal}' 执行研究...")
    search_result = call_ReActAgent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=LLM_TEMPERATURES["reasoning"]
    )
    if search_result is None:
        raise ValueError(f"Agent在为任务 '{task.id}' 执行搜索时，经过多次重试后仍然失败。")
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = search_result
    logger.info(f"搜索任务 '{task.id}' 完成。\n{search_result}")
    return updated_task


def search_aggregate(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(task.category, "search_aggregate_cn", "system_prompt", "user_prompt")
    context = get_story_rag().get_aggregate_search(task)
    messages = get_llm_messages(system_prompt=system_prompt, user_prompt=user_prompt, context_dict_user=context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
    message = llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = content
    updated_task.results["search_reasoning"] = reasoning
    return updated_task

