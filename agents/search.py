import os
from collections import defaultdict
from loguru import logger
from llama_index.core.agent import ReActAgent
from llama_index.llms.litellm import LiteLLM
from llama_index.tools.tavily_research import TavilyToolSpec
from utils.models import Task
from agents.tools.web_scraper import get_web_scraper_tool
from utils.prompt_loader import load_prompts
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from utils.rag import get_rag


async def search(task: Task) -> Task:
    logger.info(f"开始搜索任务: {task.id} - '{task.goal}'")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY 环境变量未设置，搜索任务无法执行。")
        raise ValueError("TAVILY_API_KEY is not set.")
    SYSTEM_PROMPT = load_prompts(task.category, "search_cn", "SYSTEM_PROMPT")[0]
    context = await get_rag().get_context_base(task)
    context["goal"] = task.goal
    system_prompt = SYSTEM_PROMPT.format_map(defaultdict(str, context))
    tavily_tools = TavilyToolSpec(api_key=tavily_api_key).to_tool_list()
    scraper_tool = get_web_scraper_tool()
    all_tools = tavily_tools + [scraper_tool]
    llm_params = get_llm_params(llm='reasoning', temperature=LLM_TEMPERATURES["reasoning"])
    llm = LiteLLM(**llm_params)
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        system_prompt=system_prompt,
        verbose=True
    )
    logger.info(f"ReActAgent 开始为目标 '{task.goal}' 执行研究...")
    try:
        response = await agent.achat("请开始研究。")
    except Exception as e:
        logger.error(f"ReActAgent 在为任务 '{task.id}' 执行搜索时发生错误: {e}")
        raise
    search_result = str(response)
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = search_result
    logger.info(f"搜索任务 '{task.id}' 完成。\n{search_result}")
    return updated_task

async def search_aggregate(task: Task) -> Task:
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "search_aggregate_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context = await get_rag().get_aggregate_search(task)
    messages = get_llm_messages(SYSTEM_PROMPT=SYSTEM_PROMPT, USER_PROMPT=USER_PROMPT, context_dict_user=context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
    message = await llm_acompletion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = content
    updated_task.results["search_reasoning"] = reasoning
    return updated_task