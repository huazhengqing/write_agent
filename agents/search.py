import os
import httpx
from collections import defaultdict
from loguru import logger
from llama_index.core.agent import ReActAgent
from llama_index.llms.litellm import LiteLLM
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.tools import FunctionTool
from utils.models import Task
from agents.tools.web_scraper import get_web_scraper_tool
from utils.prompt_loader import load_prompts
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from utils.rag import get_rag


async def scrape_and_extract(url: str, extraction_prompt: str) -> str:
    logger.info(f"开始使用 Jina Reader 抓取 URL: {url}，提取任务: '{extraction_prompt}'")
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            "Accept": "text/markdown"
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(jina_url, follow_redirects=True, timeout=60.0, headers=headers)
            response.raise_for_status()
        page_text = response.text
        logger.info(f"Jina Reader 抓取成功，内容长度: {len(page_text)} chars。")
        if not page_text.strip():
            logger.warning(f"Jina Reader for URL '{url}' 返回了空内容。")
            return f"错误: 抓取服务为URL '{url}' 返回了空内容。"
        max_length = 16000
        if len(page_text) > max_length:
            logger.warning(f"抓取的内容过长 ({len(page_text)} > {max_length})，将进行截断。")
            page_text = page_text[:max_length]
        logger.info(f"URL {url} 抓取完成，直接返回页面内容。")
        return page_text
    except httpx.HTTPStatusError as e:
        error_msg = f"错误: 通过 Jina Reader 抓取 URL '{url}' 失败。状态码: {e.response.status_code}, 响应: {e.response.text}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"使用 Jina Reader 抓取或提取时发生意外错误: {e}"
        logger.error(error_msg)
        return error_msg

def get_web_scraper_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=scrape_and_extract,
        name="web_scraper",
        description=(
            "使用第三方服务从指定的URL抓取网页的干净内容(Markdown格式)。"
            "当你需要从某个具体的网页(而不是通过泛泛的搜索)获取数据时使用此工具。"
            "该工具能自动处理静态和动态(JavaScript渲染)的页面。"
            "此工具接收 `url` 和 `extraction_prompt` 两个参数，但 `extraction_prompt` 会被忽略。工具会直接返回整个页面的Markdown内容。"
            "例如: `web_scraper(url='http://example.com/bestsellers', extraction_prompt='提取所有书名和它们的作者。')`"
        )
    )

async def search(task: Task) -> Task:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
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
    logger.info(f"开始搜索任务: ReActAgent 开始为目标 {task.id} - '{task.goal}' 执行研究...")
    try:
        response = await agent.achat(f"请根据系统提示中的目标 '{task.goal}' 开始你的研究。")
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