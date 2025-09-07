import os
import json
import httpx
import asyncio
import collections
import functools
from loguru import logger
from diskcache import Cache
from bs4 import BeautifulSoup
from trafilatura import extract
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Optional, Any
from sentence_transformers import SentenceTransformer, util
from langchain_community.utilities import SearxSearchWrapper
from langdetect import detect, LangDetectException
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from util.models import Task
from util.prompt_loader import load_prompts
from util.llm import get_llm_messages, get_llm_params, llm_acompletion
from util.rag import get_rag


###############################################################################


# 每个(子)任务的最大搜索-抓取-规划循环次数, 用于防止无限循环。
MAX_SEARCH_TURNS = 3
# 并发抓取网页的最大数量, 以控制资源使用和避免对目标服务器造成过大压力。
MAX_SCRAPE_CONCURRENCY = 5
# 单个网络请求（如抓取网页）的超时时间（秒）。
REQUEST_TIMEOUT = 30
# httpx 抓取时认为内容有效的最小长度, 用于判断是否需要降级到 Playwright
MIN_CONTENT_LENGTH_FOR_HTTPX = 100


###############################################################################


class Plan(BaseModel):
    thought: str = Field(description="你的思考过程, 分析已有信息和下一步计划。")
    queries: List[str] = Field(description="根据思考生成的搜索查询列表。如果认为信息足够, 则返回空列表。")

class ProcessedContent(BaseModel):
    url: str = Field(description="内容的原始URL。")
    relevance_score: float = Field(description="内容与研究焦点的相关性得分（0.0到1.0）, 分数越高越相关。")
    summary: str = Field(description="提取或生成的核心内容摘要, 应去除噪音并突出关键信息。")
    is_relevant: bool = Field(description="内容是否与研究焦点直接相关。")

class ProcessedResults(BaseModel):
    processed_contents: List[ProcessedContent] = Field(description="处理和评估后的内容列表。")


###############################################################################


class SearchAgentState(TypedDict):
    # --- 核心任务信息 ---
    task: Task                          # 传入的原始任务对象, 用于获取上下文和存储记忆。    
    current_focus: str                  # 当前研究循环的具体焦点, 在复杂任务中可能只是原始任务的一部分。
    final_report: Optional[str]         # 最终生成的研究报告, 在 synthesize_node 中填充。
    embedding_model: Any                # 根据任务语言选择的句子嵌入模型。
    # --- 研究循环状态 (简单路径和复杂路径的子任务循环共用) ---
    plan: Plan                          # 当前的行动计划（思考+查询）, 由 planner_node 生成。
    urls_to_scrape: Optional[List[str]] # 从 search_node 返回的待抓取URL列表。
    latest_scraped_content: List[dict]  # 从最新一次 scrape_node 运行中抓取的原始内容。
    latest_processed_content: List[dict] # 从最新一次 information_processor_node 运行中处理过的内容。
    processed_content_accumulator: List[dict] # 累积所有轮次中经过处理和筛选的相关信息。
    previous_rolling_summary: Optional[str] # 上一轮的滚动总结, 用于检测研究是否停滞。
    rolling_summary: Optional[str]      # 滚动总结, 由 rolling_summary_node 生成, 持续更新的知识库, 用于指导下一轮规划。
    turn_count: int                     # 当前任务/子任务的研究循环轮次计数, 用于防止无限循环。
    reasoning_history: List[str]        # 记录“思考-行动-观察”链条, 用于构建上下文和最终的推理过程展示。


###############################################################################



SEARCH_CACHE = Cache(os.path.join(".cache", 'search_cache'), size_limit=int(128 * 1024 * 1024))
SCRAPE_CACHE = Cache(os.path.join(".cache", 'scrape_cache'), size_limit=int(128 * 1024 * 1024))


search_tool = SearxSearchWrapper(searx_host=os.environ.get("SearXNG", "http://127.0.0.1:8080"))


def get_embedding_model(language: str) -> SentenceTransformer:
    if language.startswith('zh'):
        model_name = 'BAAI/bge-small-zh'
        model_local_dir = 'bge-small-zh'
    else:
        model_name = 'all-MiniLM-L6-v2'
        model_local_dir = 'all-MiniLM-L6-v2'

    model_path = f'../models/{model_local_dir}'

    if not os.path.isdir(model_path):
        logger.warning(f"本地模型路径 '{model_path}' 不存在, 将尝试从网络下载 '{model_name}'。")
        logger.warning("请考虑运行 ./start.sh 脚本中的 hf download 命令来本地化模型, 以提高加载速度和稳定性。")
        model_path = model_name
    
    logger.info(f"正在为语言 '{language}' 加载嵌入模型: {model_path}")
    return SentenceTransformer(model_path)


def async_retry(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"函数 {func.__name__} 在 {retries} 次重试后失败。")
                        raise e
                    
                    sleep = backoff_in_seconds * (2 ** x)
                    await asyncio.sleep(sleep)
                    x += 1
        return wrapper
    return decorator


async def scrape_webpages(urls: List[str]) -> List[dict]:
    """
    使用混合策略并发抓取网页内容。
    1.  首先尝试使用 `httpx` 进行快速、轻量级的抓取。
    2.  如果 `httpx` 抓取失败（如网络错误）或提取的内容过短（通常意味着页面需要JS渲染）, 
        则自动降级到使用 `Playwright` 进行深度抓取, 它可以执行JavaScript。
    这种策略旨在兼顾速度和抓取成功率。
    """
    if not urls:
        return []

    async def scrape_single_hybrid(client, browser, url: str) -> Optional[dict]:
        """对单个URL执行混合抓取策略。"""
        # 1. 检查缓存
        if url in SCRAPE_CACHE:
            logger.info(f"缓存命中 (抓取): {url}")
            return SCRAPE_CACHE[url]

        # 2. 优先使用 httpx 尝试快速抓取
        httpx_content = None
        final_url = url
        try:
            logger.info(f"使用 httpx 快速抓取: {url}")
            response = await client.get(url)
            response.raise_for_status()
            httpx_content = extract(response.text)
            final_url = str(response.url)
        except httpx.RequestError as e:
            logger.warning(f"⚠️ httpx 请求失败 ({e}), 将降级到 Playwright: {url}")
        except Exception as e:
            logger.warning(f"⚠️ httpx 处理时发生未知错误 ({e}), 将降级到 Playwright: {url}")

        # 如果 httpx 成功且内容足够, 直接返回
        if httpx_content and len(httpx_content) >= MIN_CONTENT_LENGTH_FOR_HTTPX:
            logger.info(f"✅ httpx 抓取成功: {url}")
            result = {"url": final_url, "content": httpx_content}
            SCRAPE_CACHE[url] = result
            return result
        elif httpx_content:
            logger.info(f"⚠️ httpx 内容过短, 降级到 Playwright: {url}")

        # 3. 如果 httpx 失败或内容不足, 降级到 Playwright
        page = None
        try:
            logger.info(f"使用 Playwright 深度抓取: {url}")
            page = await browser.new_page()
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in {"image", "stylesheet", "font", "media"} else route.continue_())
            await page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
            await asyncio.sleep(3)
            
            html_content = await page.content()
            text_content = extract(html_content)

            if text_content and len(text_content) >= MIN_CONTENT_LENGTH_FOR_HTTPX:
                logger.info(f"✅ Playwright 抓取成功: {url}")
                result = {"url": page.url, "content": text_content}
                SCRAPE_CACHE[url] = result
                return result
            else:
                logger.warning(f"Playwright 抓取后内容仍过短或失败: {url}")
                return None
        except (PlaywrightTimeoutError, Exception) as e:
            logger.warning(f"Playwright 处理 URL 时发生错误: {url}, 错误: {e}")
            return None
        finally:
            if page:
                await page.close()

    # --- 并发执行与上下文管理 ---
    semaphore = asyncio.Semaphore(MAX_SCRAPE_CONCURRENCY)

    async def scrape_with_semaphore(client, browser, url: str) -> Optional[dict]:
        async with semaphore:
            return await scrape_single_hybrid(client, browser, url)

    # 同时管理 httpx 和 playwright 的上下文
    # 安全地从环境变量中获取 SSL 验证设置, 默认为 True
    verify_ssl = os.environ.get("verify_ssl", "false").lower() == "true"
    async with async_playwright() as p, httpx.AsyncClient(http2=True, verify=verify_ssl, follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
        browser = await p.chromium.launch(headless=True)
        try:
            tasks = [scrape_with_semaphore(client, browser, url) for url in urls]
            results = await asyncio.gather(tasks)
            return [res for res in results if res is not None]
        finally:
            await browser.close()


###############################################################################


PROMPT_STAGNATION_DETECTION = """
# 任务: 评估信息增益
对比以下两个总结, 判断“当前总结”是否比“上一轮总结”提供了显著的、有价值的新信息。

## 上一轮总结:
{prev_summary}

## 当前总结:
{current_summary}

如果“当前总结”没有提供显著新信息 (例如, 只是重述、细化或补充了无关紧要的细节), 则判定为停滞。

停滞了吗? (只回答 true 或 false)
"""


PROMPT_SELF_CORRECTION = """
# 任务: 修正JSON输出
上次的输出因格式错误导致解析失败。

# 错误信息
{error}

# 格式错误的原始输出
{raw_output}

# 要求 
严格根据 Pydantic 模型的要求, 修正并仅返回完整的、有效的 JSON 对象。禁止任何额外解释。
"""


###############################################################################


# 图节点

async def get_structured_output_with_retry(messages: List[dict], response_model: BaseModel, retries: int = 1):
    response = None # 初始化 response 以避免 UnboundLocalError
    for i in range(retries + 1):
        try:
            llm_params = get_llm_params(messages, response_model=response_model)

            message = await llm_acompletion(llm_params)

            # 增加对LLM响应格式的健壮性检查, 防止因缺少 tool_calls 导致崩溃
            if not message.tool_calls or len(message.tool_calls) == 0:
                raise ValueError("LLM响应中缺少预期的工具调用 (tool_calls)。")

            tool_call = message.tool_calls[0]
            # litellm v1.34.0+ 会自动解析, 但为增加兼容性和鲁棒性, 添加手动解析作为后备
            if hasattr(tool_call.function, 'parsed_arguments') and tool_call.function.parsed_arguments:
                parsed_args = tool_call.function.parsed_arguments
            else:
                parsed_args = json.loads(tool_call.function.arguments)
            return response_model(**parsed_args)
        except Exception as e:
            logger.warning(f"调用LLM或解析输出失败 (尝试 {i+1}/{retries+1}): {e}")
            if i == retries:
                logger.error("达到最大重试次数, 解析失败。")
                return None
            
            # 仅当收到响应但解析失败时, 才尝试自我纠错
            if response:
                try:
                    # 尝试获取原始输出以进行纠错, 同时进行安全检查
                    message = response.choices[0].message
                    raw_output = ""
                    if message.tool_calls and len(message.tool_calls) > 0:
                        raw_output = message.tool_calls[0].function.arguments
                    elif message.content: # 如果LLM未按工具调用格式返回, 尝试从内容中获取
                        raw_output = message.content

                    correction_prompt = PROMPT_SELF_CORRECTION.format(error=str(e), raw_output=raw_output)
                    messages = messages + [response.choices[0].message, {"role": "user", "content": correction_prompt}]
                    logger.info("...正在尝试自我纠错...")
                except Exception as format_e:
                    logger.error(f"构建纠错提示时发生错误: {format_e}. 将进行常规重试。")
            else:
                logger.warning("未收到LLM响应, 将进行常规重试。")

async def planner_node(state: SearchAgentState) -> dict:
    """
    规划节点, 是研究循环的核心。
    它聚合了最关键的上下文信息（任务描述、滚动总结、最近的历史记录等）, 
    然后调用 LLM 遵循“分析-策略-行动”的框架, 生成下一步的思考和搜索查询。
    如果 LLM 认为当前焦点的信息已足够, 它将返回一个空的查询列表, 从而触发研究循环的终止。
    """
    turn = state['turn_count'] + 1
    task = state['task']

    # 1. 准备 Prompt 所需的核心上下文
    context_dict = {
        'current_focus': state['current_focus'], 
        'rolling_summary': state.get('rolling_summary') or "无, 这是第一次研究, 请开始探索。",
        # 只传递最近2轮的思考历史, 以精简上下文, 避免LLM在长对话中迷失
        'reasoning_history': "\n\n".join(state['reasoning_history'][-2:]) or "无",
    }

    # 2. 确定当前的研究焦点
    logger.info(f"▶️ 1. 进入规划节点 (第 {turn} 轮)...")
  
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "search_cn", "SYSTEM_PROMPT", "USER_PROMPT")

    context = await get_rag().get_context_base(task)

    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    
    plan = await get_structured_output_with_retry(messages, Plan)
    
    # 如果解析失败, 创建一个空的 Plan 对象以避免下游节点出错
    if not plan:
        plan = Plan(thought="规划失败, 尝试终止当前研究。", queries=[])

    logger.info(f"🤔 思考: {plan.thought}")
    logger.info(f"🔍 生成查询: {plan.queries}")

    # 5. 更新图状态, 包括新的计划和增加的轮次计数
    return {"plan": plan, "turn_count": turn}

@async_retry(retries=2, backoff_in_seconds=2)
async def _search_with_retry(query: str):
    return search_tool.results(query)

async def search_node(state: SearchAgentState) -> dict:
    queries = state['plan'].queries
    logger.info(f"▶️ 2. 进入搜索节点, 执行查询: {queries}...")
    
    try:
        # 1. 为每个查询创建一个异步搜索任务, 并处理缓存
        search_tasks = []
        for query in queries:
            if query in SEARCH_CACHE:
                logger.info(f"缓存命中 (搜索): {query}")
                future = asyncio.Future()
                future.set_result(SEARCH_CACHE[query])
                search_tasks.append(future)
            else:
                search_tasks.append(_search_with_retry(query))

        # 2. 使用 asyncio.gather 并行执行所有搜索
        # 使用 return_exceptions=True, 这样即使个别搜索失败, 也不会中断整个批次。
        results_or_exceptions = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 3. 提取所有URL并格式化结果用于日志和历史记录
        all_urls = []
        all_search_results_str_parts = []
        for query, result_or_exc in zip(queries, results_or_exceptions):
            if isinstance(result_or_exc, Exception):
                logger.error(f"搜索查询 '{query}' 失败: {result_or_exc}")
                all_search_results_str_parts.append(f"查询 '{query}' 的结果: 失败 ({result_or_exc})")
                continue
            
            results = result_or_exc
            # 更新缓存
            SEARCH_CACHE[query] = results
            
            query_urls = [res["link"] for res in results if res.get("link")]
            all_urls.extend(query_urls)
            
            # 格式化字符串结果用于历史记录
            formatted_results = "\n".join([f"Title: {res['title']}, Link: {res['link']}" for res in results])
            all_search_results_str_parts.append(f"查询 '{query}' 的结果:\n{formatted_results}")

        all_search_results_str = "\n\n".join(all_search_results_str_parts)
        history_entry = f"--- 轮次 {state['turn_count']} ---\n思考: {state['plan'].thought}\n行动: 执行搜索查询 {queries}\n观察: (搜索结果摘要)\n{all_search_results_str}"

        # 5. 更新图状态
        return {
            "urls_to_scrape": all_urls, # 直接传递URL列表
            "reasoning_history": state['reasoning_history'] + [history_entry]
        }
    except Exception as e:
        # 捕获搜索过程中的任何异常
        logger.error(f"执行网络搜索时发生错误: {e}")
        return {"urls_to_scrape": []}

async def scrape_node(state: SearchAgentState) -> dict:
    """
    抓取节点, 负责从搜索结果中提取 URL 并并发抓取网页内容。
    - 使用正则表达式稳健地解析 URL。
    - 使用 `scrape_webpages` 辅助函数进行带缓存、重试和并发控制的异步抓取。
    - 将抓取到的内容摘要记录到推理历史中。
    """
    logger.info("▶️ 3. 进入抓取节点...")
    try:
        # 1. 直接从 search_node 获取 URL 列表
        urls_to_scrape = state.get('urls_to_scrape')
        if not urls_to_scrape:
            logger.warning("在搜索结果中未找到可抓取的 URL。")
            return {}

        logger.info(f"🔍 发现 {len(urls_to_scrape)} 个 URL, 开始抓取...")

        # 2. 异步抓取所有 URL
        scraped_data = await scrape_webpages(urls_to_scrape)
        logger.info(f"✅ 成功抓取 {len(scraped_data)} 个页面。")

        # 3. 更新图状态, 用新抓取的内容覆盖旧的
        # 注意: 此处的推理历史记录已被简化, 其核心作用由 search_node 和 rolling_summary_node 承担, 
        # 以避免历史记录冗余。
        return {
            "latest_scraped_content": scraped_data,
        }
    except Exception as e:
        # 捕获抓取过程中的任何异常
        logger.error(f"抓取网页时发生错误: {e}")
        return {}

async def information_processor_node(state: SearchAgentState) -> dict:
    """
    信息处理节点, 对抓取到的原始网页内容进行评估和提炼。
    - 调用 LLM 对每份内容进行相关性打分和判断。
    - 提取相关内容的核心摘要, 去除噪音。
    - 将处理后的相关信息追加到 `processed_content_accumulator` 中, 以供后续节点使用。
    """
    logger.info("▶️ 4. 进入信息处理节点...")
    
    if not state.get('latest_scraped_content'):
        logger.info("... 没有新的抓取内容需要处理, 跳过。")
        return {}
    
    # 准备 Prompt
    # 将抓取内容列表转换为 JSON 字符串以便注入 Prompt
    scraped_content_json = json.dumps(
        [{"url": item["url"], "content": item["content"]} for item in state['latest_scraped_content']],
        ensure_ascii=False,
        indent=2
    )

    task = state['task']
    
    PROMPT_INFORMATION_PROCESSOR = load_prompts(task.category, "search_cn", "PROMPT_INFORMATION_PROCESSOR")
    prompt = PROMPT_INFORMATION_PROCESSOR.format(
        research_focus=state["current_focus"],
        rolling_summary=state["rolling_summary"] or "",
        content=scraped_content_json
    )

    # 调用 LLM
    messages = [{"role": "user", "content": prompt}]
    processed_results = await get_structured_output_with_retry(messages, ProcessedResults)

    if not processed_results or not processed_results.processed_contents:
        logger.warning("信息处理节点未能从LLM获得有效的处理结果。")
        return {}

    # 过滤不相关的, 并按相关性排序
    relevant_content = [item.model_dump() for item in processed_results.processed_contents if item.is_relevant]
    relevant_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    logger.info(f"✅ 信息处理完成, 获得 {len(relevant_content)} 条相关信息。")

    # 更新状态, 将新处理的内容追加到累加器中, 并单独存放以供滚动总结节点使用
    return {
        "processed_content_accumulator": state['processed_content_accumulator'] + relevant_content,
        "latest_processed_content": relevant_content
    }

async def rolling_summary_node(state: SearchAgentState) -> dict:
    """
    滚动总结节点, 实现“反思-规划”循环的关键。采用“增量精炼”模式。
    - 在每轮信息处理后调用。
    - 它接收“上一轮的摘要”和“本轮新增的信息”。
    - 调用 LLM 将新信息整合进旧摘要, 生成一份更新、去重后的摘要, 并识别信息缺口。
    - 这种方式避免了将所有历史信息重复发送给LLM, 有效控制了上下文长度。
    """
    # 仅当有新处理过的内容时才触发此节点。
    if not state.get('latest_processed_content'):
        logger.info("... 没有新处理的信息, 跳过滚动总结。")
        # 即使跳过, 也要确保 previous_rolling_summary 被传递, 避免状态丢失
        return {}

    logger.info("▶️ 5. 进入滚动总结节点...")

    # 在生成新总结之前, 保存当前的总结作为“上一轮”的总结
    previous_summary = state.get('rolling_summary') or "无, 这是研究的开始。"

    # 只使用本轮新增的信息进行精炼
    new_info_list = state.get('latest_processed_content', [])
    new_info_str = "\n\n---\n\n".join(
        f"来源 URL: {item['url']}\n相关性: {item.get('relevance_score', 'N/A')}\n摘要:\n{item['summary']}"
        for item in new_info_list
    )

    if not new_info_str:
        logger.info("... 本轮未发现新的有效信息, 跳过滚动总结。")
        return {"previous_rolling_summary": previous_summary}

    task = state['task']
    
    PROMPT_ROLLING_SUMMARY = load_prompts(task.category, "search_cn", "PROMPT_ROLLING_SUMMARY")
    prompt = PROMPT_ROLLING_SUMMARY.format(
        research_focus=state["current_focus"],
        previous_summary=previous_summary,
        new_information=new_info_str
    )

    llm_params = get_llm_params([{"role": "user", "content": prompt}])

    message = await llm_acompletion(llm_params)
    summary = message.content

    logger.info(f"🔄 生成滚动总结: {summary[:200]}...")
    return {"rolling_summary": summary, "previous_rolling_summary": previous_summary}

async def synthesize_node(state: SearchAgentState) -> dict:
    """
    综合报告节点, 是工作流的终点之一。
    - 基于最终的滚动总结作为主要内容, 并辅以所有处理过的摘要作为补充材料。
    - 这种方法在保证报告质量的同时, 显著减少了最终Prompt的Token消耗。
    - 调用 LLM 生成一份全面、连贯的最终研究报告, 并要求指出信息冲突。
    """
    logger.info("▶️ 6. 进入综合报告节点...")

    supplementary_summaries = "\n\n---\n\n".join(
        f"来源 URL: {item['url']}\n摘要:\n{item['summary']}"
        for item in state['processed_content_accumulator']
    ) or "无补充材料。"

    task = state['task']
    context_dict = {
        'current_focus': state['current_focus'],
        'rolling_summary': state.get('rolling_summary') or "研究未能生成有效摘要。", 
        'supplementary_summaries': supplementary_summaries, 
    }
    
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "search_cn", "SYSTEM_PROMPT_SYNTHESIZE", "USER_PROMPT_SYNTHESIZE")

    context = await get_rag().get_context_base(task)
    context.update(context_dict)
    
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)

    llm_params = get_llm_params(messages, temperature=0.4)

    message = await llm_acompletion(llm_params)
    final_report = message.content

    logger.info("✅ 报告生成完毕。")

    # 3. 更新图状态
    return {"final_report": final_report}


###############################################################################


# 条件路由函数

async def should_continue_search(state: SearchAgentState) -> str:
    """
    条件路由函数: 在规划后决定是继续研究循环还是结束。
    - 如果 `plan.queries` 为空, 表示规划器认为信息足够, 结束循环。
    - 如果达到最大搜索轮次, 为防止无限循环, 强制结束。
    - 新增: 如果研究停滞（新旧总结无显著差异）, 也结束循环。这通过两步实现: 
      1. 轻量级的Jaccard相似度检查, 快速过滤掉几乎相同的总结。
      2. 如果不够相似, 则通过LLM进行更深层次的语义判断。
    """
    # 条件3: 研究停滞检测
    prev_summary = state.get('previous_rolling_summary')
    current_summary = state.get('rolling_summary')

    # 只有在有两轮总结可比较时才进行
    if prev_summary and current_summary and state['turn_count'] > 1:
        # 优化: 在调用昂贵的LLM之前, 先进行高效的语义相似度检查。
        # 这比简单的词汇匹配（如Jaccard）更准确, 能更好地判断内容是否真的没有新意。
        similarity_threshold = 0.98
        
        embedding_model = state['embedding_model']
        # 1. 将新旧摘要编码为向量
        embeddings = embedding_model.encode([prev_summary, current_summary], convert_to_tensor=True) # type: ignore
        # 2. 计算余弦相似度
        cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        
        logger.info(f"新旧摘要的语义相似度: {cosine_sim:.4f}")

        if cosine_sim > similarity_threshold:
            logger.info(f"⏹️ 研究停滞, 新旧摘要语义相似度 ({cosine_sim:.4f}) 高于阈值 ({similarity_threshold}), 结束研究。")
            return "end_task"

        # 如果语义相似度不高, 再使用LLM进行最终的、更深层次的判断
        logger.info("...摘要相似度不高, 使用 LLM 进行深度停滞检测...")
        prompt = PROMPT_STAGNATION_DETECTION.format(
            prev_summary=prev_summary,
            current_summary=current_summary
        )
        llm_params = get_llm_params([{"role": "user", "content": prompt}], temperature=0)
        
        message = await llm_acompletion(llm_params)
        is_stagnant = message.content.strip().lower() == 'true'
        if is_stagnant:
            logger.info("⏹️ 研究停滞（LLM判断）, 新一轮未发现显著信息, 结束当前任务研究。")
            return "end_task"

    # 条件1: 规划器认为信息已足够, 主动停止。
    if not state['plan'].queries:
        logger.info("⏹️ 规划器决定结束当前任务研究。")
        return "end_task"
    
    # 条件2: 达到预设的最大搜索轮次, 被动停止。
    if state['turn_count'] >= MAX_SEARCH_TURNS:
        logger.info(f"⏹️ 已达到最大搜索轮次 ({MAX_SEARCH_TURNS}), 结束当前任务研究。")
        return "end_task"
    
    # 默认: 继续研究循环。
    logger.info("↪️ 继续为当前任务进行新一轮搜索。")
    return "continue_search"


###############################################################################


async def search(task: Task) -> Task:
    """
    1.  启动 (Entry Point): 工作流从 `planner` 节点开始。
    2.  研究循环 (The Main Loop):
        -   `planner`: 制定搜索计划。
        -   `should_continue_search` (条件边): 判断是否继续。
            -   若继续 (`continue_search`): 进入 `search`。
            -   若结束 (`end_task`): 跳转到 `synthesize`。
        -   `search` -> `scrape` -> `information_processor` -> `rolling_summary` -> `planner`: 构成研究循环。
    3.  报告生成与结束 (Termination):
        -   `synthesize`: 汇集所有信息, 生成最终报告, 流程结束 (`END`)。
    """
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if task.task_type != "search":
        raise ValueError("Task type must be 'search'.")
    
    # 根据任务目标检测语言, 并加载相应的嵌入模型
    try:
        lang = detect(task.goal)
    except LangDetectException:
        lang = 'zh'
        logger.warning(f"无法检测任务 '{task.goal}' 的语言")
    
    embedding_model = get_embedding_model(lang)
    
    # 1. 定义工作流图 (StateGraph)
    workflow = StateGraph(SearchAgentState)

    # 2. 向图中添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("scrape", scrape_node)    
    workflow.add_node("information_processor", information_processor_node)
    workflow.add_node("rolling_summary", rolling_summary_node) # 新增滚动总结节点
    workflow.add_node("synthesize", synthesize_node)
    
    # 3. 定义流程的边 (Edges), 即节点之间的连接关系
    workflow.set_entry_point("planner")
    
    # "planner" 节点后的条件分支, 决定是继续搜索、结束简单任务还是结束子任务。
    workflow.add_conditional_edges(
        "planner", # type: ignore
        should_continue_search,
        {
            "continue_search": "search", # 继续搜索循环
            "end_task": "synthesize",    # 任务结束, 生成报告
        }
    )
    
    # "search" -> "scrape" -> "information_processor" -> "rolling_summary" -> "planner" 构成研究循环的主体。
    workflow.add_edge("search", "scrape")
    workflow.add_edge("scrape", "information_processor")
    workflow.add_edge("information_processor", "rolling_summary")
    workflow.add_edge("rolling_summary", "planner") # 总结后返回规划, 形成闭环
    
    # "synthesize" (生成报告) 是终点节点, 流程在此结束。
    workflow.add_edge("synthesize", END)
    
    # 4. 编译图
    app = workflow.compile()
    
    # 5. 初始化状态并运行图
    initial_state = SearchAgentState(
        task=task,                              # 核心: 传入的任务对象
        embedding_model=embedding_model,        # 核心: 根据任务语言加载的嵌入模型
        current_focus=task.goal,                # 核心: 当前研究循环的焦点, 初始为任务目标
        final_report=None,                      # 最终报告, 初始为空
        plan=Plan(thought="", queries=[]),      # 当前的行动计划, 初始为空
        urls_to_scrape=[],                      # 待抓取的URL列表
        latest_scraped_content=[],              # 最新抓取的网页内容列表
        latest_processed_content=[],            # 最新处理的内容
        processed_content_accumulator=[],       # 累积的处理后内容
        previous_rolling_summary=None,          # 上一轮总结, 初始为空
        rolling_summary="",                     # 滚动总结, 初始为空字符串
        turn_count=0,                           # 当前任务/子任务的搜索轮次计数
        reasoning_history=[]                    # 记录思考-行动-观察的链条
    )
    
    try:
        final_state = await app.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"执行研究任务 '{task.goal}' 时发生意外错误: {e}", exc_info=True)

        updated_task = task.model_copy(deep=True)
        updated_task.results = {
            "result": f"任务执行失败: {e}",
            "reasoning": "",
        }
        return updated_task
    
    # 6. 汇总推理历史, 生成可读的执行轨迹
    reasoning_str = "\n\n".join(final_state.get('reasoning_history', []))

    # 7. 更新任务对象并返回结果
    updated_task = task.model_copy(deep=True)
    updated_task.results = {
        "result": final_state['final_report'],
        "reasoning": reasoning_str,
    }

    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task


###############################################################################
