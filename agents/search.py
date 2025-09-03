import os
import json
import httpx
import asyncio
import litellm
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
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from ..memory import memory
from ..util.models import Task
from ..util.llm import get_llm_params


"""




分析、审查当前文件的代码，找出bug并改正， 指出可以优化的地方。


根据以上分析，改进建议， 请直接修改 文件，并提供diff。





为 `scrape_webpages` 函数增加单元测试，特别是针对混合抓取策略（httpx 成功、httpx 失败后 playwright 成功、两者都失败）的场景。






`should_continue_search` 函数中的停滞检测逻辑是否可以进一步优化？例如，除了Jaccard相似度，是否可以引入其他更高效的文本相似度算法？





当前 `information_processor_node` 是一次性处理所有抓取到的内容，如果内容非常多，可能会超出LLM的上下文窗口。如何优化这个节点以处理大量内容？








# model
中文：
shibing624/text2vec-base-chinese（专为中文优化的通用模型）
BAAI/bge-small-zh（中文语义理解能力强，适合长文本）
英文：
all-MiniLM-L6-v2（轻量高效，适合大多数场景）
all-mpnet-base-v2（精度更高，但速度稍慢）
多语言：
paraphrase-multilingual-MiniLM-L12-v2（轻量，支持 100 + 语言）
xlm-r-bert-base-nli-stsb-mean-tokens（支持语言更多，精度较高）








"""


###############################################################################


# 每个(子)任务的最大搜索-抓取-规划循环次数，用于防止无限循环。
MAX_SEARCH_TURNS = 3
# 并发抓取网页的最大数量，以控制资源使用和避免对目标服务器造成过大压力。
MAX_SCRAPE_CONCURRENCY = 5
# 单个网络请求（如抓取网页）的超时时间（秒）。
REQUEST_TIMEOUT = 30


###############################################################################


class Plan(BaseModel):
    thought: str = Field(description="你的思考过程，分析已有信息和下一步计划。")
    queries: List[str] = Field(description="根据思考生成的搜索查询列表。如果认为信息足够，则返回空列表。")

class ProcessedContent(BaseModel):
    url: str = Field(description="内容的原始URL。")
    relevance_score: float = Field(description="内容与研究焦点的相关性得分（0.0到1.0），分数越高越相关。")
    summary: str = Field(description="提取或生成的核心内容摘要，应去除噪音并突出关键信息。")
    is_relevant: bool = Field(description="内容是否与研究焦点直接相关。")

class ProcessedResults(BaseModel):
    processed_contents: List[ProcessedContent] = Field(description="处理和评估后的内容列表。")


###############################################################################


class SearchAgentState(TypedDict):
    # --- 核心任务信息 ---
    task: Task                          # 传入的原始任务对象，用于获取上下文和存储记忆。
    task_description: str               # 用户的原始任务描述，整个工作流的起点。
    final_report: Optional[str]         # 最终生成的研究报告，在 synthesize_node 中填充。

    # --- 研究循环状态 (简单路径和复杂路径的子任务循环共用) ---
    plan: Plan                          # 当前的行动计划（思考+查询），由 planner_node 生成。
    urls_to_scrape: Optional[List[str]] # 从 search_node 返回的待抓取URL列表。
    latest_scraped_content: List[dict]  # 从最新一次 scrape_node 运行中抓取的原始内容。
    latest_processed_content: List[dict] # 从最新一次 information_processor_node 运行中处理过的内容。
    processed_content_accumulator: List[dict] # 累积所有轮次中经过处理和筛选的相关信息。
    previous_rolling_summary: Optional[str] # 上一轮的滚动总结，用于检测研究是否停滞。
    rolling_summary: Optional[str]      # 滚动总结，由 rolling_summary_node 生成，持续更新的知识库，用于指导下一轮规划。
    turn_count: int                     # 当前任务/子任务的研究循环轮次计数，用于防止无限循环。
    reasoning_history: List[str]        # 记录“思考-行动-观察”链条，用于构建上下文和最终的推理过程展示。


###############################################################################


cache_dir = os.path.join("output", ".cache")
os.makedirs(cache_dir, exist_ok=True)
SEARCH_CACHE = Cache(os.path.join(cache_dir, 'search_cache'), size_limit=int(128 * (1024**2)))
SCRAPE_CACHE = Cache(os.path.join(cache_dir, 'scrape_cache'), size_limit=int(128 * (1024**2)))


"""
初始化一个 SearxNG 搜索工具的实例。
SearxSearchWrapper 是 LangChain 提供的一个工具类，用于与 SearxNG 这个元搜索引擎进行交互。
searx_host 从环境变量 "SearXNG" 中读取，如果未设置，则默认为本地地址。
这与 docker-compose.yml 中配置的 searxng 服务相对应。
"""
search_tool = SearxSearchWrapper(searx_host=os.environ.get("SearXNG", "http://127.0.0.1:8080"))


"""
初始化一个轻量级的句子嵌入模型，用于计算语义相似度。
这比纯词汇匹配（如Jaccard相似度）更能理解文本的真实含义，
同时成本远低于调用一次完整的LLM。
'all-MiniLM-L6-v2' 是一个在速度和性能上表现均衡的优秀模型。

# model
中文：
shibing624/text2vec-base-chinese（专为中文优化的通用模型）
BAAI/bge-small-zh（中文语义理解能力强，适合长文本）
英文：
all-MiniLM-L6-v2（轻量高效，适合大多数场景）
all-mpnet-base-v2（精度更高，但速度稍慢）
多语言：
paraphrase-multilingual-MiniLM-L12-v2（轻量，支持 100 + 语言）
xlm-r-bert-base-nli-stsb-mean-tokens（支持语言更多，精度较高）
"""
model_path = '../models/all-MiniLM-L6-v2'
if not os.path.isdir(model_path):
    logger.warning(f"本地模型路径 '{model_path}' 不存在，将尝试从网络下载 'all-MiniLM-L6-v2'。")
    logger.warning("请考虑运行 ./start.sh 脚本中的 hf download 命令来本地化模型，以提高加载速度和稳定性。")
    # 回退到在线下载
    model_path = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_path)


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
                    
                    sleep = backoff_in_seconds * 2 ** x
                    logger.warning(f"函数 {func.__name__} 失败，错误: {e}。将在 {sleep} 秒后重试...")
                    await asyncio.sleep(sleep)
                    x += 1
        return wrapper
    return decorator


async def scrape_webpages(urls: List[str]) -> List[dict]:
    """
    使用混合策略并发抓取网页内容。
    1.  首先尝试使用 `httpx` 进行快速、轻量级的抓取。
    2.  如果 `httpx` 抓取失败（如网络错误）或提取的内容过短（通常意味着页面需要JS渲染），
        则自动降级到使用 `Playwright` 进行深度抓取，它可以执行JavaScript。
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
            logger.warning(f"⚠️ httpx 请求失败 ({e})，将降级到 Playwright: {url}")
        except Exception as e:
            logger.warning(f"⚠️ httpx 处理时发生未知错误 ({e})，将降级到 Playwright: {url}")

        # 如果 httpx 成功且内容足够，直接返回
        if httpx_content and len(httpx_content) >= 100:
            logger.info(f"✅ httpx 抓取成功: {url}")
            result = {"url": final_url, "content": httpx_content}
            SCRAPE_CACHE[url] = result
            return result
        elif httpx_content:
            logger.info(f"⚠️ httpx 内容过短，降级到 Playwright: {url}")

        # 3. 如果 httpx 失败或内容不足，降级到 Playwright
        page = None
        try:
            logger.info(f"使用 Playwright 深度抓取: {url}")
            page = await browser.new_page()
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in {"image", "stylesheet", "font", "media"} else route.continue_())
            await page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
            await asyncio.sleep(3)
            
            html_content = await page.content()
            text_content = extract(html_content)

            if text_content and len(text_content) >= 100:
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
    async with async_playwright() as p, httpx.AsyncClient(http2=True, verify=False, follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
        browser = await p.chromium.launch(headless=True)
        try:
            tasks = [scrape_with_semaphore(client, browser, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return [res for res in results if res is not None]
        finally:
            await browser.close()


###############################################################################


PLANNER_PROMPT_TEMPLATE = """你是一个世界一流的研究员，擅长为当前的研究目标制定详细的搜索计划。

### 1. 最终研究目标
{task_description}

### 3. 当前研究焦点
**{current_focus}**

### 4. 已有知识与信息缺口 (来自本轮研究的滚动总结)
这是目前关于 **当前研究焦点** 的所有已知信息。请仔细阅读，特别是“信息缺口”部分。
{rolling_summary}

### 5. 最近的思考历史 (避免重复)
{reasoning_history}

---
### 你的任务
基于以上信息，为 **当前研究焦点** 制定下一步的行动计划。请严格遵循以下思考框架：

1.  **结果分析 (Analysis)**: 回顾“滚动总结与信息缺口”和“当前研究的历史记录”，简要评估上一轮搜索（如果存在）是否有效。信息是不足、充分还是出现了偏差？
2.  **策略调整 (Strategy)**: 基于分析，决定本轮的策略。是继续深入某个方向，还是转换角度，或是补充缺失的信息？
3.  **行动计划 (Action)**: 阐述你接下来的具体思考过程，并生成用于执行的搜索查询列表。

请将你的思考过程（包含上述分析和策略）写入 `thought` 字段。
如果经过分析认为当前研究焦点的信息已足够，请在 `thought` 中说明理由，并返回空的 `queries` 列表以结束当前焦点的研究。
"""

INFORMATION_PROCESSOR_PROMPT = """你是一位顶尖的情报分析师，任务是筛选和提炼信息。
对于下面给出的每一份从网页抓取的内容，请根据当前的“研究焦点”进行评估和处理。

**当前研究焦点**: {current_focus}

**抓取到的内容列表 (JSON格式)**:
{scraped_content_json}

---
请为每一份内容完成以下任务：
1.  **相关性评估**: 判断内容与“研究焦点”的匹配程度，给出一个 0.0 到 1.0 的分数。
2.  **相关性判断**: 明确指出内容是否真的相关 (is_relevant: true/false)。
3.  **内容提取与摘要**:
    - 如果内容相关，请提取其核心信息，生成一份简洁、精炼的摘要。
    - 必须去除广告、导航链接、页脚等所有噪音信息，只保留正文的关键部分。
    - 如果原文过长，请进行压缩总结，而不是简单截断。
    - 如果内容不相关，摘要可以为空或简要说明其无关。

请严格按照指定的JSON格式返回处理结果。
"""


ROLLING_SUMMARY_PROMPT = """你是一个信息整合专家，负责持续更新一份研究摘要。
你的任务是根据“上一轮的摘要”和“本轮新增的信息”，生成一份更新、整合、去重后的新摘要，并识别当前的信息缺口。

**当前研究焦点**: {current_focus}

### 1. 上一轮的研究摘要 (这是你工作的起点)
{previous_summary}

### 2. 本轮新增的信息 (请将这些信息整合进去)
{new_info}

---
请执行以下操作:
1.  **生成更新后的综合摘要**: 将“本轮新增的信息”整合进“上一轮的研究摘要”中。请确保新的摘要连贯、全面，并移除了重复信息。
2.  **识别信息缺口**: 基于更新后的摘要和“当前研究焦点”，明确指出当前还缺少哪些关键信息。
3.  **输出**: 你的输出应该只包含更新后的综合摘要和信息缺口分析。
"""


SYNTHESIZE_PROMPT = """你是一个专业的报告撰写者。你的任务是基于一份核心摘要和一系列补充材料，为用户的原始任务生成一份全面、流畅、结构清晰的最终报告。

### 1. 原始任务
{task_description}

### 2. 核心研究摘要 (主要依据)
这份摘要是整个研究过程的精华总结，请以此为基础构建报告的主体框架和核心论点。
{rolling_summary}

### 3. 补充材料 (用于丰富细节和发现冲突)
这是从各个信息来源提取的原始摘要列表。请用它们来：
- 丰富报告的细节。
- 验证核心摘要中的事实。
- 发现并明确指出不同来源之间的信息矛盾或冲突。
 
{supplementary_summaries}
 
---
### 你的任务
请生成最终报告。报告应以“核心研究摘要”为骨架，并用“补充材料”中的信息进行填充和佐证。如果在补充材料中发现与核心摘要或与其他材料相冲突的内容，请在报告中明确指出。
"""

SELF_CORRECTION_PROMPT = """你上次的输出格式不正确，导致了解析错误。
错误信息: {error}

原始的、格式错误的输出:
```json
{raw_output}
```

请严格按照 Pydantic 模型的要求，修正你的输出，并只返回修正后的、完整的、有效的 JSON 对象。不要添加任何额外的解释或文本。
"""


###############################################################################


# 图节点

async def get_structured_output_with_retry(messages: List[dict], response_model: BaseModel, retries: int = 1):
    """
    调用 LLM 以获取结构化输出，并在解析失败时自动尝试纠错。
    """
    response = None # 初始化 response 以避免 UnboundLocalError
    for i in range(retries + 1):
        try:
            llm_params = get_llm_params(messages, response_model=response_model)
            response = await litellm.acompletion(**llm_params)

            tool_call = response.choices[0].message.tool_calls[0]
            parsed_args = tool_call.function.parsed_arguments
            return response_model(**parsed_args)

        except Exception as e:
            logger.warning(f"调用LLM或解析输出失败 (尝试 {i+1}/{retries+1}): {e}")
            if i == retries:
                logger.error("达到最大重试次数，解析失败。")
                return None
            
            # 仅当收到响应但解析失败时，才尝试自我纠错
            if response:
                try:
                    raw_output = response.choices[0].message.tool_calls[0].function.arguments
                    correction_prompt = SELF_CORRECTION_PROMPT.format(error=str(e), raw_output=raw_output)
                    messages = messages + [response.choices[0].message, {"role": "user", "content": correction_prompt}]
                    logger.info("...正在尝试自我纠错...")
                except Exception as format_e:
                    logger.error(f"构建纠错提示时发生错误: {format_e}. 将进行常规重试。")
            else:
                logger.warning("未收到LLM响应，将进行常规重试。")

async def planner_node(state: SearchAgentState) -> dict:
    """
    规划节点，是研究循环的核心。
    它聚合了最关键的上下文信息（任务描述、滚动总结、最近的历史记录等），
    然后调用 LLM 遵循“分析-策略-行动”的框架，生成下一步的思考和搜索查询。
    此节点的上下文经过精简，以提高效率和准确性。
    如果 LLM 认为当前焦点的信息已足够，它将返回一个空的查询列表，从而触发研究循环的终止。
    """
    turn = state['turn_count'] + 1
    context_dict = {}

    # 1. 准备 Prompt 所需的核心上下文
    context_dict['task_description'] = state['task_description']
    
    # 只传递最近2轮的思考历史，以精简上下文，避免LLM在长对话中迷失
    recent_history = state['reasoning_history'][-2:]
    context_dict['reasoning_history'] = "\n\n".join(recent_history) or "无"

    # 滚动总结是规划的核心依据，它包含了当前已知信息和信息缺口
    context_dict['rolling_summary'] = state.get('rolling_summary') or "无，这是第一次研究，请开始探索。"

    # 2. 确定当前的研究焦点
    context_dict['current_focus'] = state['task_description']
    logger.info(f"▶️ 1. 进入规划节点 (第 {turn} 轮)...")

    # 3. 使用精简后的上下文格式化 Prompt
    prompt = PLANNER_PROMPT_TEMPLATE.format(**context_dict)
    
    # 4. 调用 LLM，要求返回 `Plan` 结构的输出
    messages = [{"role": "user", "content": prompt}]
    plan = await get_structured_output_with_retry(messages, Plan)
    
    # 如果解析失败，创建一个空的 Plan 对象以避免下游节点出错
    if not plan:
        plan = Plan(thought="规划失败，尝试终止当前研究。", queries=[])

    logger.info(f"🤔 思考: {plan.thought}")
    logger.info(f"🔍 生成查询: {plan.queries}")

    # 5. 更新图状态，包括新的计划和增加的轮次计数
    return {"plan": plan, "turn_count": turn}

@async_retry(retries=2, backoff_in_seconds=2)
async def _search_with_retry(query: str):
    """带重试逻辑的搜索函数包装器，返回结构化结果"""
    return search_tool.results(query)

async def search_node(state: SearchAgentState) -> dict:
    queries = state['plan'].queries
    logger.info(f"▶️ 2. 进入搜索节点，执行查询: {queries}...")
    
    try:
        # 1. 为每个查询创建一个异步搜索任务，并处理缓存
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
        results_per_query = await asyncio.gather(*search_tasks)

        # 3. 提取所有URL并格式化结果用于日志和历史记录
        all_urls = []
        all_search_results_str_parts = []
        for query, results in zip(queries, results_per_query):
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
    抓取节点，负责从搜索结果中提取 URL 并并发抓取网页内容。
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

        logger.info(f"🔍 发现 {len(urls_to_scrape)} 个 URL，开始抓取...")

        # 2. 异步抓取所有 URL
        scraped_data = await scrape_webpages(urls_to_scrape)
        logger.info(f"✅ 成功抓取 {len(scraped_data)} 个页面。")

        # 3. 更新图状态，用新抓取的内容覆盖旧的
        # 注意：此处的推理历史记录已被简化，其核心作用由 search_node 和 rolling_summary_node 承担，
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
    信息处理节点，对抓取到的原始网页内容进行评估和提炼。
    - 调用 LLM 对每份内容进行相关性打分和判断。
    - 提取相关内容的核心摘要，去除噪音。
    - 将处理后的相关信息追加到 `processed_content_accumulator` 中，以供后续节点使用。
    """
    logger.info("▶️ 4. 进入信息处理节点...")
    
    if not state.get('latest_scraped_content'):
        logger.info("... 没有新的抓取内容需要处理，跳过。")
        return {}

    current_focus = state['task_description']
    
    # 准备 Prompt
    # 将抓取内容列表转换为 JSON 字符串以便注入 Prompt
    scraped_content_json = json.dumps(
        [{"url": item["url"], "content": item["content"]} for item in state['latest_scraped_content']],
        ensure_ascii=False,
        indent=2
    )
    prompt = INFORMATION_PROCESSOR_PROMPT.format(
        current_focus=current_focus,
        scraped_content_json=scraped_content_json
    )

    # 调用 LLM
    messages = [{"role": "user", "content": prompt}]
    processed_results = await get_structured_output_with_retry(messages, ProcessedResults)

    if not processed_results or not processed_results.processed_contents:
        logger.warning("信息处理节点未能从LLM获得有效的处理结果。")
        return {}

    # 过滤不相关的，并按相关性排序
    relevant_content = [item.model_dump() for item in processed_results.processed_contents if item.is_relevant]
    relevant_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    logger.info(f"✅ 信息处理完成，获得 {len(relevant_content)} 条相关信息。")

    # 更新状态，将新处理的内容追加到累加器中，并单独存放以供滚动总结节点使用
    return {
        "processed_content_accumulator": state['processed_content_accumulator'] + relevant_content,
        "latest_processed_content": relevant_content
    }

async def rolling_summary_node(state: SearchAgentState) -> dict:
    """
    滚动总结节点，实现“反思-规划”循环的关键。采用“增量精炼”模式。
    - 在每轮信息处理后调用。
    - 它接收“上一轮的摘要”和“本轮新增的信息”。
    - 调用 LLM 将新信息整合进旧摘要，生成一份更新、去重后的摘要，并识别信息缺口。
    - 这种方式避免了将所有历史信息重复发送给LLM，有效控制了上下文长度。
    """
    # 仅当有新处理过的内容时才触发此节点。
    if not state.get('latest_processed_content'):
        logger.info("... 没有新处理的信息，跳过滚动总结。")
        # 即使跳过，也要确保 previous_rolling_summary 被传递，避免状态丢失
        return {}

    logger.info("▶️ 5. 进入滚动总结节点...")

    # 在生成新总结之前，保存当前的总结作为“上一轮”的总结
    previous_summary = state.get('rolling_summary') or "无，这是研究的开始。"
    current_focus = state['task_description']

    # 只使用本轮新增的信息进行精炼
    new_info_list = state.get('latest_processed_content', [])
    new_info_str = "\n\n---\n\n".join(
        f"来源 URL: {item['url']}\n相关性: {item.get('relevance_score', 'N/A')}\n摘要:\n{item['summary']}"
        for item in new_info_list
    )

    if not new_info_str:
        logger.info("... 本轮未发现新的有效信息，跳过滚动总结。")
        return {"previous_rolling_summary": previous_summary}

    prompt = ROLLING_SUMMARY_PROMPT.format(
        current_focus=current_focus,
        previous_summary=previous_summary,
        new_info=new_info_str
    )

    llm_params = get_llm_params([{"role": "user", "content": prompt}])
    response = await litellm.acompletion(**llm_params)
    summary = response.choices[0].message.content

    logger.info(f"🔄 生成滚动总结: {summary[:200]}...")
    return {"rolling_summary": summary, "previous_rolling_summary": previous_summary}

async def synthesize_node(state: SearchAgentState) -> dict:
    """
    综合报告节点，是工作流的终点之一。
    - 基于最终的滚动总结作为主要内容，并辅以所有处理过的摘要作为补充材料。
    - 这种方法在保证报告质量的同时，显著减少了最终Prompt的Token消耗。
    - 调用 LLM 生成一份全面、连贯的最终研究报告，并要求指出信息冲突。
    """
    logger.info("▶️ 6. 进入综合报告节点...")
    
    # 1. 准备上下文
    # 核心摘要
    rolling_summary = state.get('rolling_summary') or "研究未能生成有效摘要。"

    # 补充材料：仅包含URL和摘要，更精简
    supplementary_summaries = "\n\n---\n\n".join(
        f"来源 URL: {item['url']}\n摘要:\n{item['summary']}"
        for item in state['processed_content_accumulator']
    ) or "无补充材料。"

    prompt = SYNTHESIZE_PROMPT.format(
        task_description=state['task_description'],
        rolling_summary=rolling_summary,
        supplementary_summaries=supplementary_summaries
    )
    
    # 2. 调用 LLM 生成最终报告
    llm_params = get_llm_params([{"role": "user", "content": prompt}], temperature=0.4)
    response = await litellm.acompletion(**llm_params)
    final_report = response.choices[0].message.content

    logger.info("✅ 报告生成完毕。")

    # 3. 更新图状态
    return {"final_report": final_report}


###############################################################################


# 条件路由函数

async def should_continue_search(state: SearchAgentState) -> str:
    """
    条件路由函数：在规划后决定是继续研究循环还是结束。
    - 如果 `plan.queries` 为空，表示规划器认为信息足够，结束循环。
    - 如果达到最大搜索轮次，为防止无限循环，强制结束。
    - **新增**: 如果研究停滞（新旧总结无显著差异），也结束循环。这通过两步实现：
      1. 轻量级的Jaccard相似度检查，快速过滤掉几乎相同的总结。
      2. 如果不够相似，则通过LLM进行更深层次的语义判断。
    """
    # 条件3: 研究停滞检测
    prev_summary = state.get('previous_rolling_summary')
    current_summary = state.get('rolling_summary')

    # 只有在有两轮总结可比较时才进行
    if prev_summary and current_summary and state['turn_count'] > 1:
        # 优化：在调用昂贵的LLM之前，先进行高效的语义相似度检查。
        # 这比简单的词汇匹配（如Jaccard）更准确，能更好地判断内容是否真的没有新意。
        similarity_threshold = 0.98
        
        # 1. 将新旧摘要编码为向量
        embeddings = embedding_model.encode([prev_summary, current_summary], convert_to_tensor=True)
        # 2. 计算余弦相似度
        cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        
        logger.info(f"新旧摘要的语义相似度: {cosine_sim:.4f}")

        if cosine_sim > similarity_threshold:
            logger.info(f"⏹️ 研究停滞，新旧摘要语义相似度 ({cosine_sim:.4f}) 高于阈值 ({similarity_threshold})，结束研究。")
            return "end_task"

        # 如果语义相似度不高，再使用LLM进行最终的、更深层次的判断
        logger.info("...摘要相似度不高，使用 LLM 进行深度停滞检测...")
        prompt = f"""
        判断“当前总结”相对于“上一轮总结”是否包含了显著的、有价值的新信息。

        上一轮总结:\n{prev_summary}\n\n当前总结:\n{current_summary}

        如果“当前总结”只是对旧信息的重述、细化，或者新增的信息无关紧要，则返回 "true" (表示停滞)。否则返回 "false"。
        只返回 "true" 或 "false"。
        """
        llm_params = get_llm_params([{"role": "user", "content": prompt}], temperature=0)
        response = await litellm.acompletion(**llm_params)
        is_stagnant = response.choices[0].message.content.strip().lower() == 'true'
        if is_stagnant:
            logger.info("⏹️ 研究停滞（LLM判断），新一轮未发现显著信息，结束当前任务研究。")
            return "end_task"

    # 条件1: 规划器认为信息已足够，主动停止。
    if not state['plan'].queries:
        logger.info("⏹️ 规划器决定结束当前任务研究。")
        return "end_task"
    
    # 条件2: 达到预设的最大搜索轮次，被动停止。
    if state['turn_count'] >= MAX_SEARCH_TURNS:
        logger.info(f"⏹️ 已达到最大搜索轮次 ({MAX_SEARCH_TURNS})，结束当前任务研究。")
        return "end_task"
    
    # 默认: 继续研究循环。
    logger.info("↪️ 继续为当前任务进行新一轮搜索。")
    return "continue_search"


###############################################################################


async def search(task: Task) -> Task:
    """
    Workflow:
        1.  **启动 (Entry Point)**: 工作流从 `planner` 节点开始。
        2.  **研究循环 (The Main Loop)**:
            -   `planner`: 制定搜索计划。
            -   `should_continue_search` (条件边): 判断是否继续。
                -   若继续 (`continue_search`): 进入 `search`。
                -   若结束 (`end_task`): 跳转到 `synthesize`。
            -   `search` -> `scrape` -> `information_processor` -> `rolling_summary` -> `planner`: 构成研究循环。
        3.  **报告生成与结束 (Termination)**:
            -   `synthesize`: 汇集所有信息，生成最终报告，流程结束 (`END`)。
    """
    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if task.task_type != "search":
        raise ValueError("Task type must be 'search'.")
    
    # 1. 定义工作流图 (StateGraph)
    # StateGraph 是 LangGraph 的核心组件，用于定义一个状态机。
    # SearchAgentState 是一个 TypedDict，它定义了在图的节点之间传递的状态对象的结构。
    workflow = StateGraph(SearchAgentState)

    # 2. 向图中添加节点
    # 每个节点都是一个函数或可调用对象，它接收当前状态并返回一个状态更新的字典。
    # 这些节点代表了研究过程中的各个独立步骤。
    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("scrape", scrape_node)    
    workflow.add_node("information_processor", information_processor_node)
    workflow.add_node("rolling_summary", rolling_summary_node) # 新增滚动总结节点
    workflow.add_node("synthesize", synthesize_node)
    
    # 3. 定义流程的边 (Edges)，即节点之间的连接关系
    # set_entry_point 指定了工作流的起始节点。
    workflow.set_entry_point("planner")
    
    # "planner" 节点后的条件分支，决定是继续搜索、结束简单任务还是结束子任务。
    workflow.add_conditional_edges(
        "planner", # type: ignore
        should_continue_search,
        {
            "continue_search": "search", # 继续搜索循环
            "end_task": "synthesize",    # 任务结束，生成报告
        }
    )
    
    # "search" -> "scrape" -> "information_processor" -> "rolling_summary" -> "planner" 构成研究循环的主体。
    workflow.add_edge("search", "scrape")
    workflow.add_edge("scrape", "information_processor")
    workflow.add_edge("information_processor", "rolling_summary")
    workflow.add_edge("rolling_summary", "planner") # 总结后返回规划，形成闭环
    
    # "synthesize" (生成报告) 是终点节点，流程在此结束。
    workflow.add_edge("synthesize", END)
    
    # 4. 编译图
    # compile() 方法将定义好的节点和边编译成一个可执行的 LangChain 可调用对象 (Runnable)。
    app = workflow.compile()
    
    # 5. 初始化状态并运行图
    # 创建一个初始状态字典，为图的执行提供必要的初始数据。
    initial_state = SearchAgentState(
        task=task,                              # 核心：传入的任务对象
        task_description=task.goal,             # 核心：用户的原始任务目标
        final_report=None,                      # 最终报告，初始为空
        plan=Plan(thought="", queries=[]),      # 当前的行动计划，初始为空
        urls_to_scrape=[],                      # 待抓取的URL列表
        latest_scraped_content=[],              # 最新抓取的网页内容列表
        latest_processed_content=[],            # 最新处理的内容
        processed_content_accumulator=[],       # 累积的处理后内容
        previous_rolling_summary=None,          # 上一轮总结，初始为空
        rolling_summary="",                     # 滚动总结，初始为空字符串
        turn_count=0,                           # 当前任务/子任务的搜索轮次计数
        reasoning_history=[]                    # 记录思考-行动-观察的链条
    )
    
    try:
        # ainvoke 是异步调用图执行的方法。它会从入口点开始，根据状态和边的逻辑，
        # 依次执行各个节点，直到到达 END 节点。
        # final_state 将包含图执行完毕后的最终状态。
        final_state = await app.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"执行研究任务 '{task.goal}' 时发生意外错误: {e}", exc_info=True)
        updated_task = task.model_copy(deep=True)
        updated_task.results = {
            "result": f"任务执行失败: {e}",
            "reasoning": "由于在执行过程中发生意外错误，无法生成推理历史。",
        }
        return updated_task
    
    # 6. 汇总推理历史，生成可读的执行轨迹
    # 这个步骤是为了提高代理工作过程的透明度。
    reasoning_str = "\n\n".join(final_state.get('reasoning_history', []))

    # 7. 更新任务对象并返回结果
    # 使用 model_copy(deep=True) 创建一个原始任务的深拷贝，以避免副作用。
    updated_task = task.model_copy(deep=True)
    # 将最终报告和推理历史记录存入任务的 results 字段。
    updated_task.results = {
        "result": final_state['final_report'],
        "reasoning": reasoning_str,
    }
    return updated_task


###############################################################################


"""
# 角色与任务
你是信息检索专家，任务是为下游的写作任务高效、准确地收集信息。


# 工作流程与输出格式
你通过多轮搜索迭代式完成任务。每一轮的输出必须严格遵循指定的标签结构。

## 第 1 轮: 规划与首次搜索
必须按顺序输出以下3个标签：
- `<global_plan>`:
    - 1. 任务拆解: 将当前检索任务 `{to_run_question}` 拆解为带编号的子任务列表。
    - 2. 依赖分析: 简要说明子任务间的依赖关系。
- `<query_strategy>`:
    - 1. 目标子任务: 明确本轮要执行的子任务编号。
    - 2. 查询设计: 为目标子任务设计具体的搜索查询词。
- `<search_queries>`:
    - 格式: 提供一个JSON数组格式的搜索查询词列表。
    - 示例: `["关键词1", "关键词2"]`

## 第 2 轮及以后: 迭代与评估
必须按顺序输出以下4个标签：
- `<summary_and_analysis>`:
    - 1. 滚动总结: 整合并去重至今所有轮次的关键信息，形成一个持续更新的报告。必须引用信息来源的索引号 `[doc_id]`。若发现信息冲突，需明确指出冲突点。
    - 2. 信息缺口: 基于滚动总结和全局计划，明确指出当前缺失的关键信息。
- `<turn_plan>`:
    - 1. 充分性评估: 基于滚动总结，评估信息是否足够支撑“关联写作任务” `{to_run_outer_write_task}`，并给出置信度分数（0-100%）及判断依据。
    - 2. 行动决策: 决定下一步行动：`CONTINUE` (继续深入/切换/补充) 或 `TERMINATE` (终止任务)。
    - 3. 计划更新: 若行动为 `CONTINUE`，明确指出要处理的子任务编号，并更新全局计划中子任务的完成状态 (例如：`1. [已完成]`, `2. [进行中]`)。如果发现初始计划有缺陷，可在此处修正全局计划。
- `<query_strategy>`:
    - 前提: 仅在 `<turn_plan>` 决策为 `CONTINUE` 时输出。
    - 1. 结果分析: 简要分析上一轮搜索结果的有效性。
    - 2. 策略调整: 若结果不佳，必须分析原因（如：关键词不当、角度错误）并调整策略。
    - 3. 查询设计: 基于分析，设计或优化本轮的搜索查询词。
- `<search_queries>`:
    - 格式: JSON数组。
    - CONTINUE: 输出本轮的搜索查询词列表。
    - TERMINATE: 必须输出空数组 `[]` 以终止任务。

## 最终产出
- 当你决定终止任务时，`<summary_and_analysis>` 的内容将作为最终报告，直接用于下游任务。请确保其格式干净、独立完整、面向最终用户，并移除 `[doc_id]` 等过程性标记。


# 核心规则
- 依赖处理: 有依赖关系的任务，必须分步搜索。
- 并行搜索: 无依赖关系的任务，可在一轮内并行搜索（最多4个）。
- 查询优化: 若搜索效果不佳，必须在 `<query_strategy>` 中归因并调整。
- 终止条件: 当置信度 >95% 或达到4轮搜索上限时，必须在 `<turn_plan>` 中决策为 `TERMINATE`。目标是用最少轮次完成任务。



# 任务背景与上下文

## 1. 背景
- 总目标：{to_run_root_question}
- 关联写作任务：{to_run_outer_write_task}
- 当前检索任务：{to_run_question} (你只需聚焦并完成此任务)

## 2. 动态信息
- 当前轮次: {to_run_turn}
- 最终产出要求：你的最终总结必须直接服务于“关联写作任务”。
- 历史决策: 
{to_run_action_history}
- 上一轮搜索结果:
{to_run_tool_result}

----
请严格遵循你在系统指令中定义的角色和规则，完成第 {to_run_turn} 轮任务。








Agent正在通过多轮的搜索解决用户问题，你将对**其中一轮搜索的结果**进行评估。会提供给你Agent发起该轮搜索的思考、判断和目的，以及该轮搜索得到的一条网页。

您的任务包括以下步骤：
1. 分析网页：
    - 仔细检查网页中的每一句话
    - 根据你的知识，认真推理网页与该轮搜索的目的之间的可能联系。
2. 判断网页内容能否满足该轮搜索目的，判定属于“非常丰富且完全满足”， “完全满足”，或“部分满足”，或“不满足”
请首先在<think></think>中进行思考，然后在<answer></answer>中进行判断，如下：
<think>
思考
</think>
<answer>
非常丰富且完全满足/完全满足/部分满足/不满足
</answer>

---
发起该轮搜索的思考为：**{to_run_think}**

---
以下是网页内容：
{passage}

按照要求进行判断和输出，判断网页内容能否满足该轮搜索目的







你是网页搜索和信息理解专家，Agent正在通过多轮的搜索解决用户问题，你将被给到一条搜索结果，你需要对该条搜索结果的内容，完整且详尽地整理/抽取出与**用户原始问题**，以及**该轮搜索目的**相关的内容。你所整理和抽取的内容将替代原始网页内容提供给Agent用于回答用户问题和进行之后的搜索，因此需要保证 **正确性** 和 **完整性**。

- 不要遗漏任何信息
- 网页中很可能并不存在相关内容，不要虚构任何不存在的内容
- 专注于在网页中找到相关的信息，并完整地进行整理
- 只要做内容整理和抽取，不要给出下一步搜索建议，有任何存在相关性或提示性的内容也可以进行整理
- 若完全无相关内容，直接输出 无内容
- 注意内容的时效性，并需要清晰指出描述的实体防止误解。


输出格式：
请首先在<think></think>中进行简要的思考，然后在<content></content>中进行整理和内容抽取，具体如下：
<think>
思考
</think>
<content>
具体内容
</content>


用户原始问题为：**{to_run_question}**

发起该轮搜索的思考为（你需要理解该轮搜索的目的）：**{to_run_think}**

---
以下是网页内容：
{passage}

按照要求进行整理和抽取，关注于网页内容本身，不要给出建议：


"""
