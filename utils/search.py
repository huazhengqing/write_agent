from functools import lru_cache
import os
from loguru import logger
from typing import List, Optional, Callable, Any, Tuple
from diskcache import Cache
from llama_index.core.tools import FunctionTool
from utils.file import cache_dir



cache_search_dir = cache_dir / "search"
cache_search_dir.mkdir(parents=True, exist_ok=True)
cache_searh = Cache(str(cache_search_dir), expire=60 * 60 * 24 * 7)



async def search_with_searxng(query: str, max_results: int) -> str:
    import httpx
    async with httpx.AsyncClient(follow_redirects=True) as client:
        data = {"q": query, "format": "json"}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'X-Real-IP': '127.0.0.1'
        }
        search_url = os.getenv("SEARXNG_BASE_URL").rstrip('/')
        response = await client.post(
            search_url,
            data=data,
            headers=headers,
            timeout=45.0
        )
        response.raise_for_status()
        data = response.json()
    
    results = data.get("results", [])
    if not results:
        raise ValueError("SearXNG 返回了空结果")
    
    results = results[:max_results]
    
    formatted_results = []
    for i, result in enumerate(results):
        title = result.get('title')
        href = result.get('url')
        body = result.get('content', '').strip()
        formatted_results.append(f"搜索结果 {i+1}:\n标题: {title}\n链接: {href}\n摘要: {body}")
    
    return "\n\n---\n\n".join(formatted_results)



def search_with_ddg(query: str, max_results: int) -> str:
    params = {
        "query": query,
        "max_results": max_results,
    }
    from ddgs import DDGS
    with DDGS() as ddg:
        results = list(ddg.text(**params))
    if not results:
        raise ValueError("DuckDuckGo 返回了空结果")
    formatted_results = []
    for i, result in enumerate(results):
        title = result.get('title')
        href = result.get('href')
        body = result.get('body', '').strip()
        formatted_results.append(f"搜索结果 {i+1}:\n标题: {title}\n链接: {href}\n摘要: {body}")
    return "\n\n---\n\n".join(formatted_results)



async def web_search(query: str, max_results: int = 10) -> str:
    normalized_query = query.lower().strip()
    cache_key = f"web_search:{normalized_query}:{max_results}"
    cached_result = cache_searh.get(cache_key)
    if cached_result:
        return cached_result
    
    search_strategies: List[Tuple[str, Callable[[str, int], Any]]] = [
        ("SearXNG", search_with_searxng),
        ("DuckDuckGo", search_with_ddg),
    ]

    last_exception = None
    import asyncio
    for name, strategy_func in search_strategies:
        try:
            logger.info(f"搜索策略: 正在尝试使用 {name} 执行搜索: '{query}'")
            if asyncio.iscoroutinefunction(strategy_func):
                search_results = await strategy_func(query, max_results)
            else:
                search_results = strategy_func(query, max_results)
            if search_results:
                cache_searh.set(cache_key, search_results)
                logger.info(f"使用 {name} 搜索成功")
                return search_results
        except Exception as e:
            last_exception = e
            logger.warning("使用 {} 搜索失败: {}", name, e)
            logger.warning("{} 搜索失败的错误信息: {}", name, last_exception)
            continue
        
    error_msg = f"所有搜索策略均失败。查询: '{query}'。最后错误: {last_exception}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_exception



@lru_cache(maxsize=None)
def get_web_search_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        async_fn=web_search,
        name="web_search",
        description=(
            "功能: 执行通用网络搜索, 返回搜索结果列表(标题、链接、摘要)。\n"
            "使用时机: 当你需要对一个主题进行初步研究, 或寻找信息来源时使用。\n"
            "注意: 本工具只返回搜索摘要, 不抓取网页全文。如需阅读页面详细内容, 请在获得URL后使用 `web_scraper` 工具。\n"
            "参数: `query` (str, 必需) - 搜索关键词或问题。"
        )
    )



###############################################################################



async def scrape_static(url: str) -> Optional[str]:
    import httpx
    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        html_content = response.text

    if not html_content:
        logger.warning(f"静态抓取未能获取到 HTML 内容: {url}")
        return None
    
    import trafilatura
    page_text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False
    )

    if page_text and len(page_text.strip()) > 200:
        return page_text
    
    return None



async def scrape_dynamic(url: str) -> Optional[str]:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=45000, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        html_content = await page.content()
        await browser.close()

    if not html_content:
        raise RuntimeError(f"Playwright for URL '{url}' 未能获取到任何 HTML 内容。")
    
    import trafilatura
    page_text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False
    )

    if page_text and len(page_text.strip()) > 50:
        return page_text

    logger.warning(f"动态渲染抓取后, Trafilatura 仍未能提取到有效内容, 将返回原始HTML作为后备。 URL: {url}")
    return html_content



async def scrape_and_extract(url: str) -> str:
    logger.info(f"开始抓取 URL: {url}")
    cached_content = cache_searh.get(url)
    if cached_content:
        logger.info(f"从缓存中获取 URL 内容: {url}")
        return cached_content
    
    scrape_strategies: List[Tuple[str, Callable[[str], Any]]] = [
        ("静态抓取", scrape_static),
        ("动态渲染抓取", scrape_dynamic),
    ]
    scraped_text = None
    last_exception = None
    for name, strategy_func in scrape_strategies:
        try:
            logger.info(f"抓取策略: 正在尝试 {name}: {url}")
            scraped_text = await strategy_func(url)
            if scraped_text:
                logger.success(f"使用 '{name}' 成功抓取 URL: {url}")
                break
        except Exception as e:
            last_exception = e
            logger.warning("抓取策略 '{}' 失败: {}", name, e)
            continue

    if scraped_text:
        max_length = 16000
        original_length = len(scraped_text)
        if original_length > max_length:
            logger.info(f"抓取内容长度为 {original_length}, 超过最大长度 {max_length}, 将进行截断。")
            end_pos = scraped_text.rfind('。', 0, max_length)
            if end_pos == -1:
                end_pos = scraped_text.rfind('.', 0, max_length)
            final_text = scraped_text[:end_pos + 1] if end_pos != -1 else scraped_text[:max_length]
            logger.info(f"内容已截断至 {len(final_text)} 字符。")
        else:
            final_text = scraped_text
            
        logger.info(f"正在缓存 URL 内容: {url}")
        cache_searh.set(url, final_text)
        return final_text
    
    logger.error("错误: 所有抓取策略均未能从URL '{}' 提取到有效内容。最后错误: {}", url, last_exception)
    return ""



@lru_cache(maxsize=None)
def get_web_scraper_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        async_fn=scrape_and_extract,
        name="web_scraper",
        description=(
            "功能: 抓取并提取指定URL网页的正文内容。\n"
            "使用时机: 通过 `web_search` 获得网页URL后, 用此工具读取其详细内容。\n"
            "参数: `url` (str, 必需) - 完整的网页地址。\n"
            "失败处理: 如果工具执行失败并抛出异常, 说明该URL无法访问或无有效内容。**禁止重试**, 应放弃该URL, 寻找其他信息源。"
        )
    )


###############################################################################


web_search_tools = [
    get_web_search_tool(),
    get_web_scraper_tool(),
]
