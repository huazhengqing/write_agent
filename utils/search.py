import os
import sys
import httpx
from loguru import logger
import asyncio
from typing import List, Optional, Callable, Any, Tuple
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from ddgs import DDGS
from diskcache import Cache
from llama_index.core.tools import FunctionTool
import trafilatura
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.file import cache_dir


load_dotenv()


SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080/search")


cache_search_dir = cache_dir / "search"
cache_search_dir.mkdir(parents=True, exist_ok=True)
cache_searh = Cache(str(cache_search_dir), expire=60 * 60 * 24 * 7)


async def _search_with_searxng(query: str, max_results: int) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        data = {"q": query, "format": "json"}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'X-Real-IP': '127.0.0.1'
        }
        search_url = SEARXNG_BASE_URL.rstrip('/')
        if not search_url.endswith('/search'):
            search_url += '/search'
        
        response = await client.post(
            search_url,
            data=data,
            headers=headers,
            timeout=15.0
        )
        response.raise_for_status()  # 处理HTTP错误状态码
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


def _search_with_ddg(query: str, max_results: int) -> str:
    params = {
        "query": query,
        "max_results": max_results,
    }
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
    cache_key = f"web_search:{query}:{max_results}"
    cached_result = cache_searh.get(cache_key)
    if cached_result:
        return cached_result
    
    search_strategies: List[Tuple[str, Callable[[str, int], Any]]] = [
        ("SearXNG", _search_with_searxng),
        ("DuckDuckGo", _search_with_ddg),
    ]

    last_exception = None
    for name, strategy_func in search_strategies:
        try:
            logger.info(f"搜索策略: 正在尝试使用 {name} 执行搜索: '{query}'")
            if asyncio.iscoroutinefunction(strategy_func):
                search_results = await strategy_func(query, max_results)
            else:
                search_results = strategy_func(query, max_results)
            if search_results:
                cache_searh.set(cache_key, search_results)
                return search_results
        except Exception as e:
            last_exception = e
            logger.warning(f"使用 {name} 搜索失败: {e}")
            continue
        
    error_msg = f"所有搜索策略均失败。查询: '{query}'。最后错误: {last_exception}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_exception


def get_web_search_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=web_search,
        name="web_search",
        description=(
            "功能: 执行通用网络搜索，返回搜索结果列表（标题、链接、摘要）。\n"
            "使用时机: 当你需要对一个主题进行初步研究，或寻找信息来源时使用。\n"
            "注意: 本工具只返回搜索摘要，不抓取网页全文。如需阅读页面详细内容，请在获得URL后使用 `web_scraper` 工具。\n"
            "参数: `query` (str, 必需) - 搜索关键词或问题。"
        )
    )

###############################################################################

platform_categories = {
    # 社交与内容
    "社交": ["B站", "微博", "抖音", "知乎", "小红书", "豆瓣", "贴吧", "Reddit", "Twitter", "X", "Facebook", "YouTube", "Discord", "Patreon", "Quora"],
    "国内社交": ["B站", "微博", "抖音", "知乎", "小红书", "豆瓣", "贴吧"],
    "国际社交": ["Reddit", "Twitter", "X", "Facebook", "YouTube", "Discord", "Patreon", "Quora"],
    
    # 创作与出版
    "小说": ["起点中文网", "晋江文学城", "番茄小说", "飞卢小说网", "七猫免费小说", "纵横中文网", "17K小说网", "刺猬猫", "SF轻小说", "掌阅", "书旗小说", "咪咕阅读", "逐浪网", "红袖添香", "潇湘书院", "POPO原創", "Penana", "鏡文學", "Wattpad", "RoyalRoad", "AO3", "Webnovel", "Scribble Hub", "Tapas"],
    "国内小说": ["起点中文网", "晋江文学城", "番茄小说", "飞卢小说网", "七猫免费小说", "纵横中文网", "17K小说网", "刺猬猫", "SF轻小说", "掌阅", "书旗小说", "咪咕阅读", "逐浪网", "红袖添香", "潇湘书院"],
    "港台小说": ["POPO原創", "Penana", "鏡文學"],
    "国际小说": ["Wattpad", "RoyalRoad", "AO3", "Webnovel", "Scribble Hub", "Tapas"],
    "图书出版": ["当当", "京东读书", "微信读书", "得到", "豆瓣阅读", "Amazon KDP", "Google Play Books", "Apple Books", "Gumroad", "Leanpub"],
    "国内图书": ["当当", "京东读书", "微信读书", "得到", "豆瓣阅读"],
    "国际图书": ["Amazon KDP", "Google Play Books", "Apple Books", "Gumroad", "Leanpub"],

    # 研究与分析
    "技术": ["GitHub", "Stack Overflow", "CSDN", "掘金", "博客园", "SegmentFault", "V2EX", "IT之家"],
    "学术": ["知网", "万方数据", "维普", "谷歌学术", "arXiv", "PubMed", "ResearchGate"],
    "趋势与数据": ["百度指数", "微博指数", "巨量算数", "Google Trends", "新榜", "七麦数据", "蝉大师", "飞瓜数据", "SimilarWeb", "Ahrefs", "Semrush", "Exploding Topics"],
    "行业报告": ["艾瑞咨询", "QuestMobile", "易观分析", "亿欧", "IT桔子", "Gartner", "Forrester", "Statista", "MarketResearch.com", "CB Insights"],
    "国内报告": ["艾瑞咨询", "QuestMobile", "易观分析", "亿欧", "IT桔子"],
    "国际报告": ["Gartner", "Forrester", "Statista", "MarketResearch.com", "CB Insights"],

    # 资讯与评论
    "资讯": ["今日头条", "36氪", "虎嗅", "品玩", "少数派"],
    "社区与评论": ["龙空", "橙瓜", "优书网", "什么值得买", "酷安", "Product Hunt", "Trustpilot", "G2"],
    "作家社区": ["龙空", "橙瓜", "优书网"],
}

platform_site_map = {
    # 国内主流社交和内容平台 (Mainland China Social & Content)
    "B站": "bilibili.com",
    "微博": "weibo.com",
    "抖音": "douyin.com",
    "知乎": "zhihu.com",
    "小红书": "xiaohongshu.com",
    "豆瓣": "douban.com",
    "贴吧": "tieba.baidu.com",
    "今日头条": "toutiao.com",

    # 国际主流社交和内容平台 (International Social & Content)
    "Reddit": "reddit.com",
    "Twitter": "twitter.com",
    "X": "x.com",  # Alias for Twitter
    "Quora": "quora.com",
    "Facebook": "facebook.com",
    "YouTube": "youtube.com",
    "Discord": "discord.com",
    "Patreon": "patreon.com",

    # 国内网络小说平台 (Mainland China Web Novel Platforms)
    "起点中文网": "qidian.com",
    "晋江文学城": "jjwxc.net",
    "番茄小说": "fanqienovel.com",
    "飞卢小说网": "faloo.com",
    "七猫免费小说": "qimao.com",
    "纵横中文网": "zongheng.com",
    "17K小说网": "17k.com",
    "刺猬猫": "ciweimao.com",
    "SF轻小说": "sfacg.com",
    "掌阅": "ireader.com.cn",
    "书旗小说": "shuqi.com",
    "咪咕阅读": "cmread.com",
    "逐浪网": "zhulang.com",
    "红袖添香": "hongxiu.com",
    "潇湘书院": "xxsy.net",

    # 港台小说平台 (Hong Kong & Taiwan Novel Platforms)
    "POPO原創": "popo.tw",
    "Penana": "penana.com",
    "鏡文學": "mirrorfiction.com",

    # 国际网络小说平台 (International Web Novel Platforms)
    "Wattpad": "wattpad.com",
    "RoyalRoad": "royalroad.com",
    "AO3": "archiveofourown.org",  # Archive of Our Own
    "Webnovel": "webnovel.com",  # 起点国际
    "Scribble Hub": "scribblehub.com",
    "Tapas": "tapas.io",

    # 国内图书与知识付费 (Domestic Books & Knowledge Payment)
    "当当": "dangdang.com",
    "京东读书": "e.jd.com",
    "微信读书": "weread.qq.com",
    "得到": "dedao.cn",
    "豆瓣阅读": "read.douban.com",

    # 国际图书与自出版 (International Books & Self-Publishing)
    "Amazon KDP": "kdp.amazon.com",
    "Google Play Books": "play.google.com/books",
    "Apple Books": "apple.com/apple-books",
    "Gumroad": "gumroad.com",
    "Leanpub": "leanpub.com",

    # 数据与趋势分析 (Data & Trend Analysis)
    "百度指数": "index.baidu.com",
    "微博指数": "data.weibo.com",
    "巨量算数": "trendinsight.oceanengine.com",
    "Google Trends": "trends.google.com",
    "新榜": "newrank.cn",
    "七麦数据": "qimai.cn",
    "蝉大师": "chandashi.com",
    "飞瓜数据": "feigua.cn",
    "SimilarWeb": "similarweb.com",
    "Ahrefs": "ahrefs.com",
    "Semrush": "semrush.com",
    "Exploding Topics": "explodingtopics.com",

    # 国内行业报告与数据 (Domestic Industry Reports & Data)
    "艾瑞咨询": "iresearch.com.cn",
    "QuestMobile": "questmobile.com.cn",
    "易观分析": "analysys.cn",
    "亿欧": "iyiou.com",
    "IT桔子": "itjuzi.com",

    # 国际行业报告与数据 (International Industry Reports & Data)
    "Gartner": "gartner.com",
    "Forrester": "forrester.com",
    "Statista": "statista.com",
    "MarketResearch.com": "marketresearch.com",
    "CB Insights": "cbinsights.com",

    # 专业与技术社区 (Professional & Tech Communities)
    "GitHub": "github.com",
    "Stack Overflow": "stackoverflow.com",
    "CSDN": "csdn.net",
    "掘金": "juejin.cn",
    "博客园": "cnblogs.com",
    "SegmentFault": "segmentfault.com",
    "V2EX": "v2ex.com",
    "IT之家": "ithome.com",

    # 学术与论文 (Academic & Papers)
    "知网": "cnki.net",
    "万方数据": "wanfangdata.com.cn",
    "维普": "cqvip.com",
    "谷歌学术": "scholar.google.com",
    "arXiv": "arxiv.org",
    "PubMed": "pubmed.ncbi.nlm.nih.gov",
    "ResearchGate": "researchgate.net",

    # 新闻与资讯 (News & Information)
    "36氪": "36kr.com",
    "虎嗅": "huxiu.com",
    "品玩": "pingwest.com",
    "少数派": "sspai.com",

    # 社区与评论 (Communities & Reviews)
    "龙空": "lkong.com",  # 龙的天空
    "橙瓜": "chenggua.com",
    "优书网": "yousuu.com",
    "什么值得买": "smzdm.com",
    "酷安": "coolapk.com",
    "Product Hunt": "producthunt.com",
    "Trustpilot": "trustpilot.com",
    "G2": "g2.com",
}


async def targeted_search(query: str, platforms: Optional[List[str]] = None, sites: Optional[List[str]] = None) -> str:
    cache_key = f"targeted_search:{query}:{platforms}:{sites}"
    cached_result = cache_searh.get(cache_key)
    if cached_result:
        return cached_result

    all_sites = set(sites or [])
    if platforms:
        for p_user in platforms:
            matched_sites_for_p = set()
            if p_user in platform_categories:
                for platform_name in platform_categories[p_user]:
                    site = platform_site_map.get(platform_name)
                    if site:
                        matched_sites_for_p.add(site)
            for p_key, site in platform_site_map.items():
                if p_user in p_key:
                    matched_sites_for_p.add(site)
            if matched_sites_for_p:
                all_sites.update(matched_sites_for_p)
            else:
                logger.warning(f"未找到与 '{p_user}' 匹配的平台或分类，将被忽略。")

    if not all_sites:
        search_query = query
        logger.info(f"未指定平台或网站，执行通用搜索: '{search_query}'")
    else:
        site_query_part = " OR ".join([f"site:{s}" for s in all_sites])
        search_query = f"({site_query_part}) {query}"
        logger.info(f"执行站内搜索: '{search_query}'")

    search_result = await web_search(query=search_query, max_results=5)

    cache_searh.set(cache_key, search_result)
    return search_result


def get_targeted_search_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=targeted_search,
        name="targeted_search",
        description=(
            "功能: 在一个或多个特定网站、平台或网站类别上进行定向搜索。\n"
            "使用时机: 当你需要从特定来源（如'知乎'）、特定领域（如'小说'、'学术'）获取信息时，此工具比通用 `web_search` 更精确。\n"
            "核心参数: `platforms` 是此工具的关键，用于指定搜索范围。\n"
            "可用分类: '社交', '国内社交', '国际社交', '小说', '国内小说', '港台小说', '国际小说', '技术', '学术', '趋势', '资讯', '作家社区' 等。\n"
            "参数:\n"
            "- `query` (str, 必需): 核心搜索查询。\n"
            "- `platforms` (Optional[List[str]]): 平台或分类名称的列表。可使用分类名（如 '小说'）或平台名（如 '知乎', 'B站'）。\n"
            "- `sites` (Optional[List[str]]): 网站域名列表，用于补充 `platforms` 未覆盖的网站。例如: ['some-forum.com']。\n"
            "使用指南:\n"
            "1. 优先使用 `platforms` 参数进行范围限定。\n"
            "2. 如果 `platforms` 和 `sites` 均未提供，则退化为通用搜索。\n"
            "示例:\n"
            "- 查B站热点: `targeted_search(query='科幻小说流行元素', platforms=['B站'])`\n"
            "- 查小说网站和社区的差评套路: `targeted_search(query='近期网文差评套路', platforms=['小说', '龙空'])`\n"
            "- 查学术论文: `targeted_search(query='large language model reasoning', platforms=['学术'])`"
        )
    )


###############################################################################


async def _scrape_static(url: str) -> Optional[str]:
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
    
    page_text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False
    )

    if page_text and len(page_text.strip()) > 200:
        return page_text
    
    return None


async def _scrape_dynamic(url: str) -> Optional[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=45000, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        html_content = await page.content()
        await browser.close()

    if not html_content:
        raise RuntimeError(f"Playwright for URL '{url}' 未能获取到任何 HTML 内容。")
    
    page_text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False
    )

    if page_text and len(page_text.strip()) > 50:
        return page_text

    logger.warning(f"动态渲染抓取后，Trafilatura 仍未能提取到有效内容，将返回原始HTML作为后备。 URL: {url}")
    return html_content


async def scrape_and_extract(url: str) -> str:
    cached_content = cache_searh.get(url)
    if cached_content:
        return cached_content
    
    scrape_strategies: List[Tuple[str, Callable[[str], Any]]] = [
        ("静态抓取", _scrape_static),
        ("动态渲染抓取", _scrape_dynamic),
    ]
    scraped_text = None
    last_exception = None
    for name, strategy_func in scrape_strategies:
        try:
            logger.info(f"抓取策略: 正在尝试 {name}: {url}")
            scraped_text = await strategy_func(url)
            if scraped_text:
                break
        except Exception as e:
            last_exception = e
            logger.warning(f"策略 '{name}' 失败: {e}")
            continue

    if scraped_text:
        max_length = 16000
        if len(scraped_text) > max_length:
            end_pos = scraped_text.rfind('。', 0, max_length)
            if end_pos == -1:
                end_pos = scraped_text.rfind('.', 0, max_length)
            final_text = scraped_text[:end_pos + 1] if end_pos != -1 else scraped_text[:max_length]
        else:
            final_text = scraped_text
            
        cache_searh.set(url, final_text)
        return final_text
    
    logger.error(f"错误: 所有抓取策略均未能从URL '{url}' 提取到有效内容。最后错误: {last_exception}")
    return ""


def get_web_scraper_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=scrape_and_extract,
        name="web_scraper",
        description=(
            "功能: 抓取并提取指定URL网页的正文内容。\n"
            "使用时机: 通过 `web_search` 或 `targeted_search` 获得网页URL后，用此工具读取其详细内容。\n"
            "参数: `url` (str, 必需) - 完整的网页地址。\n"
            "失败处理: 如果工具执行失败并抛出异常，说明该URL无法访问或无有效内容。**禁止重试**，应放弃该URL，寻找其他信息源。"
        )
    )


web_search_tools = [
    get_web_search_tool(),
    get_targeted_search_tool(),
    get_web_scraper_tool(),
]
