import os
import httpx
import json
from loguru import logger
from typing import List, Optional
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from market_analysis.story.market import index

###############################################################################

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY 环境变量未设置。")
agent_tavily_tools = TavilyToolSpec(api_key=tavily_api_key, max_results=7).to_tool_list()

###############################################################################

async def scrape_and_extract(url: str) -> str:
    """从指定的URL抓取并提取网页的主要文本内容。"""
    logger.info(f"开始使用 Jina Reader 抓取 URL: {url}")
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
            logger.warning(f"抓取的内容过长 ({len(page_text)} > {max_length})，将进行智能截断。")
            end_pos = page_text.rfind('。', 0, max_length)
            if end_pos == -1:
                end_pos = page_text.rfind('.', 0, max_length)
            if end_pos != -1:
                page_text = page_text[:end_pos + 1]
            else:
                page_text = page_text[:max_length]
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
    """创建并返回一个网页抓取工具。"""
    return FunctionTool.from_defaults(
        fn=scrape_and_extract,
        name="web_scraper",
        description=(
            "从指定的URL抓取并提取网页的主要文本内容。"
            "当你通过搜索找到一个包含所需信息的具体网页时，使用此工具来读取该网页的详细内容。"
            "输入是一个 `url` 参数，输出是页面的文本内容。"
            "如果此工具返回错误或空内容，说明该网页无法访问或内容为空，请不要重试同一个URL，应尝试通过搜索寻找其他信息源。"
        )
    )

###############################################################################

async def get_social_media_trends(platform: str, query: str) -> str:
    """
    从指定的社交媒体平台获取趋势信息。
    
    注意：这是一个模拟实现。在理想情况下，这里应该调用各平台的官方API或专门的爬虫工具来获取结构化、可靠的数据。
    目前，它仍然使用通用的网络搜索工具，但将查询范围限定在特定平台，并作为未来优化的接口。
    
    Args:
        platform (str): 平台名称，如 'B站', '微博', '抖音'.
        query (str): 具体的查询内容.
        
    Returns:
        str: 搜索到的趋势信息。
    """
    logger.info(f"正在从【{platform}】获取趋势信息，查询: '{query}'")
    # 构造更精确的搜索查询，限定在特定网站
    platform_site_map = {
        "B站": "bilibili.com",
        "微博": "weibo.com",
        "抖音": "douyin.com"
    }
    site = platform_site_map.get(platform)
    if not site:
        logger.warning(f"未知的社交媒体平台: {platform}，将进行通用搜索。")
        search_query = query
    else:
        search_query = f"site:{site} {query}"
    
    try:
        # 使用Tavily进行限定域名的搜索
        TAVILY_SEARCH_TOOL = TavilyToolSpec(api_key=tavily_api_key).to_tool_list()[0]
        trends_data = await TAVILY_SEARCH_TOOL.async_call("search", input=search_query)
        logger.success(f"从【{platform}】获取趋势信息成功。")
        return str(trends_data)
    except Exception as e:
        logger.error(f"从【{platform}】获取趋势信息时发生错误: {e}")
        return f"获取【{platform}】趋势失败: {e}"

def get_social_media_trends_tool() -> FunctionTool:
    """创建并返回一个社交媒体趋势搜索工具。"""
    return FunctionTool.from_defaults(
        fn=get_social_media_trends,
        name="social_media_trends_search",
        description=(
            "从指定的社交媒体平台（如'B站', '微博', '抖音'）获取关于特定查询的趋势信息。"
            "当你需要了解某个主题在社交网络上的热度、讨论或相关内容时使用此工具。"
            "输入参数包括 `platform` (平台名称) 和 `query` (查询内容)。"
            "示例: `social_media_trends_search(platform='B站', query='科幻小说最新流行元素')`"
        )
    )

###############################################################################

async def get_forum_discussions(sites: List[str], query: str) -> str:
    """
    从指定的论坛或社区网站搜索相关讨论。
    
    Args:
        sites (List[str]): 网站域名列表, 如 ['zhihu.com', 'lkong.com'].
        query (str): 具体的查询内容.
        
    Returns:
        str: 搜索到的讨论信息。
    """
    logger.info(f"正在从 {sites} 搜索讨论，查询: '{query}'")
    
    site_query_part = " OR ".join([f"site:{site}" for site in sites])
    full_query = f"({site_query_part}) {query}"
    
    try:
        TAVILY_SEARCH_TOOL = TavilyToolSpec(api_key=tavily_api_key).to_tool_list()[0]
        discussion_data = await TAVILY_SEARCH_TOOL.async_call("search", input=full_query)
        logger.success(f"从 {sites} 搜索讨论成功。")
        return str(discussion_data)
    except Exception as e:
        logger.error(f"从 {sites} 搜索讨论时发生错误: {e}")
        return f"从 {sites} 搜索讨论失败: {e}"


def get_forum_discussions_tool() -> FunctionTool:
    """创建并返回一个论坛讨论搜索工具。"""
    return FunctionTool.from_defaults(
        fn=get_forum_discussions,
        name="forum_discussion_search",
        description=(
            "从指定的论坛或社区网站列表（如['zhihu.com', 'lkong.com']）搜索相关讨论。"
            "当你需要了解专业读者或作者对某个主题的深入看法、评价或“毒点”时使用此工具。"
            "输入参数包括 `sites` (网站域名列表) 和 `query` (查询内容)。"
            "示例: `forum_discussion_search(sites=['zhihu.com', 'lkong.com'], query='近期网络小说差评套路')`"
        )
    )

###############################################################################




