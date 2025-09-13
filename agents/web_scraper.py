import httpx
from loguru import logger
from llama_index.core.tools import FunctionTool


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