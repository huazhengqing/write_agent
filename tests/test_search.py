import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from loguru import logger
from llama_index.core.tools import FunctionTool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.search import (
    search_with_searxng,
    search_with_ddg,
    web_search,
    targeted_search,
    scrape_static,
    scrape_dynamic,
    scrape_and_extract,
    get_web_search_tool,
    get_targeted_search_tool,
    get_web_scraper_tool,
    platform_site_map,
    web_search_tools
)

pytestmark = pytest.mark.asyncio


async def test_search_with_searxng_success(monkeypatch):
    """【单元测试】测试 search_with_searxng 成功时能正确格式化结果。"""
    logger.info("--- 测试 search_with_searxng (成功) ---")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": "SearXNG Test Title", "url": "http://searxng.com", "content": "SearXNG summary."}
        ]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    mock_async_client_context = AsyncMock()
    mock_async_client_context.__aenter__.return_value = mock_client
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_async_client_context)

    result = await search_with_searxng("test query", max_results=1)

    assert "SearXNG Test Title" in result
    assert "http://searxng.com" in result
    assert "SearXNG summary." in result
    assert "搜索结果 1" in result
    logger.success("--- search_with_searxng (成功) 测试通过 ---")


async def test_search_with_searxng_empty_results(monkeypatch):
    """【单元测试】测试 search_with_searxng 返回空结果时的异常处理。"""
    logger.info("--- 测试 search_with_searxng (空结果) ---")
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    mock_async_client_context = AsyncMock()
    mock_async_client_context.__aenter__.return_value = mock_client
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_async_client_context)

    with pytest.raises(ValueError, match="SearXNG 返回了空结果"):
        await search_with_searxng("test query", max_results=1)
    
    logger.success("--- search_with_searxng (空结果) 测试通过 ---")


async def test_search_with_searxng_http_error(monkeypatch):
    """【单元测试】测试 search_with_searxng HTTP错误时的异常处理。"""
    logger.info("--- 测试 search_with_searxng (HTTP错误) ---")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPError("HTTP Error")

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    mock_async_client_context = AsyncMock()
    mock_async_client_context.__aenter__.return_value = mock_client
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_async_client_context)

    with pytest.raises(httpx.HTTPError):
        await search_with_searxng("test query", max_results=1)
    
    logger.success("--- search_with_searxng (HTTP错误) 测试通过 ---")


def test_search_with_ddg_success(monkeypatch):
    """【单元测试】测试 search_with_ddg 成功时能正确格式化结果。"""
    logger.info("--- 测试 search_with_ddg (成功) ---")
    
    mock_ddg_instance = MagicMock()
    mock_ddg_instance.text.return_value = [
        {"title": "DDG Test Title", "href": "http://ddg.com", "body": "DDG summary."}
    ]

    mock_ddgs_context = MagicMock()
    mock_ddgs_context.__enter__.return_value = mock_ddg_instance
    
    monkeypatch.setattr("utils.search.DDGS", lambda **kwargs: mock_ddgs_context)

    result = search_with_ddg("test query", max_results=1)

    assert "DDG Test Title" in result
    assert "http://ddg.com" in result
    assert "DDG summary." in result
    assert "搜索结果 1" in result
    logger.success("--- search_with_ddg (成功) 测试通过 ---")


def test_search_with_ddg_empty_results(monkeypatch):
    """【单元测试】测试 search_with_ddg 返回空结果时的异常处理。"""
    logger.info("--- 测试 search_with_ddg (空结果) ---")
    mock_ddg_instance = MagicMock()
    mock_ddg_instance.text.return_value = []

    mock_ddgs_context = MagicMock()
    mock_ddgs_context.__enter__.return_value = mock_ddg_instance
    
    monkeypatch.setattr("utils.search.DDGS", lambda **kwargs: mock_ddgs_context)

    with pytest.raises(ValueError, match="DuckDuckGo 返回了空结果"):
        search_with_ddg("test query", max_results=1)
    
    logger.success("--- search_with_ddg (空结果) 测试通过 ---")


async def test_scrape_static_success(monkeypatch):
    """【单元测试】测试 scrape_static 成功抓取和提取。"""
    logger.info("--- 测试 scrape_static (成功) ---")
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Static Content</p></body></html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    mock_async_client_context = AsyncMock()
    mock_async_client_context.__aenter__.return_value = mock_client
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_async_client_context)

    long_static_content = "This is some sufficiently long static content to pass the length check. " * 20
    mock_trafilatura = MagicMock(return_value=long_static_content)
    monkeypatch.setattr("trafilatura.extract", mock_trafilatura)

    result = await scrape_static("http://static.com")

    assert result == long_static_content
    mock_trafilatura.assert_called_once()
    logger.success("--- scrape_static (成功) 测试通过 ---")


async def test_scrape_static_short_content(monkeypatch):
    """【单元测试】测试 scrape_static 提取到的内容太短时的处理。"""
    logger.info("--- 测试 scrape_static (内容太短) ---")
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Short Content</p></body></html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    mock_async_client_context = AsyncMock()
    mock_async_client_context.__aenter__.return_value = mock_client
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_async_client_context)

    short_content = "Short content that is too brief."
    mock_trafilatura = MagicMock(return_value=short_content)
    monkeypatch.setattr("trafilatura.extract", mock_trafilatura)

    result = await scrape_static("http://static.com")

    assert result is None
    mock_trafilatura.assert_called_once()
    logger.success("--- scrape_static (内容太短) 测试通过 ---")


async def test_scrape_static_empty_html(monkeypatch):
    """【单元测试】测试 scrape_static 返回空HTML时的处理。"""
    logger.info("--- 测试 scrape_static (空HTML) ---")
    mock_response = MagicMock()
    mock_response.text = ""
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    mock_async_client_context = AsyncMock()
    mock_async_client_context.__aenter__.return_value = mock_client
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_async_client_context)

    result = await scrape_static("http://static.com")

    assert result is None
    logger.success("--- scrape_static (空HTML) 测试通过 ---")


async def test_scrape_static_http_error(monkeypatch):
    """【单元测试】测试 scrape_static HTTP错误时的异常处理。"""
    logger.info("--- 测试 scrape_static (HTTP错误) ---")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPError("HTTP Error")

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    mock_async_client_context = AsyncMock()
    mock_async_client_context.__aenter__.return_value = mock_client
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_async_client_context)

    with pytest.raises(httpx.HTTPError):
        await scrape_static("http://static.com")
    
    logger.success("--- scrape_static (HTTP错误) 测试通过 ---")


async def test_scrape_dynamic_success(monkeypatch):
    """【单元测试】测试 scrape_dynamic 成功抓取和提取。"""
    logger.info("--- 测试 scrape_dynamic (成功) ---")
    
    mock_page = AsyncMock()
    mock_page.content.return_value = "<html><body><p>Dynamic Content</p></body></html>"
    
    mock_browser = AsyncMock()
    mock_browser.new_page.return_value = mock_page
    mock_browser.close = AsyncMock()
    
    mock_browser_context = AsyncMock()
    mock_browser_context.__aenter__.return_value = mock_browser
    
    mock_chromium = MagicMock()
    mock_chromium.launch.return_value = mock_browser_context
    
    mock_playwright_instance = MagicMock()
    mock_playwright_instance.chromium = mock_chromium
    
    mock_playwright_context = AsyncMock()
    mock_playwright_context.__aenter__.return_value = mock_playwright_instance
    
    monkeypatch.setattr("utils.search.async_playwright", lambda: mock_playwright_context)

    long_dynamic_content = "This is some sufficiently long dynamic content. " * 10
    mock_trafilatura = MagicMock(return_value=long_dynamic_content)
    monkeypatch.setattr("trafilatura.extract", mock_trafilatura)

    result = await scrape_dynamic("http://dynamic.com")

    assert result == long_dynamic_content
    mock_trafilatura.assert_called_once()
    mock_page.goto.assert_called_once()
    mock_browser.close.assert_called_once()
    logger.success("--- scrape_dynamic (成功) 测试通过 ---")


async def test_scrape_dynamic_short_content(monkeypatch):
    """【单元测试】测试 scrape_dynamic 提取到的内容太短时返回HTML。"""
    logger.info("--- 测试 scrape_dynamic (内容太短) ---")
    
    mock_page = AsyncMock()
    html_content = "<html><body><p>Dynamic Content</p></body></html>"
    mock_page.content.return_value = html_content
    
    mock_browser = AsyncMock()
    mock_browser.new_page.return_value = mock_page
    mock_browser.close = AsyncMock()
    
    mock_browser_context = AsyncMock()
    mock_browser_context.__aenter__.return_value = mock_browser
    
    mock_chromium = MagicMock()
    mock_chromium.launch.return_value = mock_browser_context
    
    mock_playwright_instance = MagicMock()
    mock_playwright_instance.chromium = mock_chromium
    
    mock_playwright_context = AsyncMock()
    mock_playwright_context.__aenter__.return_value = mock_playwright_instance
    
    monkeypatch.setattr("utils.search.async_playwright", lambda: mock_playwright_context)

    # 返回短内容
    short_content = "Short content"
    mock_trafilatura = MagicMock(return_value=short_content)
    monkeypatch.setattr("trafilatura.extract", mock_trafilatura)

    result = await scrape_dynamic("http://dynamic.com")

    # 当内容太短时, 应该返回原始HTML
    assert result == html_content
    mock_trafilatura.assert_called_once()
    mock_page.goto.assert_called_once()
    mock_browser.close.assert_called_once()
    logger.success("--- scrape_dynamic (内容太短) 测试通过 ---")


async def test_scrape_dynamic_empty_html(monkeypatch):
    """【单元测试】测试 scrape_dynamic 返回空HTML时的异常处理。"""
    logger.info("--- 测试 scrape_dynamic (空HTML) ---")
    
    mock_page = AsyncMock()
    mock_page.content.return_value = ""
    
    mock_browser = AsyncMock()
    mock_browser.new_page.return_value = mock_page
    mock_browser.close = AsyncMock()
    
    mock_browser_context = AsyncMock()
    mock_browser_context.__aenter__.return_value = mock_browser
    
    mock_chromium = MagicMock()
    mock_chromium.launch.return_value = mock_browser_context
    
    mock_playwright_instance = MagicMock()
    mock_playwright_instance.chromium = mock_chromium
    
    mock_playwright_context = AsyncMock()
    mock_playwright_context.__aenter__.return_value = mock_playwright_instance
    
    monkeypatch.setattr("utils.search.async_playwright", lambda: mock_playwright_context)

    with pytest.raises(RuntimeError, match="未能获取到任何 HTML 内容"):
        await scrape_dynamic("http://dynamic.com")
    
    mock_browser.close.assert_called_once()
    logger.success("--- scrape_dynamic (空HTML) 测试通过 ---")


@pytest.mark.integration
async def test_web_search_live():
    """【集成测试】测试 web_search 函数的实时通用网络搜索功能。"""
    logger.info("--- 测试 web_search (实时) ---")
    query = "大型语言模型最新进展"
    results = await web_search(query, max_results=3)
    
    assert results is not None, "搜索结果不应为 None"
    assert isinstance(results, str), "搜索结果应为字符串"
    assert "搜索结果 1" in results, "应包含'搜索结果 1'标识"
    assert "链接:" in results, "应包含'链接:'"
    assert "摘要:" in results, "应包含'摘要:'"
    logger.success("--- web_search (实时) 测试通过 ---")


async def test_web_search_cache_hit(monkeypatch):
    """【单元测试】测试 web_search 缓存命中的情况。"""
    logger.info("--- 测试 web_search (缓存命中) ---")
    
    # 模拟缓存
    mock_cache_get = MagicMock(return_value="Cached search result")
    monkeypatch.setattr("utils.search.cache_searh.get", mock_cache_get)
    
    query = "test query"
    result = await web_search(query, max_results=5)
    
    assert result == "Cached search result"
    mock_cache_get.assert_called_once_with(f"web_search:{query.lower().strip()}:5")
    logger.success("--- web_search (缓存命中) 测试通过 ---")


async def test_web_search_cache_miss(monkeypatch):
    """【单元测试】测试 web_search 缓存未命中的情况。"""
    logger.info("--- 测试 web_search (缓存未命中) ---")
    
    # 模拟缓存未命中
    mock_cache_get = MagicMock(return_value=None)
    mock_cache_set = MagicMock()
    monkeypatch.setattr("utils.search.cache_searh.get", mock_cache_get)
    monkeypatch.setattr("utils.search.cache_searh.set", mock_cache_set)
    
    # 模拟搜索成功
    expected_result = "Search result"
    mock_search_with_searxng = AsyncMock(return_value=expected_result)
    monkeypatch.setattr("utils.search.search_with_searxng", mock_search_with_searxng)
    
    query = "test query"
    result = await web_search(query, max_results=5)
    
    assert result == expected_result
    mock_search_with_searxng.assert_called_once_with(query, 5)
    mock_cache_set.assert_called_once_with(f"web_search:{query.lower().strip()}:5", expected_result)
    logger.success("--- web_search (缓存未命中) 测试通过 ---")


async def test_web_search_failure_raises_error(monkeypatch):
    """【单元测试】测试当所有搜索策略失败时, web_search 是否引发 RuntimeError。"""
    logger.info("--- 测试 web_search (全部失败) ---")
    # 模拟两种搜索策略都引发异常
    mock_searxng = AsyncMock(side_effect=Exception("SearXNG failed"))
    monkeypatch.setattr("utils.search.search_with_searxng", mock_searxng)

    mock_ddg = MagicMock(side_effect=Exception("DDG failed"))
    monkeypatch.setattr("utils.search.search_with_ddg", mock_ddg)

    with pytest.raises(RuntimeError, match="所有搜索策略均失败"):
        await web_search("some query")

    mock_searxng.assert_called_once()
    mock_ddg.assert_called_once()
    logger.success("--- web_search (全部失败) 测试通过 ---")


@pytest.mark.parametrize(
    "platforms, expected_sites",
    [
        # 测试别名扩展
        (["b站"], [f"site:{platform_site_map['B站']}"]),
        # 测试分类扩展
        (["小说"], [f"site:{platform_site_map['起点中文网']}", f"site:{platform_site_map['晋江文学城']}"]),
        # 测试混合情况
        (["b站", "小说"], [f"site:{platform_site_map['B站']}", f"site:{platform_site_map['起点中文网']}"]),
        # 测试未知平台被忽略
        (["未知平台", "知乎"], [f"site:{platform_site_map['知乎']}"]),
        # 测试空列表
        ([], []),
    ],
)
async def test_targeted_search_query_construction(monkeypatch, platforms, expected_sites):
    """【单元测试】测试 targeted_search 是否能从别名、分类和未知平台正确构建搜索查询。"""
    logger.info(f"--- 测试 targeted_search 查询构建 (platforms: {platforms}) ---")
    mock_web_search = AsyncMock(return_value="Mocked search result")
    monkeypatch.setattr("utils.search.web_search", mock_web_search)

    query = "二次元"
    await targeted_search(query=query, platforms=platforms)

    mock_web_search.assert_called_once()
    _, call_kwargs = mock_web_search.call_args
    final_query = call_kwargs.get("query")

    assert query in final_query, "原始查询字符串应保留"
    
    if expected_sites:
        for site in expected_sites:
            assert site in final_query, f"预期站点 '{site}' 未在查询中找到"
    else:
        # 如果没有提供有效平台, 不应有 site: 限制
        assert "site:" not in final_query

    if "未知平台" in platforms:
        assert "未知平台" not in final_query, "未知平台不应出现在最终查询中"
    
    logger.success(f"--- targeted_search 查询构建测试通过 (platforms: {platforms}) ---")


async def test_targeted_search_no_platforms(monkeypatch):
    """【单元测试】测试当未提供平台时, targeted_search 是否回退到通用搜索。"""
    logger.info("--- 测试 targeted_search (无平台) ---")
    mock_web_search = AsyncMock(return_value="Mocked search result")
    monkeypatch.setattr("utils.search.web_search", mock_web_search)

    query = "some query"
    await targeted_search(query=query, platforms=None, sites=None)

    # 验证它是否像通用搜索一样被调用, 没有 site: 限制
    mock_web_search.assert_called_once_with(query=query, max_results=5)
    logger.success("--- targeted_search (无平台) 测试通过 ---")


@pytest.mark.integration
async def test_targeted_search_live():
    """【集成测试】测试 targeted_search 函数的实时定向网站搜索功能。"""
    logger.info("--- 测试 targeted_search (实时) ---")
    query = "AIGC"
    platforms = ["知乎", "36氪"]
    results = await targeted_search(query=query, platforms=platforms)
    
    assert results is not None, "搜索结果不应为 None"
    assert isinstance(results, str), "搜索结果应为字符串"
    assert "搜索结果 1" in results, "应包含'搜索结果 1'标识"
    assert "链接:" in results, "应包含'链接:'"
    # 检查结果是否确实来自指定网站(或至少其中之一)
    assert "zhihu.com" in results or "36kr.com" in results, "结果中应包含指定网站的链接"
    logger.success("--- targeted_search (实时) 测试通过 ---")


@pytest.mark.integration
async def test_scrape_and_extract_live():
    """【集成测试】测试 scrape_and_extract 函数在已知URL上的实时抓取和提取功能。"""
    logger.info("--- 测试 scrape_and_extract (实时) ---")
    # 知乎专栏是一个很好的候选者, 因为它可能需要动态抓取
    url = "https://zhuanlan.zhihu.com/p/616386443"
    content = await scrape_and_extract(url)

    assert content is not None, "抓取内容不应为 None"
    assert isinstance(content, str), "抓取内容应为字符串"
    assert len(content) > 100, "应提取到有意义的文本量"
    assert "AIGC" in content or "人工智能" in content, "应包含页面核心内容"
    logger.success("--- scrape_and_extract (实时) 测试通过 ---")


async def test_scrape_content_truncation(monkeypatch):
    """【单元测试】测试过长的抓取内容是否被正确截断到句子末尾。"""
    logger.info("--- 测试 scrape_and_extract (内容截断) ---")
    max_length = 500 # Use a smaller length for easier testing
    # 创建一个带有多种句子结束符的长模拟文本
    long_text = "第一句。第二句！第三句?第四句。" * (max_length // 10)
    assert len(long_text) > max_length

    # 模拟抓取策略以返回我们的长文本
    mockscrape_static = AsyncMock(return_value=long_text)
    monkeypatch.setattr("utils.search.scrape_static", mockscrape_static)
    # 阻止动态抓取器运行
    monkeypatch.setattr("utils.search.scrape_dynamic", AsyncMock(return_value=None))

    # 模拟缓存以确保我们的逻辑被命中
    monkeypatch.setattr("utils.search.cache_searh.get", lambda key: None)
    monkeypatch.setattr("utils.search.cache_searh.set", lambda key, value: None)

    url = "http://fake-url-for-long-content.com"
    truncated_content = await scrape_and_extract(url, max_length=max_length)

    assert len(truncated_content) <= max_length + 1  # 允许结尾字符
    # 最后一个字符应该是原始文本中的标点符号
    assert truncated_content.endswith(("。", "！", "?", ".", "!", "?"))

    # 验证截断逻辑
    end_pos = long_text.rfind('。', 0, max_length)
    expected_content = long_text[:end_pos + 1]
    assert truncated_content == expected_content
    logger.success("--- scrape_and_extract (内容截断) 测试通过 ---")


async def test_scrape_failure_returns_empty_string(monkeypatch, caplog):
    """【单元测试】测试当所有抓取策略失败时, scrape_and_extract 是否返回空字符串并记录错误。"""
    logger.info("--- 测试 scrape_and_extract (全部失败) ---")
    url = "http://non-existent-url.com"
    mockscrape_static = AsyncMock(side_effect=Exception("Static scrape failed"))
    monkeypatch.setattr("utils.search.scrape_static", mockscrape_static)

    mockscrape_dynamic = AsyncMock(side_effect=Exception("Dynamic scrape failed"))
    monkeypatch.setattr("utils.search.scrape_dynamic", mockscrape_dynamic)

    # 模拟缓存
    monkeypatch.setattr("utils.search.cache_searh.get", lambda key: None)
    monkeypatch.setattr("utils.search.cache_searh.set", lambda key, value: None)

    result = await scrape_and_extract(url)

    assert result == "", "当所有抓取策略失败时, 应返回空字符串"
    mockscrape_static.assert_called_once_with(url)
    mockscrape_dynamic.assert_called_once_with(url)

    # 检查错误日志
    assert f"所有抓取策略均未能从URL '{url}' 提取到有效内容" in caplog.text
    logger.success("--- scrape_and_extract (全部失败) 测试通过 ---")


async def test_scrape_fallback_strategy(monkeypatch, caplog):
    """【单元测试】测试抓取策略的降级: 静态失败后, 动态成功。"""
    logger.info("--- 测试 scrape_and_extract (降级策略) ---")
    url = "http://requires-js.com"
    expected_content = "这是由JavaScript动态加载的内容。"

    # 模拟静态抓取失败(例如, 返回空内容或抛出异常)
    mockscrape_static = AsyncMock(return_value="")
    monkeypatch.setattr("utils.search.scrape_static", mockscrape_static)

    # 模拟动态抓取成功
    mockscrape_dynamic = AsyncMock(return_value=expected_content)
    monkeypatch.setattr("utils.search.scrape_dynamic", mockscrape_dynamic)

    # 模拟缓存
    monkeypatch.setattr("utils.search.cache_searh.get", lambda key: None)
    monkeypatch.setattr("utils.search.cache_searh.set", lambda key, value: None)

    result = await scrape_and_extract(url)

    assert result == expected_content, "应返回动态抓取器成功获取的内容"
    mockscrape_static.assert_called_once_with(url)
    mockscrape_dynamic.assert_called_once_with(url)
    assert f"静态抓取 '{url}' 失败或内容为空, 尝试动态抓取..." in caplog.text
    logger.success("--- scrape_and_extract (降级策略) 测试通过 ---")


def test_get_web_search_tool():
    """【单元测试】测试 get_web_search_tool 是否正确创建工具。"""
    logger.info("--- 测试 get_web_search_tool ---")
    tool = get_web_search_tool()
    
    assert isinstance(tool, FunctionTool)
    assert tool.metadata.name == "web_search"
    assert "执行通用网络搜索" in tool.metadata.description
    assert "query" in tool.metadata.parameters
    logger.success("--- get_web_search_tool 测试通过 ---")


def test_get_targeted_search_tool():
    """【单元测试】测试 get_targeted_search_tool 是否正确创建工具。"""
    logger.info("--- 测试 get_targeted_search_tool ---")
    tool = get_targeted_search_tool()
    
    assert isinstance(tool, FunctionTool)
    assert tool.metadata.name == "targeted_search"
    assert "在一个或多个特定网站" in tool.metadata.description
    assert "platforms" in tool.metadata.parameters
    logger.success("--- get_targeted_search_tool 测试通过 ---")


def test_get_web_scraper_tool():
    """【单元测试】测试 get_web_scraper_tool 是否正确创建工具。"""
    logger.info("--- 测试 get_web_scraper_tool ---")
    tool = get_web_scraper_tool()
    
    assert isinstance(tool, FunctionTool)
    assert tool.metadata.name == "web_scraper"
    assert "抓取并提取指定URL网页的正文内容" in tool.metadata.description
    assert "url" in tool.metadata.parameters
    logger.success("--- get_web_scraper_tool 测试通过 ---")


def test_web_search_tools():
    """【单元测试】测试 web_search_tools 是否包含所有必要的工具。"""
    logger.info("--- 测试 web_search_tools ---")
    
    assert len(web_search_tools) == 3
    
    # 检查工具类型
    for tool in web_search_tools:
        assert isinstance(tool, FunctionTool)
    
    # 检查工具名称
    tool_names = [tool.metadata.name for tool in web_search_tools]
    assert "web_search" in tool_names
    assert "targeted_search" in tool_names
    assert "web_scraper" in tool_names
    
    logger.success("--- web_search_tools 测试通过 ---")
