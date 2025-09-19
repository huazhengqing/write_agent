import pytest
import sys
import os
import logging
from pathlib import Path
import nest_asyncio

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.log import init_logger
from utils.vector import init_llama_settings, get_vector_store, vector_add_from_dir, default_file_metadata
from tests import test_data


def pytest_configure(config):
    """
    为 pytest 配置全局警告过滤器。
    """
    # 忽略来自 litellm 内部关于 importlib.resources.open_text 的弃用警告
    config.addinivalue_line(
        "filterwarnings", "ignore:open_text is deprecated:DeprecationWarning"
    )
    # 忽略 Pydantic 在序列化不完整模型实例时发出的警告，这在 litellm 的响应对象中可能发生
    config.addinivalue_line(
        "filterwarnings", "ignore:Pydantic serializer warnings:UserWarning"
    )

# -- 会话级 Fixtures --

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    为整个测试会话初始化测试环境。
    - 应用 nest_asyncio 以便在 pytest 中运行异步代码。
    - 初始化 LlamaIndex 设置 (LLM, embedding model)。
    - 为嘈杂的库设置更高的日志级别。
    """
    nest_asyncio.apply()
    logging.getLogger("litellm").setLevel(logging.WARNING)
    init_llama_settings()
    # 这个 fixture 会为会话中的所有测试自动运行。
    # 不需要 yield 任何东西。


@pytest.fixture(scope="module", autouse=True)
def setup_module_logging(request):
    """
    为每个测试模块（文件）设置独立的日志文件。
    """
    # 从请求中获取模块（测试文件）的路径，并提取文件名（不含扩展名）作为日志名
    log_filename = Path(request.module.__file__).stem
    init_logger(log_filename)


@pytest.fixture(scope="module")
def test_dirs(tmp_path_factory):
    """为每个测试模块创建临时的目录。"""
    base_dir = tmp_path_factory.mktemp("vector_tests")
    db_path = base_dir / "chroma_db"
    input_path = base_dir / "input_data"
    input_path.mkdir()
    return {"db_path": str(db_path), "input_path": str(input_path)}


def _write_test_data_to_files(input_dir: str):
    """将 vector_test_data.py 中的数据写入到临时文件中。"""
    data_map = {
        "simple.md": test_data.VECTOR_TEST_SIMPLE_MD,
        "simple.txt": test_data.VECTOR_TEST_SIMPLE_TXT,
        "simple.json": test_data.VECTOR_TEST_SIMPLE_JSON,
        "character_info.md": test_data.VECTOR_TEST_CHARACTER_INFO,
        "worldview.md": test_data.VECTOR_TEST_WORLDVIEW,
        "table_data.md": test_data.VECTOR_TEST_TABLE_DATA,
        "structured.json": test_data.VECTOR_TEST_STRUCTURED_JSON,
        "multi_paragraph.md": test_data.VECTOR_TEST_MULTI_PARAGRAPH,
        "diagram.md": test_data.VECTOR_TEST_DIAGRAM_CONTENT,
        "nested_list.md": test_data.VECTOR_TEST_NESTED_LIST,
        "special_chars.md": test_data.VECTOR_TEST_SPECIAL_CHARS,
        "novel_worldview.md": test_data.VECTOR_TEST_NOVEL_WORLDVIEW,
        "novel_characters.json": test_data.VECTOR_TEST_NOVEL_CHARACTERS,
        "novel_plot.md": test_data.VECTOR_TEST_NOVEL_PLOT_ARC,
        "report_outline.md": test_data.VECTOR_TEST_REPORT_OUTLINE,
        "report_market.json": test_data.VECTOR_TEST_REPORT_MARKET_DATA,
        "complex_markdown.md": test_data.VECTOR_TEST_COMPLEX_MARKDOWN,
    }
    for filename, content in data_map.items():
        if content and content.strip():
            (Path(input_dir) / filename).write_text(content, encoding='utf-8')


@pytest.fixture(scope="module")
def ingested_store(test_dirs):
    """
    准备一个已灌入所有测试数据的 VectorStore 实例。
    这个 fixture 的作用域是 module，意味着每个测试文件只会执行一次数据灌入。
    """
    _write_test_data_to_files(test_dirs["input_path"])
    vector_store = get_vector_store(db_path=test_dirs["db_path"], collection_name="test_collection")
    vector_add_from_dir(vector_store, test_dirs["input_path"], default_file_metadata)
    return vector_store