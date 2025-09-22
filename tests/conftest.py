import pytest
import sys
import os
import logging
from pathlib import Path
import nest_asyncio
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.log import init_logger
from utils.file import log_dir
from utils.vector import init_llama_settings, get_vector_store, vector_add_from_dir, file_metadata_default
from tests import test_data


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings", "ignore:open_text is deprecated:DeprecationWarning:litellm.*"
    )
    config.addinivalue_line(
        "filterwarnings", "ignore:Pydantic serializer warnings:UserWarning"
    )


def pytest_generate_tests(metafunc):
    """为所有需要 `llm_group` 参数的测试自动进行参数化。"""
    if "llm_group" in metafunc.fixturenames:
        # MODEL_GROUPS = ["fast", "summary", "reasoning"]
        MODEL_GROUPS = ["summary"]
        metafunc.parametrize("llm_group", MODEL_GROUPS)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    nest_asyncio.apply()
    logging.getLogger("litellm").setLevel(logging.WARNING)
    init_llama_settings()


@pytest.fixture(scope="module", autouse=True)
def setup_module_logging(request):
    log_filename_stem = Path(request.module.__file__).stem
    log_file = log_dir / f"{log_filename_stem}.log"
    if log_file.exists():
        log_file.unlink()
    sink_id = init_logger(log_filename_stem)
    yield
    logger.remove(sink_id)


@pytest.fixture(scope="module")
def test_dirs(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("vector_tests")
    db_path = base_dir / "chroma_db"
    input_path = base_dir / "input_data"
    input_path.mkdir()
    return {"db_path": str(db_path), "input_path": str(input_path)}


def _write_test_data_to_files(input_dir: str):
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
