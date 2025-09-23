import pytest
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
def test_dirs(tmp_path_factory) -> dict:
    """为所有测试创建统一的临时目录。"""
    base_dir = tmp_path_factory.mktemp(".tests_workspace")
    vector_db_path = base_dir / ".chroma_db"
    kg_db_path = base_dir / ".kuzu_db"
    input_path = base_dir / ".input_data"

    # 创建所有需要的目录
    vector_db_path.mkdir()
    kg_db_path.mkdir()
    input_path.mkdir()

    logger.info(f"创建统一测试临时目录: {base_dir}")

    paths = {
        "vector_db_path": str(vector_db_path),
        "kg_db_path": str(kg_db_path),
        "input_path": str(input_path),
    }

    yield paths

    logger.info(f"统一测试临时目录 {base_dir} 将被 pytest 自动清理。")


@pytest.fixture(scope="module")
def input_dir_with_test_files(test_dirs) -> str:
    """在模块级别的临时目录中创建并填充测试文件。"""
    input_path = test_dirs["input_path"]
    logger.info(f"为测试模块填充测试文件于: {input_path}")
    _write_test_data_to_files(input_path)
    return input_path


def _write_test_data_to_files(input_dir: str):
    data_map = {
        "simple.md": test_data.VECTOR_TEST_SIMPLE_MD,
        "simple.txt": test_data.VECTOR_TEST_SIMPLE_TXT,
        "simple.json": test_data.VECTOR_TEST_SIMPLE_JSON,
        "character_info.md": test_data.VECTOR_TEST_CHARACTER_INFO,
        "worldview.md": test_data.VECTOR_TEST_WORLDVIEW,
        "table_data.md": test_data.VECTOR_TEST_TABLE_DATA,
        "large_table_data.md": test_data.VECTOR_TEST_LARGE_TABLE_DATA,
        "structured.json": test_data.VECTOR_TEST_STRUCTURED_JSON,
        "multi_paragraph.md": test_data.VECTOR_TEST_MULTI_PARAGRAPH,
        "diagram.md": test_data.VECTOR_TEST_DIAGRAM_CONTENT,
        "complex_mermaid_diagram.md": test_data.VECTOR_TEST_COMPLEX_MERMAID_DIAGRAM,
        "composite_structure.md": test_data.VECTOR_TEST_COMPOSITE_STRUCTURE,
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


def get_all_test_data_params():
    """为数据覆盖率测试提供统一的参数。"""
    # (test_id, content, content_format, expect_success)
    params = [
        # 1. 基础与边缘用例
        ("empty", test_data.VECTOR_TEST_EMPTY, "text", False),
        ("simple_txt", test_data.VECTOR_TEST_SIMPLE_TXT, "text", True),
        ("simple_cn", test_data.VECTOR_TEST_SIMPLE_CN, "text", True),
        ("simple_md", test_data.VECTOR_TEST_SIMPLE_MD, "md", True),
        ("simple_json", test_data.VECTOR_TEST_SIMPLE_JSON, "json", True),
        ("mixed_lang", test_data.VECTOR_TEST_MIXED_LANG, "md", True),
        # 2. 结构化内容
        ("table_data", test_data.VECTOR_TEST_TABLE_DATA, "md", True),
        ("large_table_data", test_data.VECTOR_TEST_LARGE_TABLE_DATA, "md", True),
        ("nested_list", test_data.VECTOR_TEST_NESTED_LIST, "md", True),
        ("structured_json", test_data.VECTOR_TEST_STRUCTURED_JSON, "json", True),
        ("deep_hierarchy_json", test_data.VECTOR_TEST_DEEP_HIERARCHY_JSON, "json", True),
        ("multi_paragraph", test_data.VECTOR_TEST_MULTI_PARAGRAPH, "md", True),
        ("complex_markdown", test_data.VECTOR_TEST_COMPLEX_MARKDOWN, "md", True),
        ("novel_structured_info", test_data.VECTOR_TEST_NOVEL_STRUCTURED_INFO, "md", True),
        ("conversational_log", test_data.VECTOR_TEST_CONVERSATIONAL_LOG, "md", True),
        ("philosophical_text", test_data.VECTOR_TEST_PHILOSOPHICAL_TEXT, "md", True),
        ("composite_structure", test_data.VECTOR_TEST_COMPOSITE_STRUCTURE, "md", True),
        # 3. 特殊格式与代码块
        ("diagram_content", test_data.VECTOR_TEST_DIAGRAM_CONTENT, "md", True),
        ("complex_mermaid_diagram", test_data.VECTOR_TEST_COMPLEX_MERMAID_DIAGRAM, "md", True),
        ("special_chars", test_data.VECTOR_TEST_SPECIAL_CHARS, "md", True),
        ("md_with_code_block", test_data.VECTOR_TEST_MD_WITH_CODE_BLOCK, "md", True),
        ("json_with_code_block", test_data.VECTOR_TEST_JSON_WITH_CODE_BLOCK, "json", True),
        ("md_with_complex_json_code_block", test_data.VECTOR_TEST_MD_WITH_COMPLEX_JSON_CODE_BLOCK, "md", True),
        # 4. 领域场景: 小说创作
        ("character_info", test_data.VECTOR_TEST_CHARACTER_INFO, "md", True),
        ("worldview", test_data.VECTOR_TEST_WORLDVIEW, "md", True),
        ("novel_worldview", test_data.VECTOR_TEST_NOVEL_WORLDVIEW, "md", True),
        ("novel_characters", test_data.VECTOR_TEST_NOVEL_CHARACTERS, "json", True),
        ("novel_plot_arc", test_data.VECTOR_TEST_NOVEL_PLOT_ARC, "md", True),
        ("novel_magic_system", test_data.VECTOR_TEST_NOVEL_MAGIC_SYSTEM, "md", True),
        ("novel_factions", test_data.VECTOR_TEST_NOVEL_FACTIONS, "md", True),
        ("novel_chapter", test_data.VECTOR_TEST_NOVEL_CHAPTER, "md", True),
        ("novel_summary", test_data.VECTOR_TEST_NOVEL_SUMMARY, "md", True),
        ("novel_full_outline", test_data.VECTOR_TEST_NOVEL_FULL_OUTLINE, "md", True),
        # 5. 领域场景: 报告撰写
        ("report_outline", test_data.VECTOR_TEST_REPORT_OUTLINE, "md", True),
        ("detailed_report_outline", test_data.VECTOR_TEST_DETAILED_REPORT_OUTLINE, "md", True),
        ("report_market_data", test_data.VECTOR_TEST_REPORT_MARKET_DATA, "json", True),
        ("report_tech_trends", test_data.VECTOR_TEST_REPORT_TECH_TRENDS, "md", True),
        ("report_case_study", test_data.VECTOR_TEST_REPORT_CASE_STUDY, "md", True),
        # 6. 领域场景: 技术文档
        ("technical_book_chapter", test_data.VECTOR_TEST_TECHNICAL_BOOK_CHAPTER, "md", True),
    ]
    ids = [p[0] for p in params]
    return {"params": params, "ids": ids}
