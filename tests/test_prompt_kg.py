import os
import sys
import pytest
from loguru import logger
import ast
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.llm import get_llm_messages, llm_completion, get_llm_params
from story.prompts.kg import kg_extraction_prompt_design, kg_extraction_prompt_write
from tests.test_data import VECTOR_TEST_COMPLEX_MARKDOWN
from tests.test_prompt_summary import LONG_TEXT_TO_SUMMARIZE


@pytest.mark.asyncio
async def test_kg_extraction_design_prompt():
    logger.info("--- 测试 KG Extraction (Design) ---")
    context = {
        "text": VECTOR_TEST_COMPLEX_MARKDOWN
    }
    messages = [
        {"role": "user", "content": kg_extraction_prompt_design.format(**context)}
    ]
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.0,
        messages=messages
    )
    response = await llm_completion(llm_params)
    triplets_str = response.content
    logger.info(f"LLM 生成的三元组 (Design):\n{triplets_str}")

    try:
        triplets: List[Tuple[str, str, str]] = ast.literal_eval(triplets_str)
    except (ValueError, SyntaxError) as e:
        pytest.fail(f"无法将输出解析为Python列表: {e}\n输出内容: {triplets_str}")

    assert isinstance(triplets, list), "输出应为列表"
    if triplets:
        assert isinstance(triplets[0], tuple), "列表元素应为元组"
        assert len(triplets[0]) == 3, "元组应包含3个元素"

    # 检查是否提取出了一些关键关系
    found_character_relation = any("龙傲天" in t for t in triplets)
    found_faction_relation = any("青云宗" in t for t in triplets)
    found_worldview_relation = any("九霄大陆" in t for t in triplets)

    assert found_character_relation, "应提取到与'龙傲天'相关的三元组"
    assert found_faction_relation, "应提取到与'青云宗'相关的三元组"
    assert found_worldview_relation, "应提取到与'九霄大陆'相关的三元组"


@pytest.mark.asyncio
async def test_kg_extraction_write_prompt():
    logger.info("--- 测试 KG Extraction (Write) ---")
    context = {
        "text": LONG_TEXT_TO_SUMMARIZE
    }
    messages = [
        {"role": "user", "content": kg_extraction_prompt_write.format(**context)}
    ]
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.0,
        messages=messages
    )
    response = await llm_completion(llm_params)
    triplets_str = response.content
    logger.info(f"LLM 生成的三元组 (Write):\n{triplets_str}")

    try:
        triplets: List[Tuple[str, str, str]] = ast.literal_eval(triplets_str)
    except (ValueError, SyntaxError) as e:
        pytest.fail(f"无法将输出解析为Python列表: {e}\n输出内容: {triplets_str}")

    assert isinstance(triplets, list), "输出应为列表"
    if triplets:
        assert isinstance(triplets[0], tuple), "列表元素应为元组"
        assert len(triplets[0]) == 3, "元组应包含3个元素"

    # 检查是否提取出了一些关键的状态变化和因果关系
    found_status_change = any(t[1] == "状态变为" and t[2] == "重伤" for t in triplets)
    found_action = any("撤" in t[2] for t in triplets)
    found_goal = any(t[1] == "形成目标" and "复仇" in t[2] for t in triplets)

    assert found_status_change, "应提取到'赵日天'状态变为'重伤'的三元组"
    assert found_action, "应提取到黑衣人'撤退'的行动"
    assert found_goal, "应提取到龙傲天'为友复仇'的目标"