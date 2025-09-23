import os
import sys
import pytest
from loguru import logger
import asyncio
import ast
from typing import List, Tuple
from llama_index.core import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm import llm_completion, get_llm_params, get_llm_messages, llm_temperatures
from prompts.story.kg import kg_extraction_prompt_design, kg_extraction_prompt_write
from utils.kg import _get_kg_node_parser
from tests.test_data import (
    VECTOR_TEST_NOVEL_PLOT_ARC,
    VECTOR_TEST_NOVEL_CHAPTER,
    VECTOR_TEST_NOVEL_FACTIONS,
    VECTOR_TEST_NOVEL_MAGIC_SYSTEM,
)


def _validate_triplets(result_str: str) -> List[Tuple[str, str, str]]:
    """
    验证并解析LLM返回的三元组结果, 增加更健壮的错误处理
    """
    if not result_str:
        logger.warning("LLM返回内容为空")
        return []
        
    cleaned_str = result_str.strip()
    # 处理代码块标记
    if cleaned_str.startswith("```python"):
        cleaned_str = cleaned_str[len("```python"):].strip()
    if cleaned_str.startswith("```"):
        cleaned_str = cleaned_str[3:].strip()
    if cleaned_str.endswith("```"):
        cleaned_str = cleaned_str[:-3].strip()
    
    # 检查是否为列表格式
    if not (cleaned_str.startswith('[') and cleaned_str.endswith(']')):
        logger.error(f"返回内容不是一个有效的列表字符串: {cleaned_str}")
        # 尝试修复格式, 添加方括号
        try:
            cleaned_str = f"[{cleaned_str}]"
            logger.warning("已尝试自动修复格式")
        except:
            pytest.fail(f"无法将LLM输出解析为Python列表: {cleaned_str}")
    
    try:
        triplets = ast.literal_eval(cleaned_str)
    except (ValueError, SyntaxError) as e:
        logger.error(f"解析错误: {e}\n输出内容: {cleaned_str}")
        pytest.fail(f"无法将LLM输出解析为Python列表: {e}\n输出内容: {cleaned_str}")
    
    if not isinstance(triplets, list):
        pytest.fail("解析结果不是一个列表")
        
    if not triplets:
        logger.warning("LLM返回了一个空列表, 这可能是有效的, 但请检查原文是否真的不含三元组。")
        return []
    
    # 验证每个三元组
    validated_triplets = []
    for i, triplet in enumerate(triplets):
        try:
            assert isinstance(triplet, tuple), f"列表中的元素不是元组: {triplet}"
            assert len(triplet) == 3, f"元组长度不为3: {triplet}"
            assert all(isinstance(item, str) for item in triplet), f"元组中的元素不全是字符串: {triplet}"
            validated_triplets.append(triplet)
        except AssertionError as e:
            logger.warning(f"跳过无效的三元组 (索引 {i}): {e}")
    
    return validated_triplets


async def _extract_triplets_from_chunks(
    text: str,
    content_format: str,
    llm_group: str,
    prompt_template: str,
) -> List[Tuple[str, str, str]]:
    """
    模拟真实的提取过程: 先将文本分割成块, 然后从每个块中提取三元组。
    """
    # 1. 将文本分割成节点 (块)
    doc = Document(text=text)
    parser = _get_kg_node_parser(content_format, len(text))
    nodes = parser.get_nodes_from_documents([doc])

    if not nodes:
        logger.warning(f"文本未能解析出任何节点 (chunks), 无法提取。")
        return []

    logger.info(f"文本被分割为 {len(nodes)} 个节点 (chunks) 进行处理。")

    # 2. 并行从每个节点提取三元组
    llm_params = get_llm_params(llm_group=llm_group, temperature=llm_temperatures["classification"])

    async def extract_from_node(node):
        context_dict = {"text": node.get_content()}
        messages = get_llm_messages(
            system_prompt=prompt_template,
            context_dict_system=context_dict
        )
        node_llm_params = llm_params.copy()
        node_llm_params["messages"] = messages

        try:
            response = await llm_completion(node_llm_params)
            return _validate_triplets(response.content)
        except Exception as e:
            logger.error(f"从节点提取三元组时出错: {e}")
            return []

    tasks = [extract_from_node(node) for node in nodes]
    results_from_chunks = await asyncio.gather(*tasks)

    # 3. 将所有块的结果合并成一个列表
    all_triplets = [triplet for sublist in results_from_chunks for triplet in sublist]

    return all_triplets


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, text_content, expected_entities",
    [
        ("plot_arc", VECTOR_TEST_NOVEL_PLOT_ARC, ["龙傲天", "叶良辰", "海图残卷", "赵日天"]),
        ("factions", VECTOR_TEST_NOVEL_FACTIONS, ["青云宗", "北冥魔殿", "天剑阁", "天机阁"]),
        ("magic_system", VECTOR_TEST_NOVEL_MAGIC_SYSTEM, ["鸿蒙道体", "御水决"]),
    ],
    ids=["plot_arc", "factions", "magic_system"]
)
async def test_kg_extraction_design_from_chunks(llm_group, test_id, text_content, expected_entities):
    """
    测试在模拟真实分块流程下, 使用 design 提示词提取三元组。
    """
    logger.info(f"--- 测试三元组提取 (Design) | 模型组: {llm_group} | 用例: {test_id} ---")

    triplets = await _extract_triplets_from_chunks(
        text=text_content,
        content_format="md",
        llm_group=llm_group,
        prompt_template=kg_extraction_prompt_design
    )

    logger.info(f"模型组 '{llm_group}' 从用例 '{test_id}' 中提取了 {len(triplets)} 个三元组:\n{triplets}")

    assert len(triplets) > 0, f"模型组 '{llm_group}' 未能从用例 '{test_id}' 中提取出任何三元组"

    all_entities = {item for t in triplets for item in (t[0], t[2])}
    for entity in expected_entities:
        assert entity in all_entities, f"应从 '{test_id}' 提取出实体 '{entity}', 但在提取出的实体中未找到: {all_entities}"

    logger.success(f"--- 三元组提取 (Design) 测试通过 | 模型组: {llm_group}, 用例: {test_id} ---")


@pytest.mark.asyncio
async def test_kg_extraction_write_from_chunks(llm_group):
    """
    测试在模拟真实分块流程下, 使用 write 提示词从小说正文中提取三元组。
    """
    logger.info(f"--- 测试三元组提取 (Write) | 模型组: {llm_group} ---")

    triplets = await _extract_triplets_from_chunks(
        text=VECTOR_TEST_NOVEL_CHAPTER,
        content_format="md",
        llm_group=llm_group,
        prompt_template=kg_extraction_prompt_write
    )

    logger.info(f"模型组 '{llm_group}' 从小说正文中提取了 {len(triplets)} 个三元组:\n{triplets}")

    assert len(triplets) > 0, f"模型组 '{llm_group}' 未能从小说正文中提取出任何三元组"

    all_subjects = {t[0] for t in triplets}
    assert "龙傲天" in all_subjects, "应提取出主语 '龙傲天'"
    assert "赵日天" in all_subjects, "应提取出主语 '赵日天'"

    relations = {t[1] for t in triplets}
    assert any("拦住" in r or "轰出" in r or "拍了拍" in r for r in relations), "应捕捉到角色间的动态交互关系"

    logger.success(f"--- 三元组提取 (Write) 测试通过 | 模型组: {llm_group} ---")


@pytest.mark.asyncio
async def test_kg_extraction_empty_input(llm_group):
    """测试处理空输入的情况。"""
    logger.info(f"--- 测试空输入三元组提取 (模型组: {llm_group}) ---")

    triplets = await _extract_triplets_from_chunks(
        text="",
        content_format="txt",
        llm_group=llm_group,
        prompt_template=kg_extraction_prompt_write
    )

    assert len(triplets) == 0, f"空输入应返回空列表, 实际返回 {len(triplets)} 个三元组"

    logger.success(f"--- 空输入三元组提取测试通过 (模型组: {llm_group}) ---")


@pytest.mark.asyncio
async def test_kg_extraction_consistency(llm_group):
    """
    测试多次对相同内容进行分块提取时结果的一致性。
    """
    logger.info(f"--- 测试三元组提取一致性 (模型组: {llm_group}) ---")

    text_content = VECTOR_TEST_NOVEL_CHAPTER

    # 执行多次提取
    results_sets = []
    for i in range(3):
        logger.info(f"一致性测试, 第 {i+1}/3 次运行...")
        triplets = await _extract_triplets_from_chunks(
            text=text_content,
            content_format="md",
            llm_group=llm_group,
            prompt_template=kg_extraction_prompt_write
        )
        results_sets.append(set(triplets))

    # 计算结果的交集大小, 评估一致性
    if not results_sets:
        pytest.fail("所有提取运行均未返回任何结果。")

    common_triplets = set.intersection(*results_sets)
    avg_triplets = sum(len(r) for r in results_sets) / len(results_sets)

    logger.info(f"一致性测试结果: 平均提取 {avg_triplets:.1f} 个三元组, 共同三元组 {len(common_triplets)} 个")

    if avg_triplets == 0:
        logger.warning("平均提取的三元组数量为0, 无法评估一致性, 但测试通过。")
        return

    # 确保至少有一定数量的共同三元组
    min_common_ratio = 0.5  # 至少50%的三元组应该一致
    consistency_ratio = len(common_triplets) / avg_triplets if avg_triplets > 0 else 1.0

    assert consistency_ratio >= min_common_ratio, \
        f"提取结果一致性不足: 共同三元组比例 ({consistency_ratio:.2f}) < 最小要求 ({min_common_ratio})"

    logger.success(f"--- 三元组提取一致性测试通过 (模型组: {llm_group}) ---")