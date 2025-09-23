import os
import sys
import pytest
from loguru import logger
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import Task
from utils.llm import get_llm_messages, llm_completion, get_llm_params, llm_temperatures
from prompts.story.atom_write import system_prompt, user_prompt


class AtomWriteOutput(BaseModel):
    reasoning: str
    update_goal: Optional[str] = None
    update_instructions: Optional[List[str]] = None
    update_input_brief: Optional[List[str]] = None
    update_constraints: Optional[List[str]] = None
    update_acceptance_criteria: Optional[List[str]] = None
    atom_result: Literal["atom", "complex"]
    complex_reasons: Optional[List[str]] = None

    @field_validator('complex_reasons')
    def check_complex_reasons(cls, v, info):
        if info.data.get('atom_result') == 'complex' and not v:
            raise ValueError("当 atom_result 为 'complex' 时, complex_reasons 必须提供")
        if info.data.get('atom_result') == 'atom' and v:
            raise ValueError("当 atom_result 为 'atom' 时, complex_reasons 必须省略")
        return v


task_atom = Task(
    id="1.1.1",
    parent_id="1.1",
    task_type="write",
    hierarchical_position="第一章",
    goal="续写龙傲天与赵日天在黑松林遭遇埋伏后的情节。",
    length="1500",
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_atom",
)

context_atom = {
    "task": task_atom.model_dump_json(indent=2),
    "dependent_design": """
# 章节设计: 黑松林之变后续
- 核心事件: 龙傲天检查敌人留下的令牌, 发现了一个神秘符号, 决心查明真相。
- 角色动态: 龙傲天的心态从暴怒转为冷静和坚定, 确立为友复仇的新目标。赵日天重伤昏迷。
- 场景氛围: 战斗后的寂静, 月光下林间的肃杀, 龙傲天内心的焦急与冰冷。
- 结尾钩子: 龙傲天决定先带赵日天回城治疗, 再调查令牌的来历, 为下一章铺垫。
""",
    "dependent_search": "",
    "text_latest": "黑衣人头领见状, 眼中闪过一丝惊异, 随即果断下令: “撤！”黑影们迅速消失在密林深处, 只在地上留下了一枚漆黑的玄铁令牌...一个全新的目标在他心中形成: 查明真相, 为友复仇。",
    "task_list": "1.1.1 第一章 write 续写龙傲天与赵日天在黑松林遭遇埋伏后的情节。 1500"
}

expected_atom = {
    "atom_result": "atom",
    "has_updates": True
}


task_complex_length = Task(
    id="1.2",
    parent_id="1",
    task_type="write",
    hierarchical_position="第一卷",
    goal="撰写龙傲天在东海的全部冒险经历。",
    length="30000",
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_complex_len",
)
context_complex_length = {
    "task": task_complex_length.model_dump_json(indent=2),
    "dependent_design": "# 第一卷: 东海风云 - 结构规划\n- 第一幕: 初入东海 (约10000字)\n- 第二幕: 寻药风波 (约10000字)\n- 第三幕: 宗门大比 (约10000字)",
    "dependent_search": "",
    "text_latest": "龙傲天告别了家乡, 踏上了前往东海的征途。",
    "task_list": "1.2 第一卷 write 撰写龙傲天在东海的全部冒险经历。 30000"
}
expected_complex_length = {
    "atom_result": "complex",
    "complex_reasons": ["length_excessive"]
}


task_complex_design = Task(
    id="1.3.1",
    parent_id="1.3",
    task_type="write",
    hierarchical_position="第三章",
    goal="描写龙傲天探索神秘洞穴。",
    length="1800",
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_complex_design",
)
context_complex_design = {
    "task": task_complex_design.model_dump_json(indent=2),
    "dependent_design": "", # 无设计方案
    "dependent_search": "",
    "text_latest": "龙傲天在黑松林的深处, 发现了一个深不见底、散发着微光的洞穴入口。",
    "task_list": "1.3.1 第三章 write 描写龙傲天探索神秘洞穴。 1800"
}
expected_complex_design = {
    "atom_result": "complex",
    "complex_reasons": ["design_insufficient"]
}


task_complex_both = Task(
    id="2",
    parent_id="root",
    task_type="write",
    hierarchical_position="第二卷",
    goal="撰写龙傲天在中央神州的崛起之路。",
    length="100000",
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_complex_both",
)
context_complex_both = {
    "task": task_complex_both.model_dump_json(indent=2),
    "dependent_design": "", # 无设计方案
    "dependent_search": "",
    "text_latest": "龙傲天踏上了前往中央神州的传送阵, 眼前白光一闪, 已是换了人间。",
    "task_list": "2 第二卷 write 撰写龙傲天在中央神州的崛起之路。 100000"
}
expected_complex_both = {
    "atom_result": "complex",
    "complex_reasons": ["length_excessive", "design_insufficient"]
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, task, context, expected",
    [
        ("atom_task", task_atom, context_atom, expected_atom),
        ("complex_length", task_complex_length, context_complex_length, expected_complex_length),
        ("complex_design", task_complex_design, context_complex_design, expected_complex_design),
        ("complex_both", task_complex_both, context_complex_both, expected_complex_both),
    ],
    ids=["atom_task", "complex_length_excessive", "complex_design_insufficient", "complex_both_reasons"]
)
async def test_atom_write_prompt(test_id, task, context, expected):
    """
    测试 atom_write 提示词在不同场景下的分析和判定能力。
    """
    logger.info(f"--- 测试 atom_write 提示词 | 用例: {test_id} ---")

    # 1. 准备消息
    messages = get_llm_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_dict_user=context
    )

    # 2. 获取 LLM 参数
    llm_params = get_llm_params(
        llm_group="reasoning", # 使用擅长遵循指令的模型
        temperature=llm_temperatures["classification"]
    )
    llm_params["messages"] = messages

    # 3. 调用 LLM 并进行 Pydantic 验证
    response = await llm_completion(llm_params, response_model=AtomWriteOutput)
    result = response.validated_data
    
    logger.info(f"LLM 输出 (用例: {test_id}):\n{result.model_dump_json(indent=2, ensure_ascii=False)}")

    # 4. 断言验证
    assert result is not None, "Pydantic 模型验证不应返回 None"
    
    # 验证 atom_result
    assert result.atom_result == expected["atom_result"], \
        f"预期 atom_result 为 '{expected['atom_result']}', 实际为 '{result.atom_result}'"

    if result.atom_result == "complex":
        # 验证 complex_reasons
        assert result.complex_reasons is not None, "complex_reasons 不应为 None"
        assert sorted(result.complex_reasons) == sorted(expected["complex_reasons"]), \
            f"预期 complex_reasons 为 {expected['complex_reasons']}, 实际为 {result.complex_reasons}"
    
    if result.atom_result == "atom":
        # 验证 atom 任务是否进行了更新
        has_updates = any([
            result.update_goal,
            result.update_instructions,
            result.update_input_brief,
            result.update_constraints,
            result.update_acceptance_criteria
        ])
        assert has_updates == expected.get("has_updates", False), \
            "对于 atom 任务, 预期应有更新字段, 但未找到。"
        if has_updates and result.update_goal:
            assert task.goal not in result.update_goal, "update_goal 应该比原始 goal 更具体"

    logger.success(f"--- atom_write 提示词测试通过 | 用例: {test_id} ---")