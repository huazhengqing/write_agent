import pytest
import json
from unittest.mock import AsyncMock, patch

from utils.models import Task, PlanOutput, convert_plan_to_tasks
from utils.llm import get_llm_messages, llm_completion
from utils.loader import load_prompts


system_prompt, user_prompt = load_prompts("story", "plan_write", "system_prompt", "user_prompt")


base_task = Task(
    id="1",
    parent_id="root",
    task_type="write",
    hierarchical_position="全书",
    goal="写一部关于赛博修仙的百万字长篇小说",
    length="1000000",
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_123",
    results={
        "atom_result": "complex",
        "complex_reasons": ["design_insufficient"],
        "atom_reasoning": "任务目标宏大，篇幅长，且缺乏具体设计，需要先进行规划。"
    }
)


# 场景1: 设计阶段 (无结构化设计方案)
design_phase_context = {
    "task": base_task.model_dump_json(indent=2),
    "complex_reasons": base_task.results["complex_reasons"],
    "atom_reasoning": base_task.results["atom_reasoning"],
    "dependent_design": "", # 关键：没有设计方案
    "dependent_search": "",
    "text_latest": "",
    "text_summary": "",
    "task_list": "1 全书 write 写一部关于赛博修仙的百万字长篇小说 1000000",
    "upper_design": "",
    "upper_search": ""
}


# 模拟LLM在设计阶段的输出
mock_design_phase_llm_output = PlanOutput(
    id="1",
    task_type="write",
    goal="写一部关于赛博修仙的百万字长篇小说",
    length="1000000",
    hierarchical_position="全书",
    reasoning="当前为顶层写作任务，且无具体设计方案，需要进入设计阶段。根据规划维度，将任务分解为核心概念、主角、世界观、力量体系等设计任务，并创建一个占位写作任务。",
    sub_tasks=[
        {
            "id": "1.1",
            "task_type": "design",
            "goal": "核心概念设计: 定义'赛博修仙'的核心世界观钩子和独特卖点。",
            "hierarchical_position": "全书",
            "instructions": ["明确科技与修仙的结合方式。", "设计1-3个吸引读者的核心创意。"],
            "dependency": [],
            "sub_tasks": []
        },
        {
            "id": "1.2",
            "task_type": "design",
            "goal": "主角设计: 规划主角的背景、核心动机和成长弧光。",
            "hierarchical_position": "全书",
            "dependency": ["1.1"],
            "sub_tasks": []
        },
        {
            "id": "1.3",
            "task_type": "design",
            "goal": "力量体系设计: 创建'赛博灵根'和'数据洞天'的力量体系规则档案。",
            "hierarchical_position": "全书",
            "dependency": ["1.1"],
            "sub_tasks": []
        },
        {
            "id": "1.4",
            "task_type": "write",
            "goal": "[占位写作任务]: 根据所有同层级设计成果, 继承父任务'写一部关于赛博修仙的百万字长篇小说'的目标进行写作。",
            "hierarchical_position": "全书",
            "dependency": ["1.1", "1.2", "1.3"],
            "length": "1000000",
            "sub_tasks": []
        }
    ]
)


# 场景2: 分解阶段 (有结构化设计方案)
decomposition_task = base_task.model_copy(deep=True)
decomposition_task.id = "1.5"
decomposition_task.parent_id = "1"
decomposition_task.goal = "撰写第一幕：主角觉醒"
decomposition_task.length = "15000"
decomposition_task.hierarchical_position = "第一卷"
decomposition_task.results["complex_reasons"] = ["length_excessive"]
decomposition_task.results["atom_reasoning"] = "篇幅超过2000字，需要分解为更小的章节。"


decomposition_phase_context = {
    "task": decomposition_task.model_dump_json(indent=2),
    "complex_reasons": decomposition_task.results["complex_reasons"],
    "atom_reasoning": decomposition_task.results["atom_reasoning"],
    "dependent_design": """
# 第一幕：主角觉醒 结构规划
- **第一章：废柴的日常** (3000字)
  - 核心事件：主角林风在底层数据矿场被工头欺压，展现其生活的窘迫和对力量的渴望。
  - 结尾钩子：林风意外获得一块古老的生物芯片。
- **第二章：神秘芯片** (4000字)
  - 核心事件：林风研究芯片，被电击昏迷。在意识海中，他见到了芯片中的残魂“玄机子”。
  - 关键信息：玄机子揭示“赛博修仙”的可能性。
- **第三章：初窥门径** (4000字)
  - 核心事件：在玄机子指导下，林风成功将一缕灵力数据化，修复了自己受损的义体。
  - 角色动态：林风重拾信心，目标从“生存”变为“变强”。
- **第四章：新的危机** (4000字)
  - 核心事件：工头发现林风的异常，试图抢夺芯片，冲突爆发。
  - 结尾钩子：林风利用初学的能力击倒工头，但引来了矿场背后更大的势力“黑石集团”的注意。
""",
    "dependent_search": "",
    "text_latest": "林风拖着疲惫的义体回到狭窄的住处，空气中弥漫着机油和铁锈的味道。",
    "text_summary": "主角林风是一个生活在社会底层的义体改造者。",
    "task_list": "1.5 第一卷 write 撰写第一幕：主角觉醒 15000",
    "upper_design": "主角林风将通过一块神秘芯片走上赛博修仙之路。",
    "upper_search": ""
}

# 模拟LLM在分解阶段的输出
mock_decomposition_phase_llm_output = PlanOutput(
    id="1.5",
    task_type="write",
    goal="撰写第一幕：主角觉醒",
    length="15000",
    hierarchical_position="第一卷",
    reasoning="检测到设计方案中包含结构化规划，进入分解阶段。将第一幕的四个章节分别映射为四个独立的write子任务。",
    sub_tasks=[
        {
            "id": "1.5.1",
            "task_type": "write",
            "goal": "第一章：废柴的日常: 续写主角林风在数据矿场被欺压，并意外获得生物芯片的情节。",
            "hierarchical_position": "第一章",
            "length": "3000",
            "instructions": ["重点描写底层生活的窘迫感。", "刻画工头的蛮横与主角对力量的渴望。", "设计获得芯片的意外性和神秘感。"],
            "input_brief": ["参考`设计方案`中第一章的核心事件和结尾钩子。", "从`最新章节`的场景无缝衔接。"],
            "constraints": ["避免过早揭示芯片的全部功能。"],
            "acceptance_criteria": ["成功塑造主角的初始困境。", "结尾的芯片获取情节能够引发读者好奇心。"],
            "dependency": [],
            "sub_tasks": []
        },
        {
            "id": "1.5.2",
            "task_type": "write",
            "goal": "第二章：神秘芯片: 续写林风研究芯片并遇见残魂“玄机子”的情节。",
            "hierarchical_position": "第二章",
            "length": "4000",
            "dependency": [],
            "sub_tasks": []
        },
        {
            "id": "1.5.3",
            "task_type": "write",
            "goal": "第三章：初窥门径: 续写林风在指导下将灵力数据化，并修复义体的情节。",
            "hierarchical_position": "第三章",
            "length": "4000",
            "dependency": [],
            "sub_tasks": []
        },
        {
            "id": "1.5.4",
            "task_type": "write",
            "goal": "第四章：新的危机: 续写林风与工头冲突，并引来更大势力注意的情节。",
            "hierarchical_position": "第四章",
            "length": "4000", # 15000 - 3000 - 4000 - 4000 = 4000
            "dependency": [],
            "sub_tasks": []
        }
    ]
)


@pytest.mark.asyncio
@patch('agents.plan.llm_completion', new_callable=AsyncMock)
async def test_plan_write_design_phase(mock_llm_completion):
    """
    测试 plan_write 在“设计阶段”的分解能力。
    - 输入: 任务复杂，但 `dependent_design` 为空。
    - 预期: 分解出多个 `design`/`search` 子任务和一个占位的 `write` 子任务。
    """
    # 模拟 llm_completion 返回预设的 PlanOutput
    mock_message = AsyncMock()
    mock_message.validated_data = mock_design_phase_llm_output
    mock_message.get.return_value = mock_design_phase_llm_output.reasoning
    mock_llm_completion.return_value = mock_message

    # 准备输入
    task_obj = base_task
    context = design_phase_context
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    
    # 调用被测试逻辑 (这里我们直接调用 llm_completion 来模拟 agent 中的调用)
    response_message = await llm_completion(
        llm_params={"messages": messages}, 
        response_model=PlanOutput
    )
    
    # 验证结果
    result_plan = response_message.validated_data
    assert result_plan is not None
    assert result_plan.reasoning is not None
    assert len(result_plan.sub_tasks) == 4

    # 检查子任务类型
    sub_task_types = [st.task_type for st in result_plan.sub_tasks]
    assert sub_task_types.count("design") == 3
    assert sub_task_types.count("write") == 1

    # 检查占位 write 任务
    write_task = next((st for st in result_plan.sub_tasks if st.task_type == 'write'), None)
    assert write_task is not None
    assert write_task.id == "1.4"
    assert "[占位写作任务]" in write_task.goal
    assert write_task.length == "1000000"
    assert set(write_task.dependency) == {"1.1", "1.2", "1.3"}

    # 检查 design 任务
    design_task = next((st for st in result_plan.sub_tasks if st.id == '1.2'), None)
    assert design_task is not None
    assert design_task.task_type == "design"
    assert set(design_task.dependency) == {"1.1"}


@pytest.mark.asyncio
@patch('agents.plan.llm_completion', new_callable=AsyncMock)
async def test_plan_write_decomposition_phase(mock_llm_completion):
    """
    测试 plan_write 在“分解阶段”的分解能力。
    - 输入: 任务复杂，且 `dependent_design` 包含结构化规划。
    - 预期: 分解出多个 `write` 子任务，且每个子任务都包含详细的指令。
    """
    # 模拟 llm_completion
    mock_message = AsyncMock()
    mock_message.validated_data = mock_decomposition_phase_llm_output
    mock_message.get.return_value = mock_decomposition_phase_llm_output.reasoning
    mock_llm_completion.return_value = mock_message

    # 准备输入
    task_obj = decomposition_task
    context = decomposition_phase_context
    messages = get_llm_messages(system_prompt, user_prompt, None, context)

    # 调用
    response_message = await llm_completion(
        llm_params={"messages": messages},
        response_model=PlanOutput
    )

    # 验证
    result_plan = response_message.validated_data
    assert result_plan is not None
    assert len(result_plan.sub_tasks) == 4

    # 检查所有子任务都是 write 类型
    assert all(st.task_type == 'write' for st in result_plan.sub_tasks)

    # 检查字数守恒
    total_length = sum(int(st.length) for st in result_plan.sub_tasks if st.length)
    assert total_length == int(task_obj.length)

    # 检查第一个子任务的细节
    first_sub_task = result_plan.sub_tasks[0]
    assert first_sub_task.id == "1.5.1"
    assert first_sub_task.hierarchical_position == "第一章"
    assert "废柴的日常" in first_sub_task.goal
    assert first_sub_task.length == "3000"
    assert len(first_sub_task.instructions) > 0
    assert "重点描写底层生活的窘迫感" in first_sub_task.instructions
    assert len(first_sub_task.input_brief) > 0
    assert "参考`设计方案`" in first_sub_task.input_brief[0]
    assert len(first_sub_task.constraints) > 0
    assert len(first_sub_task.acceptance_criteria) > 0