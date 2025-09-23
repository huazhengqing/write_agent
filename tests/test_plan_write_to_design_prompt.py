import pytest
from loguru import logger

from utils.models import Task, PlanOutput
from utils.llm import get_llm_messages, llm_completion, get_llm_params, llm_temperatures
from utils.loader import load_prompts


# 加载被测试的提示词
system_prompt, user_prompt = load_prompts("story", "plan_write_to_design", "system_prompt", "user_prompt")


# 场景: 一个宏大的写作任务, 因为缺乏设计方案而被判定为复杂任务
base_task = Task(
    id="1",
    parent_id="root",
    task_type="write",
    hierarchical_position="全书",
    goal="写一部关于赛博朋克侦探在雨夜都市追查神秘芯片的小说",
    instructions=[
        "主角要有一个悲惨的过去, 这段经历是他追寻真相的根源。",
        "故事节奏要快, 充满反转, 每个章节都要有钩子。"
    ],
    input_brief=[
        "参考《赛博朋克2077》和《银翼杀手》的美术风格和世界观设定。"
    ],
    constraints=[
        "禁止出现魔法元素, 所有超自然现象必须有科学或伪科学解释。"
    ],
    acceptance_criteria=[
        "故事结局要出人意料, 但又在情理之中, 所有伏笔都得到回收。",
        "主角的人物弧光完整。"
    ],
    length="1000000",
    dependency=[],
    sub_tasks=[],
    results={
        "atom_result": "complex",
        "complex_reasons": ["design_insufficient"],
        "atom_reasoning": "任务目标宏大, 篇幅长, 且缺乏具体设计, 需要先进行规划。"
    },
    category="story",
    language="cn",
    root_name="侦探小说",
    day_wordcount_goal=3000,
    run_id="test_run_plan_to_design"
)


# 为提示词准备的上下文, 关键在于 `dependent_design` 为空
design_phase_context = {
    "task": base_task.model_dump_json(indent=2),
    "complex_reasons": base_task.results["complex_reasons"],
    "atom_reasoning": base_task.results["atom_reasoning"],
    "dependent_design": "", # 关键: 没有提供设计方案
    "dependent_search": "",
    "text_latest": "",
    "text_summary": "",
    "task_list": "1 全书 write 写一部关于赛博朋克侦探在雨夜都市追查神秘芯片的小说 1000000",
    "upper_design": "",
    "upper_search": ""
}


@pytest.mark.asyncio
async def test_plan_write_to_design_prompt():
    """
    测试 plan_write_to_design 提示词在“设计阶段”的分解能力。
    - 输入: 任务复杂, 且 `dependent_design` 为空。
    - 预期: 分解出多个 `design` 子任务和一个占位的 `write` 子任务。
    """
    # 准备输入
    messages = get_llm_messages(system_prompt, user_prompt, None, design_phase_context)

    # 准备LLM参数
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=llm_temperatures["reasoning"],
        messages=messages
    )
    
    # 调用被测试逻辑 (真实API调用)
    response_message = await llm_completion(
        llm_params=llm_params, 
        response_model=PlanOutput
    )
    
    # 验证结果
    result_plan = response_message.validated_data

    # 记录日志以供分析
    logger.info(f"LLM返回的规划结果:\n{result_plan.model_dump_json(indent=2, ensure_ascii=False)}")
    
    assert result_plan is not None
    assert result_plan.reasoning is not None
    assert len(result_plan.sub_tasks) > 1, "应至少分解出一个 design/search 任务和一个 write 任务"

    # 检查子任务类型
    sub_task_types = [st.task_type for st in result_plan.sub_tasks]
    assert "design" in sub_task_types or "search" in sub_task_types, "必须包含 design 或 search 任务"
    assert sub_task_types.count("write") == 1, "必须有且仅有一个占位的 write 任务"

    # 检查占位 write 任务
    write_task = next((st for st in result_plan.sub_tasks if st.task_type == 'write'), None)
    assert write_task is not None
    assert "[占位写作任务]" in write_task.goal
    assert write_task.length == "1000000"
    
    # 检查依赖关系: write 任务应该依赖所有其他同级任务
    other_task_ids = {st.id for st in result_plan.sub_tasks if st.task_type != 'write'}
    assert set(write_task.dependency) == other_task_ids, "占位 write 任务应依赖所有同级的 design/search 任务"

    # 检查 design/search 任务
    design_search_tasks = [st for st in result_plan.sub_tasks if st.task_type != 'write']
    assert len(design_search_tasks) > 0
    for task in design_search_tasks:
        assert task.goal is not None
        assert ":" in task.goal, f"任务 {task.id} 的 goal 应遵循 '[指令]: [要求]' 格式"
        # 检查依赖关系是否正确 (依赖项必须是同级的、在它之前的任务)
        if task.dependency:
            try:
                parent_prefix = task.id.rsplit('.', 1)[0] + '.'
                task_seq = int(task.id.split('.')[-1])
                for dep_id in task.dependency:
                    assert dep_id.startswith(parent_prefix), f"任务 {task.id} 的依赖 {dep_id} 不属于同一个父任务"
                    dep_seq = int(dep_id.split('.')[-1])
                    assert dep_seq < task_seq, f"任务 {task.id} 的依赖 {dep_id} 必须是前置任务"
            except (ValueError, IndexError):
                pytest.fail(f"任务ID {task.id} 或其依赖 {task.dependency} 格式不正确, 无法进行序列检查")