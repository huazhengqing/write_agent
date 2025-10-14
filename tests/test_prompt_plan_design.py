import pytest
from loguru import logger

from utils.models import Task, PlanOutput
from utils.llm import get_llm_messages, llm_completion, get_llm_params
from utils.loader import load_prompts


system_prompt, user_prompt = load_prompts("story", "plan_design", "system_prompt", "user_prompt")


base_task = Task(
    id="1.2",
    parent_id="1",
    task_type="design",
    hierarchical_position="全书",
    goal="设计主角'龙傲天'",
    instructions=[
        "主角需要符合'赛博修仙'的世界观。",
        "他的能力需要有独创性, 避免常见的金手指套路。"
    ],
    constraints=[
        "主角的背景故事不能过于复杂, 以免喧宾夺主。"
    ],
    acceptance_criteria=[
        "产出的主角设计包含完整的背景、动机、能力和成长弧光。",
        "主角设计能够支撑起至少100万字的长篇故事。"
    ],
    results={
        "atom_result": "complex",
        "complex_reasons": ["composite_goal", "need_search"],
        "atom_reasoning": "设计主角的目标是复合的, 包含背景、能力、成长等多个方面, 且'赛博修仙'的能力设计需要外部资料参考。"
    },
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_plan_design"
)


design_phase_context = {
    "task": base_task.to_context_dict(),
    "complex_reasons": base_task.results["complex_reasons"],
    "atom_reasoning": base_task.results["atom_reasoning"],
    "design_dependent": "", 
    "search_dependent": "",
    "latest_text": "",
    "text_summary": "",
    "overall_planning": """1. 全书 write 写一部关于赛博修仙的小说
  1.1. design 核心概念与世界观设计
  1.2. design 设计主角'龙傲天'""",
    "outside_design": """
# 核心概念与世界观设计
- 核心概念: 在一个高度发达的赛博朋克世界中, 人类通过植入'灵根芯片'来修炼古老的真气, 实现了科技与修仙的结合。
- 世界观钩子: 修炼的尽头不是飞升, 而是将意识上传到宇宙尺度的量子网络'道'。
""",
    "outside_search": ""
}


@pytest.mark.asyncio
async def test_plan_design_prompt():
    messages = get_llm_messages(system_prompt, user_prompt, None, design_phase_context)
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.1,
        messages=messages
    )
    response_message = await llm_completion(
        llm_params=llm_params, 
        response_model=PlanOutput
    )
    result_plan = response_message.validated_data
    logger.info(f"LLM返回的规划结果:\n{result_plan.model_dump_json(indent=2, ensure_ascii=False)}")
