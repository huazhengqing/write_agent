import pytest
from loguru import logger

from utils.models import Task, PlanOutput
from utils.llm import get_llm_messages, llm_completion, get_llm_params
from utils.loader import load_prompts


system_prompt, user_prompt = load_prompts("story", "plan_write_to_design", "system_prompt", "user_prompt")


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
    name="侦探小说",
    day_wordcount_goal=3000,
    run_id="test_run_plan_to_design"
)


design_phase_context = base_task.to_context_dict()
design_phase_context.update({
    "complex_reasons": base_task.results["complex_reasons"],
    "atom_reasoning": base_task.results["atom_reasoning"],
    "design_dependent": "", 
    "search_dependent": "",
    "latest_text": "",
    "text_summary": "",
    "overall_planning": "1 全书 write 写一部关于赛博朋克侦探在雨夜都市追查神秘芯片的小说 1000000",
    "outside_design": "",
    "outside_search": ""
})


@pytest.mark.asyncio
async def test_plan_write_to_design_prompt():
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
