0,0 @@
import os
import sys
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import Task
from utils.llm import get_llm_messages, llm_completion, get_llm_params
from utils.loader import load_prompts


system_prompt, user_prompt = load_prompts("story", "design")


task = Task(
    id="1.2",
    parent_id="1",
    task_type="design",
    hierarchical_position="第一卷 东海风云",
    goal="设计第一卷'东海风云'的详细情节大纲, 包含核心冲突、关键转折和结局钩子。",
    instructions=[
        "情节需要围绕主角'龙傲天'为挚友'赵日天'复仇并寻求自身力量突破展开。",
        "必须引入核心反派'叶良辰', 并建立他与主角的宿敌关系。",
        "卷末需要为第二卷'京城龙影'留下明确的线索。"
    ],
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_design",
)


context = {
    "task": task.model_dump_json(indent=2),
    "design_dependent": "",
    "search_dependent": "",
    "latest_text": "龙傲天望着黑衣人消失的方向, 眼神冰冷, 一个全新的目标在他心中形成: 查明真相, 为友复仇。",
    "text_summary": "在序章'黑松林之变'中, 主角龙傲天与挚友赵日天遭遇神秘黑衣人埋伏, 赵日天为保护龙傲天而重伤昏迷。龙傲天在暴怒中逼退敌人, 拾得一枚神秘令牌, 誓要为友复仇。",
    "task_list": "1. 全书 -> 1.1 核心设定 -> 1.2 第一卷 东海风云",
    "upper_level_design": """
# 核心设定
- 世界观: 赛博朋克与修仙结合的世界。人类通过植入"灵根芯片"进行修炼。
- 主角: 龙傲天, 从地球穿越而来, 身怀神秘的"鸿蒙道体", 对灵气有超凡的亲和力。
- 核心冲突: 科技与传统的冲突, 个体自由与巨型公司控制的冲突。
- [核心锚点]: 主角龙傲天必须在第一卷结尾, 发现他的"鸿蒙道体"与这个世界的"道"网络(宇宙级量子网络)的底层协议有关。
""",
    "upper_level_search": ""
}


@pytest.mark.asyncio
async def test_design_prompt():
    messages = get_llm_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.75,
        messages=messages
    )
    response = await llm_completion(llm_params)
    design_output = response.content
    logger.info(f"LLM 生成的设计方案:\n{design_output}")
    