import os
import sys
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.models import Task
from utils.llm import get_llm_messages, llm_completion, get_llm_params, llm_temperatures
from utils.loader import load_prompts


system_prompt, user_prompt = load_prompts("story", "hierarchy")


task = Task(
    id="1.2",
    parent_id="1",
    task_type="hierarchy",
    hierarchical_position="第一卷 东海风云",
    goal="为第一卷'东海风云'进行结构规划, 将其分解为若干幕。",
    length="150000",
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_hierarchy",
)


context = {
    "task": task.model_dump_json(indent=2),
    "design_dependent": """
# 第一卷 东海风云 设计方案

## 核心情节
主角龙傲天为给挚友赵日天复仇, 追查神秘令牌的来源, 来到沿海都市"东海市"。他发现令牌与东海市最大的科技公司"天穹集团"有关。龙傲天在调查中结识了反抗组织"地火"的成员林墨瞳, 并与天穹集团的少主叶良辰发生激烈冲突。

## 关键转折
1.  **中期转折**: 龙傲天发现赵日天的"死"其实是一场骗局, 他被天穹集团改造并洗脑, 成为了追杀龙傲天的杀手。
2.  **[核心锚点]**: 在卷末高潮, 龙傲天与被改造的赵日天对决, 战斗中意外触发了"鸿蒙道体"的深层力量, 暂时唤醒了赵日天的部分记忆。
3.  **结局**: 赵日天再次被天穹集团控制并带走。叶良辰向龙傲天揭示, 这一切都与一个名为"飞升计划"的更大阴谋有关, 并邀请他加入。

## 角色弧光
- 龙傲天: 从单纯的复仇者, 转变为开始质疑世界真相、寻求更高力量的探索者。

## 结尾钩子
- 为第二卷"京城龙影"铺垫, 龙傲天决定前往首都, 深入调查"飞升计划"。
""",
    "search_dependent": "",
    "latest_text": "龙傲天望着黑衣人消失的方向, 眼神冰冷, 一个全新的目标在他心中形成: 查明真相, 为友复仇。",
    "text_summary": "在序章'黑松林之变'中, 主角龙傲天与挚友赵日天遭遇神秘黑衣人埋伏, 赵日天为保护龙傲天而重伤昏迷。龙傲天在暴怒中逼退敌人, 拾得一枚神秘令牌, 誓要为友复仇。",
    "task_list": "1. 全书 -> 1.1 核心设定 -> 1.2 第一卷 东海风云",
    "upper_level_design": """
# 核心设定
- 世界观: 赛博朋克与修仙结合的世界。人类通过植入"灵根芯片"进行修炼。
- 主角: 龙傲天, 从地球穿越而来, 身怀神秘的"鸿蒙道体"。
- 核心冲突: 科技与传统的冲突, 个体自由与巨型公司控制的冲突。
""",
    "upper_level_search": ""
}


@pytest.mark.asyncio
async def test_hierarchy_prompt():
    messages = get_llm_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=llm_temperatures["reasoning"],
        messages=messages
    )
    response = await llm_completion(llm_params)
    hierarchy_output = response.content
    logger.info(f"LLM 生成的层级结构规划:\n{hierarchy_output}")


