import os
import sys
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import Task
from utils.llm import get_llm_messages, llm_completion, get_llm_params
from utils.loader import load_prompts


system_prompt, user_prompt = load_prompts("story", "write")


task = Task(
    id="1.1.1",
    parent_id="1.1",
    task_type="write",
    hierarchical_position="第一卷 破晓之章 | 第一章 黑松林之变",
    goal="续写龙傲天在黑松林击退刺客, 准备带赵日天回城治疗的情节。",
    length="1500", 
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_write",
)


context = {
    "task": task.model_dump_json(indent=2),
    "design_dependent": """
# 章节设计: 黑松林之变后续
- 核心事件: 龙傲天检查敌人留下的令牌, 发现了一个神秘符号, 决心查明真相。
- 角色动态: 龙傲天的心态从暴怒转为冷静和坚定, 确立为友复仇的新目标。赵日天重伤昏迷。
- 场景氛围: 战斗后的寂静, 月光下林间的肃杀, 龙傲天内心的焦急与冰冷。
- 结尾钩子: 龙傲天决定先带赵日天回城治疗, 再调查令牌的来历, 为下一章铺垫。
""",
    "search_dependent": "",
    "latest_text": "黑衣人头领见状, 眼中闪过一丝惊异, 随即果断下令: "撤！"黑影们迅速消失在密林深处, 只在地上留下了一枚漆黑的玄铁令牌...一个全新的目标在他心中形成: 查明真相, 为友复仇。",
    "text_summary": "龙傲天与赵日天在黑松林遭遇埋伏, 经过一番激战, 龙傲天在暴怒中爆发出强大力量逼退敌人, 但赵日天为保护他而重伤昏迷。",
    "overall_planning": "1. 全书 -> 1.1 第一卷 -> 1.1.1 第一章",
    "outside_design": """
# 全书风格设定
- 整体叙事风格: 第三章人称, 过去时。
- 文笔基调: 节奏明快, 兼具古典仙侠的诗意与赛博朋克的冷峻。
- 主角龙傲天角色声音: 平时冷静理智, 涉及朋友安危时会变得果决甚至带有一丝狠厉。
""",
    "outside_search": ""
}


@pytest.mark.asyncio
async def test_write_prompt():
    messages = get_llm_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="writing", 
        temperature=0.75, 
        messages=messages
    )
    response = await llm_completion(llm_params)
    generated_text = response.content
    logger.info(f"LLM 生成的小说内容:\n{generated_text}")
