import os
import sys
import pytest
from loguru import logger
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import Task, AtomOutput
from utils.llm import get_llm_messages, llm_completion, get_llm_params
from story.prompts.atom_write import system_prompt, user_prompt


task = Task(
    id="1.1.1",
    parent_id="1.1",
    task_type="write",
    hierarchical_position="第一章",
    goal="续写龙傲天与赵日天在黑松林遭遇埋伏后的情节。",
    length="1500",
    category="story",
    language="cn",
    name="赛博真仙",
    run_id="test_run_atom_write",
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
    "overall_planning": "1.1.1 第一章 write 续写龙傲天与赵日天在黑松林遭遇埋伏后的情节。 1500"
}


@pytest.mark.asyncio
async def test_atom_write_prompt():
    messages = get_llm_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.0
    )
    llm_params["messages"] = messages
    response = await llm_completion(llm_params, response_model=AtomOutput)
    result = response.validated_data
    logger.info(f"LLM 输出:\n{result.model_dump_json(indent=2, ensure_ascii=False)}")


