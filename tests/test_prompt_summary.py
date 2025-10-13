import os
import sys
import pytest
from loguru import logger
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import Task
from utils.llm import get_llm_messages, llm_completion, get_llm_params
from utils.loader import load_prompts


system_prompt, user_prompt = load_prompts("story", "summary")


LONG_TEXT_TO_SUMMARIZE = """
# 第一章: 黑松林之变

黄昏时分, 临海镇"听潮轩"酒楼二楼。

龙傲天将那张泛黄的羊皮卷在桌上缓缓展开, 神情专注。地图的纹路古朴而神秘, 指向城外黑松林深处的一个未知标记。

"傲天, 这玩意儿真靠谱吗?"赵日天灌下一大口烈酒, 粗声问道, "我怎么瞅着像个骗局。"

"直觉告诉我, 这地图背后不简单。"龙傲天指着地图中心的一个奇特符号, "你看这个标记, 我胸口的玄铁吊坠似乎对它有微弱的感应。"他从领口掏出一枚色泽暗沉、雕刻着云纹的吊坠, 这是他穿越时唯一伴随的物品, 多年来毫无异状。

"哦?还有这等奇事！"赵日天来了兴致, "那还等什么, **今晚**入夜后, 咱们就去探他个究竟！"

因果就此种下。入夜, 两人借着月色潜入黑松林。林中寂静无声, 唯有风吹过松针的沙沙声, 气氛诡异。突然, 数道黑影从林间阴影处暴起, 手持淬毒的短刃, 悄无声息地袭向二人。

"有埋伏！小心！"龙傲天低喝一声, 拔出背后的长剑"惊鸿", 剑光如水银泻地, 瞬间挡开两名黑衣人的攻击。战斗骤然爆发。

黑衣人身法诡异, 配合默契, 显然是训练有素的杀手。龙傲天剑法精妙, 一时还能应对, 但赵日天拳脚功夫虽猛, 却渐渐落入下风。

"噗嗤！"一声闷响, 赵日天为替龙傲天挡下一记背刺, 左肩被短刃贯穿, 鲜血瞬间染红了衣衫。他的状态从"完好"急转为"重伤"**, 豆大的汗珠从他额头滴落**。

"日天！"龙傲天双目赤红, 心态从"冷静"变为"暴怒"。他不再保留, 体内一股神秘力量涌动, 胸前的玄铁吊坠发出一阵微不可查的温热。他大喝一声, 剑招威力陡增, 一道璀璨的剑气横扫而出, 逼退了所有敌人。

黑衣人头领见状, 眼中闪过一丝惊异, 随即果断下令: "撤！"黑影们迅速消失在密林深处, 只在地上留下了一枚漆黑的玄铁令牌, 上面刻着一个与地图上相似但更为复杂的符号。

悬念由此产生。这群黑衣人是谁?为何抢夺地图?那枚令牌又代表着什么?

龙傲天来不及多想, 立刻扶起重伤昏迷的赵日天, 朋友的失散与重伤让他心急如焚。他望着黑衣人消失的方向, 眼神冰冷, **他仔细观察着那枚令牌, 上面雕刻的符号, 既像是一只眼睛, 又像是一座倒立的山峰, **一个全新的目标在他心中形成: 查明真相, 为友复仇。
"""


task = Task(
    id="1.1.summary",
    parent_id="1.1",
    task_type="summary",
    hierarchical_position="第一章 黑松林之变",
    goal="为'黑松林之变'这一章节生成一份结构化摘要。",
    length=str(len(LONG_TEXT_TO_SUMMARIZE)),
    category="story",
    language="cn",
    root_name="赛博真仙",
    run_id="test_run_summary",
)


context = {
    "task": task.model_dump_json(indent=2),
    "text": LONG_TEXT_TO_SUMMARIZE
}


@pytest.mark.asyncio
async def test_summary_prompt():
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", temperature=0.2, messages=messages)
    response = await llm_completion(llm_params)
    summary_content = response.content
    logger.info(f"LLM 生成的结构化摘要:\n{summary_content}")
