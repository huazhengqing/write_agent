import os
import sys
import pytest
from loguru import logger
import json

from llama_index.core.schema import NodeWithScore, Document
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.llms.litellm import LiteLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.indices.prompt_helper import PromptHelper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_api import get_llm_params, llm_temperatures
from utils.llm import get_llm_messages, llm_completion
from prompts.story import summary as summary_prompts
from utils.vector import text_qa_prompt, refine_prompt


# 用于摘要测试的长文本, 包含所有关键元素: 角色、情节、事件、场景、伏笔、时间、地点、转折、因果关系、对话、战斗、悬念、物品以及角色状态变化。
LONG_TEXT_TO_SUMMARIZE = """
# 第一章: 黑松林之变

黄昏时分, 临海镇“听潮轩”酒楼二楼。

龙傲天将那张泛黄的羊皮卷在桌上缓缓展开, 神情专注。地图的纹路古朴而神秘, 指向城外黑松林深处的一个未知标记。

“傲天, 这玩意儿真靠谱吗?”赵日天灌下一大口烈酒, 粗声问道, “我怎么瞅着像个骗局。”

“直觉告诉我, 这地图背后不简单。”龙傲天指着地图中心的一个奇特符号, “你看这个标记, 我胸口的玄铁吊坠似乎对它有微弱的感应。”他从领口掏出一枚色泽暗沉、雕刻着云纹的吊坠, 这是他穿越时唯一伴随的物品, 多年来毫无异状。

“哦?还有这等奇事！”赵日天来了兴致, “那还等什么, **今晚**入夜后, 咱们就去探他个究竟！”

因果就此种下。入夜, 两人借着月色潜入黑松林。林中寂静无声, 唯有风吹过松针的沙沙声, 气氛诡异。突然, 数道黑影从林间阴影处暴起, 手持淬毒的短刃, 悄无声息地袭向二人。

“有埋伏！小心！”龙傲天低喝一声, 拔出背后的长剑“惊鸿”, 剑光如水银泻地, 瞬间挡开两名黑衣人的攻击。战斗骤然爆发。

黑衣人身法诡异, 配合默契, 显然是训练有素的杀手。龙傲天剑法精妙, 一时还能应对, 但赵日天拳脚功夫虽猛, 却渐渐落入下风。

“噗嗤！”一声闷响, 赵日天为替龙傲天挡下一记背刺, 左肩被短刃贯穿, 鲜血瞬间染红了衣衫。他的状态从“完好”急转为“重伤”**, 豆大的汗珠从他额头滴落**。

“日天！”龙傲天双目赤红, 心态从“冷静”变为“暴怒”。他不再保留, 体内一股神秘力量涌动, 胸前的玄铁吊坠发出一阵微不可查的温热。他大喝一声, 剑招威力陡增, 一道璀璨的剑气横扫而出, 逼退了所有敌人。

黑衣人头领见状, 眼中闪过一丝惊异, 随即果断下令: “撤！”黑影们迅速消失在密林深处, 只在地上留下了一枚漆黑的玄铁令牌, 上面刻着一个与地图上相似但更为复杂的符号。

悬念由此产生。这群黑衣人是谁?为何抢夺地图?那枚令牌又代表着什么?

龙傲天来不及多想, 立刻扶起重伤昏迷的赵日天, 朋友的失散与重伤让他心急如焚。他望着黑衣人消失的方向, 眼神冰冷, **他仔细观察着那枚令牌, 上面雕刻的符号, 既像是一只眼睛, 又像是一座倒立的山峰, **一个全新的目标在他心中形成: 查明真相, 为友复仇。
"""

@pytest.mark.asyncio
async def test_summary_model_functionality():
    """专门测试摘要模型的功能和正确性。"""
    logger.info("--- 测试摘要模型 (Summary Model) ---")

    # 1. 准备提示词和上下文
    system_prompt = "你是一个专业的文本摘要助手。请根据用户提供的文本, 生成一段简洁、准确、涵盖核心信息的摘要。"
    user_prompt = "请为以下文本生成摘要: \n\n{text}"
    
    context_dict_user = {
        "text": LONG_TEXT_TO_SUMMARIZE
    }

    messages = get_llm_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_dict_user=context_dict_user
    )

    # 2. 获取摘要模型的 LLM 参数
    llm_params = get_llm_params(
        llm_group="summary", 
        temperature=llm_temperatures["summarization"]
    )
    llm_params["messages"] = messages

    # 3. 调用 LLM
    response = await llm_completion(llm_params)
    summary_content = response.content
    
    logger.info(f"原始文本长度: {len(LONG_TEXT_TO_SUMMARIZE)}")
    logger.info(f"摘要文本长度: {len(summary_content)}")
    logger.info(f"生成的摘要:\n{summary_content}")

    # 4. 断言验证
    assert summary_content, "摘要内容不应为空。"
    assert len(summary_content) < len(LONG_TEXT_TO_SUMMARIZE), "摘要长度应小于原文长度。"
    
    # 检查核心关键词是否被包含
    keywords = ["龙傲天", "赵日天", "黑松林", "玄铁吊坠", "玄铁令牌"]
    for keyword in keywords:
        assert keyword in summary_content, f"摘要中应包含关键词 '{keyword}'"

    logger.success("--- 摘要模型测试通过 ---")


@pytest.mark.asyncio
async def test_compact_and_refine_synthesizer():
    """
    专门测试 CompactAndRefine 响应合成器的内容整合效果。
    这个测试模拟了从向量数据库检索到多个相关但信息分散的文本块后, 
    如何将它们融合成一个连贯的答案。
    """
    logger.info("--- 测试 CompactAndRefine 内容整合 ---")

    # 1. 准备多个文本节点, 信息分散在其中
    nodes_with_score = [
        NodeWithScore(
            node=Document(text="龙傲天是一名孤傲的剑客, 他来自一个神秘的东方世家, 以一手快剑闻名于世。"),
            score=0.9
        ),
        NodeWithScore(
            node=Document(text="叶良辰是北境魔门的少主, 性格乖张, 行事不择手段。他视龙傲天为自己唯一的宿敌。"),
            score=0.88
        ),
        NodeWithScore(
            node=Document(text="在昆仑之巅的决战中, 龙傲天以微弱优势击败了叶良辰, 从而奠定了自己天下第一的地位。"),
            score=0.85
        ),
    ]

    question = "请全面介绍龙傲天和叶良辰的关系, 以及他们之间发生的关键事件。"

    # 2. 明确创建响应合成器, 确保使用正确的模型进行测试
    synthesis_llm_params = get_llm_params(
        llm_group="summary",
        temperature=llm_temperatures["synthesis"] 
    )
    synthesizer = CompactAndRefine(
        llm=LiteLLM(**synthesis_llm_params),
        text_qa_template=PromptTemplate(text_qa_prompt),
        refine_template=PromptTemplate(refine_prompt),
        prompt_helper = PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 8192),
            num_output=synthesis_llm_params.get('max_tokens', 2048),
            chunk_overlap_ratio=0.2,
        )
    )

    # 3. 调用合成器进行内容合成
    response = await synthesizer.asynthesize(query=question, nodes=nodes_with_score)
    synthesized_content = str(response)
    logger.info(f"合成后的内容:\n{synthesized_content}")

    # 4. 断言验证
    assert synthesized_content, "合成内容不应为空。"
    assert "龙傲天" in synthesized_content and "剑客" in synthesized_content, "应包含龙傲天的身份信息"
    assert "叶良辰" in synthesized_content and "北境魔门" in synthesized_content, "应包含叶良辰的背景信息"
    assert "宿敌" in synthesized_content, "应点明二人的关系"
    assert "昆仑之巅" in synthesized_content and "击败" in synthesized_content, "应包含关键事件的信息"

    logger.success("--- CompactAndRefine 内容整合测试通过 ---")


@pytest.mark.asyncio
async def test_real_world_summary_model():
    """
    使用项目中的真实提示词, 测试结构化摘要模型的真实效果。
    """
    logger.info("--- 测试真实场景下的结构化摘要模型 ---")

    # 1. 准备任务信息和上下文
    task_info = {
        "id": "1.1",
        "task_type": "summary",
        "hierarchical_position": "第一章",
        "goal": "为'黑松林之变'这一章节生成一份结构化摘要。",
        "length": len(LONG_TEXT_TO_SUMMARIZE)
    }

    context_dict_user = {
        "task": json.dumps(task_info, ensure_ascii=False, indent=2),
        "text": LONG_TEXT_TO_SUMMARIZE
    }

    messages = get_llm_messages(
        system_prompt=summary_prompts.system_prompt,
        user_prompt=summary_prompts.user_prompt,
        context_dict_user=context_dict_user
    )

    # 2. 获取摘要模型的 LLM 参数
    llm_params = get_llm_params(
        llm_group="summary", 
        temperature=llm_temperatures["summarization"]
    )
    llm_params["messages"] = messages

    # 3. 调用 LLM
    response = await llm_completion(llm_params)
    summary_content = response.content
    
    logger.info(f"生成的结构化摘要:\n{summary_content}")

    # 4. 断言验证
    assert summary_content, "结构化摘要内容不应为空。"
    assert "## 摘要" in summary_content, "摘要中应包含Markdown标题 '## 摘要'"
    assert "## 场景时间线" in summary_content, "摘要中应包含Markdown标题 '## 场景时间线'"
    assert "## 角色关系与冲突分析" in summary_content, "摘要中应包含Markdown标题 '## 角色关系与冲突分析'"
    assert "graph TD" in summary_content, "摘要中应包含Mermaid图语法 'graph TD'"
    assert "龙傲天" in summary_content and "赵日天" in summary_content, "摘要应包含原文的核心实体"

    logger.success("--- 真实场景结构化摘要模型测试通过 ---")