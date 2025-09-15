import asyncio
import json
import os
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from loguru import logger
from datetime import datetime
from llama_index.core import Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from market_analysis.story.common import index, story_output_dir, task_platform_briefing, task_new_author_opportunity, task_load_platform_profile, get_market_tools, task_store
from utils.llm import call_agent, get_llm_messages, get_llm_params, llm_acompletion
from utils.log import init_logger
from utils.prefect_utils import local_storage, readable_json_serializer, generate_readable_cache_key
from prefect import flow, task

init_logger(os.path.splitext(os.path.basename(__file__))[0])


class Candidate(BaseModel):
    platform: str
    genre: str

class FinalDecision(BaseModel):
    platform: str = Field(description="最终选择的平台")
    genre: str = Field(description="最终选择的题材")
    reasoning: str = Field(description="做出该选择的详细理由，并解释为什么它优于其他主要竞争选项。")

class RankedConcept(BaseModel):
    rank: int = Field(description="机会排名，1为最佳。")
    platform: str = Field(description="平台名称。")
    genre: str = Field(description="题材大类。")
    brief_reasoning: str = Field(description="对该选项排名的简要理由，点出其核心优劣势。")

class FinalDecisionResult(BaseModel):
    final_choice: FinalDecision = Field(description="根据综合排序最终选定的最佳方案。")
    ranking: List[RankedConcept] = Field(description="所有候选方案的完整排名列表。")

FINAL_DECISION_SYSTEM_PROMPT_JSON = """
# 角色
你是一位顶尖的网文总编，拥有敏锐的市场嗅觉和战略眼光。

# 任务
你收到了多份（N份）关于不同“平台-题材”的深度分析报告。你的任务是：
1.  对所有报告代表的机会进行综合评估和排序。
2.  选出排名第一的最佳机会作为最终决策。
3.  以一个结构化的JSON对象输出你的完整分析和决策。

# 决策维度
1.  潜力上限 (Ceiling): 哪个方向的天花板更高？考虑其题材的受众广度、付费潜力、IP衍生可能性以及是否踩在“新兴机会与蓝海方向”上。
2.  成功概率 (Probability): 哪个方向的成功确定性更高？考虑其与平台调性的契合度、竞争激烈程度、以及我们规避“常见毒点”的难易度。
3.  创新价值 (Innovation): 哪个方向更具创新性，更能带来差异化优势，避免陷入红海竞争？
4.  执行难度 (Execution): 哪个方向的实际创作门槛更低，对作者的要求更友好？

# 工作流程
1.  通读报告: 完整阅读所有深度分析报告，理解每个选项的优劣势。
2.  评估与排序: 在脑中根据上述“决策维度”对所有选项进行打分和比较，形成一个从优到劣的完整排序列表。
3.  最终决策: 确定排名第一的选项，并为其撰写详细的决策理由，说明它为何优于其他主要竞争者。
4.  格式化输出: 将你的最终决策和完整的排序列表，严格按照Pydantic模型的JSON格式输出。

# 输出要求
- 严格按照 Pydantic 模型的格式，仅输出一个完整的、有效的 JSON 对象。
- 禁止在 JSON 前后添加任何额外解释、注释或 markdown 代码块。
- JSON 结构必须符合以下定义:
  - `final_choice` (object): 最终选定的最佳方案。
    - `platform` (string): 最终选择的平台名称。
    - `genre` (string): 最终选择的题材名称。
    - `reasoning` (string): 详细阐述做出该选择的理由，并解释为什么它优于其他主要竞争选项。
  - `ranking` (array of objects): 所有候选方案的完整排名列表，按排名从高到低排序。
    - `rank` (integer): 机会排名，1为最佳。
    - `platform` (string): 平台名称。
    - `genre` (string): 题材大类。
    - `brief_reasoning` (string): 对该选项排名的简要理由，点出其核心优劣势。
"""


DEEP_DIVE_SYSTEM_PROMPT = """
# 角色
你是一名顶尖的网络小说市场分析师，专精于【{platform}】平台的【{genre}】题材。

# 任务
基于你掌握的工具和提供的上下文信息，为【{platform}】平台的【{genre}】题材生成一份深度洞察报告。

# 工作流程
1.  信息整合: 仔细阅读提供的【平台基础信息】和【市场动态简报】。
2.  深度研究: 在研究过程中，如果你需要更多关于特定概念、作品或市场趋势的背景信息，请优先使用 `story_market_vector` 工具查询内部知识库。如果内部知识库没有你需要的信息，再使用网络搜索工具进行外部研究.围绕以下“输出结构”中的要点进行深入研究。
    - 内部优先: 在研究任何概念、作品、市场趋势或读者心理时，必须首先使用 `story_market_vector` 工具查询内部知识库。知识库中包含了已归档的平台档案、市场报告和小说创意。
    - 外部补充: 只有当内部知识库信息不足或需要获取最新动态（例如，社交媒体上的实时热点）时，才使用网络搜索工具进行补充。
    - 研究要点: 围绕“输出结构”中的要点进行研究，例如搜索该题材的常见套路、读者“毒点”、相关的外部热点、流行文化等。
3.  撰写报告: 综合所有信息，严格按照指定的Markdown格式输出最终报告。

# 上下文信息
---
## 平台基础信息 ({platform})
{platform_profile}
---
## 市场动态简报 ({platform})
{broad_scan_report}
---
## 平台新人机会评估报告 ({platform})
{opportunity_report}
---

# 输出结构 (Markdown)
## 【{platform}】平台 - 【{genre}】题材深度分析报告

### 1. 核心标签与流行元素
- 高频标签: [通过搜索作品和评论，总结出3-5个最高频的标签]
- 关键元素: [具体描述这些标签在小说中的表现形式，例如“系统”标签表现为“签到流”、“神豪返现”]
- 趋势验证: [结合网络搜索趋势（如百度指数），分析这些标签的近期热度变化]

### 2. 核心爽点与读者心理
- 爽点一: [描述该题材最核心的一个爽点，例如：扮猪吃虎后瞬间打脸]
  - 读者心理: [分析该爽点满足了读者的何种心理需求。尝试结合社会心理学理论进行深度解释。例如：这满足了读者对“代理复仇”和“恢复秩序”的渴望，与“公平世界信念”相关。当主角揭示实力时，读者通过“社会比较理论”中的下行比较，从反派的失败中获得优越感和满足感。]
- 爽点二: [描述另一核心爽点，例如：获得独一无二的系统/金手指]
  - 读者心理: [同上，进行深度心理学分析。例如：这直接满足了读者的“掌控感”需求，在充满不确定性的现实世界中，一个规则明确、反馈即时的系统提供了极大的心理安全感。这也与“自我效能感”理论相关，读者通过代入主角，体验到自己有能力改变环境、达成目标。]

### 3. 关键付费点设计
- (分析该题材作品通常如何设计付费章节，例如：高潮前、悬念揭晓时、新地图展开时。如果是免费平台，则分析广告点位的设计逻辑)

### 4. 新兴机会与蓝海方向
- 题材融合: [提出有数据支撑的新颖题材融合方向]
- 跨界融合: [结合【外部热点趋势】，提出可融合的跨界创意]
- 设定创新: [提出未被滥用的创新设定/金手指]
- 切入角度: [建议新颖的主角身份或故事切入点]
- 作者友好度: [结合【平台新人机会评估报告】，分析该方向对新人作者的友好度，例如：平台有流量扶持，适合新人切入]

### 5. 主角人设迭代方向
- 流行人设分析: [分析当前题材下最受欢迎的1-2种主角人设及其核心魅力]
- 创新方向: [提出一种对流行人设进行反转或融合创新的设计，创造差异化]

### 6. 常见“毒点”与风险规避
- 毒点一: [总结一个读者普遍反感的情节或设定]
  - 规避建议: [提出具体规避方法]
- 毒点二: [总结另一个常见毒点]
  - 规避建议: [提出对应规避方法]

### 7. 报告质量自我评估
- 数据驱动度 (1-5分): [报告基于搜索数据的程度]
- 洞察深刻度 (1-5分): [报告揭示深层趋势的程度]
- 可执行性 (1-5分): [报告建议的清晰度和可用性]
- 综合评价: [总结报告优缺点]
"""
# 机会生成的提示词
OPPORTUNITY_GENERATION_SYSTEM_PROMPT = """
# 角色
金牌小说策划人。

# 任务
根据所有输入信息，构思3个全新的、有商业潜力的小说选题。

# 创作原则
- 机会导向: 核心创意必须回应【新兴机会与蓝海方向】。
- 跨界优先: 至少一个选题深度融合【跨界融合】建议，力争S级。
- 灵感融合: 充分利用【市场深度分析报告】中提到的外部热点、流行文化、视觉风格等元素，并将这些元素融入选题。
- 风险规避: 避开【常见“毒点”】。
- 爽点聚焦: 围绕【核心爽点】构建。
- 避免重复: 与【历史创意参考】显著区别。
- 创新激励: 鼓励提出“反套路指数”高的选题。
- 强制差异化: 3个选题必须在以下至少两个维度上存在显著差异，以确保多样性：
    - 核心卖点: 例如，一个主打“创新金手指”，一个主打“极致情绪冲突”，一个主打“新颖世界观”。
    - 切入角度: 例如，一个从传统主角视角，一个从反派或配角视角，一个从“物品”或“概念”的非人视角。
    - 题材融合: 例如，一个融合“科幻”，一个融合“悬疑”，一个融合“历史”。
    - 目标读者: 例如，一个面向追求极致爽感的年轻读者，一个面向偏好逻辑和智斗的成熟读者。

# 输出结构 (Markdown)
- 选题名称: [名称]
- 一句话卖点: [宣传语]
- 核心创意: [概括，明确指出融合的“蓝海方向”或“跨界热点”]
- 主角设定: [身份, 特点, 独特性]
- 核心冲突: [主要矛盾]
- 爆款潜力: [S/A/B级]
- 潜力理由: [市场契合度, 创意新颖度, 爽点强度, 跨界优势]
- 反套路指数: [高/中/低]
- 指数理由: [解释该选题在多大程度上规避了常见套路，或对套路进行了创新改造]
- 写作难度: [高/中/低]
- 难度理由: [世界观, 角色, 情节, 资料]
"""

OPPORTUNITY_GENERATION_USER_PROMPT = """
---
# 市场深度分析报告
{market_report}

---
# 历史创意参考
{historical_concepts}
"""

# 小说创意生成提示词
NOVEL_CONCEPT_SYSTEM_PROMPT = """
# 角色
顶级小说策划人。

# 任务
1.  选择: 从【初步选题列表】中选择评级最高、最具潜力的选题 (优先“跨界融合”建议)。
2.  扩展: 结合所有输入信息，将所选选题扩展为一份详细的【小说创意】文档。
    - 模式抽象与创新 (关键步骤): 不要直接模仿【历史成功案例参考】的表面情节或设定。你的任务是进行更高层次的“模式挖掘”：
        - 解构与抽象: 分析成功案例的底层成功范式。例如，不要只看“它用了什么升级体系”，而要分析“这个升级体系的节奏和反馈循环是如何设计的”、“它的核心资源循环是什么样的”。不要只看“它写了什么爽点”，要分析“这些爽点组合的顺序和内在逻辑是什么，它们满足了读者何种深层心理欲望”。
        - 转化与重塑: 将你提炼出的抽象范式（如“高频即时反馈+长线稀缺目标”的节奏模式，“身份认同危机驱动的冲突模式”等）应用到你正在构思的新选题中。你要做的是用这个范式来生成全新的、符合当前选题世界观和主角人设的具体设定，而不是复制旧的设定。
    - 知识库挖掘: 在进行“模式抽象与创新”时，主动使用 `story_market_vector` 工具，查询与你正在构思的核心设定、升级体系或角色弧光相关的历史创意和分析报告。这能帮助你发现可复用的底层范式，或避免重复已经被验证失败的模式。
    - 规避套路: 主动使用 `forum_discussion_search` 工具在知乎、龙空等社区搜索“网络小说[题材]常见过时套路”、“读者差评”、“毒点总结”，剔除这些过时元素。
    - 对标分析: 识别并分析1-2个与当前选题最直接的竞品或标杆作品。明确指出它们的优缺点，并构思你的创意如何进行差异化竞争，是“做得更好”、“做得不同”还是“开创新品类”。
    - 融合灵感: 主动使用 `social_media_trends_search` 工具在B站搜索与选题相关的视觉、观点或元素，并融入创意。

# 输出结构 (Markdown)
选择的选题: [选题名称]
选择理由: [说明选择原因，特别是“跨界融合”的优势]

---

## 小说创意：[选题名称]

### 1. 一句话简介 (Logline)
- [30-50字，概括主角、目标、冲突、独特设定]

### 2. 详细故事梗概 (Synopsis)
- [200-300字，概述起因、发展、核心冲突、高潮]

### 3. 主角设定 (Character Profile)
- 背景与动机: [出身, 职业, 内心渴望/恐惧]
- 性格与能力: [性格特点, 行事风格, 核心能力/金手指及其限制]
- 成长弧光 (Character Arc):
  - 核心缺陷/谎言: [主角在故事开始时所信奉的、限制其成长的错误信念或世界观。例如：“只有绝对的力量才能带来安全感”、“我不值得被爱”。]
  - 欲望 (Want) vs. 需求 (Need):
    - 外在欲望: [主角明确追求的外在目标。例如：成为天下第一、赚到一百亿。]
    - 内在需求: [主角未曾察觉、但真正需要的东西，通常与克服“核心缺陷”相关。例如：学会信任他人、接纳自己的不完美。]
  - 转变路径:
    - 催化事件: [什么事件迫使主角踏上旅程，并首次挑战其“核心缺陷”？]
    - 关键转折 (中点): [在故事中点，主角遭遇了什么重大失败或启示，使其开始质疑自己的“谎言”？]
    - 最终证明 (高潮): [在高潮部分，主角如何通过一个关键行动，彻底抛弃“谎言”，拥抱“内在需求”，从而战胜最终挑战？]
  - 最终状态: [故事结束时，主角的新信念和新行为模式是怎样的？]

### 4. 核心冲突矩阵 (Core Conflict Matrix)
- 原则: 所有冲突必须相互关联，外部冲突是内在冲突的映射，关系冲突是主角成长的试金石。

- 根本性冲突 (主题): [贯穿始终的哲学/价值观冲突，如: 自由 vs. 安全]
  - 核心议题: [该冲突向读者提出的、没有简单答案的两难问题，如: “为了绝对的安全，放弃个人自由是否值得？”]
  - 冲突体现: [主角代表哪一方，对手代表哪一方，以及世界观如何体现这一冲突]

- 主线情节冲突 (外部):
  - 具体目标: [主角必须完成的、具体的、可见的外部目标]
  - 强大对手:
    - 动机与目标: [对手追求什么? 为什么他的目标与主角不共戴天?]
    - “黑镜”关系: [对手在哪些方面是主角的“黑暗镜像”? (例如，拥有相似的过去，但做出了不同的选择)]
  - 冲突升级与赌注 (Stakes):
    - 初期 (试探): [冲突如何开始? 失败的初步后果是什么?]
    - 中期 (对抗): [冲突如何升级? 赌注如何提高? (例如，从个人安危上升到团队/城市存亡)]
    - 终局 (决战): [最终对决的场景是什么? 失败的最终后果是什么? (世界毁灭? 失去挚爱? 永远无法实现内在需求?)]

- 主角内在冲突 (内部):
  - 核心矛盾: [主角的“欲望(Want)”与“需求(Need)”之间的核心矛盾]
  - 艰难抉择: [在哪个关键情节中，主角必须在“欲望”和“需求”之间做出选择? 这个选择如何推动外部情节?]
  - 失败后果: [如果主角无法克服内在缺陷、无法实现“需求”，他将付出什么永久性的代价?]

- 核心关系冲突 (人际):
  - 关系对象: [与主角关系最纠结的核心配角 (盟友/爱人/导师)]
  - 冲突根源: [两人因何产生冲突? (价值观差异? 对主线目标的方法论分歧? 主角的内在缺陷导致?)]
  - 考验与张力: [这段关系如何考验主角的成长? 它如何为主线情节增加复杂性和情感张力?]

### 5. 世界观核心设定 (World-building)
- 核心概念: [作为世界观基石的“What if”问题]
- 独特法则: [1-2条与现实相悖的物理/社会法则及其影响]
- 标志性元素: [2-3个独特的地理、生物、组织或技术]
- 历史谜团与探索感:
  - 历史谜团: [贯穿始终的古老谜团]
  - 探索路径: [主角揭开世界秘密的路线]

### 6. 升级体系与核心设定 (Progression System & Core Setting)
- 创新原则: [规避常见设定 (如简单的等级制), 必须与世界观深度绑定, 并且本身要成为冲突的来源, 而不仅仅是工具。]
- 体系核心概念: [用一个独特的比喻来描述体系的本质。例如：“力量不是阶梯，而是与一个古老存在签订的、不断加码的契约。”]
- 核心资源与获取:
  - 独特资源: [命名一种非传统的“经验值”或“法力”资源，如“被遗忘的记忆”、“他人情绪的结晶”。]
  - 获取方式与风险: [描述获取该资源的独特方式，并强调其伴随的风险或道德困境。例如：“通过吞噬他人的梦境来获取，但有被梦境同化的风险。”]
- 晋升路径与质变:
  - 非线性路径: [设计包含“分支选择”、“代价置换”或“职业分化”的成长路径，而非单一线性升级。]
  - 关键阶段 (3-5个): [为关键的成长阶段命名，并描述其核心特征，避免使用“青铜、白银、黄金”等套路。]
  - 能力质变: [描述晋升如何从根本上改变主角与世界的交互方式，而不仅仅是数值提升。例如：“从‘能看到鬼魂’质变为‘能篡改鬼魂的记忆’”。]
- 体系的内在矛盾与社会影响:
  - 内在悖论: [设计体系本身的缺陷或悖论，作为核心冲突的来源。例如：“体系追求绝对秩序，但其力量来源却是混乱本身。”]
  - 社会结构: [描述该体系如何塑造了社会结构、阶级或派系。例如：“掌握不同晋升路径的派系之间互相敌视。”]
  - 可利用的漏洞: [设计一个隐藏的、非主流的规则或漏洞，为主角的“反套路”成长提供可能性。]

### 7. 关键配角设定 (Key Supporting Characters)
- 原则: [避免工具人，配角需有独立目标和内在矛盾]
- 配角一:
  - 定位与功能: [导师/对手/盟友/反派等]
  - 独立人生:
    - 个人目标: [与主角无关的个人追求]
    - 内在矛盾: [角色内心的核心挣扎]
    - 一个秘密: [角色隐瞒的关键秘密]
  - 与主角的动态关系:
    - 关系演变: [关系如何随故事发展而变化]
    - 核心互动模式: [最常见的互动方式]
- 配角二:
  - [结构同上]

### 8. 核心爽点与高光场景 (Core Appeal & Highlight Scenes)
- 主爽点: [最核心、最高频的爽点类型]
  - 高光场景示例:
    - 场景简述: [体现该爽点的具体场景]
    - 情绪顶点: [爽感最强的瞬间]
    - 关键画面/台词: [标志性的镜头或台词]
- 辅爽点: [1-2个调剂节奏的辅助爽点]
  - 高光场景示例:
    - [结构同上]
- 其他核心卖点:
  - [创新设定, 极致情绪张力, 反套路设计, 世界观探索感等]

### 9. 开篇章节构思 (Opening Chapter Idea)
- [设计“黄金三章”开篇]
- 第一章: 破局
  - 核心事件: [快速引入主角与困境]
  - 情绪目标: [压抑, 同情, 好奇]
  - 章末钩子: [强力悬念]
- 第二章: 展开
  - 核心事件: [尝试使用金手指，引出更大冲突]
  - 情绪目标: [惊喜, 期待, 展现爽点]
  - 章末钩子: [新角色登场或揭示代价]
- 第三章: 确立
  - 核心事件: [取得初步胜利，确立短期目标]
  - 情绪目标: [强烈爽感和成就感]
  - 章末钩子: [揭示更大世界观或更强对手]

### 10. 市场风险评估 (Market Risk Assessment)
- 核心风险: [识别1-2个最主要的潜在风险, 如: 创意同质化, 设定过于复杂, 慢热不符合平台调性]
- 风险分析: [简要分析风险来源, 例如: “核心设定与【历史成功案例参考】中的某作品相似度较高。”]
- 规避策略: [提出具体的规避建议, 例如: “在开篇三章内, 通过一个强反转事件, 突出与参考案例的根本性区别。”]
"""

NOVEL_CONCEPT_USER_PROMPT = """
---
# 初步选题列表
{selected_opportunity}

# 历史成功案例参考
{historical_success_cases}
"""

@task(name="final_decision",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,  # 7 天过期 (秒)
)
async def task_final_decision(reports: List[Dict[str, str]]) -> FinalDecisionResult:
    logger.info("在多个深度分析报告中进行最终决策...")
    if not reports:
        raise ValueError("task_final_decision需要至少1份报告进行评估。")

    user_prompt_parts = ["请对以下深度分析报告进行评估、排序，并选出最佳方案："]
    for i, report_data in enumerate(reports):
        user_prompt_parts.append(
            f"\n---\n\n# 候选方案 {i+1}: 【{report_data['platform']}】 - 【{report_data['genre']}】\n{report_data['report']}"
        )
    user_prompt = "".join(user_prompt_parts)

    messages = get_llm_messages(SYSTEM_PROMPT=FINAL_DECISION_SYSTEM_PROMPT_JSON, USER_PROMPT=user_prompt)
    llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=0.1)
    response_message = await llm_acompletion(llm_params=llm_params, response_model=FinalDecisionResult)
    decision = response_message.validated_data

    if not decision:
        # llm_acompletion会处理重试和错误，理论上不应该到这里。但作为保险。
        logger.error("最终决策失败，LLM未返回有效结果。")
        raise ValueError("最终决策失败。")

    logger.success(f"最终决策完成。选择: 【{decision.final_choice.platform}】 - 【{decision.final_choice.genre}】")
    return decision

@task(name="deep_dive_analysis",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,  # 7 天过期 (秒)
)
async def task_deep_dive_analysis(platform: str, genre: str, platform_profile: str, broad_scan_report: str, opportunity_report: str) -> Optional[str]:
    logger.info(f"对【{platform} - {genre}】启动深度分析...")
    system_prompt = DEEP_DIVE_SYSTEM_PROMPT.format(
        platform=platform,
        genre=genre,
        platform_profile=platform_profile,
        broad_scan_report=broad_scan_report,
        opportunity_report=opportunity_report
    )
    user_prompt = f"请开始为【{platform}】平台的【{genre}】题材生成深度分析报告。"
    report = await call_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_type='reasoning',
        tools=get_market_tools(),
        temperature=0.1
    )
    if not report:
        logger.error(f"为【{platform} - {genre}】生成深度分析报告失败。")
        return None
    await task_store.submit(
        content=report,
        doc_type="deep_dive_report",
        platform=platform,
        genre=genre
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{platform.replace(' ', '_')}_{genre.replace(' ', '_')}_{timestamp}_deep_dive.md"
    file_path = story_output_dir / file_name
    await asyncio.to_thread(file_path.write_text, report, encoding="utf-8")
    logger.success(f"报告已保存为Markdown文件: {file_path}")
    
    logger.success("深度分析完成！")
    return report

@task(name="generate_opportunities",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,  # 7 天过期 (秒)
)
async def task_generate_opportunities(market_report: str, genre: str) -> Optional[str]:
    logger.info("启动创意脑暴，生成小说选题...")
    logger.info(f"正在查询【{genre}】相关的历史创意库，避免重复...")
    retriever = index.as_retriever(
        similarity_top_k=5,
        filters=MetadataFilters(filters=[ExactMatchFilter(key="type", value="novel_concept")])
    )
    historical_concepts_docs = await retriever.aretrieve(f"{genre} 小说核心创意")
    if historical_concepts_docs:
        historical_concepts_str = "\n\n---\n\n".join([doc.get_content() for doc in historical_concepts_docs])
        logger.success(f"查询到 {len(historical_concepts_docs)} 份历史创意，将用于规避重复。")
    else:
        historical_concepts_str = "无相关历史创意可供参考。"
    user_prompt = OPPORTUNITY_GENERATION_USER_PROMPT.format(
            market_report=market_report,
            historical_concepts=historical_concepts_str
    )
    messages = get_llm_messages(SYSTEM_PROMPT=OPPORTUNITY_GENERATION_SYSTEM_PROMPT, USER_PROMPT=user_prompt)
    llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=0.5)
    response_message = await llm_acompletion(llm_params=llm_params)
    opportunities = response_message.content

    if opportunities:
        logger.success("小说选题生成完毕！")
    else:
        logger.error("生成小说选题失败。")
    return opportunities

@task(name="generate_novel_concept",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,  # 7 天过期 (秒)
)
async def task_generate_novel_concept(opportunities_report: str, platform: str, genre: str) -> Optional[str]:
    logger.info("深化选题，生成详细小说创意...")
    logger.info(f"正在查询【{platform} - {genre}】相关的历史成功案例...")
    retriever = index.as_retriever(
        similarity_top_k=3,
        filters=MetadataFilters(filters=[
            ExactMatchFilter(key="type", value="novel_concept"),
            ExactMatchFilter(key="platform", value=platform)
        ])
    )
    historical_success_docs = await retriever.aretrieve(f"{platform} {genre} 爆款成功小说创意案例")
    if historical_success_docs:
        historical_success_cases_str = "\n\n---\n\n".join([doc.get_content() for doc in historical_success_docs])
        logger.success(f"查询到 {len(historical_success_docs)} 份成功案例，将用于借鉴。")
    else:
        historical_success_cases_str = "无相关历史成功案例可供参考。"
    user_prompt = NOVEL_CONCEPT_USER_PROMPT.format(
            selected_opportunity=opportunities_report,
            historical_success_cases=historical_success_cases_str,
    )
    concept = await call_agent(
        system_prompt=NOVEL_CONCEPT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        llm_type='reasoning',
        tools=get_market_tools(),
        temperature=0.7
    )
    if not concept:
        logger.error("生成详细小说创意失败。")
        return None
    await task_store.submit(
        content=concept,
        doc_type="novel_concept",
        platform=platform,
        genre=genre
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{platform.replace(' ', '_')}_{genre.replace(' ', '_')}_{timestamp}_concept.md"
    file_path = story_output_dir / file_name
    await asyncio.to_thread(file_path.write_text, concept, encoding="utf-8")
    logger.success(f"小说创意已保存为Markdown文件: {file_path}")

    return concept


@flow(name="creative_concept_flow")
async def creative_concept(candidates_to_explore: List[Candidate]):
    if not candidates_to_explore:
        logger.error("未提供任何需要探索的方案，工作流终止。")
        return

    logger.info(f"收到手动指定的 {len(candidates_to_explore)} 个方案，开始第二部分流程: {candidates_to_explore}")
    
    # 注意：下面的任务（task_load_platform_profile, task_platform_briefing, task_new_author_opportunity）
    # 与第一阶段（platform_subject.py）中的任务相同。
    # Prefect的缓存机制（基于result_storage_key）将确保这些任务不会重复执行，而是直接从缓存加载结果，
    # 从而高效地为本流程提供必要的上下文信息。
    platforms_to_process = list(set([c.platform for c in candidates_to_explore]))

    # 重新获取上下文信息 (平台档案、广域扫描报告、新人机会报告)
    # 由于已统一缓存键(result_storage_key)，Prefect的缓存机制将避免重复执行已在第一阶段完成的任务
    profile_futures = await task_load_platform_profile.map(platforms_to_process)
    platform_profiles = {}
    for future in profile_futures:
        platform, content = await future.result()
        platform_profiles[platform] = content

    scan_futures = await task_platform_briefing.map(platforms_to_process)
    opportunity_futures = await task_new_author_opportunity.map(platforms_to_process)

    platform_reports = {}
    opportunity_reports = {}
    for i, platform_name in enumerate(platforms_to_process):
        try:
            report = await scan_futures[i].result()
            platform_reports[platform_name] = report
        except Exception as e:
            logger.error(f"为后续流程获取平台 '{platform_name}' 扫描报告失败: {e}")
            platform_reports[platform_name] = f"## {platform_name} 平台市场动态简报\n\n生成报告时出错: {e}"
        
        try:
            opportunity_report = await opportunity_futures[i].result()
            opportunity_reports[platform_name] = opportunity_report
        except Exception as e:
            logger.error(f"为后续流程获取平台 '{platform_name}' 新人机会报告失败: {e}")
            opportunity_reports[platform_name] = f"## {platform_name} 平台新人机会评估报告\n\n生成报告时出错: {e}"

    # 并行深度钻取
    deep_dive_futures = []
    for candidate in candidates_to_explore:
        future = task_deep_dive_analysis.submit(
            platform=candidate.platform,
            genre=candidate.genre,
            platform_profile=platform_profiles.get(candidate.platform, "无基础信息"),
            broad_scan_report=platform_reports.get(candidate.platform, "无动态简报"),
            opportunity_report=opportunity_reports.get(candidate.platform, "无新人机会评估报告")
        )
        deep_dive_futures.append((candidate, future))

    deep_dive_reports = []
    for candidate, future in deep_dive_futures:
        try:
            report_content = await future.result()
            if report_content:
                deep_dive_reports.append({
                    "platform": candidate.platform,
                    "genre": candidate.genre,
                    "report": report_content
                })
                logger.success(f"完成【{candidate.platform} - {candidate.genre}】的深度分析。")
            else:
                logger.error(f"【{candidate.platform} - {candidate.genre}】的深度分析返回空，将忽略此方案。")
        except Exception as e:
            logger.error(f"【{candidate.platform} - {candidate.genre}】的深度分析失败: {e}，将忽略此方案。")

    if not deep_dive_reports:
        logger.error("所有方案的深度分析均失败，工作流终止。")
        return

    # 最终决策
    if len(deep_dive_reports) == 1:
        final_choice_data = deep_dive_reports[0]
        logger.info("只有一个方案成功完成深度分析，直接采纳。")
        decision_report_content = {
            "platform": final_choice_data["platform"],
            "genre": final_choice_data["genre"],
            "reasoning": "只有一个方案成功完成深度分析，因此被直接采纳。"
        }
        await task_store.submit(
            content=json.dumps(decision_report_content, indent=2, ensure_ascii=False),
            doc_type="final_decision_report",
            platform=final_choice_data["platform"],
            genre=final_choice_data["genre"]
        )
    else:
        final_decision_result = await task_final_decision(deep_dive_reports)
        final_choice_obj = final_decision_result.final_choice
        final_choice_data = {
            "platform": final_choice_obj.platform,
            "genre": final_choice_obj.genre,
            "report": next((r['report'] for r in deep_dive_reports if r['platform'] == final_choice_obj.platform and r['genre'] == final_choice_obj.genre), None)
        }
        logger.info("--- 最终市场方向决策 ---")
        logger.info(f"选择: 【{final_choice_obj.platform}】 - 【{final_choice_obj.genre}】")
        logger.info(f"理由: {final_choice_obj.reasoning}")
        logger.info(f"完整排名:\n{json.dumps([r.model_dump() for r in final_decision_result.ranking], indent=2, ensure_ascii=False)}")
        await task_store.submit(
            content=final_decision_result.model_dump_json(indent=2, ensure_ascii=False),
            doc_type="final_decision_report",
            platform=final_choice_obj.platform,
            genre=final_choice_obj.genre
        )

    chosen_platform = final_choice_data["platform"]
    chosen_genre = final_choice_data["genre"]
    deep_dive_report = final_choice_data["report"]

    if not deep_dive_report:
        logger.error(f"最终选择的方案【{chosen_platform} - {chosen_genre}】没有有效的深度分析报告，工作流终止。")
        return

    # 后续流程 (机会生成, 创意深化)
    logger.info("--- 深度分析报告 (最终选定) ---")
    logger.info(f"\n{deep_dive_report}")

    final_opportunities = await task_generate_opportunities(market_report=deep_dive_report, genre=chosen_genre)
    if not final_opportunities:
        logger.error(f"生成小说选题失败，工作流终止。")
        return

    await task_store.submit(
        content=final_opportunities,
        doc_type="opportunity_generation_report",
        platform=chosen_platform,
        genre=chosen_genre
    )

    logger.info("--- 小说选题建议 ---")
    logger.info(f"\n{final_opportunities}")

    detailed_concept = await task_generate_novel_concept(opportunities_report=final_opportunities, platform=chosen_platform, genre=chosen_genre)
    if not detailed_concept:
        logger.error(f"深化小说创意失败，工作流终止。")
        return

    logger.info("--- 详细小说创意 ---")
    logger.info(f"\n{detailed_concept}")


if __name__ == "__main__":
    # ==================================================================
    # 用户手动配置区域
    # 请根据第一部分流程的输出，在这里填入你想要深度探索的方案
    # 格式: [Candidate(platform="平台名", genre="题材名"), ...]
    # ==================================================================
    candidates_to_run = [
        Candidate(platform="番茄小说", genre="都市脑洞"),
        Candidate(platform="起点中文网", genre="东方玄幻"),
    ]
    # ==================================================================

    flow_run_name = f"creative_concept-{datetime.now().strftime('%Y%m%d')}"
    analysis_flow = creative_concept.with_options(name=flow_run_name)
    asyncio.run(analysis_flow(candidates_to_explore=candidates_to_run))
