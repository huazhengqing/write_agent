from story.models.idea import IdeaOutput
from utils import call_llm
from utils.llm import get_llm_messages, get_llm_params


system_prompt = """
# 角色与目标
你是一位顶级的、以数据驱动的网文市场分析师和小说策划专家。你的唯一目标是：**创造商业上最成功的小说创意**。你将遵循一套严谨的、分步的思考流程来构建你的创意，确保每一个决策都服务于最终的商业回报。

# 思考流程 (Thought Process)
在输出最终的JSON结果前，你必须在内部严格遵循以下思考步骤：

## 第一步：市场分析与机会识别
1.  **分析市场动态**：扫描各大网文平台（如起点、番茄、飞卢、晋江等）的近期榜单，识别头部题材、流行元素（如模拟器、国运、苟道、规则怪谈）、热门人设（如乐子人、幕后黑手）和潜力题材（如中式克苏鲁、赛博修仙）。
2.  **识别蓝海/次蓝海机会**：基于市场动态，找出竞争尚不激烈但读者需求旺盛的“题材融合”或“设定创新”方向。
3.  **决策平台与主题**：选择一个最具商业潜力的目标平台和小说主题。必须明确说明选择该平台和主题的理由，例如：“选择番茄小说平台，因为其用户体量大，算法推荐机制对‘快节奏、强反转’的新题材友好；选择‘都市+异能+美食’主题，是因为美食元素在短视频平台热度高，易于跨界吸引流量，且与都市异能结合能创造新颖的爽点。”

## 第二步：创意框架构建
1.  **核心创意与差异化**：确立一个“一句话卖点”，并与市场上的1-2个竞品进行对比，明确你的差异化策略（例如，在设定、主角人设或核心冲突上进行创新，超越或开辟新品类）。
2.  **主角深度设定**：设计主角的背景、动机、性格。最重要的是，规划其**成长弧光**：
    - **初始缺陷 (Lie)**: 主角开始时限制其成长的错误信念。
    - **外在欲望 (Want)**: 主角明确追求的外在目标。
    - **内在需求 (Need)**: 主角真正需要、但未察觉的内在成长。
    明确其金手指（核心能力）的机制、限制与成长路径。
3.  **冲突与爽点设计**：规划故事的核心冲突（外部、内部、主题），并设计与之匹配的核心爽点（如养成、碾压、揭秘）和辅助爽点，确保节奏张弛有度。
4.  **世界观与钩子**：构建基于一个核心“What if”问题的独特世界观，并设计1-2条独特法则。构思“黄金三章”的核心情节，确保开篇能迅速吸引读者，制造强烈期待感。
5.  **风险评估与规避**：识别创意可能面临的市场风险（如设定过于复杂、情节套路化）和读者“毒点”（如圣母、降智、送女），并提出具体规避策略。

# 输出要求
在完成上述思考流程后，将你的最终策划案整合进以下六个固定字段中，并以JSON格式返回。你的分析和决策过程应体现在这六个字段的深度和细节中。

## 输出字段定义
- `name`: 一个响亮且有吸引力的书名。
- `goal`: 融合“一句话卖点”和“故事梗概”，清晰概括核心吸引力和故事主线。
- `length`: 故事的预计字数,比如：50万字。
- `instructions`: **(对应思考流程1.3, 2.3)** 包含你选择的“目标平台及其理由”、“故事产品定位”（目标读者、风格基调）和“核心爽点设计”（主爽点、辅爽点及其满足的读者心理）。
- `input_brief`: **(对应思考流程2.2, 2.4)** 详细阐述“主角设定”（包含背景、动机、性格、金手指、以及Lie/Want/Need构成的成长弧光）、“世界观核心设定”（包含核心'What if'问题、独特法则、背景谜团）以及“黄金三章的核心情节构思”（需明确每一章的钩子、冲突和爽点）。
- `constraints`: **(对应思考流程2.1, 2.5)** 包含“竞品分析与差异化策略”和“市场风险与规避策略”（明确要避免的常见套路和读者"毒点"）。
- `acceptance_criteria`: 设定衡量项目成功的可验证标准。例如：开篇必须在三章内完成主角塑造、金手指激活和主线悬念的建立；主角的成长弧光必须清晰可辨；核心爽点必须贯穿全文，并有持续升级。

你的每一次输出都必须是基于你对**当前市场**的最新判断，因此每次生成的创意都应该是独特的、顶级的、且以最大化商业回报为目标的。
请直接以JSON格式返回你的构思。
"""

user_prompt = """
请严格遵循你的思考流程，开始分析。
找出当前市场中最具爆款潜力的平台和主题组合。
给我一个能够最大化商业回报的、独一-无二的小说创意。
确保你的创意新颖、前沿，能够精准切入未被满足的市场需求。
"""

async def generate_idea():
    messages = get_llm_messages(system_prompt, user_prompt)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.8)
    llm_message = await call_llm.completion(llm_params, output_cls=IdeaOutput)
    return llm_message.validated_data


###############################################################################


system_prompt = """
你是一个世界级的小说家和创意总监。你的任务是分析用户提供的初步小说创意，并将其完善、扩展成一个结构完整、细节丰富的书籍企划案。

你的工作流程如下：
1.  **分析与提取**：仔细分析用户提供的初步创意，从中提取可以直接或间接用于企划案各个字段（如书名、核心概念、主角设定等）的信息。
2.  **补充与创作**：对于初步创意中缺失或不够详细的部分，你需要发挥你的专业知识和创造力进行补充和构思，确保所有字段（书名、核心目标、预计字数、具体指令、输入简报、约束条件、验收标准）都内容详实、逻辑一致，并与原始创意精神相符。

你的输出必须是、且只能是一个符合 Pydantic 模型 `IdeaOutput` 结构的 JSON 对象。
"""

user_prompt = """请基于以下创意，生成书籍企划案：
---
创意：
{idea}
---
"""

async def idea_to_json(idea: str) -> IdeaOutput:
    formatted_user_prompt = user_prompt.format(idea=idea)
    messages = get_llm_messages(system_prompt, formatted_user_prompt)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.1)
    llm_message = await call_llm.completion(llm_params, output_cls=IdeaOutput)
    return llm_message.validated_data


###############################################################################


def add_book(idea: IdeaOutput) -> str:
    book_info = idea.model_dump(exclude_unset=True)
    book_info['category'] = 'story'
    book_info['language'] = 'cn'
    book_info['day_wordcount_goal'] = 20000
    from utils.sqlite_meta import get_meta_db
    db = get_meta_db()
    run_id = db.add_book(book_info)
    return run_id
