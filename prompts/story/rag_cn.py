#coding: utf8
import re
import json
from typing import Dict, Any, List, Literal
from llama_index.core.base.response.schema import Response
from utils.models import Task


def get_search_config(task: Task, inquiry_plan: Dict[str, Any], search_type: Literal['text_summary', 'upper_design', 'upper_search']) -> Dict[str, Any]:
    current_level = len(task.id.split("."))
    configs = {
        'text_summary': {
            'kg_filters_list': [
                "n.status = 'active'",
                f"n.run_id = '{task.run_id}'",
                "n.content_type = 'write'"
            ],
            'vector_filters_list': [{'key': 'content_type', 'value': 'summary'}],
            'vector_tool_desc': "功能: 检索小说情节摘要、角色关系、事件发展。范围: 所有层级的历史摘要。",
            'final_instruction': "角色: 剧情连续性编辑。\n任务: 整合情节摘要(向量)与正文细节(图谱), 生成一份服务于续写的上下文报告。\n重点: 角色关系、关键伏笔、情节呼应。",
            'rules_text': """
# 整合规则
1.  冲突解决: 向量(摘要)与图谱(细节)冲突时, 以图谱为准。
2.  排序依据:
    - 向量: 相关性。
    - 图谱: 章节顺序 (时间线)。
3.  输出要求: 整合为连贯叙述, 禁止罗列。

# 输出结构
- 核心上下文: [一段连贯的叙述, 总结最重要的背景信息]
- 关键要点: (可选) [以列表形式补充说明关键的角色状态、伏笔或设定]
""",
            'vector_sort_by': 'narrative',
            'kg_sort_by': 'narrative',
        },
        'upper_design': {
            'kg_filters_list': [
                "n.status = 'active'",
                f"n.run_id = '{task.run_id}'",
                "n.content_type = 'design'",
                f"n.hierarchy_level < {current_level}"
            ],
            'vector_filters_list': [{'key': 'content_type', 'value': 'design'}],
            'vector_tool_desc': f"功能: 检索上层的宏观小说设定、故事大纲、核心概念。范围: 任务层级 < {current_level}。",
            'final_instruction': "角色: 首席故事架构师。\n任务: 整合上层设计, 提炼统一、无冲突的宏观设定和指导原则。",
            'rules_text': """
# 整合规则
1.  时序优先: 结果按时间倒序。遇直接矛盾, 采纳最新版本。
2.  矛盾 vs. 细化:
    - 矛盾: 无法共存的描述 (A是孤儿 vs A父母健在)。
    - 细化: 补充细节, 不推翻核心 (A会用剑 -> A擅长流风剑法)。细化应融合, 不是矛盾。
3.  输出要求: 融合非冲突信息, 报告被忽略的冲突旧信息, 禁止罗列, 聚焦问题。

# 输出结构
- 统一设定: [以要点形式, 清晰列出整合后的最终设定]
- 设计演变与冲突: (可选) [简要说明关键设定的演变过程, 或指出已解决的重大设计矛盾]
""",
            'vector_sort_by': 'time',
            'kg_sort_by': 'time',
        },
        'upper_search': {
            'kg_filters_list': [
                "n.status = 'active'",
                f"n.run_id = '{task.run_id}'",
                "n.content_type = 'search'",
                f"n.hierarchy_level < {current_level}"
            ],
            'vector_filters_list': [{'key': 'content_type', 'value': 'search'}],
            'vector_tool_desc': f"功能: 从外部研究资料库检索事实、概念、历史事件。范围: 任务层级 < {current_level}。",
            'final_instruction': "角色: 研究分析师。\n任务: 整合外部研究资料, 提供准确、有深度、经过批判性评估的背景支持。",
            'rules_text': """
# 整合规则
1.  评估: 批判性评估所有信息。
2.  冲突处理:
    - 识别并报告矛盾。
    - 列出冲突来源和时间戳。
    - 无法解决时, 保留不确定性并提出。
3.  时效性: `created_at`是评估因素, 但非唯一标准。结果按相关性排序。
4.  输出要求: 组织为连贯报告, 禁止罗列。

# 输出结构
- 核心结论: [提炼1-3条最重要的研究结论]
- 详细摘要: [按主题组织详细信息]
- 矛盾与不确定性: [单独列出信息冲突点或仍存在疑问的地方]
""",
            'vector_sort_by': 'relevance',
            'kg_sort_by': 'relevance',
        }
    }
    config = configs[search_type]
    config["kg_filters_str"] = " AND ".join(config['kg_filters_list'])
    config["kg_tool_desc"] = f"功能: 探索实体、关系、路径。查询必须满足过滤条件: {config['kg_filters_str']}。"
    config["query_text"] = _build_agent_query(inquiry_plan, config['final_instruction'], config['rules_text'])
    return config


def _build_agent_query(inquiry_plan: Dict[str, Any], final_instruction: str, rules_text: str) -> str:
    main_inquiry = inquiry_plan.get("main_inquiry", "请综合分析并回答以下问题。")

    # 1. 构建“核心探询目标”和“具体信息需求”部分
    query_text = f"# 核心探询目标\n{main_inquiry}\n\n# 具体信息需求 (按优先级降序排列)\n"
    has_priorities = False
    questions_dict = inquiry_plan.get("questions", {})
    # 按 high, medium, low 排序问题
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_questions = sorted(questions_dict.items(), key=lambda item: priority_order.get(item[1], 99))

    for question, priority in sorted_questions:
        if priority in ['high', 'low']:
            has_priorities = True # 检查是否存在高/低优先级, 以便后续添加规则
        query_text += f"- {question} (优先级: {priority})\n"

    # 2. 构建“任务指令与规则”部分
    instruction_block = f"\n# 任务指令与规则\n"
    instruction_block += f"## 最终目标\n{final_instruction}\n"
    instruction_block += f"\n## 执行规则\n"
    if has_priorities:
        instruction_block += "- 优先级: 你必须优先分析和回答标记为 `high` 优先级的信息需求。\n"

    # 动态调整传入规则的 Markdown 标题层级, 使其能正确嵌入到当前结构中
    adapted_rules_text = re.sub(r'^\s*#\s+', '### ', rules_text.lstrip(), count=1)
    instruction_block += adapted_rules_text

    query_text += instruction_block
    return query_text


###############################################################################


def format_response_with_sorting(response: Response, sort_by: Literal['time', 'narrative', 'relevance']) -> str:
    """
        sort_by (Literal): 排序策略: 'time' (时间倒序), 'narrative' (章节顺序), 'relevance' (相关性)。
    """
    if not response.source_nodes:
        return f"未找到相关来源信息, 但综合回答是: \n{str(response)}"

    # 默认使用 LlamaIndex 返回的顺序 (通常是按相关性)
    sorted_nodes = response.source_nodes
    sort_description = ""

    if sort_by == 'narrative':
        # 定义一个“自然排序”的 key 函数, 能正确处理 '1.2' 和 '1.10' 这样的章节号
        def natural_sort_key(s: str) -> List[Any]:
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        sorted_nodes = sorted(
            list(response.source_nodes),
            key=lambda n: natural_sort_key(n.metadata.get("task_id", "")),
            reverse=False  # 正序排列
        )
        sort_description = "按小说章节顺序排列 (从前到后)"
    elif sort_by == 'time':
        # 按元数据中的 'created_at' 时间戳对来源节点进行降序排序
        sorted_nodes = sorted(
            list(response.source_nodes),
            key=lambda n: n.metadata.get("created_at", "1970-01-01T00:00:00"),
            reverse=True  # 倒序排列, 最新的在前
        )
        sort_description = "按时间倒序排列 (最新的在前)"
    else: # 'relevance' 或其他默认情况
        sort_description = "按相关性排序"

    # 格式化每个来源节点的详细信息
    source_details = []
    for node in sorted_nodes:
        timestamp = node.metadata.get("created_at", "未知时间")
        task_id = node.metadata.get("task_id", "未知章节")
        score = node.get_score()
        score_str = f"{score:.4f}" if score is not None else "N/A"
        # 将文本中的多个空白符合并为一个空格, 以简化输出
        content = re.sub(r'\s+', ' ', node.get_content()).strip()
        source_details.append(f"来源信息 (章节: {task_id}, 时间: {timestamp}, 相关性: {score_str}):\n---\n{content}\n---")

    formatted_sources = "\n\n".join(source_details)

    # 将综合回答和详细来源信息组合成最终输出
    final_output = f"综合回答:\n{str(response)}\n\n详细来源 ({sort_description}):\n{formatted_sources}"
    return final_output


###############################################################################


kg_gen_query_prompt = """
# 角色
你是一位精通 Cypher 的图数据库查询专家。

# 任务
根据用户提供的自然语言问题和图谱 Schema, 生成一条精确、高效、且符合所有规则的 Cypher 查询语句。

# 上下文
- 用户问题: '{query_str}'
- 图谱 Schema:
---
{schema}
---

# 核心规则 (必须严格遵守)
1.  强制过滤 (最重要!): 查询必须包含 `WHERE` 子句, 且该子句必须包含过滤条件: `{kg_filters_str}`。
2.  Schema遵从: 仅使用 Schema 中定义的节点标签和关系类型。
3.  单行输出: Cypher 查询必须是单行文本, 无换行。
4.  效率优先: 生成的查询应尽可能高效。
5.  无效处理: 若问题无法基于 Schema 回答, 固定返回字符串 "INVALID_QUERY"。

# 示例
- 用户问题: '角色"龙傲天"和"赵日天"是什么关系?'
- Cypher 查询: MATCH (a:角色 {{name: "龙傲天"}})-[r]-(b:角色 {{name: "赵日天"}}) WHERE {kg_filters_str} RETURN type(r)

# 行动
现在, 请为上述用户问题生成 Cypher 查询语句。
"""


###############################################################################


react_system_prompt = """
# 角色
你是一个高级研究分析师AI, 专为复杂的小说创作背景研究而设计。

# 核心任务
严格遵循“任务指令与规则”中的“最终目标”, 拆解“核心探询目标”, 策略性地使用可用工具, 最终生成一个全面、整合、且直接回答核心目标的答案。

# 可用工具
1.  `time_aware_vector_search`:
    - 功能: 语义搜索。用于检索与情节、上下文、主题、角色动机、设计理念相关的描述性信息。
    - 适用场景: 当你需要理解“为什么”、“怎么样”或寻找模糊、概念性的信息时使用。
2.  `time_aware_knowledge_graph_search`:
    - 功能: 实体关系搜索。用于查询特定实体 (角色、地点、物品) 之间的确切关系、属性和连接。
    - 适用场景: 当你需要查询“是谁”、“在哪里”、“拥有什么”等事实性、结构化的信息时使用。

# 思考流程 (Thought Process)
1.  分解: 首先, 仔细分析“核心探询目标”和“具体信息需求”。
2.  规划: 制定一个清晰的行动计划。我应该先用哪个工具? 为了回答主要问题, 我需要先查明哪些子问题?
3.  行动 (Action): 调用最合适的工具并提出精确的问题。
    - 如果问题复杂, 我会先用 `time_aware_vector_search` 获取宏观背景, 再用 `time_aware_knowledge_graph_search` 查证具体关系。
    - 如果问题直接, 我会选择最匹配的工具直接查询。
4.  观察 (Observation): 评估工具返回的结果。信息是否足够? 是否回答了我的问题?
5.  迭代或总结:
    - 如果信息不足或引出了新问题, 我会继续思考并执行新的“行动”。
    - 如果信息已经足够, 我将停止使用工具, 并基于所有观察到的信息, 结合“任务指令与规则”, 形成最终的、整合性的答案。

# 输出要求
- 最终答案必须是针对“核心探询目标”的直接、连贯的回答, 而不是工具输出的简单罗列。
- 你的所有思考和最终答案都必须严格遵循用户在“任务指令与规则”中提供的指示。
"""


###############################################################################


synthesis_system_prompt = "角色: 信息整合分析师。任务: 遵循用户指令, 整合并提炼向量检索和知识图谱的信息。输出: 一个逻辑连贯、事实准确、完全基于所提供材料的最终回答。"


synthesis_user_prompt = """
# 任务
- 遵循“探询计划与规则”。
- 整合“向量检索”和“知识图谱检索”的信息。
- 生成一个连贯、统一、直接回应“核心探询目标”的最终回答。

# 探询计划与规则
{query_text}

# 信息源
## 向量检索 (语义与上下文)
{formatted_vector_str}
---
## 知识图谱检索 (事实与关系)
{formatted_kg_str}
---

# 输出要求
- 严格遵循“探询计划与规则”中的所有指令。
- 必须完全基于提供的信息源进行整合提炼, 禁止罗列。
- 禁止任何关于你自身或任务过程的描述 (例如, “根据您的要求...”)。
"""


###############################################################################

