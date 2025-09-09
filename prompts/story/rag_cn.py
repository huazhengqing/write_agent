#coding: utf8
import re
import json
from loguru import logger
from typing import Dict, Any, List, Literal
from llama_index.core.base.response.schema import Response
from utils.models import Task


def get_search_config(task: Task, inquiry_plan: Dict[str, Any], search_type: Literal['text_summary', 'upper_design', 'upper_search']) -> Dict[str, Any]:
    logger.info(f"开始 {task.run_id} {task.id} {search_type} \n{json.dumps(inquiry_plan, indent=2, ensure_ascii=False)}")

    current_level = len(task.id.split("."))
    configs = {
        'text_summary': {
            'kg_filters_list': [
                "n.status = 'active'",
                f"n.run_id = '{task.run_id}'",
                "n.content_type = 'write'"
            ],
            'vector_filters_list': [{'key': 'content_type', 'value': 'summary'}],
            'vector_tool_desc': "功能: 检索情节摘要、角色关系、事件发展。范围: 所有层级的历史摘要。",
            'final_instruction': "任务: 整合情节摘要(向量)与正文细节(图谱), 提供写作上下文。重点: 角色关系、关键伏笔、情节呼应。",
            'rules_text': """
# 整合规则
1.  冲突解决: 向量(摘要)与图谱(细节)冲突时, 以图谱为准。
2.  排序依据:
    - 向量: 相关性。
    - 图谱: 章节顺序 (时间线)。
3.  输出要求: 整合为连贯叙述, 禁止罗列。
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
            'vector_tool_desc': f"功能: 检索小说设定、摘要、概念。范围: 任务层级 < {current_level}。",
            'final_instruction': "任务: 整合上层设计, 提供统一、无冲突的宏观设定和指导原则。",
            'rules_text': """
# 整合规则
1.  时序优先: 结果按时间倒序。遇直接矛盾, 采纳最新版本。
2.  矛盾 vs. 细化:
    - 矛盾: 无法共存的描述 (A是孤儿 vs A父母健在)。
    - 细化: 补充细节, 不推翻核心 (A会用剑 -> A擅长流风剑法)。细化应融合, 不是矛盾。
3.  输出要求:
    - 融合非冲突信息。
    - 报告被忽略的冲突旧信息。
    - 禁止罗列, 聚焦问题。
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
            'final_instruction': "任务: 整合外部研究资料, 提供准确、有深度、经过批判性评估的背景支持。",
            'rules_text': """
# 整合规则
1.  评估: 批判性评估所有信息。
2.  冲突处理:
    - 识别并报告矛盾。
    - 列出冲突来源和时间戳。
    - 无法解决时, 保留不确定性。
3.  时效性: `created_at`是评估因素, 但非唯一标准。结果按相关性排序。
4.  输出要求: 组织为连贯报告, 禁止罗列。
""",
            'vector_sort_by': 'relevance',
            'kg_sort_by': 'relevance',
        }
    }

    config = configs[search_type]
    config["kg_filters_str"] = " AND ".join(config['kg_filters_list'])
    config["kg_tool_desc"] = f"功能: 探索实体、关系、路径。查询必须满足过滤条件: {config['kg_filters_str']}。"
    config["query_text"] = _build_agent_query(inquiry_plan, config['final_instruction'], config['rules_text'])

    logger.info(f"结束 \n{json.dumps(config, indent=2, ensure_ascii=False)}")
    return config

def _build_agent_query(inquiry_plan: Dict[str, Any], final_instruction: str, rules_text: str) -> str:
    """
    构建一个结构化的、详细的查询文本, 用于指导 ReAct Agent 或最终的合成 LLM。

    这个查询文本由三部分组成:
    1.  核心探询目标: 来自探询计划的主查询。
    2.  具体信息需求: 将探询计划中的问题列表化, 并标注优先级。
    3.  任务指令与规则: 包含最终目标和具体的执行规则 (如优先级处理、冲突解决等)。
    Args:
        inquiry_plan (Dict[str, Any]): LLM 生成的探询计划。
        final_instruction (str): 针对当前任务的最终目标描述。
        rules_text (str): 针对当前任务的特定整合规则。
    """
    logger.info(f"开始 \n{inquiry_plan}\n{final_instruction}\n{rules_text}")

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

    logger.info(f"结束 \n{query_text}")
    return query_text

def format_response_with_sorting(response: Response, sort_by: Literal['time', 'narrative', 'relevance']) -> str:
    """
    格式化查询响应, 并根据指定策略对来源节点进行排序。
    Args:
        response (Response): 查询引擎返回的响应对象。
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

    logger.info(f"结束 \n{final_output}")
    return final_output