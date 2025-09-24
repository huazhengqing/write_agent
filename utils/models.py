from functools import lru_cache
from typing import List, Optional, Dict, Literal, Any, get_args
from pydantic import BaseModel, Field, conlist



###############################################################################


CategoryType = Literal["story", "book", "report"]
TaskType = Literal["write", "design", "search"]
LanguageType = Literal["cn", "en"]


class Task(BaseModel):
    id: str = Field(..., description="任务的唯一标识符, 采用层级结构, 父任务id.子任务序号。如 '1', '1.1', '1.2.1'")
    parent_id: str = Field(..., description="父任务的ID")
    task_type: TaskType = Field(..., description="任务类型: 'write'写作, 'design'设计, 'search'搜索") 
    hierarchical_position: Optional[str] = Field(None, description="任务在书/故事结构中的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。")
    goal: str = Field(..., description="任务需要达成的[核心目标](一句话概括)。")
    instructions: List[str] = Field(default_factory=list, description="任务的[具体指令](HOW): 明确指出需要执行的步骤、包含的关键要素或信息点。")
    input_brief: List[str] = Field(default_factory=list, description="任务的[输入指引](FROM WHERE): 指导执行者应重点关注依赖项中的哪些关键信息。")
    constraints: List[str] = Field(default_factory=list, description="任务的[限制和禁忌](WHAT NOT): 明确指出需要避免的内容或必须遵守的规则。")
    acceptance_criteria: List[str] = Field(default_factory=list, description="任务的[验收标准](VERIFY HOW): 定义任务完成的衡量标准, 用于后续评审。")
    length: Optional[str] = Field(None, description="预估产出字数 (仅 'write' 任务)")
    dependency: List[str] = Field(default_factory=list, description="执行前必须完成的同级任务ID列表")
    sub_tasks: List['Task'] = Field(default_factory=list, description="所有子任务的列表")
    results: Dict[str, Any] = Field(default_factory=dict, description="任务执行后的产出结果, 可以是文本或结构化数据")
    
    category: CategoryType = Field(..., description="任务类别 (例如 'story', 'book', 'report')")
    language: LanguageType = Field(..., description="任务语言 (例如 'cn', 'en')")
    root_name: str = Field(..., description="根任务的名字, 书名, 报告名")
    day_wordcount_goal: Optional[int] = Field(0, description="每日字数目标, 0表示无限制")

    run_id: str = Field(..., description="整个流程运行的唯一ID, 用于隔离不同任务的记忆")



###############################################################################



ComplexReason = Literal[
    # write
    "design_insufficient",
    "length_excessive",
    # design
    "dependency_insufficient",
    "need_search",
    "composite_goal",
    # search
    "broad_topic",
    "requires_analysis",
    "vague_goal"
]


class AtomOutput(BaseModel):
    reasoning: Optional[str] = Field(None, description="关于任务是原子还是复杂的推理过程。")
    update_goal: Optional[str] = Field(None, description="在分析了任务后, 对原始[核心目标]的优化或澄清。如果LLM认为不需要修改, 则此字段可以省略。")
    update_instructions: Optional[List[str]] = Field(None, description="对[具体指令]的优化或补充。")
    update_input_brief: Optional[List[str]] = Field(None, description="对[输入指引]的优化或补充。")
    update_constraints: Optional[List[str]] = Field(None, description="对[限制和禁忌]的优化或补充。")
    update_acceptance_criteria: Optional[List[str]] = Field(None, description="对[验收标准]的优化或补充。")
    atom_result: Literal['atom', 'complex'] = Field(description="判断任务是否为原子任务的结果, 值必须是 'atom' 或 'complex'。")
    complex_reasons: Optional[conlist(item_type=ComplexReason, min_length=1)] = Field(None, description="当任务被判定为 'complex' 时, 此字段列出具体原因。")


###############################################################################



class PlanNode(BaseModel):
    id: str = Field(..., description="任务的唯一字符串ID, 父任务id.子任务序号。例如 '1' 或 '1.3.2'。")
    task_type: TaskType = Field(..., description="任务类型, 值必须是: 'design' 或 'write' 或 'search'。")
    hierarchical_position: Optional[str] = Field(None, description="任务在书/故事结构中的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。")
    goal: str = Field(..., description="任务的清晰、具体的[核心目标](一句话概括)。")
    instructions: List[str] = Field(default_factory=list, description="任务的[具体指令](HOW): 明确指出需要执行的步骤、包含的关键要素或信息点。")
    input_brief: List[str] = Field(default_factory=list, description="任务的[输入指引](FROM WHERE): 指导执行者应重点关注依赖项中的哪些关键信息。")
    constraints: List[str] = Field(default_factory=list, description="任务的[限制和禁忌](WHAT NOT): 明确指出需要避免的内容或必须遵守的规则。")
    acceptance_criteria: List[str] = Field(default_factory=list, description="任务的[验收标准](VERIFY HOW): 定义任务完成的衡量标准, 用于后续评审。")
    dependency: List[str] = Field(default_factory=list, description="此任务所依赖的同层的 design/search 的 id 列表。")
    length: Optional[str] = Field(None, description="对于 'write' 类型的任务, 此任务的预估长度或字数。")
    sub_tasks: List['PlanNode'] = Field(default_factory=list, description="分解出的更深层次的子任务列表。")


class PlanOutput(PlanNode):
    reasoning: Optional[str] = Field(None, description="关于任务分解的推理过程。")



###############################################################################



RouteCategory = Literal[
    "market", 
    "title", 
    "style", 
    "general", 
    "scene_atmosphere", 
    "faction_culture", 
    "power_system", 
    "narrative_pacing", 
    "thematic_imagery", 
    "hierarchy", 
    "trend_integration"
]


class RouteOutput(BaseModel):
    categories: List[RouteCategory] = Field(description=f"判断出的任务类型列表。列表中的每个元素都必须是 {get_args(RouteCategory)} 之一。对于复合任务, 可以返回多个类别。")


###############################################################################



def convert_plan_to_tasks(sub_task_outputs: List[PlanNode], parent_task: Task) -> List[Task]:
    tasks = []
    inherited_props = {
        "parent_id": parent_task.id,
        "category": parent_task.category,
        "language": parent_task.language,
        "root_name": parent_task.root_name,
        "run_id": parent_task.run_id,
        "day_wordcount_goal": parent_task.day_wordcount_goal,
    }
    for plan_item in sub_task_outputs:
        new_task = Task(
            id=plan_item.id,
            task_type=plan_item.task_type,
            goal=plan_item.goal,
            instructions=plan_item.instructions,
            input_brief=plan_item.input_brief,
            constraints=plan_item.constraints,
            acceptance_criteria=plan_item.acceptance_criteria,
            dependency=plan_item.dependency,
            hierarchical_position=plan_item.hierarchical_position,
            length=plan_item.length,
            **inherited_props
        )
        if not new_task.hierarchical_position:
            new_task.hierarchical_position = parent_task.hierarchical_position
        if plan_item.sub_tasks:
            new_task.sub_tasks = convert_plan_to_tasks(plan_item.sub_tasks, new_task)
        tasks.append(new_task)
    return tasks



###############################################################################



@lru_cache(maxsize=30)
def natural_sort_key(task_id: str) -> List[int]:
    """为任务ID字符串提供健壮的自然排序键, 处理空或格式错误的ID。"""
    if not task_id:
        return []
    try:
        # 过滤掉拆分后可能产生的空字符串(如 '1.'), 并转换为整数列表
        return [int(p) for p in task_id.split('.') if p]
    except ValueError:
        # 如果ID格式错误(包含非数字), 返回[]。
        # 在排序中, 这会使无效ID排在最后。
        return []



@lru_cache(maxsize=30)
def get_sibling_ids_up_to_current(task_id: str) -> List[str]:
    """
    根据任务ID, 生成从第一个兄弟任务到其自身的所有兄弟任务的ID列表。
    例如, 对于 '1.2.3', 它会生成 ['1.2.1', '1.2.2', '1.2.3']。
    """
    if '.' not in task_id:
        # 如果没有'.', 说明是根任务或顶级任务, 返回它自己
        return [task_id]
    parts = task_id.split('.')
    parent_id = ".".join(parts[:-1])
    try:
        current_seq = int(parts[-1])
    except (ValueError, IndexError):
        return []
    if current_seq <= 1:
        return [task_id]
    return [f"{parent_id}.{i}" for i in range(1, current_seq + 1)]
