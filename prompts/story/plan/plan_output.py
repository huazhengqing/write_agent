from typing import List, Optional
from pydantic import BaseModel, Field
from utils.models import Task, TaskType


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
