from typing import List, Literal, Optional
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
    complexity_score: Optional[int] = Field(None, description="任务的复杂度预估(1-10), 1为最简单, 10为最复杂。用于辅助原子判断。")
    length: Optional[str] = Field(None, description="对于 'write' 类型的任务, 此任务的预估长度或字数。")
    sub_tasks: List['PlanNode'] = Field(default_factory=list, description="分解出的更深层次的子任务列表。")

class PlanOutput(PlanNode):
    reasoning: Optional[str] = Field(None, description="关于任务分解的推理过程。")

    def to_task(self) -> Task:
        """将 PlanOutput 实例转换为 Task 对象。"""
        new_task = Task(
            id=self.id,
            task_type=self.task_type,
            hierarchical_position=self.hierarchical_position,
            goal=self.goal,
            instructions=self.instructions,
            input_brief=self.input_brief,
            constraints=self.constraints,
            acceptance_criteria=self.acceptance_criteria,
            complexity_score=self.complexity_score,
            length=self.length, 
            status = "pending"
        )
        return new_task



def plan_to_tasks(sub_task_outputs: List[PlanNode], parent_task: Task) -> List[Task]:
    tasks = []
    inherited_props = {
        "parent_id": parent_task.id,
        "run_id": parent_task.run_id,
    }
    for plan_item in sub_task_outputs:
        hierarchical_position = plan_item.hierarchical_position or parent_task.hierarchical_position
        new_task = Task(
            id=plan_item.id,
            task_type=plan_item.task_type,
            goal=plan_item.goal,
            instructions=plan_item.instructions,
            input_brief=plan_item.input_brief,
            constraints=plan_item.constraints,
            acceptance_criteria=plan_item.acceptance_criteria,
            hierarchical_position=hierarchical_position,
            length=plan_item.length,
            status = "pending", 
            **inherited_props
        )
        if plan_item.sub_tasks:
            new_task.sub_tasks = plan_to_tasks(plan_item.sub_tasks, new_task)
        tasks.append(new_task)
    return tasks
