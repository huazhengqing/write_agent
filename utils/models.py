from functools import lru_cache
from datetime import datetime
from typing import List, Optional, Dict, Literal, Any
from pydantic import BaseModel, Field


CategoryType = Literal["story", "book", "report"]
TaskType = Literal["write", "design", "search"]
LanguageType = Literal["cn", "en"]



"""
待处理 (Pending): 任务已创建，但尚未开始执行。
执行中 (Running): 任务正在执行。
已完成 (Completed): 任务已成功完成。
已失败 (Failed): 任务执行失败。
已取消 (Cancelled): 任务被取消执行。
已暂停 (Paused): 任务执行被暂停。
"""
TaskStatusType = Literal["pending", "running", "completed", "failed", "cancelled", "paused"]



class Task(BaseModel):
    id: str = Field(..., description="任务的唯一标识符, 采用层级结构, 父任务id.子任务序号。如 '1', '1.1', '1.2.1'")
    parent_id: str = Field(..., description="父任务的ID")
    task_type: TaskType = Field(..., description="任务类型: 'write'写作, 'design'设计, 'search'搜索") 
    hierarchical_position: Optional[str] = Field(None, description="任务在书/故事结构中的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。")
    goal: str = Field("", description="任务需要达成的[核心目标](一句话概括)。")
    instructions: str = Field("", description="任务的[具体指令](HOW): 明确指出需要执行的步骤、包含的关键要素或信息点。")
    input_brief: str = Field("", description="任务的[输入指引](FROM WHERE): 指导执行者应重点关注依赖项中的哪些关键信息。")
    constraints: str = Field("", description="任务的[限制和禁忌](WHAT NOT): 明确指出需要避免的内容或必须遵守的规则。")
    acceptance_criteria: str = Field("", description="任务的[验收标准](VERIFY HOW): 定义任务完成的衡量标准, 用于后续评审。")
    length: Optional[str] = Field(None, description="预估产出字数 (仅 'write' 任务)")
    dependency: List[str] = Field(default_factory=list, description="执行前必须完成的同级任务ID列表")
    sub_tasks: List['Task'] = Field(default_factory=list, description="所有子任务的列表")
    results: Dict[str, Any] = Field(default_factory=dict, description="任务执行后的产出结果, 可以是文本或结构化数据")
    status: TaskStatusType = Field("pending", description="任务状态: 'pending', 'running', 'completed', 'failed', 'cancelled', 'paused'")
    run_id: str = Field(..., description="整个流程运行的唯一ID, 用于隔离不同任务的记忆")

    def to_context(self) -> str:
        """将Task对象转换为用于LLM上下文的Markdown格式字符串。"""
        context_parts = []
        context_parts.append(f"### 任务: {self.id} - {self.hierarchical_position}")
        context_parts.append(f"**类型**: {self.task_type}")
        if self.length:
            context_parts.append(f"**预估长度**: {self.length}")
        if self.goal and self.goal.strip():
            context_parts.append(f"#### 核心目标\n{self.goal}")
        if self.instructions and self.instructions.strip():
            context_parts.append(f"#### 具体指令\n{self.instructions}")
        if self.input_brief and self.input_brief.strip():
            context_parts.append(f"#### 输入指引\n{self.input_brief}")
        if self.constraints and self.constraints.strip():
            context_parts.append(f"#### 限制和禁忌\n{self.constraints}")
        if self.acceptance_criteria and self.acceptance_criteria.strip():
            context_parts.append(f"#### 验收标准\n{self.acceptance_criteria}")
        return "\n\n".join(context_parts)




def get_parent_id(task_id: str) -> str:
    """
    根据任务ID, 获取其父任务ID。
    例如, 对于 '1.2.3', 它会返回 '1.2'。
    对于 '1', 它会返回 ''。
    """
    if not task_id or '.' not in task_id:
        return ""
    # 使用 rsplit 提高效率, 只分割一次
    parent_id, _ = task_id.rsplit('.', 1)
    return parent_id



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
