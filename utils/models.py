from typing import List, Optional, Dict, Literal, Any
from pydantic import BaseModel, Field


CategoryType = Literal["story", "book", "report"]
TaskType = Literal["write", "design", "search"]
LanguageType = Literal["cn", "en"]


class Task(BaseModel):
    id: str = Field(..., description="任务的唯一标识符, 采用层级结构, 父任务id.子任务序号。如 '1', '1.1', '1.2.1'")
    parent_id: str = Field(..., description="父任务的ID")
    task_type: TaskType = Field(..., description="任务类型: 'write'写作, 'design'设计, 'search'搜索") 
    hierarchical_position: Optional[str] = Field(None, description="任务在书/故事结构中的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。")
    goal: str = Field(..., description="任务需要达成的具体目标")
    length: Optional[str] = Field(None, description="预估产出字数 (仅 'write' 任务)")
    dependency: List[str] = Field(default_factory=list, description="执行前必须完成的同级任务ID列表")
    sub_tasks: List['Task'] = Field(default_factory=list, description="所有子任务的列表")
    results: Dict[str, Any] = Field(default_factory=dict, description="任务执行后的产出结果, 可以是文本或结构化数据")
    
    category: CategoryType = Field(..., description="任务类别 (例如 'story', 'book', 'report')")
    language: LanguageType = Field(..., description="任务语言 (例如 'cn', 'en')")
    root_name: str = Field(..., description="根任务的名字, 书名, 报告名")

    run_id: str = Field(..., description="整个流程运行的唯一ID, 用于隔离不同任务的记忆")


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


def get_preceding_sibling_ids(task_id: str) -> List[str]:
    """
    根据任务ID, 生成其所有前序兄弟任务的ID列表。
    例如, 对于 '1.2.3', 它会生成 ['1.2.1', '1.2.2']。
    """
    if '.' not in task_id:
        return []
    parts = task_id.split('.')
    parent_id = ".".join(parts[:-1])
    try:
        current_seq = int(parts[-1])
    except (ValueError, IndexError):
        return []
    if current_seq <= 1:
        return []
    return [f"{parent_id}.{i}" for i in range(1, current_seq)]






