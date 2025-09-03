from typing import List, Optional, Dict, Literal, Any
from pydantic import BaseModel, Field


CategoryType = Literal["story", "book", "report"]
TaskType = Literal["write", "design", "search"]
LanguageType = Literal["cn", "en"]


class Task(BaseModel):
    id: str = Field(..., description="任务的唯一标识符, 采用层级结构, 父任务id.子任务序号。如 '1', '1.1', '1.2.1'")
    parent_id: str = Field(..., description="父任务的ID")
    task_type: TaskType = Field(..., description="任务类型：'write'写作, 'design'设计, 'search'搜索")
    goal: str = Field(..., description="任务需要达成的具体目标")
    length: Optional[str] = Field(None, description="预估产出字数 (仅 'write' 任务)")
    dependency: List[str] = Field(default_factory=list, description="执行前必须完成的同级任务ID列表")
    sub_tasks: List['Task'] = Field(default_factory=list, description="所有子任务的列表")
    results: Dict[str, Any] = Field(default_factory=dict, description="任务执行后的产出结果, 可以是文本或结构化数据")
    
    category: CategoryType = Field(..., description="任务类别 (例如 'story', 'book', 'report')")
    language: LanguageType = Field(..., description="任务语言 (例如 'cn', 'en')")
    root_name: str = Field(..., description="根任务的名字, 书名, 报告名")

    run_id: str = Field(..., description="整个流程运行的唯一ID, 用于隔离不同任务的记忆")


