import os
import litellm
from loguru import logger
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, ValidationError
from ..util.models import Task, TaskStatus
from ..util.llm import get_llm_params
from ..memory import memory, get_llm_messages


class PlanOutput(BaseModel):
    id: str = Field(..., description="任务的唯一字符串ID, 父任务id.子任务序号。例如 '1' 或 '1.3.2'。")
    task_type: Literal['design', 'write', 'search'] = Field(..., description="任务类型, 值必须是: 'design' 或 'write' 或 'search'。")
    goal: str = Field(..., description="任务的清晰具体目标。")
    dependency: List[str] = Field(default_factory=list, description="此任务所依赖的同层的 design/search 的 id 列表。")
    length: Optional[str] = Field(None, description="对于 'write' 类型的任务, 此任务的预估长度或字数。")
    sub_tasks: List['PlanOutput'] = Field(default_factory=list, description="分解出的更深层次的子任务列表。")


async def plan(task: Task) -> Task:
    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    
    if task.task_type == "design":
        from ..prompts.story.plan_design_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.task_type == "search":
        from ..prompts.story.plan_search_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.task_type == "write":
        if not task.length:
            raise ValueError("Task length must be set.")
        
        if os.getenv("deployment_environment") == "test":
            from ..prompts.story.plan_write_cn import SYSTEM_PROMPT, USER_PROMPT, test_get_task_level
            messages = await get_llm_messages(task, SYSTEM_PROMPT.format(**test_get_task_level(task.goal)), USER_PROMPT)
        else:
            from ..prompts.story.plan_write_cn import SYSTEM_PROMPT, USER_PROMPT, get_task_level
            messages = await get_llm_messages(task, SYSTEM_PROMPT.format(**get_task_level(task.goal)), USER_PROMPT)
    else:
        raise ValueError(f"未知的任务类型: {task.task_type}")
    
    llm_params = get_llm_params(messages, temperature=0.1)
    llm_params['response_format'] = {"type": "json_object", "schema": PlanOutput.model_json_schema()}
    logger.info(f"{llm_params}")

    response = await litellm.acompletion(**llm_params)
    if not response.choices or not response.choices[0].message:
        raise ValueError("LLM API 调用失败, 没有返回任何有效的 choices 或 message。")
    
    message = response.choices[0].message
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
    if not content:
        raise ValueError("LLM API 调用失败, 没有返回任何 content。")
    
    data = PlanOutput.model_validate_json(content)
    updated_task = task.model_copy(deep=True)
    updated_task.sub_tasks = _convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results = {
        "result": content,
        "reasoning": reason,
    }
    
    logger.info(f"{updated_task}")
    return updated_task


def _convert_plan_to_tasks(
    sub_task_outputs: List[PlanOutput],
    parent_task: Task
) -> List[Task]:
    tasks = []
    inherited_props = {
        "parent_id": parent_task.id,
        "category": parent_task.category,
        "language": parent_task.language,
        "root_name": parent_task.root_name,
        "run_id": parent_task.run_id,
    }
    for plan_item in sub_task_outputs:
        new_task = Task(
            id=plan_item.id,
            task_type=plan_item.task_type,
            goal=plan_item.goal,
            dependency=plan_item.dependency,
            length=plan_item.length,
            **inherited_props
        )
        if plan_item.sub_tasks:
            new_task.sub_tasks = _convert_plan_to_tasks(plan_item.sub_tasks, new_task)
        tasks.append(new_task)
    return tasks



