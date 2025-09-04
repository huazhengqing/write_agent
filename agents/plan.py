import os
import importlib
import collections
from loguru import logger
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from util.models import Task
from util.llm import get_llm_params, llm_acompletion
from memory import get_llm_messages
from util.prompt_loader import load_prompts


class PlanOutput(BaseModel):
    id: str = Field(..., description="任务的唯一字符串ID, 父任务id.子任务序号。例如 '1' 或 '1.3.2'。")
    task_type: Literal['design', 'write', 'search'] = Field(..., description="任务类型, 值必须是: 'design' 或 'write' 或 'search'。")
    goal: str = Field(..., description="任务的清晰具体目标。")
    dependency: List[str] = Field(default_factory=list, description="此任务所依赖的同层的 design/search 的 id 列表。")
    length: Optional[str] = Field(None, description="对于 'write' 类型的任务, 此任务的预估长度或字数。")
    sub_tasks: List['PlanOutput'] = Field(default_factory=list, description="分解出的更深层次的子任务列表。")


async def plan(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    
    VALID_CATEGORIES = {"story", "report", "book"}
    if task.category not in VALID_CATEGORIES:
        raise ValueError(f"未知的 category: {task.category}")

    VALID_TASK_TYPES = {"design", "search", "write"}
    if task.task_type not in VALID_TASK_TYPES:
        raise ValueError(f"未知的任务类型: {task.task_type}")

    if task.task_type == "write":
        if not task.length:
            raise ValueError("Task length must be set.")

    if task.category == "story" and task.task_type == "write":
        module_path = "prompts.story.plan_write_cn"
        module = importlib.import_module(module_path)
        SYSTEM_PROMPT = module.SYSTEM_PROMPT
        USER_PROMPT = module.USER_PROMPT

        if os.getenv("deployment_environment") == "test":
            task_level_func = module.test_get_task_level
        else:
            task_level_func = module.get_task_level
        safe_context = collections.defaultdict(str, task_level_func(task.goal))
        formatted_system_prompt = SYSTEM_PROMPT.format_map(safe_context)

        messages = await get_llm_messages(task, formatted_system_prompt, USER_PROMPT)
    else:
        module_name = f"plan_{task.task_type}_cn"
        SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT")
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)

    llm_params = get_llm_params(messages, temperature=0.1)
    llm_params['response_format'] = {"type": "json_object", "schema": PlanOutput.model_json_schema()}
    message = await llm_acompletion(llm_params)
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
    data = PlanOutput.model_validate_json(content)

    updated_task = task.model_copy(deep=True)
    updated_task.sub_tasks = _convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results = {
        "result": content,
        "reasoning": reason,
    }
    
    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
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