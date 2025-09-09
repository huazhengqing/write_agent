import os
import importlib
import litellm
import collections
from loguru import logger
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


class PlanNode(BaseModel):
    id: str = Field(..., description="任务的唯一字符串ID, 父任务id.子任务序号。例如 '1' 或 '1.3.2'。")
    task_type: Literal['design', 'write', 'search'] = Field(..., description="任务类型, 值必须是: 'design' 或 'write' 或 'search'。")
    hierarchical_position: Optional[str] = Field(None, description="任务在书/故事结构中的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。")
    goal: str = Field(..., description="任务的清晰具体目标。")
    dependency: List[str] = Field(default_factory=list, description="此任务所依赖的同层的 design/search 的 id 列表。")
    length: Optional[str] = Field(None, description="对于 'write' 类型的任务, 此任务的预估长度或字数。")
    sub_tasks: List['PlanNode'] = Field(default_factory=list, description="分解出的更深层次的子任务列表。")


class PlanOutput(PlanNode):
    reasoning: Optional[str] = Field(None, description="关于任务分解的推理过程。")
    

async def plan(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    if task.category == "story" and task.task_type == "write":
        SYSTEM_PROMPT, USER_PROMPT, get_task_level, test_get_task_level = load_prompts(task.category, f"plan_{task.task_type}_cn", "SYSTEM_PROMPT", "USER_PROMPT", "get_task_level", "test_get_task_level")
        if os.getenv("deployment_environment") == "test":
            task_level_func = test_get_task_level
        else:
            task_level_func = get_task_level
        context = await get_rag().get_context(task)
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, task_level_func(task.hierarchical_position), context)
    else:
        SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, f"plan_{task.task_type}_cn", "SYSTEM_PROMPT", "USER_PROMPT")
        if task.task_type == "search":
            context = await get_rag().get_context_base(task)
        else:
            context = await get_rag().get_context(task)
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)

    llm_params = get_llm_params(messages, temperature=LLM_TEMPERATURES["reasoning"])
    message = await llm_acompletion(llm_params, response_model=PlanOutput)
    data = message.validated_data
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results["plan"] = content
    updated_task.results["plan_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    
    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task

def convert_plan_to_tasks(
    sub_task_outputs: List[PlanNode],
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
            hierarchical_position=plan_item.hierarchical_position,
            length=plan_item.length,
            **inherited_props
        )
        if plan_item.sub_tasks:
            new_task.sub_tasks = convert_plan_to_tasks(plan_item.sub_tasks, new_task)
        tasks.append(new_task)
    return tasks