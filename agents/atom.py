import os
import litellm
from loguru import logger
from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from ..util.models import Task
from ..util.llm import get_llm_params
from ..memory import memory, get_llm_messages


class AtomOutput(BaseModel):
    goal_update: Optional[str] = Field(None, description="在分析了任务后, 对原始目标的优化或澄清。如果LLM认为不需要修改, 则此字段可以省略。")
    atom_result: Literal['atom', 'complex'] = Field(description="判断任务是否为原子任务的结果, 值必须是 'atom' 或 'complex'。")


async def atom(task: Task) -> Task:
    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    
    if os.getenv("deployment_environment") == "test":
        if task.task_type == "design":
            from ..prompts.story.atom_design_cn import test_output
            data = AtomOutput.model_validate_json(test_output)
            updated_task = task.model_copy(deep=True)
            updated_task.results = {
                "result": test_output,
                "reasoning": "",
                "atom_result": data.atom_result,
            }
            return updated_task
        elif task.task_type == "search":
            from ..prompts.story.atom_search_cn import test_output
            data = AtomOutput.model_validate_json(test_output)
            updated_task = task.model_copy(deep=True)
            updated_task.results = {
                "result": test_output,
                "reasoning": "",
                "atom_result": data.atom_result,
            }
            return updated_task
    
    if task.task_type == "design":
        from ..prompts.story.atom_design_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.task_type == "search":
        from ..prompts.story.atom_search_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    elif task.task_type == "write":
        if not task.length:
            raise ValueError("Task length must be set.")
        from ..prompts.story.atom_write_cn import SYSTEM_PROMPT, USER_PROMPT
        messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
    else:
        raise ValueError(f"未知的任务类型: {task.task_type}")
    
    llm_params = get_llm_params(messages, temperature=0.1)
    llm_params['response_format'] = {"type": "json_object", "schema": AtomOutput.model_json_schema()}
    logger.info(f"{llm_params}")

    response = await litellm.acompletion(**llm_params)
    if not response.choices or not response.choices[0].message:
        raise ValueError(f"LLM API 调用失败(task: {task.id}), 没有返回任何 choices。")
    
    message = response.choices[0].message
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
    if not content:
        raise ValueError(f"LLM API 调用失败(task: {task.id}), 返回的 content 为空。")
    
    data = AtomOutput.model_validate_json(content)
    updated_task = task.model_copy(deep=True)
    updated_task.results = {
        "result": content,
        "reasoning": reason,
        "atom_result": data.atom_result,
    }
    if data.goal_update and len(data.goal_update.strip()) > 10:
        updated_task.goal = data.goal_update
        updated_task.results["goal_update"] = data.goal_update
    
    logger.info(f"{updated_task}")
    return updated_task





