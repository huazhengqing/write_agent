import os
from loguru import logger
from typing import Optional, Literal
from pydantic import BaseModel, Field
from ..util.models import Task
from ..util.llm import get_llm_params, llm_acompletion
from ..memory import get_llm_messages


class AtomOutput(BaseModel):
    goal_update: Optional[str] = Field(None, description="在分析了任务后, 对原始目标的优化或澄清。如果LLM认为不需要修改, 则此字段可以省略。")
    atom_result: Literal['atom', 'complex'] = Field(description="判断任务是否为原子任务的结果, 值必须是 'atom' 或 'complex'。")


async def atom(task: Task) -> Task:
    logger.info(f"{task}")

    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")

    if task.category == "story":
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
    elif task.category == "report":
        if task.task_type == "design":
            from ..prompts.report.atom_design_cn import SYSTEM_PROMPT, USER_PROMPT
            messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
        elif task.task_type == "search":
            from ..prompts.report.atom_search_cn import SYSTEM_PROMPT, USER_PROMPT
            messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
        elif task.task_type == "write":
            if not task.length:
                raise ValueError("Task length must be set.")
            from ..prompts.report.atom_write_cn import SYSTEM_PROMPT, USER_PROMPT
            messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
        else:
            raise ValueError(f"未知的任务类型: {task.task_type}")
    elif task.category == "book":
        if task.task_type == "design":
            from ..prompts.book.atom_design_cn import SYSTEM_PROMPT, USER_PROMPT
            messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
        elif task.task_type == "search":
            from ..prompts.book.atom_search_cn import SYSTEM_PROMPT, USER_PROMPT
            messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
        elif task.task_type == "write":
            if not task.length:
                raise ValueError("Task length must be set.")
            from ..prompts.book.atom_write_cn import SYSTEM_PROMPT, USER_PROMPT
            messages = await get_llm_messages(task, SYSTEM_PROMPT, USER_PROMPT)
        else:
            raise ValueError(f"未知的任务类型: {task.task_type}")
    else:
        raise ValueError(f"未知的 category: {task.category}")

    llm_params = get_llm_params(messages, temperature=0.1)
    llm_params['response_format'] = {"type": "json_object", "schema": AtomOutput.model_json_schema()}
    message = await llm_acompletion(llm_params)
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
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





