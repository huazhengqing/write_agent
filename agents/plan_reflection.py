import os
import importlib
import collections
from loguru import logger
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from util.models import Task
from util.llm import get_llm_messages, get_llm_params, llm_acompletion
from util.rag import get_rag
from util.prompt_loader import load_prompts
from plan import PlanNode, PlanOutput, convert_plan_to_tasks


async def plan_reflection(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["result_reflection"] = ""
        updated_task.results["reasoning_reflection"] = ""
    else:
        if task.category == "story" and task.task_type == "write":
            module_name = f"plan_{task.task_type}_reflection_cn"
            SYSTEM_PROMPT, USER_PROMPT, get_task_level, test_get_task_level = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT", "get_task_level", "test_get_task_level")

            if os.getenv("deployment_environment") == "test":
                task_level_func = test_get_task_level
            else:
                task_level_func = get_task_level
            context = await get_rag().get_context_base(task)
            context["to_reflection"] = task.results.get("result")

            messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, task_level_func(task.goal), context)
        else:
            module_name = f"plan_{task.task_type}_reflection_cn"
            SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT")

            context = await get_rag().get_context_base(task)
            context["to_reflection"] = task.results.get("result")

            messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)

        llm_params = get_llm_params(messages, temperature=0.1)
        llm_params['response_format'] = {
            "type": "json_object", 
            "schema": PlanOutput.model_json_schema()
        }
        message = await llm_acompletion(llm_params)
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        content = message.content
        data = PlanOutput.model_validate_json(content)

        updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
        updated_task.results["result_reflection"] = content
        updated_task.results["reasoning_reflection"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    
    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task


