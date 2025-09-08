import os
import importlib
import litellm
import collections
from loguru import logger
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion
from utils.rag import get_rag
from utils.prompt_loader import load_prompts
from agents.plan import PlanNode, PlanOutput, convert_plan_to_tasks


async def plan_reflection(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["plan_reflection"] = ""
        updated_task.results["plan_reflection_reasoning"] = ""
    else:
        if task.category == "story" and task.task_type == "write":
            module_name = f"plan_{task.task_type}_reflection_cn"
            SYSTEM_PROMPT, USER_PROMPT, get_task_level, test_get_task_level = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT", "get_task_level", "test_get_task_level")
            if os.getenv("deployment_environment") == "test":
                task_level_func = test_get_task_level
            else:
                task_level_func = get_task_level
            context = await get_rag().get_context_base(task)
            context["to_reflection"] = task.results.get("plan")
            messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, task_level_func(task.goal), context)
        else:
            module_name = f"plan_{task.task_type}_reflection_cn"
            SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT")
            context = await get_rag().get_context_base(task)
            context["to_reflection"] = task.results.get("plan")
            messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)

        llm_params = get_llm_params(messages, temperature=0.1)
        llm_params['response_format'] = {
            "type": "json_object", 
            "schema": PlanOutput.model_json_schema()
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                message = await llm_acompletion(llm_params)
                content = message.content
                data = PlanOutput.model_validate_json(content)
                break  # 验证成功，跳出循环
            except Exception as e:
                logger.warning(f"响应内容验证失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    try:
                        # 尝试删除错误的缓存条目
                        from litellm.caching.cache_key_generator import get_cache_key
                        cache_key = get_cache_key(**llm_params)
                        litellm.cache.delete(cache_key)
                        logger.info(f"已删除错误的缓存条目: {cache_key}。正在重试...")
                    except Exception as cache_e:
                        logger.error(f"删除缓存条目失败: {cache_e}")
                else:
                    logger.error("LLM 响应在多次重试后仍然无效，任务失败。")
                    raise

        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
        updated_task.results["plan_reflection"] = content
        updated_task.results["plan_reflection_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    
    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task
