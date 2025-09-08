import os
import importlib
import litellm
import collections
from loguru import logger
from typing import Optional, Literal
from pydantic import BaseModel, Field
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


class AtomOutput(BaseModel):
    reasoning: Optional[str] = Field(None, description="关于任务是原子还是复杂的推理过程。")
    goal_update: Optional[str] = Field(None, description="在分析了任务后, 对原始目标的优化或澄清。如果LLM认为不需要修改, 则此字段可以省略。")
    atom_result: Literal['atom', 'complex'] = Field(description="判断任务是否为原子任务的结果, 值必须是 'atom' 或 'complex'。")


async def atom(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")
    
    if os.getenv("deployment_environment") == "test":
        if task.task_type in ["design", "search"]:
            module_path = f"prompts.{task.category}.atom_{task.task_type}_cn"
            module = importlib.import_module(module_path)
            test_output = getattr(module, "test_output")
            data = AtomOutput.model_validate_json(test_output)
            updated_task = task.model_copy(deep=True)
            updated_task.results["result"] = test_output
            updated_task.results["reasoning"] = ""
            updated_task.results["atom_result"] = data.atom_result
            return updated_task

    module_name = f"atom_{task.task_type}_cn"
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT")
    context = await get_rag().get_context_base(task)
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages, temperature=0.1)
    llm_params['response_format'] = {
        "type": "json_object", 
        "schema": AtomOutput.model_json_schema()
        }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = await llm_acompletion(llm_params)
            content = message.content
            data = AtomOutput.model_validate_json(content)
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
    updated_task = task.model_copy(deep=True)
    updated_task.results["result"] = content
    updated_task.results["reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    updated_task.results["atom_result"] = data.atom_result
    if data.goal_update and len(data.goal_update.strip()) > 10 and data.goal_update != task.goal:
        updated_task.goal = data.goal_update
        updated_task.results["goal_update"] = data.goal_update
    
    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task