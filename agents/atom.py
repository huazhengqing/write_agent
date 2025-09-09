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
    # logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")
    
    if os.getenv("deployment_environment") == "test":
        if task.task_type in ["design", "search"]:
            module_path = f"prompts.{task.category}.atom_{task.task_type}_cn"
            module = importlib.import_module(module_path)
            test_output = getattr(module, "test_output")
            data = AtomOutput.model_validate_json(test_output)
            updated_task = task.model_copy(deep=True)
            updated_task.results["atom"] = test_output
            updated_task.results["atom_reasoning"] = ""
            updated_task.results["atom_result"] = data.atom_result
            return updated_task

    module_name = f"atom_{task.task_type}_cn"
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT")
    context = await get_rag().get_context_base(task)
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages, temperature=0.1)
    message = await llm_acompletion(llm_params, response_model=AtomOutput)
    data = message.validated_data
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["atom"] = content
    updated_task.results["atom_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    updated_task.results["atom_result"] = data.atom_result
    if data.goal_update and len(data.goal_update.strip()) > 10 and data.goal_update != task.goal:
        updated_task.goal = data.goal_update
        updated_task.results["goal_update"] = data.goal_update
    
    # logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task