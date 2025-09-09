import os
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from utils.rag import get_rag
from utils.prompt_loader import load_prompts
from agents.plan import PlanOutput, convert_plan_to_tasks


async def plan_reflection(task: Task) -> Task:
    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["plan_reflection"] = task.results["plan"]
        updated_task.results["plan_reflection_reasoning"] = ""
    else:
        module_name = f"plan_{task.task_type}_reflection_cn"
        SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, module_name, "SYSTEM_PROMPT", "USER_PROMPT")
        if task.task_type == "search":
            context = await get_rag().get_context_base(task)
        else:
            context = await get_rag().get_context(task)
        context["to_reflection"] = task.results.get("plan")
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
        llm_params = get_llm_params(messages, temperature=LLM_TEMPERATURES["reasoning"])
        message = await llm_acompletion(llm_params, response_model=PlanOutput)
        data = message.validated_data
        content = message.content
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
        updated_task.results["plan_reflection"] = content
        updated_task.results["plan_reflection_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    return updated_task
