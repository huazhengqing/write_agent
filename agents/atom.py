import os
from utils.models import AtomOutput, Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


async def atom(task: Task) -> Task:
    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, f"atom_{task.task_type}_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context = await get_rag().get_context_base(task)
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["classification"])
    message = await llm_acompletion(llm_params, response_model=AtomOutput)
    data = message.validated_data
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["atom"] = data.model_dump(exclude_none=True, exclude={'reasoning', 'atom_result'})
    updated_task.results["atom_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    updated_task.results["atom_result"] = data.atom_result
    if hasattr(data, 'complex_reasons') and data.complex_reasons:
        updated_task.results["complex_reasons"] = data.complex_reasons
    if data.goal_update and len(data.goal_update.strip()) > 10 and data.goal_update != task.goal:
        updated_task.goal = data.goal_update
        updated_task.results["goal_update"] = data.goal_update
    if os.getenv("deployment_environment") == "test":
        if task.task_type in ["design", "search"]:
            updated_task.results["atom_result"] = "atom"
    return updated_task