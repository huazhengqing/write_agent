import os
from prompts.story.atom_output import AtomOutput
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.story_rag import get_story_rag


async def atom(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.atom.atom_{task.task_type}", "system_prompt", "user_prompt")
    context = get_story_rag().get_context_base(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["classification"])
    message = await llm_completion(llm_params, response_model=AtomOutput)
    data = message.validated_data
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["atom_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    updated_task.results["atom_result"] = data.atom_result
    if data.complex_reasons:
        updated_task.results["complex_reasons"] = data.complex_reasons
    return updated_task
