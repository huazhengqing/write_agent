from utils.llm import get_llm_messages, get_llm_params, llm_completion
from utils.loader import load_prompts
from utils.models import Task



async def pruning(context_type: str, task: Task, context: str) -> str:
    context = {
        "task": task.to_context(),
        "context": context,
    }
    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.context.{context_type}", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.1)
    llm_message = await llm_completion(llm_params)
    return llm_message.content
