import importlib
from utils import call_llm
from utils.llm import get_llm_messages, get_llm_params
from utils.models import Task



async def pruning(context_type: str, task: Task, context: str) -> str:
    context = {
        "task": task.to_context(),
        "context": context,
    }
    module = importlib.import_module(f"story.prompts.pruning.{context_type}")
    messages = get_llm_messages(module.system_prompt, module.user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.1)
    llm_message = await call_llm.completion(llm_params)
    return llm_message.content
