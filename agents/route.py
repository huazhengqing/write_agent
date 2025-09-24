from utils.models import RouteOutput, Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts


async def route(task: Task) -> str:
    user_prompt = load_prompts(task.category, "route", "user_prompt")[0]
    context_dict_user = {
        "goal": task.goal
    }
    messages = get_llm_messages(None, user_prompt, None, context_dict_user)
    llm_params = get_llm_params(llm_group="fast", messages=messages, temperature=llm_temperatures["classification"])
    message = await llm_completion(llm_params, response_model=RouteOutput)
    return message.validated_data.category
