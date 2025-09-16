from utils.models import RouteOutput, Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, LLM_TEMPERATURES
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


def route(task: Task) -> str:
    USER_PROMPT = load_prompts(task.category, "route_cn", "USER_PROMPT")[0]
    context_dict_user = {
        "goal": task.goal
    }
    messages = get_llm_messages(None, USER_PROMPT, None, context_dict_user)
    llm_params = get_llm_params(llm="fast", messages=messages, temperature=LLM_TEMPERATURES["classification"])
    message = llm_completion(llm_params, response_model=RouteOutput)
    return message.validated_data.category