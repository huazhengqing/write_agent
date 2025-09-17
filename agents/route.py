from utils.models import RouteOutput, Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, LLM_TEMPERATURES
from story.story_rag import get_story_rag
from utils.prompt_loader import load_prompts


def route(task: Task) -> str:
    user_prompt = load_prompts(task.category, "route_cn", "user_prompt")[0]
    context_dict_user = {
        "goal": task.goal
    }
    messages = get_llm_messages(None, user_prompt, None, context_dict_user)
    llm_params = get_llm_params(llm="fast", messages=messages, temperature=LLM_TEMPERATURES["classification"])
    message = llm_completion(llm_params, response_model=RouteOutput)
    return message.validated_data.category

