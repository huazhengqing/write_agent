from prompts.story.route_cn import RouteOutput
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


async def route(task: Task) -> str:
    RouteOutput, USER_PROMPT = load_prompts(task.category, "route_cn", "RouteOutput", "USER_PROMPT")
    context_dict_user = {
        "goal": task.goal
    }
    messages = get_llm_messages(None, USER_PROMPT, None, context_dict_user)
    llm_params = get_llm_params(llm="fast", messages=messages, temperature=LLM_TEMPERATURES["classification"])
    message = await llm_acompletion(llm_params, response_model=RouteOutput)
    data = message.validated_data
    return data.category