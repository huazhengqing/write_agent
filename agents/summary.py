from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, LLM_TEMPERATURES
from utils.prompt_loader import load_prompts
from story.story_rag import get_story_rag


def summary(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(task.category, "summary_cn", "system_prompt", "user_prompt")
    context_dict_user = {
        "task": task.model_dump_json(
            indent=2,
            exclude_none=True,
            include={'goal'}
        ),
        "text": task.results.get("write_reflection")
    }
    messages = get_llm_messages(system_prompt, user_prompt, None, context_dict_user)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["summarization"])
    message = llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = content
    updated_task.results["summary_reasoning"] = reasoning
    return updated_task


def summary_aggregate(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(task.category, "summary_aggregate_cn", "system_prompt", "user_prompt")
    context = get_story_rag().get_aggregate_summary(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["summarization"])
    message = llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = content
    updated_task.results["summary_reasoning"] = reasoning
    return updated_task

