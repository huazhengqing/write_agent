from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.story_rag import get_story_rag


async def summary(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.summary.summary", "system_prompt", "user_prompt")
    context_dict_user = {
        "task": task.model_dump_json(
            indent=2,
            exclude_none=True,
            include={'id', 'hierarchical_position', 'goal', 'length', 'dependency', 'instructions', 'input_brief', 'constraints', 'acceptance_criteria'}
        ),
        "text": task.results.get("write_reflection")
    }
    messages = get_llm_messages(system_prompt, user_prompt, None, context_dict_user)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=llm_temperatures["summarization"])
    message = await llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = content
    updated_task.results["summary_reasoning"] = reasoning
    return updated_task


async def summary_aggregate(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.summary.summary_aggregate", "system_prompt", "user_prompt")
    context = get_story_rag().get_aggregate_summary(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=llm_temperatures["summarization"])
    message = await llm_completion(llm_params)
    content = message.content
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = content
    updated_task.results["summary_reasoning"] = reasoning
    return updated_task
