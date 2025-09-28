from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.story_rag import get_story_rag
from utils.sqlite_task import get_task_db


async def write_plan(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.write.write_1_plan", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["creative"])
    message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["write_plan"] = message.content
    updated_task.results["write_plan_reasoning"] = message.get("reasoning_content") or message.get("reasoning", "")
    return updated_task


async def write_draft(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.write.write_2_draft", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    context["write_plan"] = task.results.get("write_plan")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["creative"])
    message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["write_draft"] = message.content
    updated_task.results["write_draft_reasoning"] = message.get("reasoning_content") or message.get("reasoning", "")
    return updated_task


async def write_critic(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.write.write_3_critic", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    context.update({
        "write_plan": task.results.get("write_plan"),
        "write_draft": task.results.get("write_draft"),
    })
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["reasoning"])
    message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["write_critic"] = message.content
    updated_task.results["write_critic_reasoning"] = message.get("reasoning_content") or message.get("reasoning", "")
    return updated_task


async def write_refine(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.write.write_4_refine", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    context.update({
        "write_plan": task.results.get("write_plan"),
        "write_draft": task.results.get("write_draft"),
        "write_critic": task.results.get("write_critic"),
    })
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["reasoning"])
    message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["write"] = message.content
    all_reasoning = [
        f"### Plan Reasoning\n{task.results.get('plan_reasoning', '')}",
        f"### Draft Reasoning\n{task.results.get('draft_reasoning', '')}",
        f"### Critic Reasoning\n{task.results.get('critic_reasoning', '')}",
        f"### Refine Reasoning\n{message.get('reasoning_content') or message.get('reasoning', '')}",
    ]
    updated_task.results["write_reasoning"] = "\n\n".join(filter(None, all_reasoning))
    return updated_task


###############################################################################


async def write_review(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.write.write_review", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    context.update({
        "text": get_task_db(task.run_id).get_write_text(task)
    })
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["reasoning"])
    message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["write_review"] = message.content
    updated_task.results["write_review_reasoning"] = message.get("reasoning_content") or message.get("reasoning", "")
    return updated_task





