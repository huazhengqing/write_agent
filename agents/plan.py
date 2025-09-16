import os
from utils.models import PlanOutput, Task, convert_plan_to_tasks
from utils.llm import get_llm_messages, get_llm_params, llm_completion, LLM_TEMPERATURES
from story.story_rag import get_story_rag
from utils.prompt_loader import load_prompts


def plan(task: Task) -> Task:
    # if task.category == "story" and task.task_type == "write":
    #     system_prompt, user_prompt, get_task_level, test_get_task_level = load_prompts(task.category, f"plan_{task.task_type}_cn", "system_prompt", "user_prompt", "get_task_level", "test_get_task_level")
    #     if os.getenv("deployment_environment") == "test":
    #         task_level_func = test_get_task_level
    #     else:
    #         task_level_func = get_task_level
    #     context = get_story_rag().get_context(task)
    #     messages = get_llm_messages(system_prompt, user_prompt, task_level_func(task.hierarchical_position), context)
    # else:
    system_prompt, user_prompt = load_prompts(task.category, f"plan_{task.task_type}_cn", "system_prompt", "user_prompt")
    if task.task_type == "search":
        context = get_story_rag().get_context_base(task)
    else:
        context = get_story_rag().get_context(task)
    context.update({
        "atom_reasoning": task.results.get("atom_reasoning", "无"),
        "complex_reasons": task.results.get("complex_reasons") or "原因未知"
    })
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
    message = llm_completion(llm_params, response_model=PlanOutput)
    data = message.validated_data
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results["plan"] = data.model_dump(exclude_none=True, exclude={'reasoning'})
    updated_task.results["plan_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    return updated_task

def plan_reflection(task: Task) -> Task:
    updated_task = task.model_copy(deep=True)
    if os.getenv("deployment_environment") == "test":
        updated_task.results["plan_reflection"] = task.results["plan"]
        updated_task.results["plan_reflection_reasoning"] = ""
    else:
        system_prompt, user_prompt = load_prompts(task.category, f"plan_{task.task_type}_reflection_cn", "system_prompt", "user_prompt")
        if task.task_type == "search":
            context = get_story_rag().get_context_base(task)
        else:
            context = get_story_rag().get_context(task)
        context["to_reflection"] = task.results.get("plan")
        messages = get_llm_messages(system_prompt, user_prompt, None, context)
        llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
        message = llm_completion(llm_params, response_model=PlanOutput)
        data = message.validated_data
        reasoning = message.get("reasoning_content") or message.get("reasoning", "")
        updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
        updated_task.results["plan_reflection"] = data.model_dump(exclude_none=True, exclude={'reasoning'})
        updated_task.results["plan_reflection_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    return updated_task
