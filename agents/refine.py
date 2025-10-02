from prompts.story.refine_output import RefineOutput
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.story_rag import get_story_rag
from typing import Any, List, Optional

from utils.sqlite_task import get_task_db


def _update_task_field(
    task: Task,
    field_name: str,
    refined_value: Optional[Any],
):
    if refined_value:
        original_value = getattr(task, field_name)
        if isinstance(refined_value, str):
            refined_value = refined_value.strip()
            if not refined_value:
                return
        if refined_value != original_value:
            setattr(task, field_name, refined_value)



async def refine(task: Task) -> Task:
    if task.id == "1":
        return task
    db = get_task_db(run_id=task.run_id)
    if not db.has_preceding_sibling_design_tasks(task):
        return task
    system_prompt, user_prompt = load_prompts(f"prompts.{task.category}.refiner.refine_{task.task_type}", "system_prompt", "user_prompt")
    context = get_story_rag().get_context_base(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["reasoning"])
    message = await llm_completion(llm_params, response_model=RefineOutput)
    data = message.validated_data
    updated_task = task.model_copy(deep=True)
    updated_task.results["refine"] = data.model_dump(exclude_none=True, exclude={'reasoning'})
    updated_task.results["refine_reasoning"] = data.reasoning or ""
    _update_task_field(updated_task, "goal", data.refine_goal)
    _update_task_field(updated_task, "instructions", data.refine_instructions)
    _update_task_field(updated_task, "input_brief", data.refine_input_brief)
    _update_task_field(updated_task, "constraints", data.refine_constraints)
    _update_task_field(updated_task, "acceptance_criteria", data.refine_acceptance_criteria)
    return updated_task
