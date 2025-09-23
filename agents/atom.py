import os

from utils.models import AtomOutput, Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts

from story.story_rag import get_story_rag


async def atom(task: Task) -> Task:
    system_prompt, user_prompt = load_prompts(task.category, f"atom_{task.task_type}", "system_prompt", "user_prompt")
    context = get_story_rag().get_context_base(task)
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["classification"])
    message = await llm_completion(llm_params, response_model=AtomOutput)
    data = message.validated_data
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["atom"] = data.model_dump(exclude_none=True, exclude={'reasoning', 'atom_result'})
    updated_task.results["atom_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    updated_task.results["atom_result"] = data.atom_result
    if hasattr(data, 'complex_reasons') and data.complex_reasons:
        updated_task.results["complex_reasons"] = data.complex_reasons
    if data.update_goal and len(data.update_goal.strip()) > 10 and data.update_goal != task.goal:
        updated_task.goal = data.update_goal
        updated_task.results["update_goal"] = data.update_goal
    if data.update_instructions and data.update_instructions != task.instructions:
        updated_task.instructions = data.update_instructions
        updated_task.results["update_instructions"] = data.update_instructions
    if data.update_input_brief and data.update_input_brief != task.input_brief:
        updated_task.input_brief = data.update_input_brief
        updated_task.results["update_input_brief"] = data.update_input_brief
    if data.update_constraints and data.update_constraints != task.constraints:
        updated_task.constraints = data.update_constraints
        updated_task.results["update_constraints"] = data.update_constraints
    if data.update_acceptance_criteria and data.update_acceptance_criteria != task.acceptance_criteria:
        updated_task.acceptance_criteria = data.update_acceptance_criteria
        updated_task.results["update_acceptance_criteria"] = data.update_acceptance_criteria
    if os.getenv("deployment_environment") == "test":
        if task.task_type in ["design", "search"]:
            updated_task.results["atom_result"] = "atom"
    return updated_task
