from story.prompts.models.atom import AtomOutput
from story.prompts.models.plan import PlanOutput, convert_plan_to_tasks
from story.prompts.route.expert import ExpertOutput
from story.agent.context import get_outside_design, get_outside_search, get_summary
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion
from utils.loader import load_prompts
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import get_task_db, dict_to_task
from story import save



async def atom(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("atom_result") in ["atom", "complex"]:
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    task_list = task_db.get_task_list(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "task_list": task_list,
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.design.atom", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.0)
    message = await llm_completion(llm_params, response_model=AtomOutput)
    data = message.validated_data
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["atom_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
    updated_task.results["atom_result"] = data.atom_result
    if data.complex_reasons:
        updated_task.results["complex_reasons"] = data.complex_reasons

    task_db.add_result(updated_task)
    return updated_task



###############################################################################



async def decomposition(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    subtask_ids = task_db.get_subtask_ids(task.id)
    if subtask_ids:
        updated_task = task.model_copy(deep=True)
        sub_tasks = []
        for sub_id in subtask_ids:
            sub_task_data = task_db.get_task_by_id(sub_id)
            sub_tasks.append(dict_to_task(sub_task_data))
        updated_task.sub_tasks = sub_tasks
        return updated_task

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    design_dependent = task_db.get_dependent_design(task)
    search_dependent = task_db.get_dependent_search(task)
    latest_text = task_db.get_text_latest()
    task_list = task_db.get_task_list(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "task_list": task_list,
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.design.decomposition", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.1)
    llm_message = await llm_completion(llm_params, response_model=PlanOutput)
    plan_output = llm_message.validated_data

    updated_task = task.model_copy(deep=True)
    sub_tasks = convert_plan_to_tasks(plan_output.sub_tasks, parent_task=updated_task)
    updated_task.sub_tasks = sub_tasks

    task_db.add_sub_tasks(updated_task)
    return updated_task


###############################################################################


async def route(task: Task) -> ExpertOutput:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("expert"):
        return ExpertOutput(expert=db_task_data["expert"], reasoning="")

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    design_dependent = task_db.get_dependent_design(task)
    search_dependent = task_db.get_dependent_search(task)
    latest_text = task_db.get_text_latest()
    task_list = task_db.get_task_list(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "task_list": task_list,
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.route.expert", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.0)
    llm_message = await llm_completion(llm_params, response_model=ExpertOutput)
    expert_output = llm_message.validated_data

    task_db.update_task_expert(task.id, expert_output.expert)
    return expert_output


###############################################################################


async def design(task: Task, expert: str) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("design"):
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    design_dependent = task_db.get_dependent_design(task)
    search_dependent = task_db.get_dependent_search(task)
    latest_text = task_db.get_text_latest()
    task_list = task_db.get_task_list(task)
    
    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "task_list": task_list,
        "upper_level_design": await get_outside_design(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, task_list),
        "upper_level_search": await get_outside_search(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, task_list),
        "text_summary": await get_summary(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, task_list),
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.design.{expert}", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.75)
    llm_message = await llm_completion(llm_params)
    content = llm_message.content
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = content
    updated_task.results["design_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)

    meta_db = get_meta_db()
    if expert == "style":
        meta_db.update_style(task.run_id, content)
    elif expert == "synopsis":
        meta_db.update_synopsis(task.run_id, content)
    elif expert == "title":
        meta_db.update_title(task.run_id, content)

    save.design(task, content)
    return updated_task



###############################################################################



async def aggregate(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("design"):
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    task_list = task_db.get_task_list(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "task_list": task_list,
        "subtask_design": task_db.get_subtask_design(task.id),
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.design.aggregate", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.4)
    message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = message.content
    updated_task.results["design_reasoning"] = message.get("reasoning_content") or message.get("reasoning", "")
    
    task_db.add_result(updated_task)

    save.design(task, message.content)
    return updated_task



###############################################################################



async def book_level_design(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("book_level_design"):
        return dict_to_task(db_task_data)
    
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")

    design_dependent = task_db.get_dependent_design(task)
    search_dependent = task_db.get_dependent_search(task)
    latest_text = task_db.get_text_latest()
    task_list = task_db.get_task_list(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "design": task.results.get("design", ""),
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "task_list": task_list,
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.design.book_level_design", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.2)
    message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["book_level_design"] = message.content
    updated_task.results["book_level_design_reasoning"] = message.get("reasoning_content") or message.get("reasoning", "")
    
    meta_db.update_book_level_design(task.run_id, message.content)
    return updated_task
