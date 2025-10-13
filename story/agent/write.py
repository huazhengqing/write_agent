from story.prompts.models.atom import AtomOutput
from story import save
from story.agent.context import get_outside_design, get_outside_search, get_summary
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion
from utils.loader import load_prompts
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def atom(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("atom_result") in ["atom", "complex"]:
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    task_db = get_task_db(run_id=task.run_id)
    task_list = task_db.get_task_list(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "task_list": task_list,
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.atom", "system_prompt", "user_prompt")
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



async def write(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("write"):
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
        "style": book_meta.get("style", ""),
        "upper_level_design": await get_outside_design(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, task_list),
        "upper_level_search": await get_outside_search(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, task_list),
        "text_summary": await get_summary(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, task_list),
    }

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.write.write", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.75)
    llm_message = await llm_completion(llm_params)
    content = llm_message.content
    updated_task = task.model_copy(deep=True)
    updated_task.results["write"] = content
    updated_task.results["write_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)

    save.write(task, content)
    return updated_task



###############################################################################



async def review(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("review"):
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

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.write.review", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.1)
    llm_message = await llm_completion(llm_params)
    content = llm_message.content
    updated_task = task.model_copy(deep=True)
    updated_task.results["review"] = content
    updated_task.results["review_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)

    return updated_task
