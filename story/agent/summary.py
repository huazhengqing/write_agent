from story import save
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def summary(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("summary"):
        return dict_to_task(db_task_data)
    
    context = {
        "task": task.to_context(),
        "text": task.results.get("write"),
    }

    from story.prompts.summary.summary import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.2)
    llm_message = await llm_completion(llm_params)

    content = llm_message.content
    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = content
    updated_task.results["summary_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db = get_task_db(run_id=task.run_id)
    task_db.add_result(updated_task)
    
    save.summary(task, content)
    return updated_task



###############################################################################



async def aggregate(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("summary"):
        return dict_to_task(db_task_data)
    
    context = {
        "task": task.to_context(),
        "subtask_summary": task_db.get_subtask_summary(task.id),
    }

    from story.prompts.summary.aggregate import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.2)
    llm_message = await llm_completion(llm_params)

    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = llm_message.content
    updated_task.results["summary_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)
    
    save.summary(task, llm_message.content)
    return updated_task



###############################################################################



async def global_state(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("global_state"):
        return dict_to_task(db_task_data)
    
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(task.run_id) or {}
    global_state_summary = book_meta.get("global_state_summary", "")
    
    summary = task.results.get("summary", "")
    if not summary:
        raise ValueError("")

    context = {
        "task": task.to_context(),
        "global_state_summary": global_state_summary,
        "summary": summary,
    }

    from story.prompts.summary.global_state import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.2)
    llm_message = await llm_completion(llm_params)

    updated_task = task.model_copy(deep=True)
    updated_task.results["global_state"] = llm_message.content
    updated_task.results["global_state_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)
    meta_db.update_global_state_summary(task.run_id, llm_message.content)
    return updated_task


