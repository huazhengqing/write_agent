from story.context import get_context
from story.rag import save
from utils import call_llm
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def translation(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("translation"):
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    design_dependent = task_db.get_dependent_design(task)
    search_dependent = task_db.get_dependent_search(task)
    latest_text = task_db.get_text_latest()
    overall_planning = task_db.get_overall_planning(task)
    
    context = {
        "task": task.to_context(),
        "text": task.results.get("write"), 
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "overall_planning": overall_planning,
        "style": book_meta.get("style", ""),
    }

    from story.prompts.translation.translation import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group='summary', messages=messages, temperature=0.75)
    llm_message = await call_llm.completion(llm_params)

    updated_task = task.model_copy(deep=True)
    updated_task.results["translation"] = llm_message.content
    updated_task.results["translation_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)
    return updated_task

