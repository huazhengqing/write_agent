from loguru import logger
from story.prompts.models.atom import AtomOutput
from story.prompts.models.plan import PlanOutput, convert_plan_to_tasks
from story.rag import save
from utils import call_llm
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def atom(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("atom_result") in ["atom", "complex"]:
        return dict_to_task(db_task_data)

    if task.parent_id:
        parent_task_data = task_db.get_task_by_id(task.parent_id)
        if parent_task_data and parent_task_data.get("task_type") == "search":
            updated_task = task.model_copy(deep=True)
            updated_task.results["atom"] = "atom"
            updated_task.results["atom_reasoning"] = "父任务是 search 类型，为防止无限分解，此任务被强制设为原子任务。"
            task_db.add_result(updated_task)
            return updated_task
    
    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    overall_planning = task_db.get_overall_planning(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "overall_planning": overall_planning,
    }

    from story.prompts.search.atom import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.0)
    llm_message = await call_llm.completion(llm_params, output_cls=AtomOutput)

    data = llm_message.validated_data
    reasoning = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["atom"] = data.atom_result
    updated_task.results["atom_reasoning"] = "\n\n".join(filter(None, [reasoning, data.reasoning]))
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
    overall_planning = task_db.get_overall_planning(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "overall_planning": overall_planning,
    }

    from story.prompts.search.decomposition import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.1)
    llm_message = await call_llm.completion(llm_params, output_cls=PlanOutput)

    plan_output = llm_message.validated_data
    updated_task = task.model_copy(deep=True)
    sub_tasks = convert_plan_to_tasks(plan_output.sub_tasks, parent_task=updated_task)
    updated_task.sub_tasks = sub_tasks
    updated_task.results["decomposition_reasoning"] = plan_output.reasoning

    task_db.add_sub_tasks(updated_task)
    return updated_task


###############################################################################


async def search(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("search"):
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
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "overall_planning": overall_planning,
    }
    
    from story.prompts.search.search import system_prompt, user_prompt
    from utils.search import web_search_tools
    
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    final_system_prompt = messages[0]["content"]
    final_user_prompt = messages[1]["content"]
    search_result = await call_llm.react(
        system_prompt=final_system_prompt,
        user_prompt=final_user_prompt,
        tools=web_search_tools,
    )
    if search_result is None:
        raise ValueError(f"Agent在为任务 '{task.id}' 执行搜索时, 经过多次重试后仍然失败。")
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = search_result
 
    task_db.add_result(updated_task)

    save.search(task, search_result)
    return updated_task



###############################################################################



async def aggregate(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("search"):
        return dict_to_task(db_task_data)
    
    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    overall_planning = task_db.get_overall_planning(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "overall_planning": overall_planning,
        "subtask_search": task_db.get_subtask_search(task.id),
    }

    from story.prompts.search.aggregate import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.4)
    llm_message = await call_llm.completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["search"] = llm_message.content
    updated_task.results["search_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)

    save.search(task, llm_message.content)
    return updated_task
