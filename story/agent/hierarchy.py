import os
from typing import Optional
from story.agent.context import get_outside_design, get_outside_search, get_summary
from utils.models import Task
from story.prompts.models.plan import PlanOutput, convert_plan_to_tasks, plan_to_task
from utils.llm import get_llm_messages, get_llm_params, llm_completion
from utils.loader import load_prompts
from story.base import hybrid_query_react
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def all(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("hierarchy"):
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

    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.hierarchy.all", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.75)
    llm_message = await llm_completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["hierarchy"] = llm_message.content
    updated_task.results["hierarchy_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")

    task_db.add_result(updated_task)
    return updated_task



###############################################################################



async def next(parent_task: Task, pre_task: Optional[Task]) -> Optional[Task]:
    task_db = get_task_db(run_id=parent_task.run_id)
    # 如果 pre_task 为空, 尝试从数据库中查找父任务的最后一个子任务
    if not pre_task:
        subtask_ids = task_db.get_subtask_ids(parent_task.id)
        if subtask_ids:
            last_subtask_id = subtask_ids[-1]
            pre_task_data = task_db.get_task_by_id(last_subtask_id)
            if pre_task_data:
                from utils.sqlite_task import dict_to_task
                pre_task = dict_to_task(pre_task_data)

    if not pre_task:
        raise ValueError(f"找不到前一个任务。")

    book_meta = get_meta_db().get_book_meta(parent_task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    design_dependent = task_db.get_dependent_design(pre_task)
    search_dependent = task_db.get_dependent_search(pre_task)
    latest_text = task_db.get_text_latest()
    task_list = task_db.get_task_list(pre_task)
    
    context = {
        "parent_task": parent_task.to_context(),
        "pre_task": pre_task.to_context if pre_task.task_type == "write" else "",
        "hierarchy": parent_task.results.get("hierarchy", ""),
        "task_list": task_list,
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
    }

    system_prompt, user_prompt = load_prompts(f"prompts.{parent_task.category}.hierarchy.next", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    final_system_prompt = messages[0]["content"]
    final_user_prompt = messages[1]["content"]
    llm_message = await hybrid_query_react(
        run_id=parent_task.run_id,
        system_prompt=final_system_prompt,
        user_prompt=final_user_prompt,
        response_model=PlanOutput
    )

    plan_next = llm_message.validated_data
    if not plan_next or not plan_next.goal:
        return None
        
    task_next = plan_to_task(plan_next)

    # 根据 pre_task 和 parent_task 补全 task_next 的字段
    if pre_task:
        # 如果有前一个任务, ID 在其基础上加1
        parts = pre_task.id.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        task_next.id = ".".join(parts)
    else:
        # 如果没有前一个任务, 这是第一个子任务
        task_next.id = f"{parent_task.id}.1"
    task_next.parent_id = parent_task.id
    task_next.category = parent_task.category
    task_next.language = parent_task.language
    task_next.root_name = parent_task.root_name
    task_next.run_id = parent_task.run_id
    task_next.day_wordcount_goal = parent_task.day_wordcount_goal
    task_next.status = "pending"
    task_next.results["cache_key"] = llm_message.cache_key

    # 将新任务存入数据库
    task_db.add_task(task_next)
    return task_next
