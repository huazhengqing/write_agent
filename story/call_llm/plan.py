from typing import Optional
from story.prompts.models.plan import PlanOutput, plan_to_task
from story.context import get_context
from utils import call_llm
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params
from utils.sqlite_task import get_task_db, dict_to_task
from utils.sqlite_meta import get_meta_db



async def all(task: Task) -> Task:
    if task.task_type != "write":
        raise ValueError(f"plan.all 只能处理 'write' 类型的任务, 但收到了 '{task.task_type}' 类型。")

    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("plan"):
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
        "outside_design": await get_context.design(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, overall_planning),
        "outside_search": await get_context.search(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, overall_planning),
        "text_summary": await get_context.summary(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, overall_planning),
    }

    from story.prompts.plan.all import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.75)
    llm_message = await call_llm.completion(llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan"] = llm_message.content
    updated_task.results["plan_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")
    
    task_db.add_result(updated_task)
    return updated_task



###############################################################################



async def next(parent_task: Task, pre_task: Optional[Task]) -> Optional[Task]:
    if parent_task.task_type != "write":
        raise ValueError(f"plan.next 只能处理 'write' 类型的任务, 但收到了 '{parent_task.task_type}' 类型。")

    task_db = get_task_db(run_id=parent_task.run_id)
    subtask_ids = task_db.get_subtask_ids(parent_task.id)
    if len(subtask_ids) > 30:
        raise ValueError(f"父任务 '{parent_task.id}' 的 子任务数量已超过30个的限制。")

    # 如果 pre_task 为空, 尝试从数据库中查找父任务的最后一个子任务
    if not pre_task:
        if subtask_ids:
            last_subtask_id = subtask_ids[-1]
            pre_task_data = task_db.get_task_by_id(last_subtask_id)
            if pre_task_data:
                pre_task = dict_to_task(pre_task_data)

    book_meta = get_meta_db().get_book_meta(parent_task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    design_dependent = task_db.get_dependent_design(pre_task)
    search_dependent = task_db.get_dependent_search(pre_task)
    latest_text = task_db.get_text_latest()
    overall_planning = task_db.get_overall_planning(pre_task or parent_task)
    
    context = {
        "parent_task": parent_task.to_context(),
        "pre_task": pre_task.to_context if pre_task else "",
        "plan": parent_task.results.get("plan", ""),
        "overall_planning": overall_planning,
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
    }

    from story.prompts.plan.next import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    final_system_prompt = messages[0]["content"]
    final_user_prompt = messages[1]["content"]
    llm_message = await call_llm.react.react(
        run_id=parent_task.run_id,
        system_prompt=final_system_prompt,
        user_prompt=final_user_prompt,
        output_cls=PlanOutput
    )
    if not llm_message:
        return None

    plan_next = llm_message.validated_data if hasattr(llm_message, 'validated_data') else None
    if not plan_next or not plan_next.goal:
        return None

    task_next = plan_to_task(plan_next)

    if pre_task:
        parts = pre_task.id.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        task_next.id = ".".join(parts)
    else:
        task_next.id = f"{parent_task.id}.1"
    task_next.parent_id = parent_task.id
    task_next.run_id = parent_task.run_id

    task_db.add_task(task_next)
    return task_next
