from typing import Optional
import story
from story.context import get_context
from story.prompts.models.plan import PlanOutput, plan_to_task
from utils import call_llm
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, template_fill
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def all(task: Task) -> Task:
    if task.task_type != "write":
        raise ValueError(f"hierarchy.all 只能处理 'write' 类型的任务, 但收到了 '{task.task_type}' 类型。")

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

    from story.prompts.hierarchy.all import system_prompt, user_prompt
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(messages=messages, temperature=0.75)
    llm_message = await call_llm.completion(llm_params)

    updated_task = task.model_copy(deep=True)
    updated_task.results["hierarchy"] = llm_message.content
    updated_task.results["hierarchy_reasoning"] = llm_message.get("reasoning_content") or llm_message.get("reasoning", "")

    task_db.add_result(updated_task)
    return updated_task



###############################################################################



async def next(parent_task: Task, pre_task: Optional[Task]) -> Optional[Task]:
    if parent_task.task_type != "write":
        raise ValueError(f"hierarchy.next 只能处理 'write' 类型的任务, 但收到了 '{parent_task.task_type}' 类型。")

    task_db = get_task_db(run_id=parent_task.run_id)
    write_subtasks_count = len(task_db.get_subtask_ids(parent_task.id, task_type="write"))
    if write_subtasks_count > 30:
        raise ValueError(f"父任务 '{parent_task.id}' 的 'write' 类型子任务数量已超过30个的限制。")

    # 如果 pre_task 为空, 尝试从数据库中查找父任务的最后一个子任务
    if not pre_task:
        subtask_ids = task_db.get_subtask_ids(parent_task.id)
        if subtask_ids:
            last_subtask_id = subtask_ids[-1]
            pre_task_data = task_db.get_task_by_id(last_subtask_id)
            if pre_task_data:
                pre_task = dict_to_task(pre_task_data)

    if not pre_task:
        raise ValueError(f"无法为父任务 '{parent_task.id}' 生成下一个子任务，因为找不到任何前置任务。")

    book_meta = get_meta_db().get_book_meta(parent_task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    design_dependent = task_db.get_dependent_design(pre_task)
    search_dependent = task_db.get_dependent_search(pre_task)
    latest_text = task_db.get_text_latest()
    overall_planning = task_db.get_overall_planning(pre_task)
    
    context = {
        "parent_task": parent_task.to_context(),
        "pre_task": pre_task.to_context if pre_task.task_type == "write" else "",
        "hierarchy": parent_task.results.get("hierarchy", ""),
        "overall_planning": overall_planning,
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
    }
    
    structured_response = None
    # 根任务的层级规划使用直接调用, 确保稳定性和效率
    if parent_task.id == "1":
        from story.prompts.hierarchy.next import system_prompt, user_prompt
        messages = get_llm_messages(system_prompt, user_prompt, None, context)
        llm_params = get_llm_params(messages=messages, temperature=0.1)
        llm_message = await call_llm.completion(llm_params, output_cls=PlanOutput)
        structured_response = llm_message.validated_data
    else:
        # 深层任务使用 react 模式, 允许动态检索信息
        from story.prompts.hierarchy.next_react import system_prompt, user_prompt
        structured_response = await story.rag.react.react(
            run_id=parent_task.run_id,
            system_header=system_prompt,
            user_prompt=template_fill(user_prompt, context),
            output_cls=PlanOutput
        )
    
    if not structured_response:
        return None

    # llm_message 已经是验证过的 PlanOutput 实例了
    if not structured_response.goal:
        return None
        
    task_next = plan_to_task(structured_response)

    # pre_task 在此函数逻辑中必然存在
    parts = pre_task.id.split('.')
    parts[-1] = str(int(parts[-1]) + 1)
    task_next.id = ".".join(parts)
    task_next.parent_id = parent_task.id
    task_next.run_id = parent_task.run_id

    task_db.add_task(task_next)
    return task_next
