from utils.models import Task
from utils.sqlite_task import get_task_db
from story.agent import design, hierarchy, plan, search, summary, write



async def do_write(current_task: Task):
    if not current_task.id or not current_task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if current_task.task_type != "write":
        raise ValueError("")
    if not current_task.length:
        raise ValueError("写作任务没有长度要求")
    
    if current_task.status == "completed":
        return
    
    if current_task.status == "pending":
        current_task.status = "running"
        get_task_db(current_task.run_id).update_task_status(current_task.id, "running")

    await do_plan(current_task)
    design.aggregate(current_task)
    task_result = write.atom(current_task)
    if task_result.results["atom_result"] == "atom":
        task_result = write.write(task_result)
        task_result = summary.summary(task_result)
        task_result = summary.global_state(task_result)
    elif task_result.results["atom_result"] == "complex":
        await do_hierarchy(task_result)
        summary.aggregate(task_result)
    else:
        raise ValueError("")
    
    current_task.status = "completed"
    get_task_db(current_task.run_id).update_task_status(current_task.id, "completed")



async def do_plan(parent_task: Task):
    if not parent_task.results.get("plan", ""):
        parent_task = plan.all(parent_task)
    if not parent_task.results.get("plan", ""):
        raise ValueError("")
    sub_task = plan.next(parent_task, None)
    while sub_task:
        if sub_task.task_type == "design":
            sub_task = await do_design(sub_task)
        elif sub_task.task_type == "search":
            sub_task = await do_search(sub_task)
        else:
            raise ValueError("")
        sub_task = plan.next(parent_task, sub_task)



async def do_hierarchy(parent_task: Task):
    if not parent_task.results.get("hierarchy", ""):
        parent_task = hierarchy.all(parent_task)
    if not parent_task.results.get("hierarchy", ""):
        raise ValueError("")
    sub_task = hierarchy.next(parent_task, None)
    while sub_task:
        if sub_task.task_type == "write":
            sub_task = await do_write(sub_task)
        else:
            raise ValueError("")
        sub_task = hierarchy.next(parent_task, sub_task)



async def do_design(current_task: Task):
    if not current_task.id or not current_task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if current_task.task_type != "design":
        raise ValueError("")
    
    if current_task.status == "completed":
        return
    
    if current_task.status == "pending":
        current_task.status = "running"
        get_task_db(current_task.run_id).update_task_status(current_task.id, "running")

    task_result = design.atom(current_task)
    if task_result.results["atom_result"] == "atom":
        route_result = design.route(task_result)
        task_result = design.design(task_result, route_result.expert)
        current_level = len(task_result.id.split("."))
        if current_level <= 2:
            design.book_level_design(task_result)
    elif task_result.results["atom_result"] == "complex":
        task_result = design.decomposition(task_result)
        if task_result.sub_tasks:
            for sub_task in task_result.sub_tasks:
                if sub_task.task_type == "design":
                    await do_design(sub_task)
                elif sub_task.task_type == "search":
                    await do_search(sub_task)
                else:
                    raise ValueError("")
            task_result = design.aggregate(task_result)
            current_level = len(task_result.id.split("."))
            if current_level <= 2:
                design.book_level_design(task_result)
        else:
            raise ValueError("")
    else:
        raise ValueError("")
    
    current_task.status = "completed"
    get_task_db(current_task.run_id).update_task_status(current_task.id, "completed")



async def do_search(current_task: Task):
    if not current_task.id or not current_task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if current_task.task_type != "search":
        raise ValueError("")
    
    if current_task.status == "completed":
        return
    
    if current_task.status == "pending":
        current_task.status = "running"
        get_task_db(current_task.run_id).update_task_status(current_task.id, "running")

    task_result = search.atom(current_task)
    if task_result.results["atom_result"] == "atom":
        search.search(task_result)
    elif task_result.results["atom_result"] == "complex":
        task_result = search.decomposition(task_result)
        if task_result.sub_tasks:
            for sub_task in task_result.sub_tasks:
                await do_search(sub_task)
            search.aggregate(task_result)
        else:
            raise ValueError("")
    else:
        raise ValueError("")
    
    current_task.status = "completed"
    get_task_db(current_task.run_id).update_task_status(current_task.id, "completed")
