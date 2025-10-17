from contextlib import contextmanager
from loguru import logger
from utils import call_llm
from utils.log import ensure_task_logger
from utils.models import Task
from utils.sqlite_meta import BookMetaDB, get_meta_db
from utils.sqlite_task import get_task_db



def create_root_task(run_id: str):
    if not run_id:
        raise ValueError("run_id 不能为空。")
    ensure_task_logger(run_id)
    with logger.contextualize(run_id=run_id):
        task_db = get_task_db(run_id)
        if task_db.get_task_by_id("1"):
            return
        book_meta_db = get_meta_db()
        book_meta = book_meta_db.get_book_meta(run_id)
        if not book_meta:
            raise ValueError(f"在数据库中找不到 run_id '{run_id}' 对应的书籍元数据。")
        root_task = Task(
            id="1",
            parent_id="",
            task_type="write",
            status="pending",
            hierarchical_position="全书",
            goal=book_meta.get("goal", ""),
            instructions=book_meta.get("instructions", ""),
            input_brief=book_meta.get("input_brief", ""),
            constraints=book_meta.get("constraints", ""),
            acceptance_criteria=book_meta.get("acceptance_criteria", ""),
            length=book_meta.get("length", "根据任务要求确定"),
            run_id=run_id,
        )
        task_db.add_task(root_task)



###############################################################################



@contextmanager
def manage_project_status(run_id: str):
    """
    一个上下文管理器, 用于在任务执行期间管理项目的运行状态。
    进入时将项目状态设置为 'running', 退出时(无论成功或异常)设置为 'idle'。
    """
    meta_db: BookMetaDB = get_meta_db()
    meta_db.update_status(run_id, "running")
    try:
        yield
    finally:
        meta_db.update_status(run_id, "idle")



async def do_task(task: Task):
    """
    外部调用接口, 统一的入口
    """
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(task.run_id)
    if book_meta and book_meta.get("status") == "running":
        return
    ensure_task_logger(task.run_id)
    with manage_project_status(task.run_id), logger.contextualize(run_id=task.run_id):
        if task.task_type == "write":
            await do_write(task)
        elif task.task_type == "design":
            await do_design(task)
        elif task.task_type == "search":
            await do_search(task)
        else:
            raise ValueError(f"不支持的任务类型: {task.task_type}")



###############################################################################



@contextmanager
def track_task_execution(task: Task):
    """
    更新任务状态
    """
    task_db = get_task_db(task.run_id)
    task.status = "running"
    task_db.update_task_status(task.id, "running")
    try:
        yield
        task.status = "completed"
        task_db.update_task_status(task.id, "completed")
    except Exception:
        task.status = "failed"
        task_db.update_task_status(task.id, "failed")
        raise
    finally:
        pass



###############################################################################



def is_daily_word_goal_reached(run_id: str) -> bool:
    """
    检查指定项目的每日写作字数目标是否已达到。
    """
    task_db = get_task_db(run_id)
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(run_id)
    if not book_meta:
        return False
    day_wordcount_goal = book_meta.get("day_wordcount_goal", 0)
    if day_wordcount_goal <= 0:
        return False
    word_count_last_24h = task_db.get_word_count_last_24h()
    if word_count_last_24h >= day_wordcount_goal:
        return True
    return False



async def do_write(current_task: Task):
    if not current_task.id or not current_task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if current_task.task_type != "write":
        raise ValueError(f"do_write 只能处理 'write' 类型的任务, 但收到了 '{current_task.task_type}' 类型。")
    if not current_task.length:
        raise ValueError("写作任务没有长度要求")
    
    if is_daily_word_goal_reached(current_task.run_id):
        return

    if current_task.status == "completed":
        return
        
    with track_task_execution(current_task):
        await do_plan(current_task)
        task_result = call_llm.design.aggregate(current_task)
        task_result = call_llm.write.atom(task_result)
        if task_result.results["atom"] == "atom":
            task_result = call_llm.write.write(task_result)
            task_result = call_llm.summary.summary(task_result)
            task_result = call_llm.summary.global_state(task_result)
        elif task_result.results["atom"] == "complex":
            await do_hierarchy(task_result)
            call_llm.summary.aggregate(task_result)
        else:
            raise ValueError(f"未知的 'atom' 类型: '{task_result.results.get('atom')}'")
        


async def do_plan(parent_task: Task):
    if not parent_task.results.get("plan", ""):
        parent_task = await call_llm.plan.all(parent_task)
    if not parent_task.results.get("plan", ""):
        raise ValueError(f"任务 '{parent_task.id}' 在执行 plan.all 后未能生成计划。")
    sub_task = await call_llm.plan.next(parent_task, None)
    while sub_task:
        if sub_task.task_type == "design":
            sub_task = await do_design(sub_task)
        elif sub_task.task_type == "search":
            sub_task = await do_search(sub_task)
        else:
            raise ValueError(f"do_plan 无法处理类型为 '{sub_task.task_type}' 的子任务。")
        sub_task = await call_llm.plan.next(parent_task, sub_task)



async def do_hierarchy(parent_task: Task):
    if not parent_task.results.get("hierarchy", ""):
        parent_task = await call_llm.hierarchy.all(parent_task)
    if not parent_task.results.get("hierarchy", ""):
        raise ValueError(f"任务 '{parent_task.id}' 在执行 hierarchy.all 后未能生成层级结构。")
    sub_task = await call_llm.hierarchy.next(parent_task, None)
    while sub_task:
        if sub_task.task_type == "write":
            if is_daily_word_goal_reached(parent_task.run_id):
                return
            sub_task = await do_write(sub_task)
        else:
            raise ValueError(f"do_hierarchy 无法处理类型为 '{sub_task.task_type}' 的子任务。")
        sub_task = await call_llm.hierarchy.next(parent_task, sub_task)



###############################################################################



async def do_design(current_task: Task):
    if not current_task.id or not current_task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if current_task.task_type != "design":
        raise ValueError(f"do_design 只能处理 'design' 类型的任务, 但收到了 '{current_task.task_type}' 类型。")
    
    if current_task.status == "completed":
        return
        
    with track_task_execution(current_task):
        task_result = call_llm.design.atom(current_task)
        if task_result.results["atom"] == "atom":
            route_result = call_llm.design.route(task_result)
            task_result = call_llm.design.design(task_result, route_result.expert)
            if len(task_result.id.split(".")) <= 2:
                call_llm.design.book_level_design(task_result)
        elif task_result.results["atom"] == "complex":
            task_result = call_llm.design.decomposition(task_result)
            if task_result.sub_tasks:
                for sub_task in task_result.sub_tasks:
                    if sub_task.task_type == "design":
                        await do_design(sub_task)
                    elif sub_task.task_type == "search":
                        await do_search(sub_task)
                    else:
                        raise ValueError(f"do_design 无法处理分解出的类型为 '{sub_task.task_type}' 的子任务。")
                task_result = call_llm.design.aggregate(task_result)
                if len(task_result.id.split(".")) <= 2:
                    call_llm.design.book_level_design(task_result)
            else:
                raise ValueError(f"复杂设计任务 '{task_result.id}' 分解后没有产生子任务。")
        else:
            raise ValueError(f"未知的 'atom' 类型: '{task_result.results.get('atom')}'")
        

###############################################################################



async def do_search(current_task: Task):
    if not current_task.id or not current_task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if current_task.task_type != "search":
        raise ValueError(f"do_search 只能处理 'search' 类型的任务, 但收到了 '{current_task.task_type}' 类型。")
        
    if current_task.status == "completed":
        return
    
    with track_task_execution(current_task):
        task_result = call_llm.search.atom(current_task)
        if task_result.results["atom"] == "atom":
            call_llm.search.search(task_result)
        elif task_result.results["atom"] == "complex":
            task_result = call_llm.search.decomposition(task_result)
            if task_result.sub_tasks:
                for sub_task in task_result.sub_tasks:
                    await do_search(sub_task)
                call_llm.search.aggregate(task_result)
            else:
                raise ValueError(f"复杂搜索任务 '{task_result.id}' 分解后没有产生子任务。")
        else:
            raise ValueError(f"未知的 'atom' 类型: '{task_result.results.get('atom')}'")
        
