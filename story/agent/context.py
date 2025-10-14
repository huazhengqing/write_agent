from loguru import logger
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from story.agent import inquiry
from utils.models import Task, get_sibling_ids_up_to_current
from rag.vector_query import get_vector_query_engine, index_query_batch
from story.base import get_story_vector_store
from utils.sqlite_task import get_task_db



async def get_outside_search(
    task: Task, 
    book_level_design: str, 
    global_state_summary: str, 
    design_dependent: str, 
    search_dependent: str, 
    latest_text: str, 
    overall_planning: str
) -> str:
    task_db = get_task_db(task.run_id)
    task_data = task_db.get_task_by_id(task.id)
    if task_data and (existing_context := task_data.get("context_search")):
        return existing_context

    current_level = len(task.id.split("."))
    if current_level <= 2:
        return ""
    
    inquiry_result = await inquiry(
        'search', 
        task,
        book_level_design,
        global_state_summary,
        design_dependent,
        search_dependent,
        latest_text,
        overall_planning,
    )
    all_questions = (
        inquiry_result.causality_probing +
        inquiry_result.setting_probing +
        inquiry_result.state_probing
    )
    if not all_questions:
        logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过上层搜索。")
        return ""
    
    active_filters = []
    preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
    if preceding_sibling_ids:
        active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
    filters = MetadataFilters(filters=active_filters) if active_filters else None
    
    query_engine = get_vector_query_engine(
        vector_store=get_story_vector_store(task.run_id, "search"),
        filters=filters,
        similarity_top_k=150,
        top_n=50,
    )

    results = await index_query_batch(query_engine, all_questions)
    result = "\n\n---\n\n".join(results)

    if result:
        task_db.update_task_context(task.id, "search", result)
    return result


async def get_outside_design(
    task: Task, 
    book_level_design: str, 
    global_state_summary: str, 
    design_dependent: str, 
    search_dependent: str, 
    latest_text: str, 
    overall_planning: str
) -> str:
    task_db = get_task_db(task.run_id)
    task_data = task_db.get_task_by_id(task.id)
    if task_data and (existing_context := task_data.get("context_design")):
        return existing_context

    current_level = len(task.id.split("."))
    if current_level <= 2:
        return ""
    
    inquiry_result = await inquiry(
        'design', 
        task,
        book_level_design,
        global_state_summary,
        design_dependent,
        search_dependent,
        latest_text,
        overall_planning,
    )
    all_questions = (
        inquiry_result.causality_probing +
        inquiry_result.setting_probing +
        inquiry_result.state_probing
    )
    if not all_questions:
        logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过上层设计检索。")
        return ""

    active_filters = []
    preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
    if preceding_sibling_ids:
        active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
    vector_filters = MetadataFilters(filters=active_filters) if active_filters else None

    from story.base import hybrid_query_design
    result = await hybrid_query_design(task.run_id, all_questions, vector_filters)

    if result:
        task_db.update_task_context(task.id, "design", result)
    return result


async def get_summary(
    task: Task, 
    book_level_design: str, 
    global_state_summary: str, 
    design_dependent: str, 
    search_dependent: str, 
    latest_text: str, 
    overall_planning: str
) -> str:
    task_db = get_task_db(task.run_id)
    task_data = task_db.get_task_by_id(task.id)
    if task_data and (existing_context := task_data.get("context_summary")):
        return existing_context

    current_level = len(task.id.split("."))
    if current_level <= 2:
        return ""

    total_word_count = task_db.get_total_write_word_count()
    if total_word_count < 200:
        return ""
    
    inquiry_result = await inquiry(
        'summary', 
        task,
        book_level_design,
        global_state_summary,
        design_dependent,
        search_dependent,
        latest_text,
        overall_planning,
    )
    all_questions = (
        inquiry_result.causality_probing +
        inquiry_result.setting_probing +
        inquiry_result.state_probing
    )
    if not all_questions:
        logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过历史情节概要检索。")
        return ""

    active_filters = []
    preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
    if preceding_sibling_ids:
        active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
    vector_filters = MetadataFilters(filters=active_filters) if active_filters else None

    from story.base import hybrid_query_write
    result = await hybrid_query_write(task.run_id, all_questions, vector_filters)

    if result:
        task_db.update_task_context(task.id, "summary", result)
    return result
