import os
from typing import Optional
from loguru import logger
from utils.models import Task
from utils.llm import clean_markdown_fences, template_fill
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

    latest_text = task_db.get_text_latest()
    overall_planning = task_db.get_overall_planning(task)
    
    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "latest_text": latest_text,
        "overall_planning": overall_planning,
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.rag.tools import get_design_tool, get_plot_history_tool, get_search_tool
    from story.prompts.plan.all import system_prompt, user_prompt

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.75, 
        max_tokens = None,
        max_retries = 10,
        api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        api_base = os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
    )
    agent = FunctionAgent(
        system_prompt = system_prompt,
        tools = [get_design_tool(task.run_id), get_plot_history_tool(task.run_id), get_search_tool(task.run_id)],
        llm = llm,
        output_cls = None, 
        streaming = False,
        timeout = 600,
        verbose= False
    )
    user_msg = template_fill(user_prompt, context)
    handler = agent.run(user_msg)

    logger.info(f"system_prompt=\n{system_prompt}")
    logger.info(f"user_msg=\n{user_msg}")

    agentOutput = await handler

    raw_output = clean_markdown_fences(agentOutput.response)
    if not raw_output:
        raise ValueError("")
    logger.info(f"完成 \n{raw_output}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["plan"] = raw_output
    updated_task.results["plan_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")

    task_db.add_result(updated_task)
    return updated_task



###############################################################################



async def next(parent_task: Task, pre_task: Optional[Task]) -> Optional[Task]:
    if parent_task.task_type != "write":
        raise ValueError(f"plan.next 只能处理 'write' 类型的任务, 但收到了 '{parent_task.task_type}' 类型。")

    task_db = get_task_db(run_id=parent_task.run_id)

    # 从数据库获取最新的任务状态, 如果 plan 已完成, 直接返回 None
    db_task_data = task_db.get_task_by_id(parent_task.id)
    if db_task_data and db_task_data.get("plan_completed"):
        return None

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
    
    latest_text = task_db.get_text_latest()
    overall_planning = task_db.get_overall_planning(pre_task or parent_task)
    design_dependent = task_db.get_dependent_design(pre_task)
    search_dependent = task_db.get_dependent_search(pre_task)
    
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

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.rag.tools import get_design_tool, get_plot_history_tool, get_search_tool
    from story.prompts.plan.next import system_prompt, user_prompt
    from story.models.plan import PlanOutput

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.1, 
        max_tokens = None,
        max_retries = 10,
        api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        api_base = os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
    )
    agent = FunctionAgent(
        system_prompt = system_prompt,
        tools = [get_design_tool(parent_task.run_id), get_plot_history_tool(parent_task.run_id), get_search_tool(parent_task.run_id)],
        llm = llm,
        output_cls = PlanOutput, 
        streaming = False,
        timeout = 600,
        verbose= False
    )
    user_msg = template_fill(user_prompt, context)
    handler = agent.run(user_msg)
    
    logger.info(f"system_prompt=\n{system_prompt}")
    logger.info(f"user_msg=\n{user_msg}")

    agentOutput = await handler

    # 优先检查原始输出是否为 'null'
    raw_output = clean_markdown_fences(agentOutput.response)
    if raw_output.strip().lower() == 'null':
        logger.info("完成, Agent 返回 'null', 表示任务完成。")
        parent_task.results["plan_completed"] = 1
        task_db.add_result(parent_task)
        return None
 
    if not agentOutput.structured_response:
        raise ValueError("") 
    if not agentOutput.structured_response.goal:
        raise ValueError("")

    logger.info(f"完成 \n{agentOutput.structured_response.model_dump_json(indent=2, ensure_ascii=False)}")

    task_next = agentOutput.structured_response.to_task()
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
