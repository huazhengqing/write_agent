import os
from loguru import logger
from story.rag import save
from utils.models import Task
from utils.llm import clean_markdown_fences, template_fill
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def atom(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("atom_result") in ["atom", "complex"]:
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    task_db = get_task_db(run_id=task.run_id)
    overall_planning = task_db.get_overall_planning(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "overall_planning": overall_planning,
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.atom import system_prompt, user_prompt
    from story.models.atom import AtomOutput

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.0, 
        max_tokens = None,
        max_retries = 10,
        api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        api_base = os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
    )
    agent = FunctionAgent(
        system_prompt = system_prompt,
        tools = [],
        llm = llm,
        output_cls = AtomOutput, 
        streaming = False,
        timeout = 600,
        verbose= False
    )
    user_msg = template_fill(user_prompt, context)
    handler = agent.run(user_msg)

    logger.info(f"system_prompt=\n{system_prompt}")
    logger.info(f"user_msg=\n{user_msg}")

    agentOutput = await handler

    if not agentOutput.structured_response:
        raise ValueError(f"Agent在为任务 '{task.id}' 执行原子性判断时, 经过多次重试后仍然失败。")
    logger.info(f"完成 \n{agentOutput.structured_response.model_dump_json(indent=2, ensure_ascii=False)}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["atom"] = agentOutput.structured_response.atom_result
    updated_task.results["atom_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "") or agentOutput.structured_response.reasoning
    if agentOutput.structured_response.complex_reasons:
        updated_task.results["complex_reasons"] = agentOutput.structured_response.complex_reasons

    task_db.add_result(updated_task)
    return updated_task



###############################################################################



async def write(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("write"):
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    latest_text = task_db.get_text_latest()
    overall_planning = task_db.get_overall_planning(task)
    design_dependent = task_db.get_subtask_design(task)
    search_dependent = task_db.get_subtask_search(task)
    
    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "overall_planning": overall_planning,
        "style": book_meta.get("style", ""),
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.write.write import system_prompt, user_prompt
    from story.rag.tools import get_design_tool, get_plot_history_tool, get_search_tool

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
        raise ValueError(f"Agent在为任务 '{task.id}' 执行写作时, 经过多次重试后仍然失败。")
    logger.info(f"完成 \n{raw_output}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["write"] = raw_output
    updated_task.results["write_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)

    save.write(task, raw_output)
    return updated_task



###############################################################################



async def review(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("review"):
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
        # "outside_design": await get_context.design(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, overall_planning),
        # "outside_search": await get_context.search(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, overall_planning),
        # "text_summary": await get_context.summary(task, book_level_design, global_state_summary, design_dependent, search_dependent, latest_text, overall_planning),
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.write.review import system_prompt, user_prompt

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
        tools = [],
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
        raise ValueError(f"Agent在为任务 '{task.id}' 执行审阅时, 经过多次重试后仍然失败。")
    logger.info(f"完成 \n{raw_output}")
    updated_task = task.model_copy(deep=True)
    updated_task.results["review"] = raw_output
    updated_task.results["review_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)

    save.design(task, raw_output)
    return updated_task
