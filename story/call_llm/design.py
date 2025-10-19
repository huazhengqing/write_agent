import importlib
import os
from loguru import logger
from story.prompts.route.expert import RouteExpertOutput
from utils.models import Task
from utils.llm import clean_markdown_fences, template_fill
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import get_task_db, dict_to_task
from story.rag import save



async def atom(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("atom_result") in ["atom", "complex"]:
        return dict_to_task(db_task_data)

    if task.parent_id:
        parent_task_data = task_db.get_task_by_id(task.parent_id)
        if parent_task_data and parent_task_data.get("task_type") == "design":
            grandparent_id = parent_task_data.get("parent_id")
            if grandparent_id:
                grandparent_task_data = task_db.get_task_by_id(grandparent_id)
                if grandparent_task_data and grandparent_task_data.get("task_type") == "design":
                    updated_task = task.model_copy(deep=True)
                    updated_task.results["atom"] = "atom"
                    updated_task.results["atom_reasoning"] = "父任务和祖父任务均为 design 类型, 为防止无限分解, 此任务被强制设为原子任务。"
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

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.design.atom import system_prompt, user_prompt
    from story.models.atom import AtomOutput

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0, 
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
        raise ValueError("") 

    logger.info(f"完成 \n{agentOutput.structured_response.model_dump_json(indent=2, ensure_ascii=False)}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["atom"] = agentOutput.structured_response.atom_result
    updated_task.results["atom_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "") or agentOutput.structured_response.reasoning
    if agentOutput.structured_response.complex_reasons:
        updated_task.results["complex_reasons"] = agentOutput.structured_response.complex_reasons

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
        "complex_reasons": task.results.get("complex_reasons", ""), 
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "overall_planning": overall_planning,
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.design.decomposition import system_prompt, user_prompt
    from story.models.plan import PlanOutput, plan_to_tasks

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

    if not agentOutput.structured_response:
        raise ValueError("") 

    logger.info(f"完成 \n{agentOutput.structured_response.model_dump_json(indent=2, ensure_ascii=False)}")

    updated_task = task.model_copy(deep=True)
    sub_tasks = plan_to_tasks(agentOutput.structured_response.sub_tasks, parent_task=updated_task)
    updated_task.sub_tasks = sub_tasks
    updated_task.results["decomposition_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "") or agentOutput.structured_response.reasoning

    task_db.add_result(updated_task)
    task_db.add_sub_tasks(updated_task)
    return updated_task



###############################################################################


async def route(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("expert"):
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

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.route.expert import system_prompt, user_prompt

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
        output_cls = RouteExpertOutput, 
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
        raise ValueError("") 

    logger.info(f"完成 \n{agentOutput.structured_response.model_dump_json(indent=2, ensure_ascii=False)}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["expert"] = agentOutput.structured_response.expert
    updated_task.results["expert_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "") or agentOutput.structured_response.reasoning

    task_db.add_result(updated_task)
    return updated_task



###############################################################################


async def design(task: Task, expert: str) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("design"):
        return dict_to_task(db_task_data)

    book_meta = get_meta_db().get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    global_state_summary = book_meta.get("global_state_summary", "")
    
    latest_text = task_db.get_text_latest()
    overall_planning = task_db.get_overall_planning(task)
    design_dependent = task_db.get_dependent_design(task)
    search_dependent = task_db.get_dependent_search(task)
    
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
    from story.rag.tools import get_design_tool, get_plot_history_tool, get_search_tool
    module = importlib.import_module(f"story.prompts.design.{expert}")

    tools = [get_design_tool(task.run_id), get_plot_history_tool(task.run_id), get_search_tool(task.run_id)]
    if expert in ["strategist", "style", "title", "synopsis"]:
        tools = []

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.75, 
        max_tokens = None,
        max_retries = 10,
        api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        api_base = os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
    )
    agent = FunctionAgent(
        system_prompt = module.system_prompt,
        tools = tools,
        llm = llm,
        output_cls = None, 
        streaming = False,
        timeout = 600,
        verbose= False
    )
    user_msg = template_fill(module.user_prompt, context)
    handler = agent.run(user_msg)

    logger.info(f"system_prompt=\n{module.system_prompt}")
    logger.info(f"user_msg=\n{user_msg}")

    agentOutput = await handler

    raw_output = clean_markdown_fences(agentOutput.response)
    if not raw_output:
        raise ValueError("")
    logger.info(f"完成 \n{raw_output}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = raw_output
    updated_task.results["design_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)
    meta_db = get_meta_db()
    if expert == "style":
        meta_db.update_style(task.run_id, raw_output)
    elif expert == "synopsis":
        meta_db.update_synopsis(task.run_id, raw_output)
    elif expert == "title":
        meta_db.update_title(task.run_id, raw_output)

    save.design(task, raw_output)
    return updated_task



###############################################################################



async def aggregate(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("design"):
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
        "subtask_design": task_db.get_subtask_design(task.id),
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.design.aggregate import system_prompt, user_prompt

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.4, 
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
        raise ValueError("")
    logger.info(f"完成 \n{raw_output}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = raw_output
    updated_task.results["design_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)
    
    save.design(task, raw_output)
    return updated_task



###############################################################################



async def book_level_design(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("book_level_design"):
        return dict_to_task(db_task_data)
    
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(task.run_id) or {}
    book_level_design = book_meta.get("book_level_design", "")
    
    overall_planning = task_db.get_overall_planning(task)

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "design": task.results.get("design", ""),
        "overall_planning": overall_planning,
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.design.book_level_design import system_prompt, user_prompt

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.2, 
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
        raise ValueError("")
    logger.info(f"完成 \n{raw_output}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["book_level_design"] = raw_output
    updated_task.results["book_level_design_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)
    meta_db.update_book_level_design(task.run_id, raw_output)
    return updated_task
