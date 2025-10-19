import os
from loguru import logger
from story.rag import save
from utils.models import Task
from utils.llm import clean_markdown_fences, template_fill
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import dict_to_task, get_task_db



async def summary(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("summary"):
        return dict_to_task(db_task_data)
    
    context = {
        "task": task.to_context(),
        "text": task.results.get("write"),
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.summary.summary import system_prompt, user_prompt

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
        raise ValueError(f"Agent在为任务 '{task.id}' 执行总结时, 经过多次重试后仍然失败。")
    logger.info(f"完成 \n{raw_output}")

    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = raw_output
    updated_task.results["summary_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)
    
    save.summary(task, raw_output)
    return updated_task



###############################################################################



async def aggregate(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("summary"):
        return dict_to_task(db_task_data)
    
    context = {
        "task": task.to_context(),
        "subtask_summary": task_db.get_subtask_summary(task.id),
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.summary.aggregate import system_prompt, user_prompt

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
        raise ValueError(f"Agent在为任务 '{task.id}' 执行聚合总结时, 经过多次重试后仍然失败。")
    logger.info(f"完成 \n{raw_output}")
    
    updated_task = task.model_copy(deep=True)
    updated_task.results["summary"] = raw_output
    updated_task.results["summary_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)
    
    save.summary(task, raw_output)
    return updated_task



###############################################################################



async def global_state(task: Task) -> Task:
    task_db = get_task_db(run_id=task.run_id)
    db_task_data = task_db.get_task_by_id(task.id)
    if db_task_data and db_task_data.get("global_state"):
        return dict_to_task(db_task_data)
    
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(task.run_id) or {}
    global_state_summary = book_meta.get("global_state_summary", "")
    
    summary = task.results.get("summary", "")
    if not summary:
        raise ValueError(f"任务 '{task.id}' 缺少 'summary' 结果, 无法生成全局状态。")

    context = {
        "task": task.to_context(),
        "global_state_summary": global_state_summary,
        "summary": summary,
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from story.prompts.summary.global_state import system_prompt, user_prompt

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
        raise ValueError(f"Agent在为任务 '{task.id}' 生成全局状态时, 经过多次重试后仍然失败。")
    logger.info(f"完成 \n{raw_output}")
    updated_task = task.model_copy(deep=True)
    updated_task.results["global_state"] = raw_output
    updated_task.results["global_state_reasoning"] = agentOutput.raw.get("reasoning_content", "") or agentOutput.raw.get("reasoning", "")
    
    task_db.add_result(updated_task)
    meta_db.update_global_state_summary(task.run_id, raw_output)
    return updated_task
