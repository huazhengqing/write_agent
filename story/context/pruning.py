import os
import importlib
from loguru import logger
from utils.models import Task
from utils.llm import template_fill, clean_markdown_fences



async def pruning(context_type: str, task: Task, context: str) -> str:
    if not context:
        return ""

    context_dict = {
        "task": task.to_context(),
        "context": context,
    }
    module = importlib.import_module(f"story.prompts.pruning.{context_type}")

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.1, 
        max_tokens = None,
        max_retries = 10,
        api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        api_base = os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
    )
    agent = FunctionAgent(
        system_prompt = module.system_prompt,
        tools = [],
        llm = llm,
        output_cls = None, 
        streaming = False,
        timeout = 600,
        verbose= False
    )
    user_msg = template_fill(module.user_prompt, context_dict)
    handler = agent.run(user_msg)

    logger.info(f"system_prompt=\n{module.system_prompt}")
    logger.info(f"user_msg=\n{user_msg}")

    agentOutput = await handler

    raw_output = clean_markdown_fences(agentOutput.response)
    if not raw_output:
        logger.warning(f"Agent在为任务 '{task.id}' 进行上下文剪枝({context_type})时, 返回了空内容。")
        return ""
    
    return raw_output
