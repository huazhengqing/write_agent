import os
from typing import List
from loguru import logger
from llama_index.core.base.base_query_engine import BaseQueryEngine
from rag.hybrid_prompts import synthesis_system_prompt, synthesis_user_prompt



async def hybrid_query(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    question: str,
    synthesis_system_prompt: str = synthesis_system_prompt,
    synthesis_user_prompt: str = synthesis_user_prompt,
) -> str:
    if not question or not isinstance(question, str):
        logger.warning("查询问题为空或类型不正确, 无法执行混合查询。")
        return ""

    logger.info(f"开始对问题 '{question}' 执行混合查询...")
    
    from rag.vector_query import index_query
    import asyncio

    vector_task = index_query(vector_query_engine, question)
    kg_task = index_query(kg_query_engine, question)
    vector_content, kg_content = await asyncio.gather(vector_task, kg_task)

    if not vector_content and not kg_content:
        logger.warning(f"对于问题 '{question}', 向量和知识图谱查询均未返回任何内容。")
        return ""

    formatted_vector_str = vector_content or ""
    formatted_kg_str = kg_content or ""
    context_dict_user = {
        "vector_str": formatted_vector_str, 
        "kg_str": formatted_kg_str, 
        "question": question
    }

    from llama_index.llms.litellm import LiteLLM
    from llama_index.core.agent.workflow import FunctionAgent
    from utils.llm import template_fill, clean_markdown_fences

    llm = LiteLLM(
        model = f"openai/summary",
        temperature = 0.4, 
        max_tokens = None,
        max_retries = 10,
        api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        api_base = os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
    )
    agent = FunctionAgent(
        system_prompt = synthesis_system_prompt,
        tools = [],
        llm = llm,
        output_cls = None, 
        streaming = False,
        timeout = 600,
        verbose= False
    )
    user_msg = template_fill(synthesis_user_prompt, context_dict_user)
    handler = agent.run(user_msg)

    logger.info(f"system_prompt=\n{synthesis_system_prompt}")
    logger.info(f"user_msg=\n{user_msg}")

    agentOutput = await handler

    raw_output = clean_markdown_fences(agentOutput.response)
    if not raw_output:
        logger.warning(f"Agent在为问题 '{question}' 综合答案时, 经过多次重试后仍然失败。")
        return ""
    logger.success(f"混合查询答案综合完成。")
    return raw_output.strip()



async def hybrid_query_batch(
    vector_query_engine: BaseQueryEngine,
    kg_query_engine: BaseQueryEngine,
    questions: List[str],
    synthesis_system_prompt: str = synthesis_system_prompt,
    synthesis_user_prompt: str = synthesis_user_prompt,
) -> List[str]:
    if not questions:
        return []

    logger.info(f"开始执行 {len(questions)} 个问题的批量混合查询...")

    import asyncio
    sem = asyncio.Semaphore(3)

    async def safe_hybrid_query(question: str) -> str:
        async with sem:
            try:
                return await hybrid_query(
                    vector_query_engine,
                    kg_query_engine,
                    question,
                    synthesis_system_prompt,
                    synthesis_user_prompt,
                )
            except Exception as e:
                logger.error("批量混合查询中, 问题 '{}' 失败: {}", question, e, exc_info=True)
                return ""

    tasks = [safe_hybrid_query(q) for q in questions]
    results = await asyncio.gather(*tasks)
    logger.success(f"批量混合查询完成。")
    return results
