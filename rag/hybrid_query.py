from typing import List
from loguru import logger
from llama_index.core.base.base_query_engine import BaseQueryEngine
from utils.llm import get_llm_params, get_llm_messages, llm_completion
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
    vector_task = index_query(vector_query_engine, question)
    kg_task = index_query(kg_query_engine, question)
    import asyncio
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
    messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)
    final_llm_params = get_llm_params(llm_group='summary', messages=messages, temperature=0.4)
    final_message = await llm_completion(final_llm_params)
    final_answer = final_message.content.strip()
    return final_answer



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


