from loguru import logger
import collections
import litellm
from litellm.caching.cache_key_generator import get_cache_key
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_acompletion
from utils.rag import get_rag
from utils.prompt_loader import load_prompts


async def design_aggregate(task: Task) -> Task:
    logger.info(f"开始\n{task.model_dump_json(indent=2, exclude_none=True)}")

    SYSTEM_PROMPT, USER_PROMPT = load_prompts(task.category, "design_aggregate_cn", "SYSTEM_PROMPT", "USER_PROMPT")
    context = await get_rag().get_context_aggregate_design(task)
    messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context)
    llm_params = get_llm_params(messages, temperature=0.75)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = await llm_acompletion(llm_params)
            content = message.content
            if not content or len(content.strip()) < 20: # 简单验证：内容不能为空或过短
                raise ValueError("生成的内容为空或过短。")
            break  # 验证成功，跳出循环
        except Exception as e:
            logger.warning(f"响应内容验证失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                try:
                    cache_key = get_cache_key(**llm_params)
                    litellm.cache.delete(cache_key)
                    logger.info(f"已删除错误的缓存条目: {cache_key}。正在重试...")
                except Exception as cache_e:
                    logger.error(f"删除缓存条目失败: {cache_e}")
            else:
                logger.error("LLM 响应在多次重试后仍然无效，任务失败。")
                raise
    
    reasoning = message.get("reasoning_content") or message.get("reasoning", "")
    updated_task = task.model_copy(deep=True)
    updated_task.results["design"] = content
    updated_task.results["design_reasoning"] = reasoning

    logger.info(f"完成\n{updated_task.model_dump_json(indent=2, exclude_none=True)}")
    return updated_task