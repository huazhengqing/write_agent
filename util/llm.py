import copy
import litellm
import collections
import json
from loguru import logger
from litellm.caching.caching import Cache
from typing import List, Dict, Any, Optional


litellm.cache = Cache(type='disk')
# litellm.input_callback = ["lunary"]
# litellm.success_callback = ["lunary"]
# litellm.failure_callback = ["lunary"]

litellm.enable_json_schema_validation=True

LLM_PARAMS_reasoning = {
    'model': 'openrouter/deepseek/deepseek-r1-0528:free',
    'temperature': 0.1,
    'caching': True,
    'max_tokens': 10000,
    'max_completion_tokens': 10000,
    'timeout': 900,
    'num_retries': 20,
    'respect_retry_after': True,
    'disable_moderation': True,
    'disable_safety_check': True,
    'safe_mode': False,
    'safe_prompt': False,
    'fallbacks': [
        'openai/deepseek-ai/DeepSeek-R1-0528',
        # 'openrouter/deepseek/deepseek-r1-0528-qwen3-8b',
        # 'openrouter/qwen/qwen3-32b',
        # 'openrouter/qwen/qwen3-30b-a3b',
        # 'openrouter/deepseek/deepseek-r1-distill-llama-70b',
    ]
}

LLM_PARAMS_fast= {
    "model": 'openrouter/deepseek/deepseek-chat-v3-0324:free',
    "temperature": 0.0,
    "max_tokens": 5000,
    "max_completion_tokens": 5000,
    "caching": True,
    "timeout": 300,
    "num_retries": 30,
    "respect_retry_after": True,
    "fallbacks": [
        'openai/deepseek-ai/DeepSeek-V3',
        # 'openrouter/deepseek/deepseek-r1-0528-qwen3-8b', 
    ]
}

def get_llm_messages(SYSTEM_PROMPT: str, USER_PROMPT: str, context_dict: Dict[str, Any] = None) -> list[dict]:
    context = context_dict or {}
    safe_context = collections.defaultdict(str, context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format_map(safe_context)}
    ]
    return messages

def get_llm_params(
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    llm_params = LLM_PARAMS_reasoning.copy()
    llm_params.update(kwargs)
    if temperature is not None:
        llm_params['temperature'] = temperature
    if tools is not None:
        llm_params['tools'] = tools
    llm_params['messages'] = copy.deepcopy(messages)
    return llm_params

def _format_json_content(content: str) -> str:
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content

def _format_message_content(content: str) -> str:
    if content.strip().startswith('{') or content.strip().startswith('['):
        return _format_json_content(content)
    return content

async def llm_acompletion(llm_params):
    messages = llm_params.get('messages', [])
    system_prompt = ""
    user_prompt = ""
    
    for message in messages:
        if message.get("role") == "system":
            system_prompt = message.get("content", "")
        elif message.get("role") == "user":
            user_prompt = message.get("content", "")
    
    logger.info("=" * 50)
    logger.info("系统提示词:")
    logger.info("-" * 30)
    logger.info(f"\n{system_prompt}")
    logger.info("-" * 30)
    logger.info("用户提示词:")
    logger.info("-" * 30)
    logger.info(f"\n{user_prompt}")
    logger.info("=" * 50)

    response = await litellm.acompletion(**llm_params)

    if not response.choices or not response.choices[0].message:
        raise ValueError(f"{llm_params}")
    
    message = response.choices[0].message
    
    content = message.content
    if not content:
        raise ValueError(f"{message}")
    
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    
    logger.info("=" * 50)
    logger.info("推理过程:")
    logger.info("-" * 30)
    if reason:
        logger.info(f"\n{reason}")
    else:
        logger.error("没有推理过程")
    logger.info("-" * 30)
    
    logger.info("返回内容:")
    logger.info("-" * 30)
    formatted_content = _format_message_content(content)
    logger.info(f"\n{formatted_content}")
    logger.info("=" * 50)

    return message