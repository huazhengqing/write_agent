import copy
import litellm
import re
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

def get_llm_messages(SYSTEM_PROMPT: str, USER_PROMPT: str, context_dict_system: Dict[str, Any] = None, context_dict_user: Dict[str, Any] = None) -> list[dict]:
    if not SYSTEM_PROMPT and not USER_PROMPT:
        raise ValueError("SYSTEM_PROMPT 和 USER_PROMPT 不能同时为空")

    messages = []

    system_content = SYSTEM_PROMPT
    if context_dict_system:
        safe_context_system = collections.defaultdict(str, context_dict_system)
        system_content = SYSTEM_PROMPT.format_map(safe_context_system)
    
    if system_content and system_content.strip():
        messages.append({"role": "system", "content": system_content})

    user_content = USER_PROMPT
    if context_dict_user:
        safe_context_user = collections.defaultdict(str, context_dict_user)
        user_content = USER_PROMPT.format_map(safe_context_user)

    if user_content and user_content.strip():
        messages.append({"role": "user", "content": user_content})

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

def _clean_markdown_fences(content: str) -> str:
    """如果内容被Markdown代码块包裹, 则移除它们。"""
    if not content:
        return ""
    
    text = content.strip()
    
    # 检查是否以 ``` 开头
    if not text.startswith("```"):
        return text

    # 移除开头的 ```lang\n
    text = re.sub(r"^```[^\n]*\n?", "", text, count=1)
    
    # 检查是否以 ``` 结尾
    if text.endswith("```"):
        text = text[:-3]
        
    return text.strip()

async def llm_acompletion(llm_params):
    params_to_log = llm_params.copy()
    params_to_log.pop('messages', None)
    logger.info(f"LLM 参数:\n{json.dumps(params_to_log, indent=2, ensure_ascii=False, default=str)}")

    system_prompt = ""
    user_prompt = ""
    messages = llm_params.get('messages', [])
    for message in messages:
        if message.get("role") == "system":
            system_prompt = message.get("content", "")
        elif message.get("role") == "user":
            user_prompt = message.get("content", "")
    logger.info(f"系统提示词:\n{system_prompt}")
    logger.info(f"用户提示词:\n{user_prompt}")

    response = await litellm.acompletion(**llm_params)

    if not response.choices or not response.choices[0].message:
        raise ValueError(f"{llm_params}")
    
    message = response.choices[0].message
    content = message.content
    if not content:
        raise ValueError(f"{message}")
    
    # 如果响应格式不是json_object, 则清理Markdown代码块
    response_format = llm_params.get('response_format', {})
    if response_format.get('type') != 'json_object':
        cleaned_content = _clean_markdown_fences(content)
        message.content = cleaned_content
        content = cleaned_content # 更新 content 以便日志记录

    reason = message.get("reasoning_content") or message.get("reasoning", "")
    if reason:
        logger.info(f"推理过程:\n{reason}")
    
    formatted_content = _format_message_content(content)
    logger.info(f"返回内容:\n{formatted_content}")

    return message