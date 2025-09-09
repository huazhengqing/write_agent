import copy
import litellm
import re
import collections
import json
from loguru import logger
from litellm.caching.caching import Cache
from typing import List, Dict, Any, Optional, Type, Callable
from pydantic import BaseModel
from litellm.caching.cache_key_generator import get_cache_key

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
    'temperature': 0.1,
    'caching': True,
    'max_tokens': 5000,
    'max_completion_tokens': 5000,
    'timeout': 900,
    'num_retries': 20,
    'respect_retry_after': True,
    'disable_moderation': True,
    'disable_safety_check': True,
    'safe_mode': False,
    'safe_prompt': False,
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

def _default_text_validator(content: str):
    if not content or len(content.strip()) < 20:
        raise ValueError("生成的内容为空或过短。")

async def llm_acompletion(llm_params: Dict[str, Any], response_model: Optional[Type[BaseModel]] = None, validator: Optional[Callable[[Any], None]] = None):
    params_to_log = llm_params.copy()
    if response_model:
        params_to_log['response_model'] = response_model.__name__
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
    if system_prompt:
        logger.info(f"系统提示词:\n{system_prompt}")
    if user_prompt:
        logger.info(f"用户提示词:\n{user_prompt}")

    llm_params_for_api = llm_params.copy()
    if response_model:
        llm_params_for_api['response_model'] = response_model

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(**llm_params_for_api)

            if not response.choices or not response.choices[0].message:
                raise ValueError("LLM响应中缺少 choices 或 message。")
            
            message = response.choices[0].message

            if response_model:
                validated_data = None
                # 优先从 tool_calls 解析
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    if hasattr(tool_call.function, 'parsed_arguments') and tool_call.function.parsed_arguments:
                        parsed_args = tool_call.function.parsed_arguments
                    else:
                        parsed_args = json.loads(tool_call.function.arguments)
                    validated_data = response_model(**parsed_args)
                # 降级到从 content 解析 (用于 json_mode)
                elif message.content:
                    validated_data = response_model.model_validate_json(message.content)
                
                if validated_data:
                    message.validated_data = validated_data
                else:
                    raise ValueError("LLM响应既无tool_calls也无有效content可供解析。")
            else:
                cleaned_content = _clean_markdown_fences(message.content)
                if message.content != cleaned_content:
                    message.content = cleaned_content
                
                # 如果没有 response_model，则进行文本验证。
                # 优先使用传入的自定义 validator，否则使用默认的简单文本验证器。
                active_validator = validator or _default_text_validator
                active_validator(message.content)
                
            reasoning = message.get("reasoning_content") or message.get("reasoning", "")
            if reasoning:
                logger.info(f"推理过程:\n{reasoning}")

            logger.info(f"返回内容 (尝试 {attempt + 1}):\n{_format_message_content(message.content)}")
            return message

        except Exception as e:
            logger.warning(f"LLM调用或验证失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                try:
                    cache_key = get_cache_key(**llm_params_for_api)
                    litellm.cache.delete(cache_key)
                    logger.info(f"已删除错误的缓存条目: {cache_key}。正在重试...")
                except Exception as cache_e:
                    logger.error(f"删除缓存条目失败: {cache_e}")
            else:
                logger.error("LLM 响应在多次重试后仍然无效, 任务失败。")
                raise

    raise RuntimeError("llm_acompletion 在所有重试后失败, 这是一个不应出现的情况。")
