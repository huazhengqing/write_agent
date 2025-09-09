import copy
import re
import collections
import json
import hashlib
import os
import threading
from loguru import logger
from typing import List, Dict, Any, Optional, Type, Callable
from pydantic import BaseModel, ValidationError


LLM_TEMPERATURES = {
    "creative": 0.75,
    "reasoning": 0.1,
    "summarization": 0.2,
    "synthesis": 0.4,
    "classification": 0.0,
}

LLM_PARAMS_reasoning = {
    'model': 'openrouter/deepseek/deepseek-r1-0528:free',
    'api_key': os.getenv("OPENROUTER_API_KEY"),
    'temperature': 0.1,
    'caching': True,
    'max_tokens': 10000,
    'max_completion_tokens': 10000,
    'timeout': 900,
    'num_retries': 3,
    'respect_retry_after': True,
    'disable_moderation': True,
    'disable_safety_check': True,
    'safe_mode': False,
    'safe_prompt': False,
    'fallbacks': [
        {
            "model": 'openai/deepseek-ai/DeepSeek-R1-0528',
            "api_base": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        # 'openrouter/deepseek/deepseek-r1-0528-qwen3-8b',
        # 'openrouter/qwen/qwen3-32b',
        # 'openrouter/qwen/qwen3-30b-a3b',
        # 'openrouter/deepseek/deepseek-r1-distill-llama-70b',
    ]
}

LLM_PARAMS_fast= {
    "model": 'openrouter/deepseek/deepseek-chat-v3-0324:free',
    'api_key': os.getenv("OPENROUTER_API_KEY"),
    'temperature': 0.1,
    'caching': True,
    'max_tokens': 5000,
    'max_completion_tokens': 5000,
    'timeout': 300,
    'num_retries': 3,
    'respect_retry_after': True,
    'disable_moderation': True,
    'disable_safety_check': True,
    'safe_mode': False,
    'safe_prompt': False,
    "fallbacks": [
        {
            "model": 'openai/deepseek-ai/DeepSeek-V3',
            "api_base": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ]
}

def custom_get_cache_key(**kwargs):
    """
    自定义缓存键生成逻辑。
    仅根据 'messages' 和 'temperature' 生成缓存键。
    """
    messages = kwargs.get("messages", [])
    temperature = kwargs.get("temperature", LLM_PARAMS_reasoning.get('temperature'))
    messages_str = json.dumps(messages, sort_keys=True)
    key_data = {
        "messages": messages_str,
        "temperature": temperature
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()


_litellm_setup_lock = threading.Lock()
_litellm_initialized = False

def setup_litellm():
    """
    延迟初始化 litellm 库。
    此函数是线程安全的, 确保 litellm 的配置只执行一次。
    """
    global _litellm_initialized
    if _litellm_initialized:
        return

    with _litellm_setup_lock:
        if _litellm_initialized:
            return

        import litellm
        from litellm.caching.caching import Cache
        
        logger.info("正在延迟加载和配置 litellm...")
        litellm.enable_json_schema_validation=True
        litellm.cache = Cache(type='disk', get_cache_key=custom_get_cache_key)
        _litellm_initialized = True
        logger.info("litellm 加载和配置完成。")

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


PROMPT_SELF_CORRECTION = """
# 任务: 修正JSON输出
你上次的输出因为格式错误导致解析失败。请根据原始任务和错误信息, 重新生成。

# 错误信息
{error}

# 你上次格式错误的输出
{raw_output}

# 这是你需要完成的原始任务
{original_task}

# 新的要求
1.  严格遵循原始任务的所有指令。
2.  严格根据 Pydantic 模型的要求, 修正并仅返回完整的、有效的 JSON 对象。
3.  禁止在 JSON 前后添加任何额外解释或 markdown 代码块。
"""


async def llm_acompletion(llm_params: Dict[str, Any], response_model: Optional[Type[BaseModel]] = None, validator: Optional[Callable[[Any], None]] = None):
    setup_litellm()
    import litellm

    params_to_log = llm_params.copy()
    params_to_log.pop('messages', None)
    logger.info(f"LLM 参数:\n{json.dumps(params_to_log, indent=2, ensure_ascii=False, default=str)}")

    llm_params_for_api = llm_params.copy()
    if response_model:
        llm_params_for_api['response_format'] = {
            "type": "json_object",
            "schema": response_model.model_json_schema()
        }

    max_retries = 5
    for attempt in range(max_retries):
        system_prompt = ""
        user_prompt = ""
        messages = llm_params_for_api.get('messages', [])
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
            elif message.get("role") == "user":
                user_prompt = message.get("content", "")
        if system_prompt:
            logger.info(f"系统提示词:\n{system_prompt}")
        if user_prompt:
            logger.info(f"用户提示词:\n{user_prompt}")

        raw_output_for_correction = None
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
                    raw_output_for_correction = _clean_markdown_fences(tool_call.function.arguments)
                    if hasattr(tool_call.function, 'parsed_arguments') and tool_call.function.parsed_arguments:
                        parsed_args = tool_call.function.parsed_arguments
                    else:
                        parsed_args = json.loads(raw_output_for_correction)
                    validated_data = response_model(**parsed_args)
                elif message.content:
                    raw_output_for_correction = _clean_markdown_fences(message.content)
                    validated_data = response_model.model_validate_json(raw_output_for_correction)

                if validated_data:
                    message.validated_data = validated_data
                else:
                    raise ValueError("LLM响应既无tool_calls也无有效content可供解析。")
            else:
                message.content = _clean_markdown_fences(message.content)
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
                    cache_key = litellm.cache.get_cache_key(**llm_params_for_api)
                    litellm.cache.delete(key=cache_key)
                except Exception as cache_e:
                    logger.error(f"删除缓存条目失败: {cache_e}")

                # 针对JSON解析/验证错误的自我修正逻辑
                if response_model and isinstance(e, (ValidationError, json.JSONDecodeError)) and raw_output_for_correction:
                    logger.info("检测到JSON错误, 下次尝试将进行自我修正...")
                    # 提取原始用户任务内容
                    original_user_content = ""
                    for msg in reversed(llm_params['messages']):
                        if msg['role'] == 'user':
                            original_user_content = msg.get('content', '')
                            break
                    
                    correction_prompt = PROMPT_SELF_CORRECTION.format(
                        error=str(e), 
                        raw_output=raw_output_for_correction,
                        original_task=original_user_content
                    )
                    system_message = [m for m in llm_params['messages'] if m['role'] == 'system']
                    llm_params_for_api['messages'] = system_message + [{"role": "user", "content": correction_prompt}]
                else:
                    llm_params_for_api['messages'] = llm_params['messages']

                logger.info("正在准备重试...")
            else:
                logger.error("LLM 响应在多次重试后仍然无效, 任务失败。")
                raise

    raise RuntimeError("llm_acompletion 在所有重试后失败, 这是一个不应出现的情况。")
