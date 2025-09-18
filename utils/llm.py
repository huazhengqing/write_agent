import asyncio
import copy
import os
import re
import collections
import json
import hashlib
import sys
import litellm
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Literal, Optional, Type, Callable, Union
from litellm.caching.caching import Cache
from litellm import RateLimitError, Timeout, APIConnectionError, ServiceUnavailableError


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.file import cache_dir


import logging
litellm_logger = logging.getLogger("litellm")
litellm_logger.setLevel(logging.WARNING)


load_dotenv()


llm_temperatures = {
    "creative": 0.75,
    "reasoning": 0.1,
    "summarization": 0.2,
    "synthesis": 0.4,
    "classification": 0.0,
}


llms_api = {
    "reasoning": {
        "model": "openrouter/deepseek/deepseek-r1-0528:free",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "context_window": 163840,
        "fallbacks": [
            {
                "model": "openai/deepseek-ai/DeepSeek-R1-0528",
                "api_base": "https://api-inference.modelscope.cn/v1/",
                "api_key": os.getenv("modelscope_API_KEY"), 
                "context_window": 163840,
            }, 
            {
                "model": "gemini/gemini-2.5-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY"), 
                "context_window": 1048576,
            }, 
            {
                "model": "groq/llama-3.1-8b-instant",
                "api_key": os.getenv("GROQ_API_KEY"), 
                "context_window": 131072,
            }, 
            {
                "model": "groq/qwen/qwen3-32b",
                "api_key": os.getenv("GROQ_API_KEY"), 
                "context_window": 131072,
            }
            # "openrouter/deepseek/deepseek-r1-0528-qwen3-8b",
            # "openrouter/qwen/qwen3-32b",
            # "openrouter/qwen/qwen3-30b-a3b",
            # "openrouter/deepseek/deepseek-r1-distill-llama-70b",
        ]
    },
    "fast": {
        "model": "openrouter/deepseek/deepseek-chat-v3-0324:free",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "context_window": 163840,
        "fallbacks": [
            {
                "model": "openai/deepseek-ai/DeepSeek-V3",
                "api_base": "https://api-inference.modelscope.cn/v1/",
                "api_key": os.getenv("modelscope_API_KEY"), 
                "context_window": 163840,
            }, 
            {
                "model": "gemini/gemini-2.5-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY"), 
                "context_window": 1048576,
            }, 
            {
                "model": "groq/llama-3.1-8b-instant",
                "api_key": os.getenv("GROQ_API_KEY"), 
                "context_window": 131072,
            }, 
            {
                "model": "groq/qwen/qwen3-32b",
                "api_key": os.getenv("GROQ_API_KEY"), 
                "context_window": 131072,
            }
        ]
    }
}


llm_params_general = {
    "temperature": llm_temperatures["reasoning"],
    "caching": True,
    "max_tokens": 8000,
    "max_completion_tokens": 10000,
    "timeout": 900,
    "num_retries": 3,
    "respect_retry_after": True,
    "disable_moderation": True,
    "disable_safety_check": True,
    "safe_mode": False,
    "safe_prompt": False,
    "exceptions_to_fallback_on": [
        RateLimitError,
        Timeout,
        APIConnectionError,
        ServiceUnavailableError,
    ]
    # "context_window_fallback_dict": {
    #     "openai/deepseek-ai/DeepSeek-R1-0528": "openrouter/deepseek/deepseek-r1-0528:free", 
    #     "openrouter/deepseek/deepseek-r1-0528:free": "openai/deepseek-ai/DeepSeek-R1-0528", 
    #     "openai/deepseek-ai/DeepSeek-V3": "openrouter/deepseek/deepseek-chat-v3-0324:free", 
    #     "openrouter/deepseek/deepseek-chat-v3-0324:free": "openai/deepseek-ai/DeepSeek-V3", 
    # }
}


def get_llm_messages(
    system_prompt: str = None, 
    user_prompt: str = None, 
    context_dict_system: Dict[str, Any] = None, 
    context_dict_user: Dict[str, Any] = None
) -> list[dict]:
    if not system_prompt and not user_prompt:
        raise ValueError("system_prompt 和 user_prompt 不能同时为空")

    messages = []

    system_content = system_prompt
    if context_dict_system:
        safe_context_system = collections.defaultdict(str, context_dict_system)
        system_content = system_prompt.format_map(safe_context_system)
    
    if system_content and system_content.strip():
        messages.append({"role": "system", "content": system_content})

    user_content = user_prompt
    if context_dict_user:
        safe_context_user = collections.defaultdict(str, context_dict_user)
        user_content = user_prompt.format_map(safe_context_user)

    if user_content and user_content.strip():
        messages.append({"role": "user", "content": user_content})

    return messages


def get_llm_params(
    llm_group: Literal['reasoning', 'fast'] = 'reasoning',
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = llm_temperatures["reasoning"],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    llm_params = llms_api[llm_group].copy()
    llm_params.update(**llm_params_general)
    llm_params.update(kwargs)
    llm_params["temperature"] = temperature
    if tools is not None:
        llm_params["tools"] = tools
    if messages is not None:
        llm_params["messages"] = copy.deepcopy(messages)
    return llm_params


###############################################################################


embeddings_api = {
    "bge-m3": {
        "model": "openai/BAAI/bge-m3",
        "api_base": "https://api.siliconflow.cn/v1/",
        "api_key": os.getenv("siliconflow_API_KEY"),
        # "dims": 1024,
    },
    "gemini": {
        "model": "gemini/gemini-embedding-001",
        "api_key": os.getenv("GEMINI_API_KEY"),
        # "dims": 3072,
    }
}

embedding_params_general = {
    "caching": True,
    "timeout": 300,
    "num_retries": 3,
    "respect_retry_after": True
}


def get_embedding_params(
        embedding: Literal['bge-m3', 'gemini'] = 'bge-m3',
        **kwargs: Any
    ) -> Dict[str, Any]:
    embedding_params = embeddings_api[embedding].copy()
    embedding_params.update(**embedding_params_general)
    embedding_params.update(kwargs)
    return embedding_params


###############################################################################


rerank_api = {
    "bge": {
        "model": "BAAI/bge-reranker-v2-m3",
        "api_base": "https://api.siliconflow.cn/v1/",
        "api_key": os.getenv("siliconflow_API_KEY"),
        "context_window": 8000,
    }
}


def get_rerank_params(
        rerank: Literal['bge'] = 'bge',
        **kwargs: Any
    ) -> Dict[str, Any]:
    rerank_params = rerank_api[rerank].copy()
    rerank_params.update(**embedding_params_general)
    rerank_params.update(kwargs)
    return rerank_params


###############################################################################


def custom_get_cache_key(**kwargs):
    """
    仅根据 "messages" 和 "temperature" 生成缓存键。
    """
    messages = kwargs.get("messages", [])
    temperature = kwargs.get("temperature", llm_temperatures["reasoning"])
    messages_str = json.dumps(messages, sort_keys=True)
    key_data = {
        "messages": messages_str,
        "temperature": temperature
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


cache = Cache(type="disk", disk_cache_dir=cache_dir)
cache.get_cache_key = custom_get_cache_key
litellm.cache = cache
litellm.enable_cache()
litellm.enable_json_schema_validation=True
litellm.drop_params = True
litellm.telemetry = False
litellm.REPEATED_STREAMING_CHUNK_LIMIT = 20
# litellm._turn_on_debug()


###############################################################################


def format_json_content(content: str) -> str:
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content


def format_message_content(content: str) -> str:
    if content.strip().startswith("{") or content.strip().startswith("["):
        return format_json_content(content)
    return content


def clean_markdown_fences(content: str) -> str:
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


def text_validator_default(content: str):
    if not content or len(content.strip()) < 20:
        raise ValueError("生成的内容为空或过短。")


self_correction_prompt = """
# 任务: 修正JSON输出
你上次的输出因为格式错误导致解析失败。请根据原始任务和错误信息, 重新生成。

# 错误信息
{error}

# 你上次格式错误的输出
{raw_output}

# 这是你需要完成的原始任务
{original_task}

# 新的要求
1.  严格遵循原始任务的所有指令, 尤其是关于JSON Schema的指令。
2.  严格根据 Pydantic 模型的要求, 修正并仅返回完整的、有效的 JSON 对象。
3.  禁止在 JSON 前后添加任何额外解释或代码块。
"""


async def llm_completion(
    llm_params: Dict[str, Any], 
    response_model: Optional[Type[BaseModel]] = None, 
    validator: Optional[Callable[[Any], None]] = None
) -> Dict[str, Any]:
    logger.info("开始执行 llm_completion...")

    params_to_log = llm_params.copy()
    params_to_log.pop("messages", None)
    logger.info(f"LLM 参数:\n{json.dumps(params_to_log, indent=2, ensure_ascii=False, default=str)}")

    llm_params_for_api = llm_params.copy()

    if response_model:
        logger.info(f"启用 JSON 模式, 目标模型: {response_model.__name__}")
        llm_params_for_api["response_format"] = {
            "type": "json_object",
            "schema": response_model.model_json_schema()
        }

    max_retries = 6
    for attempt in range(max_retries):

        system_prompt = ""
        user_prompt = ""
        messages = llm_params_for_api.get("messages", [])
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
            elif message.get("role") == "user":
                user_prompt = message.get("content", "")
        
        if system_prompt:
            logger.info(f"系统提示词:\n{system_prompt}")
        if user_prompt:
            logger.info(f"用户提示词:\n{user_prompt}")
        
        logger.info(f"开始 LLM 调用 (尝试 {attempt + 1}/{max_retries})...")

        raw_output_for_correction = None
        try:
            response = await litellm.acompletion(**llm_params_for_api)
            logger.debug(f"LLM 原始响应: {response}")

            if not response.choices or not response.choices[0].message:
                raise ValueError("LLM响应中缺少 choices 或 message。")
            
            message = response.choices[0].message

            if response_model:
                validated_data = None
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    raw_output_for_correction = tool_call.function.arguments
                    logger.debug(f"从 tool_calls 中提取的原始参数: {raw_output_for_correction}")
                    cleaned_args = clean_markdown_fences(raw_output_for_correction)
                    if hasattr(tool_call.function, "parsed_arguments") and tool_call.function.parsed_arguments:
                        parsed_args = tool_call.function.parsed_arguments
                    else:
                        parsed_args = json.loads(cleaned_args)
                    validated_data = response_model(**parsed_args)
                elif message.content:
                    logger.debug(f"从 message.content 中提取的原始内容: {message.content}")
                    raw_output_for_correction = clean_markdown_fences(message.content)
                    validated_data = response_model.model_validate_json(raw_output_for_correction)

                if validated_data:
                    message.validated_data = validated_data
                else:
                    raise ValueError("LLM响应既无tool_calls也无有效content可供解析。")
            else:
                message.content = clean_markdown_fences(message.content)
                active_validator = validator or text_validator_default
                active_validator(message.content)
            
            logger.success("LLM 响应成功通过验证。")

            reasoning = message.get("reasoning_content") or message.get("reasoning", "")
            if reasoning:
                logger.info(f"推理过程:\n{reasoning}")

            final_content_to_log = message.validated_data if response_model else message.content
            logger.info(f"LLM 成功返回内容 (尝试 {attempt + 1}):\n{final_content_to_log}")

            return message
        except Exception as e:
            logger.warning(f"LLM调用或验证失败 (尝试 {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                try:
                    cache_key = litellm.cache.get_cache_key(**llm_params_for_api)
                    litellm.cache.cache.delete_cache(cache_key)
                except Exception as cache_e:
                    logger.error(f"删除缓存条目失败: {cache_e}")

                # 针对JSON解析/验证错误的自我修正逻辑
                if response_model and isinstance(e, (ValidationError, json.JSONDecodeError)) and raw_output_for_correction:
                    logger.info("检测到JSON错误, 下次尝试将进行自我修正...")
                    # 提取原始用户任务内容
                    original_user_content = ""
                    for msg in reversed(llm_params["messages"]):
                        if msg["role"] == "user":
                            original_user_content = msg.get("content", "")
                            break
                    correction_prompt = self_correction_prompt.format(
                        error=str(e), 
                        raw_output=raw_output_for_correction,
                        original_task=original_user_content
                    )
                    system_message = [m for m in llm_params["messages"] if m["role"] == "system"]
                    llm_params_for_api["messages"] = system_message + [{"role": "user", "content": correction_prompt}]
                else:
                    llm_params_for_api["messages"] = llm_params["messages"]

                logger.info("正在准备重试...")
            else:
                logger.error("LLM 响应在多次重试后仍然无效, 任务失败。")
                raise

    raise RuntimeError("llm_completion 在所有重试后失败, 这是一个不应出现的情况。")


###############################################################################


extraction_system_prompt = """
你是一个数据提取专家。你的任务是根据用户提供的文本，严格按照给定的 Pydantic JSON Schema 提取信息并生成一个 JSON 对象。
你的输出必须是、且只能是一个完整的、有效的 JSON 对象。
不要添加任何解释、注释或代码块。
"""

extraction_user_prompt = """
请从以下文本中提取信息并生成 JSON 对象:
{{cleaned_output}}
"""


async def txt_to_json(cleaned_output: str, response_model: Optional[Type[BaseModel]]):
    logger.info(f"开始将文本转换为 JSON, 目标模型: {response_model.__name__}")
    logger.debug(f"待转换的输入文本:\n{cleaned_output}")

    extraction_messages = get_llm_messages(
        system_prompt=extraction_system_prompt,
        user_prompt=extraction_user_prompt.format(cleaned_output=cleaned_output)
    )
    extraction_llm_params = get_llm_params(
        llm_group='reasoning',
        messages=extraction_messages,
        temperature=llm_temperatures["classification"]
    )
    extraction_response = await llm_completion(llm_params=extraction_llm_params, response_model=response_model)
    validated_data = extraction_response.validated_data

    logger.success("文本成功转换为 JSON 对象。")
    logger.debug(f"转换后的 JSON 对象: {validated_data}")
    return validated_data


###############################################################################


if __name__ == '__main__':
    import asyncio
    from pydantic import Field
    from utils.log import init_logger
    from unittest.mock import patch, AsyncMock
    from litellm.types.llms.base import LLMMessage, Choice, ModelResponse


    init_logger("llm_test")

    class UserInfo(BaseModel):
        name: str = Field(..., description="用户姓名")
        age: int = Field(..., description="用户年龄")
        is_student: bool = Field(..., description="是否是学生")

    async def main():
        # --- Test 1: Simple text completion ---
        logger.info("--- 测试 llm_completion (返回字符串) ---")
        try:
            messages1 = get_llm_messages(
                system_prompt="你是一个乐于助人的助手。",
                user_prompt="你好！请介绍一下你自己。"
            )
            params1 = get_llm_params(messages=messages1, temperature=0.1)
            result1 = await llm_completion(params1)
            logger.success(f"文本返回成功:\n{result1.content}")
        except Exception as e:
            logger.error(f"测试1失败: {e}", exc_info=True)

        # --- Test 2: JSON completion with response_model ---
        logger.info("\n--- 测试 llm_completion (返回 Pydantic 模型) ---")
        try:
            messages2 = get_llm_messages(
                system_prompt="请根据用户信息生成JSON。",
                user_prompt="用户信息：姓名张三，年龄25岁，是一名学生。"
            )
            params2 = get_llm_params(messages=messages2, temperature=0.1)
            result2 = await llm_completion(params2, response_model=UserInfo)
            validated_data2 = result2.validated_data
            logger.success(f"Pydantic 模型返回成功: {validated_data2}")
            if isinstance(validated_data2, UserInfo):
                logger.info(f"成功解析为 UserInfo 对象: name={validated_data2.name}, age={validated_data2.age}")
            else:
                logger.error("返回结果不是 UserInfo 对象！")
        except Exception as e:
            logger.error(f"测试2失败: {e}", exc_info=True)

        # --- Test 3: txt_to_json function ---
        logger.info("\n--- 测试 txt_to_json ---")
        try:
            text_input = '这里是一些无关的文字, 然后是JSON: {"name": "李四", "age": 40, "is_student": false}'
            cleaned_text = clean_markdown_fences(text_input) # 模拟清理
            result3 = await txt_to_json(cleaned_text, UserInfo)
            logger.success(f"txt_to_json 转换成功: {result3}")
            if isinstance(result3, UserInfo):
                logger.info(f"成功解析为 UserInfo 对象: name={result3.name}, age={result3.age}")
            else:
                logger.error("返回结果不是 UserInfo 对象！")
        except Exception as e:
            logger.error(f"测试3失败: {e}", exc_info=True)

        # --- Test 4: Self-correction for Pydantic model ---
        logger.info("\n--- 测试 llm_completion (JSON自我修正) ---")
        
        # 1. 定义格式错误和正确的响应
        malformed_json_str = '{"name": "Bad JSON", "age": 30, "is_student": tru' # 无效的布尔值
        correct_json_str = '{"name": "Corrected JSON", "age": 30, "is_student": true}'

        mock_response_fail = ModelResponse(
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=LLMMessage(
                        content=f"```json\n{malformed_json_str}\n```",
                        role="assistant"
                    )
                )
            ],
            model="mock_model"
        )

        mock_response_success = ModelResponse(
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=LLMMessage(
                        content=correct_json_str,
                        role="assistant"
                    )
                )
            ],
            model="mock_model"
        )

        # 2. 使用 patch 模拟 acompletion
        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            # 3. 设置模拟的副作用 (第一次返回失败，第二次返回成功)
            mock_acompletion.side_effect = [mock_response_fail, mock_response_success]

            # 4. 运行测试
            try:
                messages4 = get_llm_messages(
                    system_prompt="请根据用户信息生成JSON。",
                    user_prompt="用户信息：姓名Corrected JSON，年龄30岁，是一名学生。"
                )
                params4 = get_llm_params(messages=messages4, temperature=0.1)
                result4 = await llm_completion(params4, response_model=UserInfo)
                validated_data4 = result4.validated_data

                logger.success(f"自我修正测试成功: {validated_data4}")
                assert isinstance(validated_data4, UserInfo)
                assert validated_data4.name == "Corrected JSON"
                assert mock_acompletion.call_count == 2
                
            except Exception as e:
                logger.error(f"测试4失败: {e}", exc_info=True)
                assert False, "自我修正测试不应抛出异常"

    asyncio.run(main())
