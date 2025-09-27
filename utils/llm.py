import json
import os
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Literal, Type, Callable
import litellm
import copy

from dotenv import load_dotenv
load_dotenv()


llm_temperatures = {
    "creative": 0.75,
    "reasoning": 0.1,
    "summarization": 0.2,
    "synthesis": 0.4,
    "classification": 0.0,
}



def get_llm_params(
    llm_group: Literal['reasoning', 'fast', 'summary'] = 'reasoning',
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = llm_temperatures["reasoning"],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    llm_params = {
        "model": f"openai/{llm_group}",
        "api_base": os.getenv("LITELLM_PROXY_URL"),
        "api_key": os.getenv("LITELLM_MASTER_KEY")
    }
    llm_params.update(kwargs)
    llm_params["temperature"] = temperature
    if tools is not None:
        llm_params["tools"] = tools
    if messages is not None:
        llm_params["messages"] = copy.deepcopy(messages)
    return llm_params



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
        import collections
        safe_context_system = collections.defaultdict(str, context_dict_system)
        system_content = system_prompt.format_map(safe_context_system)
    
    if system_content and system_content.strip():
        messages.append({"role": "system", "content": system_content})

    user_content = user_prompt
    if context_dict_user:
        import collections
        safe_context_user = collections.defaultdict(str, context_dict_user)
        user_content = user_prompt.format_map(safe_context_user)

    if user_content and user_content.strip():
        messages.append({"role": "user", "content": user_content})

    return messages



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
    import re
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


def _clear_proxy_cache(cache_key: Optional[str]):
    if not cache_key:
        logger.debug("没有可用的 cache_key, 跳过缓存删除。")
        return

    proxy_base_url = os.getenv("LITELLM_PROXY_URL")
    if not proxy_base_url:
        logger.warning("无法获取 LITELLM_PROXY_URL, 跳过缓存删除。")
        return

    delete_url = f"{proxy_base_url}/cache/delete"
    headers = {"Authorization": f"Bearer {os.getenv('LITELLM_MASTER_KEY')}"}
    data = {"keys": [cache_key]}

    try:
        import httpx
        with httpx.Client() as client:
            response = client.post(delete_url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info(f"成功从 litellm proxy 删除缓存: key='{cache_key}'")
    except Exception as http_e:
        logger.warning(f"从 litellm proxy 删除缓存失败: key='{cache_key}', 错误: {http_e}")



def _handle_llm_failure(
    e: Exception,
    llm_params: Dict[str, Any],
    llm_params_for_api: Dict[str, Any],
    cache_key: Optional[str],
    response_model: Optional[Type[BaseModel]],
    raw_output_for_correction: Optional[str],
):
    _clear_proxy_cache(cache_key)
    from pydantic import ValidationError
    if response_model and isinstance(e, (ValidationError, json.JSONDecodeError)) and raw_output_for_correction:
        logger.info("检测到JSON错误, 下次尝试将进行自我修正...")
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
        # 如果不是可修正的错误, 或者没有 response_model, 则重置为原始消息
        llm_params_for_api["messages"] = llm_params["messages"]

    logger.info("正在准备重试...")



async def llm_completion(
    llm_params: Dict[str, Any], 
    response_model: Optional[Type[BaseModel]] = None, 
    validator: Optional[Callable[[Any], None]] = None,
    max_retries: int = 6
) -> Dict[str, Any]:
    logger.info("开始执行 llm_completion...")

    params_to_log = llm_params.copy()
    params_to_log.pop("messages", None)
    logger.info(f"LLM 参数:\n{json.dumps(params_to_log, indent=2, ensure_ascii=False, default=str)}")

    llm_params_for_api = llm_params.copy()

    if response_model:
        llm_params_for_api["response_format"] = {
            "type": "json_object",
            "schema": response_model.model_json_schema()
        }
    
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
        cache_key_from_response = None
        try:
            response = await litellm.acompletion(**llm_params_for_api)
            logger.debug(f"LLM 原始响应: {response}")
            if not response.choices or not response.choices[0].message:
                raise ValueError("LLM响应中缺少 choices 或 message。")

            message = response.choices[0].message
            cache_key_from_response = response.get("x-litellm-cache-key")

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
            logger.warning("LLM调用或验证失败 (尝试 {}/{}): {}", attempt + 1, max_retries, e)
            if attempt >= max_retries - 1:
                logger.error("LLM 响应在多次重试后仍然无效, 任务失败。")
                raise e

            _handle_llm_failure(
                e=e,
                llm_params=llm_params,
                llm_params_for_api=llm_params_for_api,
                cache_key=cache_key_from_response,
                response_model=response_model,
                raw_output_for_correction=raw_output_for_correction,
            )

    raise RuntimeError("llm_completion 在所有重试后失败, 这是一个不应出现的情况。")



###############################################################################



extraction_system_prompt = """
你是一个数据提取专家。你的任务是根据用户提供的文本, 严格按照给定的 Pydantic JSON Schema 提取信息并生成一个 JSON 对象。
你的输出必须是、且只能是一个完整的、有效的 JSON 对象。
不要添加任何解释、注释或代码块。
"""


extraction_user_prompt = """
请从以下文本中提取信息并生成 JSON 对象:
{{cleaned_output}}
"""



async def txt_to_json(
    cleaned_output: str, 
    response_model: Type[BaseModel],
    max_retries: int = 6
):
    logger.info(f"开始将文本转换为 JSON, 目标模型: {response_model.__name__}")
    logger.debug(f"待转换的输入文本:\n{cleaned_output}")

    extraction_messages = get_llm_messages(
        system_prompt=extraction_system_prompt,
        user_prompt=extraction_user_prompt.format(cleaned_output=cleaned_output)
    )
    from utils.llm import get_llm_params, llm_temperatures
    extraction_llm_params = get_llm_params(
        llm_group='reasoning',
        messages=extraction_messages,
        temperature=llm_temperatures["classification"]
    )
    extraction_response = await llm_completion(
        llm_params=extraction_llm_params, 
        response_model=response_model,
        max_retries=max_retries
    )
    validated_data = extraction_response.validated_data

    logger.success("文本成功转换为 JSON 对象。")
    logger.debug(f"转换后的 JSON 对象: {validated_data}")
    return validated_data
