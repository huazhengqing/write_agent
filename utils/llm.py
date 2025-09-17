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
from litellm.caching.caching import Cache
from loguru import logger
from typing import List, Dict, Any, Literal, Optional, Type, Callable, Union
from pydantic import BaseModel, ValidationError
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms_api.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.file import cache_dir
from utils.search import web_search_tools


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
    },
}

llms_api_2 = {
    "reasoning": {
        "model": "openai/deepseek-ai/DeepSeek-R1-0528",
        "api_base": "https://api-inference.modelscope.cn/v1/",
        "api_key": os.getenv("modelscope_API_KEY"), 
        "context_window": 163840,
        "fallbacks": [
            {
                "model": "openrouter/deepseek/deepseek-r1-0528:free",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
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
        "model": "openai/deepseek-ai/DeepSeek-V3",
        "api_base": "https://api-inference.modelscope.cn/v1/",
        "api_key": os.getenv("modelscope_API_KEY"), 
        "context_window": 163840,
        "fallbacks": [
            {
                "model": "openrouter/deepseek/deepseek-chat-v3-0324:free",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
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
    },
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
}


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
litellm.enable_json_schema_validation=True
litellm.drop_params = True
litellm.telemetry = False


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
    llm: Literal['reasoning', 'fast'] = 'reasoning',
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = llm_temperatures["reasoning"],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    llm_params = llms_api[llm].copy()
    # llm_params = llms_api_2[llm].copy()
    llm_params.update(**llm_params_general)
    llm_params.update(kwargs)
    llm_params["temperature"] = temperature
    if tools is not None:
        llm_params["tools"] = tools
    if messages is not None:
        llm_params["messages"] = copy.deepcopy(messages)
    return llm_params


def _format_json_content(content: str) -> str:
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content


def _format_message_content(content: str) -> str:
    if content.strip().startswith("{") or content.strip().startswith("["):
        return _format_json_content(content)
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


def _default_text_validator(content: str):
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
    params_to_log = llm_params.copy()
    params_to_log.pop("messages", None)
    logger.info(f"LLM 参数:\n{json.dumps(params_to_log, indent=2, ensure_ascii=False, default=str)}")
    llm_params_for_api = llm_params.copy()
    if response_model:
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
                    raw_output_for_correction = clean_markdown_fences(tool_call.function.arguments)
                    if hasattr(tool_call.function, "parsed_arguments") and tool_call.function.parsed_arguments:
                        parsed_args = tool_call.function.parsed_arguments
                    else:
                        parsed_args = json.loads(raw_output_for_correction)
                    validated_data = response_model(**parsed_args)
                elif message.content:
                    raw_output_for_correction = clean_markdown_fences(message.content)
                    validated_data = response_model.model_validate_json(raw_output_for_correction)
                if validated_data:
                    message.validated_data = validated_data
                else:
                    raise ValueError("LLM响应既无tool_calls也无有效content可供解析。")
            else:
                message.content = clean_markdown_fences(message.content)
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
    extraction_messages = get_llm_messages(
        system_prompt=extraction_system_prompt,
        user_prompt=extraction_user_prompt.format(cleaned_output=cleaned_output)
    )
    extraction_llm_params = get_llm_params(
        llm='reasoning',
        messages=extraction_messages,
        temperature=llm_temperatures["classification"]
    )
    extraction_response = await llm_completion(llm_params=extraction_llm_params, response_model=response_model)
    validated_data = extraction_response.validated_data
    return validated_data


async def call_react_agent(
    system_prompt: Optional[str],
    user_prompt: str,
    tools: List[Any] = web_search_tools,
    llm_type: Literal['reasoning', 'fast'] = 'reasoning',
    temperature: float = llm_temperatures["reasoning"],
    response_model: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    llm_params = get_llm_params(llm=llm_type, temperature=temperature)
    llm = LiteLLM(**llm_params)
    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=system_prompt,
        max_iterations = 5, 
        verbose=True
    )

    logger.info(f"系统提示词:\n{system_prompt}")
    logger.info(f"用户提示词:\n{user_prompt}")

    handler = agent.run(user_prompt)
    # response_text = ""
    # async for ev in handler.stream_events():
    #     if hasattr(ev, 'delta'):
    #         delta = ev.delta
    #         if delta is not None:
    #             response_text += str(delta)
    #             print(f"{delta}", end="", flush=True)
    final_response = await handler
    # if response_text:
    #     return response_text
    raw_output = ""
    if hasattr(final_response, 'response'):
        raw_output = str(final_response.response)
    elif hasattr(final_response, 'content'):
        raw_output = str(final_response.content)
    else:
        raw_output = str(final_response)
    cleaned_output = clean_markdown_fences(raw_output)
    
    logger.info(f"llm 返回\n{cleaned_output}")

    if response_model:
        return await txt_to_json(cleaned_output, response_model)
    else:
        _default_text_validator(cleaned_output)
        return cleaned_output


if __name__ == '__main__':
    from pydantic import BaseModel, Field
    from typing import List
    from utils.log import init_logger

    init_logger("llm_test")

    # 定义一个用于测试的 Pydantic 模型
    class CharacterInfo(BaseModel):
        name: str = Field(description="角色姓名")
        abilities: List[str] = Field(description="角色的能力列表")
        goal: str = Field(description="角色的主要目标")

    async def main():
        # 1. 测试 get_llm_messages
        logger.info("--- 测试 get_llm_messages ---")
        messages = get_llm_messages(
            system_prompt="你是一个角色设定助手。",
            user_prompt="请介绍一下角色：{name}",
            context_dict_user={"name": "龙傲天"}
        )
        logger.info(f"get_llm_messages 生成的消息:\n{messages}")

        # 2. 测试 llm_completion (纯文本)
        logger.info("--- 测试 llm_completion (纯文本) ---")
        text_params = get_llm_params(
            llm='fast',
            messages=get_llm_messages(user_prompt="写一句关于宇宙的诗。"),
            temperature=0.7
        )
        text_response = await llm_completion(text_params)
        logger.info(f"llm_completion (纯文本) 结果:\n{text_response.content}")

        # 3. 测试 llm_completion (JSON Schema)
        logger.info("--- 测试 llm_completion (JSON) ---")
        json_prompt = "根据以下描述生成一个角色信息JSON：'龙傲天是一位强大的法师，他能操控火焰和冰霜，他的目标是找到失落的古代神器。'"
        json_params = get_llm_params(
            llm='reasoning',
            messages=get_llm_messages(user_prompt=json_prompt),
            temperature=0.1
        )
        json_response = await llm_completion(json_params, response_model=CharacterInfo)
        logger.info(f"llm_completion (JSON) 验证后的Pydantic对象:\n{json_response.validated_data.model_dump_json(indent=2, ensure_ascii=False)}")

        # 4. 测试 call_react_agent
        logger.info("--- 测试 call_react_agent ---")
        agent_prompt = "2024年AI领域有什么好玩的新东西？请用中文回答。"
        agent_system_prompt = "你是一个AI科技观察家，利用工具来回答用户问题。"
        agent_response = await call_react_agent(
            system_prompt=agent_system_prompt,
            user_prompt=agent_prompt,
            tools=web_search_tools, # from utils.search
            llm_type='reasoning'
        )
        logger.info(f"call_react_agent 结果:\n{agent_response}")

    asyncio.run(main())
