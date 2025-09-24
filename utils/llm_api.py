import copy
import os
from typing import Any, Dict, List, Literal, Optional


import logging
litellm_logger = logging.getLogger("litellm")
litellm_logger.setLevel(logging.ERROR)
for handler in litellm_logger.handlers:
    litellm_logger.removeHandler(handler)


from dotenv import load_dotenv
load_dotenv()


llm_temperatures = {
    "creative": 0.75,
    "reasoning": 0.1,
    "summarization": 0.2,
    "synthesis": 0.4,
    "classification": 0.0,
}


###############################################################################


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
            }, 
            # {
            #     "model": "openrouter/deepseek/deepseek-r1-0528",
            #     "api_key": os.getenv("OPENROUTER_API_KEY"),
            #     "context_window": 163840,
            # },
            # {
            #     "model": "openrouter/deepseek/deepseek-r1-0528-qwen3-8b",
            #     "api_key": os.getenv("OPENROUTER_API_KEY"),
            #     "context_window": 131072,
            # },
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
            }, 
            # {
            #     "model": "openrouter/deepseek/deepseek-chat-v3-0324",
            #     "api_key": os.getenv("OPENROUTER_API_KEY"),
            #     "context_window": 163840,
            # },
            # {
            #     "model": "openrouter/deepseek/deepseek-r1-0528-qwen3-8b",
            #     "api_key": os.getenv("OPENROUTER_API_KEY"),
            #     "context_window": 131072,
            # },
        ]
    },
    "summary": {
        "model": "openai/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "api_base": "https://api.siliconflow.cn/v1/",
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "context_window": 131072,
        "fallbacks": [
            {
                "model": "openrouter/google/gemini-2.0-flash-exp:free",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "context_window": 1048576,
            },
            {
                "model": "openai/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                "api_base": "https://api-inference.modelscope.cn/v1/",
                "api_key": os.getenv("modelscope_API_KEY"), 
                "context_window": 131072,
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
            # {
            #     "model": "openrouter/deepseek/deepseek-r1-0528-qwen3-8b",
            #     "api_key": os.getenv("OPENROUTER_API_KEY"),
            #     "context_window": 32000,
            # },
        ]
    }
}

llm_api_params = {
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
        "RateLimitError",
        "Timeout",
        "APIConnectionError",
        "ServiceUnavailableError",
        "APIError",
    ]
}

def get_llm_params(
    llm_group: Literal['reasoning', 'fast', 'summary'] = 'reasoning',
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = llm_temperatures["reasoning"],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    llm_params = llms_api[llm_group].copy()

    llm_params.update(**llm_api_params)
    llm_params.update(kwargs)

    llm_params["temperature"] = temperature

    if tools is not None:
        llm_params["tools"] = tools

    if messages is not None:
        llm_params["messages"] = copy.deepcopy(messages)

    # `max_retries` 用于 llama-index LiteLLM 包装器的重试机制。
    # 我们将其设置为 1 (即尝试1次, 不重试), 以便让 litellm 自身更复杂的回退和重试逻辑优先执行。
    # llm_params 中的 `num_retries` 参数会被传递给 litellm, 用于控制其内部重试。
    llm_params["max_retries"] = 1

    return llm_params


###############################################################################


embeddings_api = {
    "bge-m3": {
        "model": "openai/BAAI/bge-m3",
        "api_base": "https://api.siliconflow.cn/v1/",
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        # "dims": 1024,
    },
    "gemini": {
        "model": "gemini/gemini-embedding-001",
        "api_key": os.getenv("GEMINI_API_KEY"),
        # "dims": 3072,
    }
}

embeddings_api_params = {
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
    embedding_params.update(**embeddings_api_params)
    embedding_params.update(kwargs)
    return embedding_params
