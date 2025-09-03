import copy
import litellm
from litellm.caching.caching import Cache
from typing import List, Dict, Any, Optional


"""


分析、审查当前文件的代码，找出bug并改正， 指出可以优化的地方。


根据以上分析，改进建议， 请直接修改 文件，并提供diff。



"""


###############################################################################


litellm.cache = Cache(type='disk')
# litellm.input_callback = ["lunary"]
# litellm.success_callback = ["lunary"]
# litellm.failure_callback = ["lunary"]


litellm.enable_json_schema_validation=True


###############################################################################


DEFAULT_LLM_PARAMS = {
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
        'openrouter/deepseek/deepseek-r1-0528-qwen3-8b',
        # 'openrouter/qwen/qwen3-32b',
        # 'openrouter/qwen/qwen3-30b-a3b',
        # 'openrouter/deepseek/deepseek-r1-distill-llama-70b',
    ]
}


def get_llm_params(
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    llm_params = DEFAULT_LLM_PARAMS.copy()
    llm_params.update(kwargs)
    if temperature is not None:
        llm_params['temperature'] = temperature
    if tools is not None:
        llm_params['tools'] = tools
    llm_params['messages'] = copy.deepcopy(messages)
    return llm_params

