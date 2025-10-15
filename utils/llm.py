import os
from loguru import logger
from typing import Dict, Any, Optional, List, Literal
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


llm_group_type = Literal['reasoning', 'fast', 'summary', 'formatter']


def get_llm_params(
    llm_group: llm_group_type = 'reasoning',
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
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



def format_json_content(content: str) -> str:
    import json
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content

def format_message_content(content: str) -> str:
    if content.strip().startswith("{") or content.strip().startswith("["):
        return format_json_content(content)
    return content
