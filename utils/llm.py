import os
from loguru import logger
from typing import Dict, Any, Optional, List, Literal
import copy

import collections

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
        "api_base": os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
        "api_key": os.getenv("LITELLM_MASTER_KEY", "sk-1234")
    }
    llm_params.update(kwargs)
    llm_params["temperature"] = temperature
    if tools is not None:
        llm_params["tools"] = tools
    if messages is not None:
        llm_params["messages"] = copy.deepcopy(messages)
    return llm_params



def template_fill(template: str, context: Optional[Dict[str, Any]]) -> str:
    content = template
    if context:
        safe_context = collections.defaultdict(str, context)
        content = template.format_map(safe_context)
    return content


def get_llm_messages(
    system_prompt: str = None, 
    user_prompt: str = None, 
    context_dict_system: Dict[str, Any] = None, 
    context_dict_user: Dict[str, Any] = None
) -> list[dict]:
    messages = []
    if system_prompt:
        system_content = template_fill(system_prompt, context_dict_system)
        if system_content and system_content.strip():
            messages.append({"role": "system", "content": system_content})
    if user_prompt:
        user_content = template_fill(user_prompt, context_dict_user)
        if user_content and user_content.strip():
            messages.append({"role": "user", "content": user_content})
    return messages



def log_llm_params(llm_params: Dict[str, Any]):
    params_to_log = llm_params.copy()
    params_to_log.pop("messages", None)
    import json
    logger.info(f"LLM 参数:\n{json.dumps(params_to_log, indent=2, ensure_ascii=False, default=str)}")

    messages = llm_params.get("messages", [])
    system_prompt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user_prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if system_prompt:
        logger.info(f"系统提示词:\n{system_prompt}")
    if user_prompt:
        logger.info(f"用户提示词:\n{user_prompt}")
    


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
