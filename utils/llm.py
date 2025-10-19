import os
from loguru import logger
from typing import Dict, Any, Optional, List, Literal
import copy


llm_temperatures = {
    "creative": 0.75,
    "reasoning": 0.1,
    "summarization": 0.2,
    "synthesis": 0.4,
    "classification": 0.0,
}


llm_group_type = Literal['reasoning', 'fast', 'summary', 'formatter']


def template_fill(template: str, context: Optional[Dict[str, Any]]) -> str:
    content = template
    if context:
        import collections
        safe_context = collections.defaultdict(str, context)
        content = template.format_map(safe_context)
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

