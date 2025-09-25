import asyncio
import os
import sys
from typing import List

import pytest
from pydantic import BaseModel, Field

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.log import init_logger
from utils.react_agent import call_react_agent

# 初始化测试日志
init_logger("test_react_agent")


# 为 JSON 输出测试定义一个简单的 Pydantic 模型
class TestResponse(BaseModel):
    answer: str = Field(description="用户问题的最终答案。")
    sources: List[str] = Field(description="用于生成答案的来源 URL 列表。")


@pytest.mark.asyncio
async def test_call_react_agent_text_output():
    """
    测试 call_react_agent 函数返回纯文本输出。
    这个测试会进行真实的 LLM API 调用。
    """
    user_prompt = "法国的首都是哪里？"
    system_prompt = "你是一个地理问答助手。"

    result = await call_react_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    assert result is not None, "Agent 返回的结果不应为 None"
    assert isinstance(result, str), "结果应为字符串类型"
    assert "巴黎" in result, "结果中应包含'巴黎'"


@pytest.mark.asyncio
async def test_call_react_agent_json_output():
    """
    测试 call_react_agent 函数返回 Pydantic 模型（JSON）输出。
    这个测试会进行真实的 LLM API 调用。
    """
    user_prompt = "中国的首都是哪里？并列出1个相关的信息来源URL。"
    system_prompt = "你是一个地理知识问答助手，你需要以JSON格式回答问题。"

    result = await call_react_agent(
        system_prompt=system_prompt, user_prompt=user_prompt, response_model=TestResponse
    )

    assert result is not None, "Agent 返回的结果不应为 None"
    assert isinstance(result, TestResponse), "结果应为 TestResponse Pydantic 模型实例"
    assert "北京" in result.answer, "答案中应包含'北京'"
    assert isinstance(result.sources, list), "sources 字段应为列表"