import os
import sys
import pytest
from loguru import logger
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.agent import call_react_agent


# 定义一个简单的 Pydantic 模型用于测试
class CityInfo(BaseModel):
    city: str = Field(..., description="城市名称")
    population: int = Field(..., description="城市人口")


@pytest.fixture
def mock_tool():
    """提供一个模拟的城市信息查询工具。"""
    def get_city_info(city: str) -> str:
        """获取城市信息"""
        if city == "北京":
            return "北京的人口是2189万。"
        if city == "上海":
            # 故意返回一个不完整的JSON，用于测试错误处理
            return '{"city": "上海", "population": 24870000'
        return f"找不到城市 {city} 的信息。"
    return FunctionTool.from_defaults(fn=get_city_info)


@pytest.fixture
def mock_failing_tool():
    """提供一个总是会失败的模拟工具。"""
    def failing_tool(query: str) -> str:
        """这是一个总是会失败并抛出异常的工具。"""
        raise ValueError(f"工具故意失败，输入为: '{query}'")
    return FunctionTool.from_defaults(fn=failing_tool, name="failing_tool")


@pytest.mark.asyncio
async def test_call_react_agent_success_string(mock_tool):
    """测试 Agent 成功调用工具并返回字符串结果。"""
    logger.info("--- 测试：Agent 成功返回字符串 ---")
    question = "北京的人口是多少？"
    system_prompt = "你是一个城市信息查询助手，请使用工具查询并简洁地回答问题。"
    
    result = await call_react_agent(
        system_prompt=system_prompt,
        user_prompt=question,
        tools=[mock_tool]
    )
    logger.success(f"对 '{question}' 的回答 (字符串):\n{result}")
    assert "2189" in result


@pytest.mark.asyncio
async def test_call_react_agent_pydantic_failure_invalid_json(mock_tool):
    """测试因工具返回无效JSON，导致Pydantic模型解析失败。"""
    logger.info("--- 测试：Agent 因JSON无效导致Pydantic模型解析失败 ---")
    question = "查询上海的人口，并以JSON格式返回城市和人口。"
    system_prompt = "你是一个城市信息查询助手，请使用工具查询并严格按照用户要求的JSON格式返回信息。"
    
    with pytest.raises(Exception) as exc_info:
        await call_react_agent(
            system_prompt=system_prompt,
            user_prompt=question,
            tools=[mock_tool],
            response_model=CityInfo
        )
    logger.success("成功捕获到预期的Pydantic解析错误: {}", exc_info.value)


@pytest.mark.asyncio
async def test_call_react_agent_validation_failure_short_content():
    """测试因Agent回复内容过短，导致文本验证失败。"""
    logger.info("--- 测试：Agent 因内容过短导致文本验证失败 ---")
    question = "嗨"  # 一个过于简单的问题，可能导致Agent只回复 "你好"
    system_prompt = "简单回复即可"
    
    with pytest.raises(ValueError, match="内容为空或过短"):
        await call_react_agent(
            system_prompt=system_prompt, user_prompt=question, tools=[]
        )
    logger.success("成功捕获到预期的内容过短错误。")


@pytest.mark.asyncio
async def test_call_react_agent_handles_tool_failure(mock_failing_tool):
    """测试当工具调用失败时，Agent能够捕获异常并报告，而不是崩溃。"""
    logger.info("--- 测试：Agent 处理工具调用失败 ---")
    question = "使用 failing_tool 查询一些信息。"
    system_prompt = "你必须使用 failing_tool 来回答问题。"
    
    result = await call_react_agent(
        system_prompt=system_prompt,
        user_prompt=question,
        tools=[mock_failing_tool]
    )
    logger.success(f"Agent 在工具失败后返回: {result}")
    # Agent 应该能观察到工具的错误并报告它
    assert "失败" in result or "错误" in result or "failed" in result
