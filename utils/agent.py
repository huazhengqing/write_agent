import os
import sys
from loguru import logger
from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.llm import clean_markdown_fences, llm_temperatures, get_llm_params, text_validator_default, txt_to_json
from utils.search import web_search_tools


async def call_react_agent(
    system_prompt: Optional[str],
    user_prompt: str,
    tools: List[Any] = web_search_tools,
    llm_group: Literal['reasoning', 'fast'] = 'reasoning',
    temperature: float = llm_temperatures["reasoning"],
    response_model: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    
    llm_params = get_llm_params(llm_group=llm_group, temperature=temperature)
    llm = LiteLLM(**llm_params)

    tool_names = [tool.metadata.name for tool in tools]
    logger.info(f"为 ReAct Agent 配置 {len(tool_names)} 个工具: {tool_names}")

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=system_prompt,
        max_iterations = 5, 
        verbose=True
    )

    logger.info(f"系统提示词:\n{system_prompt}")
    logger.info(f"用户提示词:\n{user_prompt}")

    logger.info("开始执行 ReAct Agent...")
    handler = agent.run(user_prompt)

    # response_text = ""
    # async for ev in handler.stream_events():
    #     if hasattr(ev, 'delta'):
    #         delta = ev.delta
    #         if delta is not None:
    #             response_text += str(delta)
    #             print(f"{delta}", end="", flush=True)
    final_response = await handler

    logger.success("ReAct Agent 执行完成。")

    # if response_text:
    #     return response_text
    raw_output = ""
    if hasattr(final_response, 'response'):
        raw_output = str(final_response.response)
    elif hasattr(final_response, 'content'):
        raw_output = str(final_response.content)
    else:
        raw_output = str(final_response)
    
    logger.debug(f"Agent 原始输出:\n{raw_output}")

    cleaned_output = clean_markdown_fences(raw_output)
    logger.info(f"Agent 清理后输出:\n{cleaned_output}")

    if response_model:
        logger.info(f"检测到 response_model, 尝试将输出解析为 {response_model.__name__} 模型...")
        json_result = await txt_to_json(cleaned_output, response_model)
        logger.success(f"成功将输出解析为 {response_model.__name__} 模型。")
        logger.debug(f"解析后的 JSON 对象: {json_result}")
        return json_result
    else:
        logger.info("未提供 response_model, 校验并返回文本结果。")
        text_validator_default(cleaned_output)
        logger.success("文本结果校验通过。")
        return cleaned_output


###############################################################################


if __name__ == '__main__':
    import asyncio
    from pydantic import Field
    from llama_index.core.tools import FunctionTool
    from utils.log import init_logger

    init_logger("agent_test")

    # 定义一个简单的 Pydantic 模型用于测试
    class CityInfo(BaseModel):
        city: str = Field(..., description="城市名称")
        population: int = Field(..., description="城市人口")

    # 创建一个模拟工具，避免网络依赖
    def get_city_info(city: str) -> str:
        """获取城市信息"""
        if city == "北京":
            return "北京的人口是2189万。"
        if city == "上海":
            # 故意返回一个不完整的JSON，用于测试错误处理
            return '{"city": "上海", "population": 24870000'
        return f"找不到城市 {city} 的信息。"

    mock_tool = FunctionTool.from_defaults(fn=get_city_info)

    async def main():
        # 1. 测试成功返回字符串
        logger.info("--- 1. 测试 call_react_agent (成功返回字符串) ---")
        question1 = "北京的人口是多少？"
        system_prompt1 = "你是一个城市信息查询助手，请使用工具查询并简洁地回答问题。"
        
        try:
            result1 = await call_react_agent(
                system_prompt=system_prompt1,
                user_prompt=question1,
                tools=[mock_tool]
            )
            logger.success(f"对 '{question1}' 的回答 (字符串):\n{result1}")
            assert "2189" in result1
        except Exception as e:
            logger.error(f"测试1失败: {e}", exc_info=True)

        # 2. 测试因JSON格式错误导致返回Pydantic模型失败
        logger.info("\n--- 2. 测试 call_react_agent (因JSON无效导致返回Pydantic模型失败) ---")
        question2 = "查询上海的人口，并以JSON格式返回城市和人口。"
        system_prompt2 = "你是一个城市信息查询助手，请使用工具查询并严格按照用户要求的JSON格式返回信息。"
        try:
            result2 = await call_react_agent(
                system_prompt=system_prompt2,
                user_prompt=question2,
                tools=[mock_tool],
                response_model=CityInfo
            )
            logger.error(f"测试2本应失败，但意外成功: {result2}")
        except Exception as e:
            logger.success(f"测试2成功捕获到预期错误: {e}")

        # 3. 测试因内容过短导致文本验证失败
        logger.info("\n--- 3. 测试 call_react_agent (因内容过短导致文本验证失败) ---")
        question3 = "嗨" # 一个过于简单的问题，可能导致Agent只回复 "你好"
        system_prompt3 = "简单回复即可"
        try:
            await call_react_agent(
                system_prompt=system_prompt3, user_prompt=question3, tools=[]
            )
            logger.error("测试3本应失败，但意外成功")
        except ValueError as e:
            logger.success(f"测试3成功捕获到预期错误: {e}")
            assert "内容为空或过短" in str(e)

    asyncio.run(main())
