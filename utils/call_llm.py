import json
import os
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Literal, Type, Callable, Union
from utils.llm import get_llm_params, llm_group_type, llm_completion, clean_markdown_fences
from utils.search import web_search_tools




def delete_cache(cache_key: Optional[str]):
    if not cache_key:
        return
    proxy_base_url = os.getenv("LITELLM_PROXY_URL")
    if not proxy_base_url:
        return
    delete_url = f"{proxy_base_url}/cache/delete"
    headers = {"Authorization": f"Bearer {os.getenv('LITELLM_MASTER_KEY')}"}
    data = {"keys": [cache_key]}
    import httpx
    with httpx.Client() as client:
        response = client.post(delete_url, json=data, headers=headers, timeout=10)
        response.raise_for_status()



###############################################################################


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



def _handle_llm_failure(
    e: Exception,
    llm_params: Dict[str, Any],
    llm_params_for_api: Dict[str, Any],
    output_cls: Optional[Type[BaseModel]],
    raw_output_for_correction: Optional[str],
):
    from pydantic import ValidationError
    if output_cls and isinstance(e, (ValidationError, json.JSONDecodeError)) and raw_output_for_correction:
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
        # 如果不是可修正的错误, 或者没有 output_cls, 则重置为原始消息
        llm_params_for_api["messages"] = llm_params["messages"]



async def completion(
    llm_params: Dict[str, Any], 
    output_cls: Optional[Type[BaseModel]] = None
) -> Dict[str, Any]:
    params_to_log = llm_params.copy()
    params_to_log.pop("messages", None)
    logger.info(f"LLM 参数:\n{json.dumps(params_to_log, indent=2, ensure_ascii=False, default=str)}")

    llm_params_for_api = llm_params.copy()

    if output_cls:
        llm_params_for_api["response_format"] = {
            "type": "json_object",
            "schema": output_cls.model_json_schema()
        }
    
    for attempt in range(6):
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
        
        logger.info(f"开始 LLM 调用 (尝试 {attempt + 1}/6)...")

        cache_key = None
        raw_output_for_correction = None
        try:
            import litellm
            response = await litellm.acompletion(**llm_params_for_api)
            # logger.debug(f"LLM 原始响应: {response}")
            if not response.choices or not response.choices[0].message:
                raise ValueError("LLM响应中缺少 choices 或 message。")

            message = response.choices[0].message
            cache_key = response.get("x-litellm-cache-key")

            if output_cls:
                validated_data = None
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    raw_output_for_correction = tool_call.function.arguments
                    cleaned_args = clean_markdown_fences(raw_output_for_correction)
                    if hasattr(tool_call.function, "parsed_arguments") and tool_call.function.parsed_arguments:
                        parsed_args = tool_call.function.parsed_arguments
                    else:
                        parsed_args = json.loads(cleaned_args)
                    validated_data = output_cls(**parsed_args)
                elif message.content:
                    raw_output_for_correction = clean_markdown_fences(message.content)
                    validated_data = output_cls.model_validate_json(raw_output_for_correction)

                if validated_data:
                    message.validated_data = validated_data
                else:
                    raise ValueError("LLM响应既无tool_calls也无有效content可供解析。")
            else:
                message.content = clean_markdown_fences(message.content)
            
            # logger.success("LLM 响应成功通过验证。")

            reasoning = message.get("reasoning_content") or message.get("reasoning", "")
            if reasoning:
                logger.info(f"推理过程:\n{reasoning}")

            final_content_to_log = message.validated_data if output_cls else message.content
            logger.info(f"LLM 成功返回内容 (尝试 {attempt + 1}):\n{final_content_to_log}")

            return message
        except Exception as e:
            logger.warning("LLM调用或验证失败 (尝试 {}/{}): {}", attempt + 1, 6, e)
            if attempt >= 6 - 1:
                logger.error("LLM 响应在多次重试后仍然无效, 任务失败。")
                raise e

            delete_cache(cache_key)
            _handle_llm_failure(
                e=e,
                llm_params=llm_params,
                llm_params_for_api=llm_params_for_api,
                output_cls=output_cls,
                raw_output_for_correction=raw_output_for_correction,
            )

    raise RuntimeError("llm_completion 在所有重试后失败, 这是一个不应出现的情况。")



###############################################################################



txt_to_json_prompt = """
你是一个数据提取专家。你的任务是根据用户提供的文本, 严格按照给定的 Pydantic JSON Schema 提取信息并生成一个 JSON 对象。
你的输出必须是、且只能是一个完整的、有效的 JSON 对象。
不要添加任何解释、注释或代码块。
请从以下文本中提取信息并生成 JSON 对象:
{text}
"""

async def txt_to_json(text: str, output_cls: Type[BaseModel]):
    messages = [{"role": "user", "content": txt_to_json_prompt.format(text=text)}]
    llm_params = get_llm_params(
        llm_group='reasoning',
        messages=messages,
        temperature=0.0
    )
    response = await llm_completion(
        llm_params=llm_params, 
        output_cls=output_cls
    )
    return response.validated_data



###############################################################################



async def react(
    llm_group: llm_group_type = 'reasoning',
    temperature = 0.1, 
    system_header: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: str = "",
    tools: List[Any] = web_search_tools,
    output_cls: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    logger.info(f"system_header=\n{system_header}")
    logger.info(f"system_prompt=\n{system_prompt}")
    logger.info(f"user_prompt=\n{user_prompt}")

    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.llms.litellm import LiteLLM

    llm_params = get_llm_params(llm_group=llm_group, temperature=temperature)
    agent = ReActAgent(
        tools=tools,
        llm=LiteLLM(**llm_params),
        output_cls = output_cls, 
        system_prompt=system_prompt,
    )
    if system_header:
        agent.update_prompts({"react_header": system_header})
    handler = agent.run(user_prompt)
    # async for ev in handler.stream_events():
    #     if hasattr(ev, 'delta'):
    #         delta = ev.delta
    #         if delta is not None:
    #             print(f"{delta}", end="", flush=True)
    agentOutput = await handler
    if output_cls:
        logger.info(f"完成 output_cls=\n{agentOutput.structured_response}")
        return agentOutput.structured_response
    else:
        output = clean_markdown_fences(agentOutput.response)
        logger.info(f"完成 output=\n{output}")
        return output

