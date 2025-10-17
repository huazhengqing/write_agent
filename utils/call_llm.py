import json
import os
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Literal, Type, Callable, Union
from utils.llm import get_llm_params, llm_group_type, clean_markdown_fences, log_llm_params



def delete_cache(cache_key: Optional[str]):
    if not cache_key:
        return
    proxy_base_url = os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000")
    if not proxy_base_url:
        return
    delete_url = f"{proxy_base_url}/cache/delete"
    headers = {"Authorization": f"Bearer {os.getenv('LITELLM_MASTER_KEY', "sk-1234")}"}
    data = {"keys": [cache_key]}
    import httpx
    with httpx.Client() as client:
        response = client.post(delete_url, json=data, headers=headers, timeout=10)
        response.raise_for_status()



###############################################################################



retry_output_cls_prompt = """
# 任务: 修正JSON输出
你上次的输出因为格式错误导致解析失败。请根据原始任务和错误信息, 重新生成。

# 你上次格式错误的输出
{error_output}

# 错误信息
{error}

# 这是你需要完成的原始任务
{original_task}

# 新的要求
1.  严格遵循原始任务的所有指令, 尤其是关于JSON Schema的指令。
2.  严格根据 Pydantic 模型的要求, 修正并仅返回完整的、有效的 JSON 对象。
3.  禁止在 JSON 前后添加任何额外解释或代码块。
"""


async def retry_output_cls(
    llm_params: Dict[str, Any],
    error_output: str,
    error: str,
    output_cls: Type[BaseModel]
) -> Dict[str, Any]:
    """
    独立的接口, 负责完成一次完整的JSON自我修正交互。
    它会生成修正提示, 调用LLM, 并解析返回的结果。
    """
    original_messages = llm_params["messages"]
    original_user_content = ""
    for msg in reversed(original_messages):
        if msg["role"] == "user":
            original_user_content = msg.get("content", "")
            break
    
    prompt = retry_output_cls_prompt.format(
        error=error, 
        error_output=error_output,
        original_task=original_user_content
    )
    
    system_message = [m for m in original_messages if m["role"] == "system"]
    new_messages = system_message + [{"role": "user", "content": prompt}]

    correction_llm_params = llm_params.copy()
    correction_llm_params["messages"] = new_messages
    correction_llm_params["response_format"] = {
        "type": "json_object",
        "schema": output_cls.model_json_schema()
    }

    import litellm
    response = await litellm.acompletion(**correction_llm_params)
    if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
        raise ValueError("LLM自我修正响应无效。")

    message = response.choices[0].message
    cleaned_content = clean_markdown_fences(message.content)
    validated_data = output_cls.model_validate_json(cleaned_content)
    message.validated_data = validated_data
    return message



###############################################################################



def _parse_and_validate_response(message: Any, output_cls: Type[BaseModel]) -> tuple[BaseModel, str]:
    """
    从 LLM 响应中解析和验证数据。
    它处理 tool_calls 或 content, 并根据 output_cls 进行验证。
    如果解析或验证失败, 它会引发一个带有原始输出的异常。
    返回:
        - validated_data: 验证后的 Pydantic 模型实例。
        - raw_output: 从 LLM 响应中提取的原始字符串输出。
    """
    validated_data = None
    raw_output = ""
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        raw_output = tool_call.function.arguments
        cleaned_args = clean_markdown_fences(raw_output)
        parsed_args = getattr(tool_call.function, "parsed_arguments", None) or json.loads(cleaned_args)
        validated_data = output_cls(**parsed_args)
    elif message.content:
        # 优先检查原始输出是否为 'null'
        raw_output = message.content
        cleaned_output = clean_markdown_fences(raw_output).strip()
        if cleaned_output.lower() == 'null':
            return None, raw_output
        # 如果不是 'null', 则正常进行 JSON 验证
        raw_output = message.content
        validated_data = output_cls.model_validate_json(clean_markdown_fences(raw_output))
    return validated_data, raw_output



async def completion_once(llm_params: Dict[str, Any], output_cls: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
    llm_params_for_api = llm_params.copy()
    if output_cls:
        llm_params_for_api["response_format"] = {
            "type": "json_object",
            "schema": output_cls.model_json_schema()
        }
    log_llm_params(llm_params_for_api)

    import litellm
    response = await litellm.acompletion(**llm_params_for_api)

    if not response.choices or not response.choices[0].message:
        raise ValueError("LLM响应中缺少 choices 或 message。")
    
    message = response.choices[0].message

    cache_key = response.get("x-litellm-cache-key", "")
    if cache_key:
        message.cache_key = cache_key
        logger.info(f"cache_key={message.cache_key}")

    if output_cls:
        from pydantic import ValidationError
        try:
            validated_data, raw_output = _parse_and_validate_response(message, output_cls)
            message.validated_data = validated_data
            if message.validated_data:
                logger.success(f"LLM 成功返回内容 output_cls=\n{message.validated_data.model_dump_json(indent=2, ensure_ascii=False)}")
            else:
                logger.info("完成, LLM 返回 'null', 表示任务完成。")

        except (json.JSONDecodeError, ValidationError) as e:
            delete_cache(message.cache_key)
            e.raw_output = raw_output
            raise e
    else:
        message.content = clean_markdown_fences(message.content)
        logger.success(f"LLM 成功返回内容 content=\n{message.content}")

    return message



async def completion(
    llm_params: Dict[str, Any], 
    output_cls: Optional[Type[BaseModel]] = None
) -> Dict[str, Any]:
    """
    调用 LLM, 支持 JSON 输出和自动修正。
    它首先尝试一次获取结果。如果因为格式问题失败, 它会进入一个重试循环, 
    尝试让 LLM 自我修正其输出。
    """
    from pydantic import ValidationError

    # 1. 首次尝试
    try:
        return await completion_once(llm_params, output_cls)
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning(f"首次尝试解析LLM输出失败: {e}")
        if not (output_cls and hasattr(e, 'raw_output') and e.raw_output):
            logger.error("无法进行修正重试: 缺少 output_cls 或原始输出。")
            raise e
        
        # 准备进入修正重试循环
        error_output = e.raw_output
        error_str = str(e)
    except Exception as e:
        logger.error(f"LLM 调用时发生意外错误: {e}")
        raise e

    # 如果首次尝试失败, 进入修正重试循环
    max_retries = 6
    for attempt in range(max_retries):
        try:
            logger.info(f"正在进行第 {attempt + 1} 次修正重试...")
            return await retry_output_cls(llm_params=llm_params, error_output=error_output, error=error_str, output_cls=output_cls)
        except (ValidationError, json.JSONDecodeError) as retry_e:
            logger.warning(f"第 {attempt + 1} 次修正重试失败: {retry_e}")
            error_output = getattr(retry_e, 'raw_output', error_output) 
            error_str = str(retry_e)
        except Exception as retry_e:
            logger.error(f"修正重试期间发生意外错误: {retry_e}")
            raise retry_e

    logger.error("LLM 响应在多次重试后仍然无效, 任务失败。")
    raise RuntimeError("LLM completion failed after all retries.")



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
    response = await completion(
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
    tools: List[Any] = [],
    output_cls: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    if not tools:
        raise ValueError("tools 参数必须提供, 且不能为空列表。")

    if not system_header and not system_prompt:
        raise ValueError("system_header 和 system_prompt 必须至少提供一个。")
    
    logger.info(f"system_header=\n{system_header}")
    logger.info(f"system_prompt=\n{system_prompt}")
    logger.info(f"user_prompt=\n{user_prompt}")

    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.llms.litellm import LiteLLM

    llm_params = get_llm_params(llm_group=llm_group, temperature=temperature)
    agent = ReActAgent(
        tools=tools,
        llm=LiteLLM(**llm_params),
        system_prompt=system_prompt,
        output_cls = output_cls, 
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

    # 优先检查原始输出是否为 'null'
    raw_output = clean_markdown_fences(agentOutput.response)
    if raw_output.strip().lower() == 'null':
        logger.info("完成, Agent 返回 'null', 表示任务完成。")
        return None

    if output_cls:
        if agentOutput.structured_response:
            logger.info(f"完成 output_cls=\n{agentOutput.structured_response.model_dump_json(indent=2, ensure_ascii=False)}")
            return agentOutput.structured_response
        else:
            logger.warning("Agent 配置了 output_cls 但 structured_response 为空, 且原始响应不是 'null'。")
            raise ValueError("")
    else:
        logger.info(f"完成 output=\n{raw_output}")
        return raw_output
