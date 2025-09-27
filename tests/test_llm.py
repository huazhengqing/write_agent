import os
import sys
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import litellm
from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank
from utils.llm import get_llm_params


@pytest.mark.asyncio
async def test_proxy_completion(llm_group: str):
    llm_params = get_llm_params(
        llm_group=llm_group,
        messages=[{"role": "user", "content": "你好, 请做个自我介绍"}]
    )
    response = await litellm.acompletion(**llm_params)
    content = response.choices[0].message.content
    logger.success(f"成功从 litellm proxy (模型组: '{llm_group}') 获得响应:")
    logger.info(content)


@pytest.mark.asyncio
async def test_proxy_embedding():
    model_name = "openai/embedding"
    text_to_embed = ["你好, 世界"]
    response = await litellm.aembedding(
        model=model_name,
        input=text_to_embed,
        api_base=os.getenv("LITELLM_PROXY_URL"),
        api_key=os.getenv("LITELLM_MASTER_KEY"),
    )
    embedding = response.data[0]["embedding"]
    logger.success(f"成功从 litellm proxy (模型: '{model_name}') 获得 embedding:")
    logger.info(f"维度: {len(embedding)}, 前5个值: {embedding[:5]}")





