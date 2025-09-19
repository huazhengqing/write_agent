import os
import sys
import numpy as np
import pytest
import pytest_asyncio
from loguru import logger

from llama_index.core import Settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



@pytest_asyncio.fixture(scope="module")
async def embedding_test_data():
    """
    准备嵌入模型测试所需的数据和模型实例。
    这个 fixture 在模块级别运行一次，以减少 API 调用。
    """
    logger.info("--- (Fixture) 准备嵌入模型测试数据 ---")
    embed_model = Settings.embed_model
    text1 = "这是一个关于人工智能的句子。"
    text2 = "这是一个关于自然语言处理的句子。"
    
    embedding1_list = await embed_model.aget_text_embedding(text1)
    embedding1 = np.array(embedding1_list)
    
    return {
        "embed_model": embed_model,
        "text1": text1,
        "text2": text2,
        "embedding1": embedding1,
    }


@pytest.mark.asyncio
async def test_embedding_different_texts(embedding_test_data):
    """测试不同文本是否产生不同的、合理的向量。"""
    logger.info("--- 测试：不同文本的向量差异性 ---")
    embed_model = embedding_test_data["embed_model"]
    embedding1 = embedding_test_data["embedding1"]
    text2 = embedding_test_data["text2"]

    embedding2_list = await embed_model.aget_text_embedding(text2)
    embedding2 = np.array(embedding2_list)

    assert np.any(embedding1 != 0), "嵌入向量1不应为全零向量。"
    assert np.any(embedding2 != 0), "嵌入向量2不应为全零向量。"
    assert not np.array_equal(embedding1, embedding2), "不同文本不应产生完全相同的嵌入向量。"

    norm1 = np.linalg.norm(embedding1) # type: ignore
    norm2 = np.linalg.norm(embedding2) # type: ignore
    assert norm1 > 0 and norm2 > 0, "向量模长不能为零。" # type: ignore
    
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    logger.info(f"两个不同但相关句子的余弦相似度: {similarity:.4f}")
    assert 0.5 < similarity < 0.999, "相关句子的相似度应在合理范围内。"
    logger.success("--- 不同文本向量差异性测试通过 ---")


@pytest.mark.asyncio
async def test_embedding_same_text(embedding_test_data):
    """测试相同文本是否产生一致的向量。"""
    logger.info("--- 测试：相同文本的向量一致性 ---")
    embed_model = embedding_test_data["embed_model"]
    text1 = embedding_test_data["text1"]
    embedding1 = embedding_test_data["embedding1"]

    embedding1_again_list = await embed_model.aget_text_embedding(text1)
    embedding1_again = np.array(embedding1_again_list)
    
    similarity_same = np.dot(embedding1, embedding1_again) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding1_again))
    logger.info(f"相同文本两次嵌入的余弦相似度: {similarity_same:.6f}")
    assert similarity_same > 0.999, f"相同文本的嵌入向量应该非常相似，但实际为 {similarity_same:.6f}"
    logger.success("--- 相同文本向量一致性测试通过 ---")


@pytest.mark.asyncio
async def test_embedding_batch(embedding_test_data):
    """测试批量嵌入功能是否正常工作。"""
    logger.info("--- 测试：批量嵌入 ---")
    embed_model = embedding_test_data["embed_model"]
    text1 = embedding_test_data["text1"]
    text2 = embedding_test_data["text2"]
    embedding1 = embedding_test_data["embedding1"]

    texts_batch = [text1, text2, "第三个完全不同的句子。"]
    embeddings_batch_list = await embed_model.aget_text_embedding_batch(texts_batch)
    embeddings_batch = [np.array(e) for e in embeddings_batch_list]
    
    assert len(embeddings_batch) == 3, f"批量嵌入应返回3个向量，但返回了 {len(embeddings_batch)} 个。"
    
    similarity_batch = np.dot(embeddings_batch[0], embedding1) / (np.linalg.norm(embeddings_batch[0]) * np.linalg.norm(embedding1))
    logger.info(f"批量嵌入与单个嵌入结果的余弦相似度: {similarity_batch:.6f}")
    assert similarity_batch > 0.999, f"批量嵌入的第一个结果应与单个嵌入结果非常相似，但实际为 {similarity_batch:.6f}"
    logger.success("--- 批量嵌入测试通过 ---")