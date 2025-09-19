import os
import sys
import pytest
from loguru import logger

from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



@pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="SILICONFLOW_API_KEY is not set")
@pytest.mark.asyncio
async def test_reranker_functionality():
    """专门测试重排服务的功能和正确性。"""
    logger.info("--- 测试重排服务 (Reranker) ---")

    query = "哪部作品是关于一个男孩发现自己是巫师的故事？"
    documents = [
        "《沙丘》是一部关于星际政治和巨型沙虫的史诗科幻小说。",
        "《哈利·波特与魔法石》讲述了一个名叫哈利·波特的年轻男孩，他发现自己是一个巫师，并被霍格沃茨魔法学校录取。",
        "《魔戒》讲述了霍比特人佛罗多·巴金斯摧毁至尊魔戒的旅程。",
        "《神经漫游者》是一部赛博朋克小说，探讨了人工智能和虚拟现实。",
        "一个男孩在魔法学校学习的故事，他最好的朋友是一个红发男孩和一个聪明的女孩。",
    ]

    reranker = SiliconFlowRerank(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        top_n=3,
    )

    nodes = [NodeWithScore(node=Document(text=d), score=1.0) for d in documents]
    query_bundle = QueryBundle(query_str=query)

    reranked_nodes = await reranker.apostprocess_nodes(nodes, query_bundle=query_bundle)
    
    assert len(reranked_nodes) <= 3, f"重排后应返回最多 {reranker.top_n} 个节点, 但返回了 {len(reranked_nodes)} 个。"
    assert len(reranked_nodes) > 0, "Reranker 返回了空列表，服务可能未正常工作或所有文档相关性均为0。"

    reranked_texts = [node.get_content() for node in reranked_nodes]
    reranked_scores = [node.score for node in reranked_nodes]
    logger.info(f"重排后的文档顺序及分数: {list(zip(reranked_texts, reranked_scores))}")

    assert "哈利·波特" in reranked_texts[0], "最相关的文档《哈利·波特》没有排在第一位。"

    for i in range(len(reranked_scores) - 1):
        assert reranked_scores[i] >= reranked_scores[i+1], f"重排后分数没有按预期递减: {reranked_scores}"

    logger.success("--- 重排服务测试通过 ---")


@pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="SILICONFLOW_API_KEY is not set")
@pytest.mark.asyncio
async def test_reranker_empty_input():
    """测试当输入文档列表为空时，重排器的行为。"""
    logger.info("--- 测试重排服务 (空输入) ---")
    reranker = SiliconFlowRerank(api_key=os.getenv("SILICONFLOW_API_KEY"), top_n=3)
    query_bundle = QueryBundle(query_str="任意查询")
    
    reranked_nodes = await reranker.apostprocess_nodes([], query_bundle=query_bundle)
    
    assert len(reranked_nodes) == 0, "当输入为空列表时，重排器应返回空列表。"
    logger.success("--- 重排服务 (空输入) 测试通过 ---")


@pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="SILICONFLOW_API_KEY is not set")
@pytest.mark.asyncio
async def test_reranker_top_n_handling():
    """测试当 top_n 大于文档数量时，重排器的行为。"""
    logger.info("--- 测试重排服务 (top_n > 文档数) ---")
    query = "关于魔法的故事"
    documents = [
        Document(text="文档1：魔法师的学徒"),
        Document(text="文档2：科幻小说"),
        Document(text="文档3：历史传记"),
    ]
    nodes = [NodeWithScore(node=d, score=1.0) for d in documents]
    
    # 设置 top_n (5) 大于文档数 (3)
    reranker = SiliconFlowRerank(api_key=os.getenv("SILICONFLOW_API_KEY"), top_n=5)
    query_bundle = QueryBundle(query_str=query)

    reranked_nodes = await reranker.apostprocess_nodes(nodes, query_bundle=query_bundle)

    assert len(reranked_nodes) == len(documents), f"当 top_n 大于文档数时，应返回所有文档，预期 {len(documents)}，实际 {len(reranked_nodes)}。"
    
    reranked_scores = [node.score for node in reranked_nodes]
    for i in range(len(reranked_scores) - 1):
        assert reranked_scores[i] >= reranked_scores[i+1], f"即使返回所有文档，分数也应递减: {reranked_scores}"

    logger.success("--- 重排服务 (top_n > 文档数) 测试通过 ---")