import os
import sys
from typing import Optional
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])
from market_analysis.story.common import index


def query_reports(query: str, date: Optional[str] = None, n_results: int = 1) -> list[NodeWithScore]:
    filters = None
    if date:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="date", value=date)])
        logger.info(f"  - 按日期筛选: {date}")
    retriever = index.as_retriever(similarity_top_k=n_results, filters=filters)
    results = retriever.retrieve(query)
    logger.info(f"✅ 查询到 {len(results)} 份相关报告。")
    return results


if __name__ == "__main__":
    query_text = "番茄小说平台对AI生成内容的政策"
    query_date = None  # 可选, 如需指定日期, 格式为 "YYYY-MM-DD"
    num_results = 30

    logger.info(f"🚀 开始查询，查询内容: '{query_text}'")
    found_docs = query_reports(query=query_text, date=query_date, n_results=num_results)
    if not found_docs:
        logger.warning("🤷 未找到相关文档。")
    else:
        logger.info("\n--- 查询结果 ---")
        for i, node_with_score in enumerate(found_docs):
            doc = node_with_score.node
            score = node_with_score.score
            logger.info(f"\n📄 文档 {i+1}: (相似度得分: {score:.4f})")
            logger.info(f"  - 元数据: {doc.metadata}")
            logger.info(f"  - 内容:\n{doc.text}")
        logger.info("\n--- 查询结束 ---")
