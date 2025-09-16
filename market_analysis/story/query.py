import os
import sys
from typing import Optional, List, Tuple
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])
from market_analysis.story.common import get_market_vector_store
from utils.vector import vector_query


def query_market_reports(
    query_text: str,
    query_date: Optional[str] = None,
    num_results: int = 3,
) -> Tuple[Optional[str], Optional[List[NodeWithScore]]]:
    """
    Args:
        query_text (str): 核心查询问题。
        query_date (Optional[str], optional): 按日期筛选，格式 "YYYY-MM-DD". Defaults to None.
        num_results (int, optional): 最终用于合成答案的文档数量. Defaults to 3.
    Returns:
        Tuple[Optional[str], Optional[List[NodeWithScore]]]: 返回一个元组，包含 (合成的答案字符串, 来源节点列表)。如果失败则返回 (None, None)。
    """
    logger.info(f"🚀 开始市场报告查询: '{query_text}'")
    
    vector_store = get_market_vector_store()

    filters = None
    if query_date:
        logger.info(f"  - 配置元数据过滤器，按日期筛选: {query_date}")
        filters = MetadataFilters(filters=[ExactMatchFilter(key="date", value=query_date)])

    answer, source_nodes = vector_query(
        vector_store=vector_store,
        query_text=query_text,
        filters=filters,
        rerank_top_n=num_results
    )

    if not answer or not source_nodes:
        logger.warning("🤷 查询未返回答案或来源文档。")
        return None, None
    
    logger.success("\n--- ✅ 最终答案 ---")
    logger.info(answer)
    
    logger.info("\n---  答案来源 (经重排序) ---")
    for i, node in enumerate(source_nodes):
        score = node.score
        logger.info(f"\n📄 文档 {i+1}: (相关性得分: {score:.2f})")
        logger.info(f"  - 元数据: {node.metadata}")
        logger.info(f"  - 内容:\n{node.text}")

    return answer, source_nodes


if __name__ == "__main__":
    query = "起点小说平台的 签约审核流程 "
    query_market_reports(query_text=query)

