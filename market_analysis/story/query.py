import nest_asyncio
nest_asyncio.apply()
import os
import sys
import asyncio
from typing import Optional, List
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])
from market_analysis.story.base import get_market_vector_store
from utils.vector import get_vector_query_engine, index_query


async def market_query(
    query_texts: List[str],
    query_date: Optional[str] = None
) -> Optional[str]:
    filters = None
    if query_date:
        logger.info(f"  - 配置元数据过滤器，按日期筛选: {query_date}")
        filters = MetadataFilters(filters=[ExactMatchFilter(key="date", value=query_date)])

    query_engine = get_vector_query_engine(
        vector_store=get_market_vector_store(),
        filters=filters
    )

    node_contents = await index_query(
        query_engine=query_engine,
        questions=query_texts
    )

    if not node_contents:
        logger.warning("🤷 查询未返回任何相关文档内容。")
        return None

    answer = "\n\n---\n\n".join(node_contents)
    logger.info(answer)
    return answer


if __name__ == "__main__":
    queries = [
        "起点中文网的签约流程和要求是什么？",
        "番茄小说对新人的扶持政策有哪些？",
        "分析一下都市脑洞题材在番茄小说的市场机会",
        "给我一个关于东方玄幻题材的详细小说创意",
        "晋江文学城的读者画像是怎样的？",
        "飞卢的最新热门题材有哪些？",
    ]
    print(f"\n\n{'='*20} 批量查询 {'='*20}")
    asyncio.run(market_query(query_texts=queries))

