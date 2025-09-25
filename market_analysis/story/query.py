import os
import sys
from typing import Optional, List
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])



async def market_query(query_texts: List[str]) -> Optional[str]:
    from rag.vector_query import get_vector_query_engine, index_query_batch
    from market_analysis.story.base import get_market_vector_store
    query_engine = get_vector_query_engine(
        vector_store=get_market_vector_store()
    )
    node_contents = await index_query_batch(
        query_engine=query_engine, 
        questions=query_texts
    )
    if not node_contents:
        logger.warning("🤷 查询未返回任何相关文档内容。")
        return None
    answer = "\n\n---\n\n".join(node_contents)
    logger.success(f"查询成功，返回总长度为 {len(answer)} 的内容。")
    return answer



if __name__ == "__main__":
    queries = [
        "起点中文网的签约流程和要求是什么?",
        "番茄小说对新人的扶持政策有哪些?",
        "分析一下都市脑洞题材在番茄小说的市场机会",
        "给我一个关于东方玄幻题材的详细小说创意",
        "晋江文学城的读者画像是怎样的?",
        "飞卢的最新热门题材有哪些?",
    ]
    logger.info(f"{'='*20} 批量查询 {'='*20}")
    import asyncio
    result = asyncio.run(market_query(query_texts=queries))
    if result:
        logger.info(f"查询结果:\n{result}")
