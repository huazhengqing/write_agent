import nest_asyncio
nest_asyncio.apply()


import os
import sys
from typing import List, Optional
from functools import lru_cache
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import VectorStoreInfo, MetadataInfo
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.react_agent import call_react_agent
from utils.file import data_market_dir
from utils.search import web_search_tools
from rag.vector import get_vector_store
from rag.vector_query import get_vector_query_engine, index_query



###############################################################################



@lru_cache(maxsize=None)
def get_market_vector_store() -> ChromaVectorStore:
    return get_vector_store(
        db_path=str(data_market_dir / "story"),
        collection_name="story_market"
    )



###############################################################################


def get_vector_query_tool() -> QueryEngineTool:
    vector_store_info = VectorStoreInfo(
        content_info="关于网络小说市场的各种研究文档, 包括平台档案、市场动态、新人机会、外部趋势、深度分析、决策报告和小说创意等。",
        metadata_info=[
            MetadataInfo(
                name="type",
                type="str",
                description=(
                    "文档类型。可选值为 'platform_profile' (平台档案), "
                    "'broad_scan_report' (市场动态简报), 'new_author_opportunity_report' (新人机会评估报告), "
                    "'external_trend_report' (外部趋势分析报告), 'market_analysis_result' (初步市场机会决策报告), "
                    "'deep_dive_report' (深度分析报告), 'final_decision_report' (最终决策报告), "
                    "'opportunity_generation_report' (小说选题建议), 'novel_concept' (详细小说创意)。"
                ),
            ),
            MetadataInfo(name="platform", type="str", description="平台名称, 例如 '番茄小说', '起点中文网'。"),
            MetadataInfo(name="genre", type="str", description="题材名称, 例如 '都市脑洞', '东方玄幻'。"),
        ]
    )

    query_engine = get_vector_query_engine(
        vector_store=get_market_vector_store(),
        use_auto_retriever=True,
        vector_store_info=vector_store_info,
    )

    tool_description = """
功能: 在内部知识库中搜索已归档的市场报告和小说创意。这是研究时的首选工具。
使用指南:
1. 自然语言查询: 你可以直接在查询中描述你想要的文档类型、平台或题材, 工具会自动进行过滤。例如: '查找关于番茄小说的深度分析报告', '搜索都市脑洞题材的小说创意'。
2. 结果解读: 返回的内容是综合了多个相关文档的答案。如果需要原始片段, 请在后续思考中说明。
3. 空结果处理: 如果返回"未找到相关信息", 则内部无数据, 此时才应使用外部搜索工具。
参数:
- `input` (str, 必需): 核心搜索查询, 可以包含过滤条件。例如: '番茄小说的新人作者机会', '科幻题材的创新世界观设定'。
"""

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="story_market_vector",
        description=tool_description,
    )


def get_market_tools() -> List:
    return web_search_tools + [get_vector_query_tool()]


async def query_react(
    query_str: str,
    agent_system_prompt: Optional[str] = None,
) -> str:
    result = await call_react_agent(
        system_prompt=agent_system_prompt,
        user_prompt=query_str,
        tools=get_market_tools()
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)
    return result
