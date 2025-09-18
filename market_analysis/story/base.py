import json
import os
import sys
from typing import List, Optional, Tuple
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import VectorStoreInfo
from llama_index.core.schema import NodeWithScore
from loguru import logger
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, get_leaf_nodes
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.llm import llm_temperatures
from utils.agent import call_react_agent
from utils.file import data_market_dir
from utils.search import web_search_tools
from utils.vector import get_vector_query_engine, get_vector_store


_vector_store: Optional[ChromaVectorStore] = None
def get_market_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        logger.info("正在初始化故事市场分析的向量库...")
        _vector_store = get_vector_store(
            db_path=str(data_market_dir / "story"),
            collection_name="story_market"
        )
    return _vector_store


BROAD_SCAN_system_prompt = """
# 角色
你是一名专业的网络小说市场动态分析师。

# 任务
为平台【{platform}】生成一份聚焦于当前市场动态的简报。你需要利用工具（网络搜索、网页抓取）收集信息，并严格按照指定的Markdown格式输出。

# 工作流程
1.  研究: 搜索并分析与平台相关的最新信息，重点关注以下核心指标。
    - 热门题材: 搜索平台的热销榜、新书榜、推荐榜，找出当前最受欢迎的3-5个题材大类。
    - 官方动向: 搜索平台的官方公告、作者后台、征文活动页面，总结近期的官方活动方向（如特定题材的征文、新的激励计划）。
2.  总结: 将你的发现综合成一份完整的Markdown报告。如果某个要点确实找不到信息，请在该标题下明确指出“未找到相关信息”。

# 输出要求
## {platform} 平台市场动态简报

### 1. 热门题材
- (列出3-5个当前最热门的题材大类，并简要说明判断依据，例如：根据XX榜单)

### 2. 官方动向
- (总结近期的官方征文、激励活动方向。如果没有则明确写出“近期未发现明确的官方活动导向”)
"""


ASSESS_NEW_AUTHOR_OPPORTUNITY_system_prompt = """
# 角色
你是一名专业的网络小说行业研究员，专注于评估各平台对新人作者的友好度。

# 任务
为平台【{platform}】生成一份详细的新人机会评估报告。你需要利用工具（网络搜索、网页抓取）收集信息，并严格按照指定的Markdown格式输出。

# 工作流程
1.  研究: 综合搜索到的信息，从以下几个维度评估新人作者在该平台发展的机会：
    - 流量扶持: 搜索平台是否有明确的新书推荐位、新人流量池或“新书期”保护机制？
    - 变现门槛: 搜索新人作者签约后多久可以开始获得收入（如广告分成、稿费），以及签约的难易程度。
    - 竞争环境: 搜索平台新书榜的更新频率和上榜难度，估算新人作品脱颖而出的竞争激烈程度。
    - 编辑支持: 搜索作者论坛（如龙的天空）、知乎等，了解新人作者获得编辑指导和反馈的普遍情况。
2.  总结: 将你的发现综合成一份完整的Markdown报告。如果某个要点确实找不到信息，请在该标题下明确指出“未找到相关信息”。

# 输出要求
## {platform} 平台新人机会评估报告

- 综合评级: [高/中/低]
- 评级理由:
  - 流量机会: (总结平台对新书的流量支持情况，例如：有独立新书榜和算法推荐，流量机会中等。)
  - 变现速度: (总结新人作者的变现路径和速度，例如：签约即有广告分成，变现速度快。)
  - 竞争压力: (总结新人面临的竞争情况，例如：头部效应明显，新书榜竞争激烈，压力大。)
  - 编辑生态: (总结编辑对新人的支持情况，例如：编辑回复较慢，主要靠作者自己摸索。)
- 核心建议: (给新人作者一句核心建议，例如：建议从平台重点扶持的XX题材切入，利用好新书期流量。)
"""


def get_vector_query_tool() -> QueryEngineTool:
    vector_store_info = VectorStoreInfo(
        content_info="关于网络小说市场的各种研究文档，包括平台档案、市场动态、新人机会、外部趋势、深度分析、决策报告和小说创意等。",
        metadata_info=[
            {
                "name": "type",
                "type": "str",
                "description": (
                    "文档类型。可选值为 'platform_profile' (平台档案), "
                    "'broad_scan_report' (市场动态简报), 'new_author_opportunity_report' (新人机会评估报告), "
                    "'external_trend_report' (外部趋势分析报告), 'market_analysis_result' (初步市场机会决策报告), "
                    "'deep_dive_report' (深度分析报告), 'final_decision_report' (最终决策报告), "
                    "'opportunity_generation_report' (小说选题建议), 'novel_concept' (详细小说创意)。"
                ),
            },
            {"name": "platform", "type": "str", "description": "平台名称，例如 '番茄小说', '起点中文网'。"},
            {"name": "genre", "type": "str", "description": "题材名称，例如 '都市脑洞', '东方玄幻'。"},
        ]
    )

    query_engine = get_vector_query_engine(
        vector_store=get_market_vector_store(),
        use_auto_retriever=True,
        vector_store_info=vector_store_info,
        similarity_top_k=25,
        rerank_top_n=5,
    )

    tool_description = (
        "功能: 在内部知识库中搜索已归档的市场报告和小说创意。这是研究时的首选工具。\n"
        "使用指南:\n"
        "1. **自然语言查询**: 你可以直接在查询中描述你想要的文档类型、平台或题材，工具会自动进行过滤。\n"
        "   例如: '查找关于番茄小说的深度分析报告', '搜索都市脑洞题材的小说创意'。\n"
        "2. **结果解读**: 返回的内容是综合了多个相关文档的答案。如果需要原始片段，请在后续思考中说明。\n"
        "3. **空结果处理**: 如果返回“未找到相关信息”，则内部无数据，此时才应使用外部搜索工具。\n"
        "参数:\n"
        "- `input` (str, 必需): 核心搜索查询，可以包含过滤条件。例如: '番茄小说的新人作者机会', '科幻题材的创新世界观设定'。"
    )
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
        tools=get_market_tools(),
        llm_group="reasoning",
        temperature=llm_temperatures["reasoning"]
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)
    return result
