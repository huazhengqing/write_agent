import json
import os
import sys
import chromadb
import asyncio
from typing import List, Optional, Tuple
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import NodeWithScore
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.file import input_dir, output_dir, chroma_dir
from utils.search import web_search_tools
from utils.vector import vector_query, get_chroma_vector_store


input_platform_dir = input_dir / "story" / "platform"
input_platform_dir.mkdir(parents=True, exist_ok=True)

output_market_dir = output_dir / "story" / "market"
output_market_dir.mkdir(parents=True, exist_ok=True)

chroma_market_dir = chroma_dir / "story" / "market"
chroma_market_dir.mkdir(parents=True, exist_ok=True)

chroma_collection_market_name = "market"

_vector_store: Optional[ChromaVectorStore] = None

def get_market_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = get_chroma_vector_store(
            db_path=str(chroma_market_dir),
            collection_name=chroma_collection_market_name
        )
    return _vector_store


BROAD_SCAN_SYSTEM_PROMPT = """
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

ASSESS_NEW_AUTHOR_OPPORTUNITY_SYSTEM_PROMPT = """
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


async def story_market_vector(
        query: str, 
        document_type: Optional[str] = None, 
        platform: Optional[str] = None,
        genre: Optional[str] = None
    ) -> str:
    """
    在向量数据库中搜索相关信息，支持按类型、平台和题材进行过滤。
    Args:
        query (str): 核心搜索查询。
        document_type (Optional[str]): 文档类型, 可选值为 'platform_profile', 'deep_dive_report', 'novel_concept'。
        platform (Optional[str]): 平台名称, 如 '番茄小说'。
        genre (Optional[str]): 题材名称, 如 '都市脑洞'。 
    Returns:
        str: 格式化的搜索结果，包含元数据和内容。
    """
    logger.info(f"向量数据库搜索: query='{query}', type='{document_type}', platform='{platform}', genre='{genre}'")
    try:
        filters = []
        if document_type: filters.append(ExactMatchFilter(key="type", value=document_type))
        if platform: filters.append(ExactMatchFilter(key="platform", value=platform))
        if genre: filters.append(ExactMatchFilter(key="genre", value=genre))
        
        metadata_filters = MetadataFilters(filters=filters) if filters else None

        # 使用通用的 vector_query 函数执行查询和重排序
        # 在异步函数中调用同步函数，使用 asyncio.to_thread
        _, final_results = await asyncio.to_thread(
            vector_query,
            vector_store=get_market_vector_store(),
            query_text=query,
            filters=metadata_filters,
            similarity_top_k=25, # 初步检索25个
            rerank_top_n=5,      # 重排后保留5个
        )

        if not final_results:
            logger.warning("向量数据库未找到匹配的结果。")
            return "在内部知识库中未找到相关信息。"

        content_parts = []
        for i, node_with_score in enumerate(final_results):
            node = node_with_score.node
            metadata_str = json.dumps(node.metadata, ensure_ascii=False)
            content_parts.append(f"--- 结果 {i+1} ---\n元数据: {metadata_str}\n内容:\n{node.get_content()}")
        
        final_content = "\n\n".join(content_parts)
        logger.success(f"向量数据库搜索和重排序完成，找到 {len(final_results)} 个结果。")
        return final_content

    except Exception as e:
        logger.error(f"向量数据库搜索出错: {e}")
        return f"向量数据库搜索失败: {e}"

def get_story_market_search_tool() -> FunctionTool:
    tool_description = (
        "功能: 在内部知识库（向量数据库）中搜索已归档的报告和创意。这是研究时的首选工具，应在进行外部网络搜索之前使用。\n"
        "知识库内容:\n"
        "- `platform_profile`: 平台静态信息（背景、用户、商业模式、作者政策）。\n"
        "- `deep_dive_report`: 特定平台和题材的深度市场分析（核心标签、爽点、用户画像、新兴机会）。\n"
        "- `novel_concept`: 详细的小说创意（梗概、人设、世界观、爽点设计、风险评估）。\n"
        "使用指南:\n"
        "1. **精确过滤**: 优先使用 `document_type` 参数以获得最精确的结果。\n"
        "2. **具体查询**: `query` 参数应提供具体的关键词或问题。\n"
        "3. **结果解读**: 返回内容包含元数据（平台、题材、日期），用于判断信息的相关性和时效性。\n"
        "4. **空结果处理**: 如果返回“未找到相关信息”，则内部无数据，此时才应使用外部搜索工具。\n"
        "参数:\n"
        "- `query` (必需, str): 核心搜索查询。例如: '新人作者机会', '创新的世界观设定'。\n"
        "- `document_type` (可选, str): 文档类型。可选值: 'platform_profile', 'deep_dive_report', 'novel_concept'。\n"
        "- `platform` (可选, str): 平台名称。例如: '番茄小说'。\n"
        "- `genre` (可选, str): 题材名称。例如: '都市脑洞'。\n"
        "使用示例:\n"
        "- 精确查找最新的番茄小说深度分析报告: `story_market_vector(query='最新用户趋势', document_type='deep_dive_report', platform='番茄小说')`\n"
        "- 查找所有科幻题材的小说创意: `story_market_vector(query='创新的世界观设定', document_type='novel_concept', genre='科幻')`\n"
        "- 模糊搜索主角人设相关信息: `story_market_vector(query='主角人设创新')`"
    )
    return FunctionTool.from_defaults(fn=story_market_vector, name="story_market_vector", description=tool_description)


def get_market_tools() -> List[FunctionTool]:
    return web_search_tools + [get_story_market_search_tool()]
