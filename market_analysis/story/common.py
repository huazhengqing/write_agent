import json
import asyncio
import chromadb
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.litellm import LiteLLMEmbedding
from datetime import datetime
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.tools import FunctionTool
from loguru import logger
from utils.agent_tools import agent_tavily_tools, get_web_scraper_tool, get_social_media_trends_tool, get_forum_discussions_tool
from utils.llm import call_agent, get_embedding_params
from utils.prefect_utils import local_storage, readable_json_serializer, generate_readable_cache_key
from utils.file import input_dir, output_dir, chroma_dir
from prefect import task


platforms_cn = ["番茄小说", "起点中文网", "飞卢小说网", "晋江文学城", "七猫免费小说", "纵横中文网", "17K小说网", "刺猬猫", "掌阅"]
platforms_en = []

input_platform_dir = input_dir / "story" / "platform"
input_platform_dir.mkdir(parents=True, exist_ok=True)

output_market_dir = output_dir / "story" / "market"
output_market_dir.mkdir(parents=True, exist_ok=True)

chroma_market_dir = chroma_dir / "story" / "market"
chroma_market_dir.mkdir(parents=True, exist_ok=True)

chroma_collection_market_name = "market"

embedding_params = get_embedding_params(embedding='bge-m3')
embed_model_name = embedding_params.pop('model')
embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)

db = chromadb.PersistentClient(path=str(chroma_market_dir))
chroma_collection = db.get_or_create_collection(chroma_collection_market_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


###############################################################################


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


###############################################################################


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
        if document_type:
            filters.append(ExactMatchFilter(key="type", value=document_type))
        if platform:
            filters.append(ExactMatchFilter(key="platform", value=platform))
        if genre:
            filters.append(ExactMatchFilter(key="genre", value=genre))
        retriever_kwargs = {"similarity_top_k": 5}
        if filters:
            retriever_kwargs["filters"] = MetadataFilters(filters=filters)
        retriever = index.as_retriever(retriever_kwargs)
        results = await retriever.aretrieve(query)
        if not results:
            logger.warning("向量数据库未找到匹配的结果。")
            return "在内部知识库中未找到相关信息。"
        content_parts = []
        for i, node in enumerate(results):
            metadata_str = json.dumps(node.metadata, ensure_ascii=False)
            content_parts.append(f"--- 结果 {i+1} ---\n元数据: {metadata_str}\n内容:\n{node.get_content()}")
        final_content = "\n\n".join(content_parts)
        logger.success(f"向量数据库搜索完成，找到 {len(results)} 个结果。")
        return final_content
    except Exception as e:
        logger.error(f"向量数据库搜索出错: {e}")
        return f"向量数据库搜索失败: {e}"

def get_story_market_search_tool() -> FunctionTool:
    tool_description = (
        "在内部知识库（向量数据库）中搜索已归档的报告和创意。这是进行研究时的首选工具，应在进行外部网络搜索之前使用。\n"
        "知识库内容:\n"
        "- 平台档案 (platform_profile): 各个平台的背景、用户特征、商业模式、作者政策等静态信息。\n"
        "- 市场分析报告 (deep_dive_report): 针对特定平台和题材的深度市场分析报告，包含核心标签、爽点、用户画像、新兴机会等。\n"
        "- 小说创意 (novel_concept): 详细的小说创意，包含一句话简介、故事梗概、人物设定、世界观、升级体系、爽点设计、风险评估等。\n"
        "使用指南:\n"
        "1.  **优先过滤**: 当你知道要找的文档类型时，必须使用 `document_type` 参数进行过滤，以获得最精确的结果。\n"
        "2.  **结合查询**: `query` 参数应提供具体的关键词或问题。\n"
        "3.  **解读结果**: 返回内容包含元数据（如平台、题材、日期）和文本内容。请利用元数据判断信息的相关性和时效性。\n"
        "4.  **空结果处理**: 如果返回“在内部知识库中未找到相关信息”，则意味着内部没有相关数据，此时你才应该考虑使用 `web_scraper` 或其他搜索工具进行外部研究。\n"
        "参数说明:\n"
        "- `query` (必需, 字符串): 核心搜索查询，例如 '新人作者机会' 或 '创新的世界观设定'。\n"
        "- `document_type` (可选, 字符串): 文档类型，可选值为 'platform_profile', 'deep_dive_report', 'novel_concept'。\n"
        "- `platform` (可选, 字符串): 平台名称，例如 '番茄小说'。\n"
        "- `genre` (可选, 字符串): 题材名称，例如 '都市脑洞'。\n"
        "使用示例:\n"
        "- 示例1 (精确查找): 查找最新的关于'番茄小说'的深度分析报告 -> `story_market_vector(query='最新用户趋势', document_type='deep_dive_report', platform='番茄小说')`\n"
        "- 示例2 (类型查找): 查找所有关于'科幻'题材的小说创意 -> `story_market_vector(query='创新的世界观设定', document_type='novel_concept', genre='科幻')`\n"
        "- 示例3 (模糊搜索): 搜索关于'主角人设'的所有信息 -> `story_market_vector(query='主角人设创新')`"
    )
    return FunctionTool.from_defaults(fn=story_market_vector, name="story_market_vector", description=tool_description)


###############################################################################


def get_market_tools() -> List[FunctionTool]:
    return agent_tavily_tools + [get_web_scraper_tool(), get_social_media_trends_tool(), get_forum_discussions_tool(), get_story_market_search_tool()]


###############################################################################


@task(
    name="load_platform_profile",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/common/load_platform_profile_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,  # 7 天过期 (秒)
)
async def task_load_platform_profile(platform: str) -> Tuple[str, str]:
    logger.info(f"正在从向量库加载平台 '{platform}' 的基础信息...")
    profile_content = f"# {platform} 平台档案\n\n未在知识库中找到该平台的基础信息。"
    try:
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="type", value="platform_profile"),
            ExactMatchFilter(key="platform", value=platform)
        ])
        retriever = index.as_retriever(similarity_top_k=1, filters=filters)
        results = await retriever.aretrieve(f"{platform} 平台档案")
        if results:
            profile_content = results[0].get_content()
            logger.success(f"已加载 '{platform}' 的基础信息。")
        else:
            logger.warning(f"在向量库中未找到 '{platform}' 的基础信息。建议先运行 `story_platform_by_search.py`。")
    except Exception as e:
        logger.error(f"加载 '{platform}' 的基础信息时出错: {e}")
        profile_content = f"# {platform} 平台档案\n\n加载基础信息时出错: {e}"
    return platform, profile_content

@task(name="platform_briefing",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/common/platform_briefing_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,  # 7 天过期 (秒)
)
async def task_platform_briefing(platform: str) -> str:
    logger.info(f"为平台 '{platform}' 生成市场动态简报...")
    system_prompt = BROAD_SCAN_SYSTEM_PROMPT.format(platform=platform)
    user_prompt = f"请开始为平台 '{platform}' 生成市场动态简报。"
    report = await call_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=get_market_tools(),
        temperature=0.1
    )
    if report:
        logger.success(f"Agent为 '{platform}' 完成了简报生成，报告长度: {len(report)}。")
        return report
    else:
        error_msg = f"为平台 '{platform}' 生成市场动态简报时Agent调用失败或返回空。"
        logger.error(error_msg)
        return f"## {platform} 平台市场动态简报\n\n生成报告时出错: {error_msg}"

@task(
    name="new_author_opportunity",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/common/new_author_opportunity_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,  # 7 天过期 (秒)
)
async def task_new_author_opportunity(platform: str) -> str:
    logger.info(f"为平台 '{platform}' 生成新人机会评估报告...")
    system_prompt = ASSESS_NEW_AUTHOR_OPPORTUNITY_SYSTEM_PROMPT.format(platform=platform)
    user_prompt = f"请开始为平台 '{platform}' 生成新人机会评估报告。"
    report = await call_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=get_market_tools(),
        temperature=0.1
    )
    if report:
        logger.success(f"Agent为 '{platform}' 完成了新人机会评估报告生成，报告长度: {len(report)}。")
        return report
    else:
        error_msg = f"为平台 '{platform}' 生成新人机会评估报告时Agent调用失败或返回空。"
        logger.error(error_msg)
        return f"## {platform} 平台新人机会评估报告\n\n生成报告时出错: {error_msg}"

@task(
    name="task_store",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    cache_expiration=604800,  # 7 天过期 (秒)
    retries=2,
    retry_delay_seconds=10
)
async def task_store(content: str, doc_type: str, content_format: str = "text", **metadata: Any) -> bool:
    if not content or not content.strip() or "生成报告时出错" in content:
        logger.warning(f"内容为空或包含错误，跳过存入向量库。类型: {doc_type}, 元数据: {metadata}")
        return False
    logger.info(f"正在将类型为 '{doc_type}' (格式: {content_format}) 的报告存入向量库...")
    final_metadata = metadata.copy()
    final_metadata["type"] = doc_type
    final_metadata["date"] = datetime.now().strftime("%Y-%m-%d")
    doc = Document(text=content, metadata=final_metadata)
    if content_format == "markdown":
        md_parser = MarkdownNodeParser(include_metadata=True)
        nodes = md_parser.get_nodes_from_documents([doc])
        await asyncio.to_thread(index.insert_nodes, nodes)
    elif content_format == "json":
        await asyncio.to_thread(index.insert, doc)
    else:
        parser = SentenceSplitter(include_metadata=True)
        nodes = parser.get_nodes_from_documents([doc])
        await asyncio.to_thread(index.insert_nodes, nodes)
    logger.success(f"类型为 '{doc_type}' 的报告已成功存入向量库。元数据: {final_metadata}")
    return True


###############################################################################

