import os
import sys
from typing import Optional, Tuple
from loguru import logger

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from market_analysis.story.base import get_market_vector_store, query_react
from rag.vector_query import get_vector_query_engine, index_query
from rag.vector_add import vector_add
from utils.file import data_market_dir
from utils.prefect import local_storage, readable_json_serializer
from prefect import task


###############################################################################



@task(
    name="load_platform_profile",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/common/load_platform_profile_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_load_platform_profile(platform: str) -> Tuple[str, str]:
    logger.info(f"正在从向量库加载平台 '{platform}' 的基础信息...")
    profile_content = f"# {platform} 平台档案\n\n未在知识库中找到该平台的基础信息。"
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="type", value="platform_profile"),
            ExactMatchFilter(key="platform", value=platform),
        ]
    )
    query_engine = get_vector_query_engine(
        vector_store=get_market_vector_store(),
        filters=filters,
        similarity_top_k=1,
        top_n=None, # 无需重排
    )
    content = await index_query(
        query_engine=query_engine,
        question=f"{platform} 平台档案"
    )

    if content:
        profile_content = content
        logger.success(f"已加载 '{platform}' 的基础信息。")
    else:
        logger.warning(
            f"在向量库中未找到 '{platform}' 的基础信息。建议先运行 `story_platform_by_search.py`。"
        )
    return platform, profile_content



###############################################################################



broad_scan_system_prompt = """
# 任务背景与目标
你是一名专业的网络小说市场动态分析师。你的任务是为平台[{platform}]生成一份聚焦于当前市场动态的简报。

# 研究清单 (Checklist)
你必须通过"思考 -> 操作 -> 观察"的循环, 依次研究并回答以下所有问题, 以收集报告所需的全部信息。
1. 热门题材: 当前在[{platform}]平台上, 最受欢迎的3-5个题材大类是什么? 请说明判断依据(例如: 基于平台的热销榜、新书榜、推荐榜等)。
2. 官方动向: 近期在[{platform}]平台上, 官方有哪些值得关注的活动方向? 请总结平台官方公告、作者后台和征文活动页面的信息(例如: 特定题材的征文、新的激励计划等)。

# 最终答案格式
当你通过工具收集完以上所有信息后, 你的最终`答案`必须是且只能是一份严格遵循以下结构的 Markdown 报告。
如果某个要点确实找不到信息, 请在该标题下明确指出"未找到相关信息"。

# 输出要求
```markdown
## {platform} 平台市场动态简报

### 1. 热门题材
- (列出3-5个当前最热门的题材大类, 并简要说明判断依据, 例如: 根据XX榜单)

### 2. 官方动向
- (总结近期的官方征文、激励活动方向。如果没有则明确写出"近期未发现明确的官方活动导向")
```
"""



@task(
    name="platform_briefing",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/common/platform_briefing_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_platform_briefing(platform: str) -> str:
    logger.info(f"为平台 '{platform}' 生成市场动态简报...")
    system_prompt = broad_scan_system_prompt.format(platform=platform)
    user_prompt = f"请开始为平台 '{platform}' 生成市场动态简报。"
    report = await query_react(
        agent_system_prompt=system_prompt, 
        query_str=user_prompt
    )
    if report:
        logger.success(f"Agent为 '{platform}' 完成了简报生成, 报告长度: {len(report)}。")
        return report
    else:
        error_msg = f"为平台 '{platform}' 生成市场动态简报时Agent调用失败或返回空。"
        logger.error(error_msg)
        return f"## {platform} 平台市场动态简报\n\n生成报告时出错: {error_msg}"



###############################################################################



assess_new_author_opportunity_system_prompt = """
# 任务背景与目标
你是一名专业的网络小说行业研究员, 专注于评估各平台对新人作者的友好度。你的任务是为平台[{platform}]生成一份详细的新人机会评估报告。

# 研究清单 (Checklist)
你必须通过"思考 -> 操作 -> 观察"的循环, 依次研究并回答以下所有问题, 以收集报告所需的全部信息。
1. 流量扶持: 在[{platform}]平台上, 新人作者能获得哪些流量扶持? 例如: 平台是否有明确的新书推荐位、新人流量池或"新书期"保护机制?
2. 变现门槛: 在[{platform}]平台上, 新人作者达到什么条件后可以开始获得收入(如广告分成、稿费)? 签约的难度如何?
3. 竞争环境: 在[{platform}]平台上, 新书榜的更新频率和上榜难度如何? 新人作品脱颖而出的竞争激烈程度如何?
4. 编辑支持: 在[{platform}]平台上, 新人作者获得编辑指导和反馈的普遍情况如何? (请搜索作者论坛(如龙的天空)、知乎等)

# 最终答案格式
当你通过工具收集完以上所有信息后, 你的最终`答案`必须是且只能是一份严格遵循以下结构的 Markdown 报告。
如果某个要点确实找不到信息, 请在该标题下明确指出"未找到相关信息"。

# 输出要求
```markdown
## {platform} 平台新人机会评估报告

- 综合评级: [高/中/低]
- 评级理由:
  - 流量机会: (总结平台对新书的流量支持情况, 例如: 有独立新书榜和算法推荐, 流量机会中等。)
  - 变现速度: (总结新人作者的变现路径和速度, 例如: 签约即有广告分成, 变现速度快。)
  - 竞争压力: (总结新人面临的竞争情况, 例如: 头部效应明显, 新书榜竞争激烈, 压力大。)
  - 编辑生态: (总结编辑对新人的支持情况, 例如: 编辑回复较慢, 主要靠作者自己摸索。)
- 核心建议: (给新人作者一句核心建议, 例如: 建议从平台重点扶持的XX题材切入, 利用好新书期流量。)
```
"""



@task(
    name="new_author_opportunity",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/common/new_author_opportunity_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_new_author_opportunity(platform: str) -> str:
    logger.info(f"为平台 '{platform}' 生成新人机会评估报告...")
    system_prompt = assess_new_author_opportunity_system_prompt.format(platform=platform)
    user_prompt = f"请开始为平台 '{platform}' 生成新人机会评估报告。"
    report = await query_react(
        agent_system_prompt=system_prompt, 
        query_str=user_prompt
    )
    if report:
        logger.success(f"Agent为 '{platform}' 完成了新人机会评估报告生成, 报告长度: {len(report)}。")
        return report
    else:
        error_msg = f"为平台 '{platform}' 生成新人机会评估报告时Agent调用失败或返回空。"
        logger.error(error_msg)
        return f"## {platform} 平台新人机会评估报告\n\n生成报告时出错: {error_msg}"



###############################################################################



@task(
    name="task_save_vector",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    cache_expiration=604800,
    retries=1,
    retry_delay_seconds=10,
)
def task_save_vector(content: Optional[str], content_format: str = "md", **kwargs) -> bool:
    if not content:
        logger.warning(f"内容为空, 跳过保存向量。元数据: kwargs={kwargs}")
        return False
    final_metadata = kwargs.copy()
    return vector_add(
        vector_store=get_market_vector_store(),
        content=content,
        metadata=final_metadata,
        content_format=content_format,
    )



###############################################################################



@task(
    name="save_markdown",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
def task_save_markdown(filename: str, content: Optional[str]) -> Optional[str]:
    if not content:
        logger.warning(f"内容为空, 跳过保存 Markdown 文件: {filename}")
        return None
    filename_md = f"{filename.replace(' ', '_')}.md"
    file_path_md = data_market_dir / "story" / filename_md
    file_path_md.write_text(content, encoding="utf-8")
    resolved_path = str(file_path_md.resolve())
    logger.success(f"Markdown 文件已保存至: {resolved_path}")
    return resolved_path
