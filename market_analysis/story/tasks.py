import os
import sys
from typing import Any, Optional, Tuple
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from market_analysis.story.base import (
    ASSESS_NEW_AUTHOR_OPPORTUNITY_system_prompt,
    BROAD_SCAN_system_prompt,
    get_market_vector_store,
    query_react,
)
from utils.vector import vector_add, get_vector_query_engine, index_query
from utils.file import data_market_dir
from utils.prefect import local_storage, readable_json_serializer
from prefect import task


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
    contents = await index_query(
        query_engine=query_engine,
        questions=[f"{platform} 平台档案"]
    )

    if contents:
        profile_content = contents[0]
        logger.success(f"已加载 '{platform}' 的基础信息。")
    else:
        logger.warning(
            f"在向量库中未找到 '{platform}' 的基础信息。建议先运行 `story_platform_by_search.py`。"
        )
    return platform, profile_content


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
    system_prompt = BROAD_SCAN_system_prompt.format(platform=platform)
    user_prompt = f"请开始为平台 '{platform}' 生成市场动态简报。"
    report = await query_react(
        agent_system_prompt=system_prompt, query_str=user_prompt
    )
    if report:
        logger.success(f"Agent为 '{platform}' 完成了简报生成, 报告长度: {len(report)}。")
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
    cache_expiration=604800,
)
async def task_new_author_opportunity(platform: str) -> str:
    logger.info(f"为平台 '{platform}' 生成新人机会评估报告...")

    system_prompt = ASSESS_NEW_AUTHOR_OPPORTUNITY_system_prompt.format(platform=platform)
    user_prompt = f"请开始为平台 '{platform}' 生成新人机会评估报告。"

    report = await query_react(
        agent_system_prompt=system_prompt, query_str=user_prompt
    )

    if report:
        logger.success(f"Agent为 '{platform}' 完成了新人机会评估报告生成, 报告长度: {len(report)}。")
        return report
    else:
        error_msg = f"为平台 '{platform}' 生成新人机会评估报告时Agent调用失败或返回空。"
        logger.error(error_msg)
        return f"## {platform} 平台新人机会评估报告\n\n生成报告时出错: {error_msg}"


@task(
    name="task_save_vector",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    cache_expiration=604800,
    retries=1,
    retry_delay_seconds=10,
)
def task_save_vector(content: Optional[str], doc_type: str, content_format: str = "text", **metadata: Any) -> bool:
    if not content:
        logger.warning(f"内容为空, 跳过保存向量。元数据: doc_type={doc_type}, metadata={metadata}")
        return False

    final_metadata = metadata.copy()
    final_metadata["type"] = doc_type
    return vector_add(
        vector_store =get_market_vector_store(),
        content=content,
        metadata=final_metadata,
        content_format=content_format,
    )


@task(
    name="task_save_md",
    retries=1,
    retry_delay_seconds=10,
)
def task_save_md(filename: str, content: Optional[str]) -> Optional[str]:
    if not content:
        return None
    filename_md = f"{filename.replace(' ', '_')}.md"
    file_path_md = data_market_dir / "story" / filename_md
    file_path_md.write_text(content, encoding="utf-8")
    return str(file_path_md.resolve())
