import nest_asyncio
nest_asyncio.apply()
import os
import sys
from typing import Optional, Dict, List
import asyncio
import json
from pydantic import BaseModel, Field
from loguru import logger
from datetime import datetime
from llama_index.core import Document
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])
from market_analysis.story.base import query_react
from market_analysis.story.tasks import task_load_platform_profile, task_platform_briefing, task_new_author_opportunity, task_save_vector
from utils.llm import llm_completion, get_llm_params, get_llm_messages
from utils.prefect import local_storage, readable_json_serializer
from prefect import flow, task


class MarketOpportunity(BaseModel):
    rank: int = Field(description="机会排名, 1为最佳。")
    platform: str = Field(description="平台名称。")
    genre: str = Field(description="题材大类。")
    reasoning: str = Field(description="对该机会的综合评估理由, 解释其排名, 说明如何平衡动态机会、静态匹配和外部趋势。")
    actionable_advice: str = Field(description="具体的、可操作的创作建议, 例如可以结合的热门元素、切入角度或目标读者画像。")
    risk_assessment: str = Field(description="分析该机会的潜在风险, 例如市场饱和度、创作难度、政策风险等。")

class MarketAnalysisResult(BaseModel):
    summary: str = Field(description="对整个市场机会列表的一句话总结, 点明整体市场趋势或核心发现。")
    opportunities: List[MarketOpportunity] = Field(description="按综合潜力从高到低排序的市场机会列表。")

class ParsedGenres(BaseModel):
    genres: List[str] = Field(description="从报告中提取的热门题材列表。", default_factory=list)


ANALYZE_EXTERNAL_TRENDS_system_prompt = """
# 角色
你是一名敏锐的流行文化分析师, 擅长捕捉跨界趋势。

# 任务
为[{platform}]平台的[{genre}]题材, 生成一份外部热点趋势分析报告。你需要利用工具进行网络搜索, 并以简洁的Markdown格式输出。

# 工作流程
1.  研究: 使用 `social_media_trends_search` 等工具, 主动搜索与[{genre}]题材相关的外部趋势。
    - 跨界热点: 在B站、微博、抖音、小红书等社交媒体上, 搜索与该题材相关的“热门话题”、“流行文化”、“影视游戏IP”、“出圈meme”。(示例: `targeted_search(platforms=['B站', '抖音'], query='{genre} 热门话题')`)
    - 增长潜力: 搜索该题材关键词的近期热度指数(如百度指数、微信指数), 评估其大众化潜力。
    - 核心讨论: 浏览相关话题下的高赞评论和讨论, 了解大众对该题材的核心看法和期待。
2.  总结: 将你的发现综合成一份简明的Markdown报告。

# 输出要求
## [{platform}]-[{genre}]外部趋势分析报告

- 核心趋势: [总结1-2个最显著的外部流行趋势]
- 跨界潜力: [评估该题材与影视、游戏、短视频等领域结合的可能性和切入点]
- 增长预期: [基于搜索热度, 给出“高增长”、“稳定”、“热度下降”等判断]
- 关键洞察: [从大众讨论中提炼出的一个独特见解或创作建议]
"""


PARSE_GENRES_system_prompt = """
# 角色
你是一个精准的信息提取助手。

# 任务
从给定的市场动态简报中, 只提取“热门题材”部分列出的所有题材大类名称。

# 工作流程
1. 定位: 找到文本中的“### 1. 热门题材”部分。
2. 提取: 提取该部分 `- ` 开头的列表项中的题材名称。
3. 输出: 将提取的题材名称列表以指定的JSON格式输出。

# 输出要求
- 严格按照 Pydantic 模型的格式, 仅输出一个完整的、有效的 JSON 对象。
- 禁止在 JSON 前后添加任何额外解释、注释或 markdown 代码块。
- 如果报告中没有“热门题材”部分或该部分为空, 则输出一个空的 `genres` 列表。
"""


CHOOSE_BEST_OPPORTUNITY_system_prompt = """
# 角色
你是一位经验丰富的网文市场战略家。

# 任务
综合分析所有输入信息, 识别出多个具有商业潜力的“平台-题材”组合。对它们进行排序, 并以一个纯粹的、结构化的JSON对象格式输出你的完整分析报告。

# 决策维度
1.  动态机会 (广域扫描报告): 
    - 题材热度: `热门题材` 是否流行?
    - 官方动向: `官方动向` 是否与特定题材相关?
2.  新人机会 (新人机会评估报告):
    - 平台机会: `综合评级` 是高还是低?
3.  静态匹配 (平台基础信息报告):
    - 平台调性: 目标题材与平台的主流风格是否契合?
    - 商业模式: 平台的付费模式是否适合该题材的写作策略?
    - 读者画像: 目标题材的读者与平台的核心读者是否一致?
    - 内容限制: 题材是否触碰平台的内容红线?
    - 签约门槛: 平台对新人是否友好?
4.  外部趋势 (外部趋势分析报告):
    - 核心趋势: `核心趋势`是否与题材有结合点?
    - 跨界潜力: `跨界潜力`评级高吗?
    - 增长预期: `增长预期`是“高增长”吗?

# 工作流程
1.  分析: 仔细阅读所有输入材料, 包括市场动态、新人机会、平台信息和外部趋势报告。
2.  评估与排序: 对所有有潜力的“平台-题材”组合进行综合评估, 并按潜力从高到低进行排序。
3.  总结: 对所有机会进行整体评估, 给出一句总结性的陈述。
4.  输出: 将你的分析结果(一个包含总结和多个机会的有序列表)严格按照指定的JSON格式进行组织和输出。

# 输出要求
- 严格按照 Pydantic 模型的格式, 仅输出一个完整的、有效的 JSON 对象。
- 禁止在 JSON 前后添加任何额外解释、注释或 markdown 代码块。
- JSON 结构必须符合以下定义:
  - `summary` (string): 对整个市场机会列表的一句话总结, 点明整体市场趋势或核心发现。
  - `opportunities` (array of objects): 一个按综合潜力从高到低排序的机会列表。列表中的每个对象都应包含以下字段: 
    - `rank` (integer): 机会排名, 1为最佳。
    - `platform` (string): 平台名称。
    - `genre` (string): 题材大类。
    - `reasoning` (string): 对该机会的综合评估理由, 解释其排名, 说明如何平衡动态机会、静态匹配和外部趋势。
    - `actionable_advice` (string): 具体的、可操作的创作建议, 例如可以结合的热门元素、切入角度或目标读者画像。
    - `risk_assessment` (string): 分析该机会的潜在风险, 例如市场饱和度、创作难度、政策风险等。
"""


CHOOSE_OPPORTUNITY_user_prompt = """
# 综合信息
{all_reports}
"""


@task(name="analyze_external_trends",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/platform_subject/external_trends_{parameters[platform]}_{parameters[genre]}.json",
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_analyze_external_trends(platform: str, genre: str) -> tuple[str, str, str]:
    logger.info(f"为[{platform} - {genre}]分析外部趋势...")
    system_prompt = ANALYZE_EXTERNAL_TRENDS_system_prompt.format(platform=platform, genre=genre)
    user_prompt = f"请开始为[{platform}]平台的[{genre}]题材生成外部趋势分析报告。"
    report = await query_react(
        agent_system_prompt=system_prompt, query_str=user_prompt
    )
    if report:
        logger.success(f"为[{platform} - {genre}]完成了外部趋势分析。")
        return platform, genre, report
    else:
        error_msg = f"为[{platform} - {genre}]分析外部趋势时Agent调用失败或返回空。"
        logger.error(error_msg)
        return platform, genre, f"## [{platform}]-[{genre}]外部趋势分析报告\n\n生成报告时出错: {error_msg}"


@task(
    name="parse_genres_from_report", 
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_parse_genres_from_report(platform: str, report: str) -> tuple[str, List[str]]:
    logger.info(f"为平台 '{platform}' 的报告解析热门题材...")
    user_prompt = f"# 市场动态简报\n\n{report}"
    messages = get_llm_messages(system_prompt=PARSE_GENRES_system_prompt, user_prompt=user_prompt)
    llm_params = get_llm_params(llm_group='fast', messages=messages, temperature=0.0)
    response_message = await llm_completion(llm_params=llm_params, response_model=ParsedGenres)
    parsed_genres_result = response_message.validated_data
    if parsed_genres_result and parsed_genres_result.genres:
        logger.success(f"为平台 '{platform}' 解析出题材: {parsed_genres_result.genres}")
        return platform, parsed_genres_result.genres
    else:
        logger.warning(f"未能为平台 '{platform}' 的报告解析出任何题材。")
        return platform, []


@task(name="choose_best_opportunity",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_choose_best_opportunity(
    platform_reports: Dict[str, str], 
    platform_profiles: Dict[str, str], 
    new_author_reports: Dict[str, str], 
    external_trend_reports: Dict[str, str]
) -> Optional[MarketAnalysisResult]:
    logger.info("决策最佳市场机会...")
    full_context = ""
    for platform, report in platform_reports.items():
        full_context += f"\n\n---\n\n# {platform} 市场动态简报\n{report}"
        full_context += f"\n\n# {platform} 平台新人机会评估报告\n{new_author_reports.get(platform, '无新人机会评估报告')}"
        full_context += f"\n\n# {platform} 平台基础信息\n{platform_profiles.get(platform, '无基础信息')}"
    full_context += "\n\n---\n\n# 各题材外部趋势分析报告汇总\n"
    if external_trend_reports:
        for key, report in external_trend_reports.items():
            full_context += f"\n--- {key} ---\n{report}\n"
    else:
        full_context += "无外部趋势分析报告。"
    user_prompt = CHOOSE_OPPORTUNITY_user_prompt.format(all_reports=full_context)
    messages = get_llm_messages(system_prompt=CHOOSE_BEST_OPPORTUNITY_system_prompt, user_prompt=user_prompt)
    llm_params = get_llm_params(llm_group='reasoning', messages=messages, temperature=0.1)
    response_message = await llm_completion(llm_params=llm_params, response_model=MarketAnalysisResult)
    decision = response_message.validated_data
    if decision and decision.opportunities:
        top_choice = decision.opportunities[0]
        logger.success(f"决策完成, 并成功验证了输出格式。排名第一的机会: {top_choice.platform} - {top_choice.genre}")
    elif decision:
        logger.warning("决策完成, 但未返回任何市场机会。")
    return decision


@flow(name="platform_subject")
async def platform_subject(platforms_to_scan: list[str]):
    logger.info("加载平台档案...")
    profile_futures = task_load_platform_profile.map(platforms_to_scan)
    platform_profiles = {}
    for future in profile_futures:
        platform, content = await future.result()
        platform_profiles[platform] = content

    logger.info("启动广域扫描...")
    scan_futures = task_platform_briefing.map(platforms_to_scan)
    opportunity_futures = task_new_author_opportunity.map(platforms_to_scan)

    platform_reports: Dict[str, str] = {}
    new_author_reports = {}
    for i, platform_name in enumerate(platforms_to_scan):
        try:
            scan_future = scan_futures[i]
            report = await scan_future.result()
            platform_reports[platform_name] = report
            task_save_vector(
                content=report,
                doc_type="broad_scan_report",
                platform=platform_name,
                content_format="markdown"
            )
        except Exception as e:
            logger.error(f"扫描平台 '{platform_name}' 失败: {e}")
            platform_reports[platform_name] = f"## {platform_name} 平台市场动态简报\n\n生成报告时出错: {e}"
            continue

        opportunity_report = await opportunity_futures[i].result()
        new_author_reports[platform_name] = opportunity_report
        task_save_vector(
            content=opportunity_report,
            doc_type="new_author_opportunity_report",
            platform=platform_name,
            content_format="markdown"
        )

    logger.info("--- 广域扫描对比报告 ---")
    for platform, report in platform_reports.items():
        logger.info(f"\n--- {platform} ---\n{report}")
    
    logger.info("--- 新人机会评估对比报告 ---")
    for platform, report in new_author_reports.items():
        logger.info(f"\n--- {platform} ---\n{report}")

    logger.info("从市场动态简报中解析热门题材...")
    if not platform_reports:
        logger.error("没有可用的市场动态简报, 无法解析题材。")
        parse_genre_futures = []
    else:
        parse_genre_futures = task_parse_genres_from_report.map(
            platform=list(platform_reports.keys()),
            report=list(platform_reports.values())
        )

    platform_genre_pairs = []
    for future in parse_genre_futures:
        platform, genres = await future.result()
        for genre in genres:
            platform_genre_pairs.append((platform, genre))

    logger.info("分析各题材外部趋势...")
    if not platform_genre_pairs:
        logger.error("未能从任何平台报告中解析出热门题材, 无法进行外部趋势分析。将跳过此步骤。")
        external_trend_reports = {}
    else:
        trend_futures = task_analyze_external_trends.map(
            platform=[p for p, g in platform_genre_pairs],
            genre=[g for p, g in platform_genre_pairs]
        )
        external_trend_reports = {}
        for future in trend_futures:
            platform, genre, trend_report = await future.result()
            external_trend_reports[f"{platform}-{genre}"] = trend_report
            task_save_vector(
                content=trend_report,
                doc_type="external_trend_report",
                platform=platform,
                genre=genre,
                content_format="markdown"
            )

    logger.info("决策初步机会...")
    initial_decision = await task_choose_best_opportunity(platform_reports, platform_profiles, new_author_reports, external_trend_reports)
    if not initial_decision or not initial_decision.opportunities:
        logger.warning("未能从决策任务中获得有效结果, 工作流终止。")
        return

    task_save_vector(
        content=json.dumps(initial_decision.model_dump(), indent=2, ensure_ascii=False),
        doc_type="market_analysis_result",
        platform="summary",
        content_format="json"
    )

    logger.info("--- 初步市场机会决策报告 (JSON) ---")
    logger.info(f"\n{json.dumps(initial_decision.model_dump(), indent=2, ensure_ascii=False)}")



if __name__ == "__main__":
    platforms = ["番茄小说", "起点中文网"]
    flow_run_name = f"platform_subject_{datetime.now().strftime('%Y%m%d')}"
    # analysis_flow = platform_subject.with_options(name=flow_run_name)
    asyncio.run(platform_subject(platforms_to_scan=platforms))
