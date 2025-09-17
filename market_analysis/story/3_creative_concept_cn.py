import json
import os
import sys
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from loguru import logger
from datetime import datetime
from llama_index.core import Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])
from utils.file import data_market_dir
from utils.llm import call_react_agent, get_llm_messages, get_llm_params, llm_completion
from utils.vector import get_vector_query_engine, index_query, get_vector_store
from market_analysis.story.base import get_market_vector_store, get_market_tools, query_react
from market_analysis.story.tasks import task_platform_briefing, task_new_author_opportunity, task_load_platform_profile, task_save_vector
from utils.prefect_utils import local_storage, readable_json_serializer
from prefect import flow, task


class Candidate(BaseModel):
    platform: str
    genre: str

class FinalDecision(BaseModel):
    platform: str = Field(description="最终选择的平台")
    genre: str = Field(description="最终选择的题材")
    reasoning: str = Field(description="做出该选择的详细理由，并解释为什么它优于其他主要竞争选项。")

class RankedConcept(BaseModel):
    rank: int = Field(description="机会排名，1为最佳。")
    platform: str = Field(description="平台名称。")
    genre: str = Field(description="题材大类。")
    brief_reasoning: str = Field(description="对该选项排名的简要理由，点出其核心优劣势。")

class FinalDecisionResult(BaseModel):
    final_choice: FinalDecision = Field(description="根据综合排序最终选定的最佳方案。")
    ranking: List[RankedConcept] = Field(description="所有候选方案的完整排名列表。")


FINAL_DECISION_system_prompt_JSON = """
# 角色
你是一位顶尖的网文总编，拥有敏锐的市场嗅觉和战略眼光。

# 任务
综合评估N份“平台-题材”深度分析报告，选出最佳机会，并对所有机会进行排序。以指定的JSON格式输出决策和排名。

# 决策维度
1.  **潜力上限**: 题材受众、付费潜力、IP衍生可能性、蓝海机会。
2.  **成功概率**: 平台调性契合度、竞争激烈度、风险规避难度。
3.  **创新价值**: 差异化优势，避免红海竞争。
4.  **执行难度**: 创作门槛，对作者的友好度。

# 工作流程
1.  **评估排序**: 基于“决策维度”评估所有方案，并从高到低排序。
2.  **最终决策**: 确定排名第一的方案，并撰写决策理由，解释其为何优于其他竞争者。
3.  **格式化输出**: 严格按照Pydantic模型的JSON格式输出结果。

# 输出要求
- 严格按照 `FinalDecisionResult` Pydantic 模型的格式，仅输出一个完整的、有效的 JSON 对象。
- 禁止在 JSON 前后添加任何额外解释、注释或 markdown 代码块。
"""


DEEP_DIVE_system_prompt = """
# 角色
你是一名顶尖的网络小说市场分析师，专精于【{platform}】平台的【{genre}】题材。

# 任务
为【{platform}】平台的【{genre}】题材，生成一份深度洞察报告。

# 工作流程
1.  **信息整合**: 仔细阅读提供的上下文信息。
2.  **深度研究**:
    - **内部优先**: 研究时，优先使用 `story_market_vector` 工具查询内部知识库（平台档案、市场报告、小说创意）。
    - **外部补充**: 仅当内部信息不足时，才使用网络搜索工具获取最新动态。
    - **研究要点**: 围绕“输出结构”中的要点，研究题材套路、读者“毒点”、外部热点等。
3.  **撰写报告**: 综合所有信息，严格按照指定的Markdown格式输出。

# 上下文信息
---
## 平台基础信息 ({platform})
{platform_profile}
---
## 市场动态简报 ({platform})
{broad_scan_report}
---
## 平台新人机会评估报告 ({platform})
{opportunity_report}
---

# 输出结构 (Markdown)
## 【{platform}】平台 - 【{genre}】题材深度分析报告

### 1. 核心标签与流行元素
- **高频标签**: [总结3-5个最高频标签，基于作品和评论]
- **关键元素**: [描述标签的具体表现形式，如“签到流”、“神豪返现”]
- **趋势验证**: [分析标签的近期热度变化，如使用百度指数]

### 2. 核心爽点与读者心理
- **爽点一**: [描述最核心的爽点，如：扮猪吃虎后瞬间打脸]
  - **读者心理**: [分析该爽点满足的深层心理需求。可结合心理学理论，例如：满足读者对“代理复仇”和“恢复秩序”的渴望，与“公平世界信念”相关。]
- **爽点二**: [描述另一核心爽点，如：获得独一无二的金手指]
  - **读者心理**: [同上，进行深度心理学分析。例如：满足读者的“掌控感”和“自我效能感”需求，在不确定的现实中提供心理安全感。]

### 3. 关键付费点设计
- [分析该题材的付费章节设计逻辑。若是免费平台，则分析广告点位设计逻辑]

### 4. 新兴机会与蓝海方向
- **题材融合**: [提出有数据支撑的新颖题材融合方向]
- **跨界融合**: [结合外部热点，提出可融合的跨界创意]
- **设定创新**: [提出未被滥用的创新设定或金手指]
- **切入角度**: [建议新颖的主角身份或故事切入点]
- **作者友好度**: [结合新人机会报告，分析该方向对新人的友好度]

### 5. 主角人设迭代方向
- **流行人设分析**: [分析当前最受欢迎的1-2种主角人设及其核心魅力]
- **创新方向**: [提出对流行人设的反转或融合创新设计，创造差异化]

### 6. 常见“毒点”与风险规避
- **毒点一**: [总结一个读者普遍反感的情节或设定]
  - **规避建议**: [提出具体规避方法]
- **毒点二**: [总结另一个常见毒点]
  - **规避建议**: [提出对应规避方法]

### 7. 报告质量自我评估
- **数据驱动度 (1-5分)**: [报告基于搜索数据的程度]
- **洞察深刻度 (1-5分)**: [报告揭示深层趋势的程度]
- **可执行性 (1-5分)**: [报告建议的清晰度和可用性]
- **综合评价**: [总结报告优缺点]
"""

# 机会生成的提示词
OPPORTUNITY_GENERATION_system_prompt = """
# 角色
金牌小说策划人。

# 任务
根据输入信息，构思3个全新的、有商业潜力、且互相差异化的小说选题。

# 创作原则
- **机会导向**: 创意回应【新兴机会与蓝海方向】。
- **跨界优先**: 至少一个选题深度融合【跨界融合】建议。
- **灵感融合**: 融入报告中提到的外部热点、流行文化等元素。
- **风险规避**: 避开【常见“毒点”】。
- **爽点聚焦**: 围绕【核心爽点】构建。
- **避免重复**: 与【历史创意参考】显著区别。
- **强制差异化**: 3个选题需在核心卖点、切入角度、题材融合、目标读者等维度上存在显著差异，确保多样性。

# 输出结构 (Markdown)
- **选题名称**: [名称]
- **一句话卖点**: [宣传语]
- **核心创意**: [概括，明确指出融合的“蓝海方向”或“跨界热点”]
- **主角设定**: [身份, 特点, 独特性]
- **核心冲突**: [主要矛盾]
- **爆款潜力**: [S/A/B级]
- **潜力理由**: [市场契合度, 创意新颖度, 爽点强度, 跨界优势]
- **反套路指数**: [高/中/低]
- **指数理由**: [解释如何规避或创新套路]
- **写作难度**: [高/中/低]
- **难度理由**: [世界观, 角色, 情节, 资料]
"""


OPPORTUNITY_GENERATION_user_prompt = """
---
# 市场深度分析报告
{market_report}

---
# 历史创意参考
{historical_concepts}
"""


# 小说创意生成提示词
NOVEL_CONCEPT_system_prompt = """
# 角色
顶级小说策划人。

# 任务
从【初步选题列表】中选择最佳选题（优先跨界融合），并将其扩展为一份详细、创新的【小说创意】文档。

# 工作流程
1.  **选择选题**: 从列表中选择评级最高、最具潜力的选题，并说明选择理由。
2.  **模式挖掘与创新 (核心)**:
    - **解构范式**: 分析【历史成功案例参考】的底层成功模式（如节奏、反馈循环、爽点逻辑、心理满足），而不是模仿表面情节。
    - **重塑应用**: 将提炼出的成功范式，创造性地应用到新选题中，生成全新的设定。
3.  **深度研究 (工具使用)**:
    - **知识库挖掘**: 使用 `story_market_vector` 查询相关历史创意和报告，复用成功范式，避免失败模式。
    - **竞品分析**: 使用 `web_scraper` 等工具分析1-2个竞品，明确差异化策略（做得更好、不同，或开创新品类）。
    - **规避套路**: 使用 `forum_discussion_search` 在知乎、龙空等社区搜索并剔除过时套路和“毒点”。
    - **融合灵感**: 使用 `targeted_search` 在B站等平台搜索相关视觉、观点元素并融入创意 (例如: `targeted_search(platforms=['B站'], query='[题材名] 视觉灵感')`)。
4.  **撰写文档**: 严格按照“输出结构”撰写详细的小说创意文档。

# 输出结构 (Markdown)
选择的选题: [选题名称]
选择理由: [说明选择原因，特别是“跨界融合”的优势]

---

## 小说创意：[选题名称]

### 1. 一句话简介 (Logline)
- [30-50字，概括主角、目标、冲突、独特设定]

### 2. 详细故事梗概 (Synopsis)
- [200-300字，概述起因、发展、核心冲突、高潮]

### 3. 主角设定 (Character Profile)
- **背景与动机**: [出身, 职业, 内心渴望/恐惧]
- **性格与能力**: [性格特点, 行事风格, 核心能力/金手指及其限制]
- **成长弧光 (Character Arc)**:
  - **核心缺陷/谎言**: [主角开始时限制其成长的错误信念]
  - **欲望 (Want) vs. 需求 (Need)**:
    - **外在欲望**: [主角明确追求的外在目标]
    - **内在需求**: [主角真正需要、但未察觉的内在成长]
  - **转变路径**: [描述主角如何通过关键事件，从质疑到最终抛弃“谎言”，拥抱“内在需求”]
  - **最终状态**: [故事结束时主角的新信念和行为模式]

### 4. 核心冲突矩阵 (Core Conflict Matrix)
- **原则**: 所有冲突相互关联，外部冲突是内在冲突的映射。
- **根本性冲突 (主题)**: [贯穿始终的哲学/价值观冲突，如: 自由 vs. 安全]
- **主线情节冲突 (外部)**: [具体目标 vs. 强大的“黑镜”式对手，以及不断升级的赌注]
- **主角内在冲突 (内部)**: [“欲望”与“需求”的矛盾，以及艰难抉择]
- **核心关系冲突 (人际)**: [与核心配角（盟友/爱人/导师）的冲突，考验主角成长]

### 5. 世界观核心设定 (World-building)
- **核心概念**: [世界观基石的“What if”问题]
- **独特法则**: [1-2条与现实相悖的物理/社会法则及其影响]
- **标志性元素**: [2-3个独特的地理、生物、组织或技术]
- **历史谜团与探索感**: [贯穿始终的古老谜团及主角的探索路径]

### 6. 升级体系与核心设定 (Progression System & Core Setting)
- **创新原则**: [规避套路，与世界观深度绑定，体系本身成为冲突来源]
- **体系核心概念**: [用一个独特的比喻描述体系本质]
- **核心资源与获取**: [非传统的“经验值”及其获取方式、风险、道德困境]
- **晋升路径与质变**: [非线性的成长路径，关键阶段的能力质变，而非简单数值提升]
- **体系的内在矛盾与社会影响**: [体系本身的悖论如何塑造社会结构与冲突]

### 7. 关键配角设定 (Key Supporting Characters)
- **原则**: [避免工具人，配角需有独立目标、内在矛盾和秘密]
- **配角一/二**:
  - **定位与功能**: [导师/对手/盟友等]
  - **独立人生**: [个人目标, 内在矛盾, 秘密]
  - **与主角的动态关系**: [关系如何演变，核心互动模式]

### 8. 核心爽点与高光场景 (Core Appeal & Highlight Scenes)
- **主爽点**: [最核心、最高频的爽点类型，并提供高光场景示例（简述、情绪顶点、关键画面/台词）]
- **辅爽点**: [1-2个调剂节奏的辅助爽点及场景示例]
- **其他核心卖点**: [创新设定, 极致情绪, 反套路, 世界观探索感等]

### 9. 开篇章节构思 (Opening Chapter Idea)
- **“黄金三章”设计**:
- **第一章 (破局)**: [引入主角与困境，制造悬念钩子]
- **第二章 (展开)**: [使用金手指，引出更大冲突，展现爽点]
- **第三章 (确立)**: [初步胜利，确立短期目标，制造强烈爽感]

### 10. 市场风险评估 (Market Risk Assessment)
- **核心风险**: [识别1-2个主要风险，如创意同质化、设定复杂、慢热]
- **风险分析**: [分析风险来源，可对标参考案例]
- **规避策略**: [提出具体的规避建议]
"""


NOVEL_CONCEPT_user_prompt = """
---
# 初步选题列表
{selected_opportunity}

# 历史成功案例参考
{historical_success_cases}
"""


@task(name="final_decision",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_final_decision(reports: List[Dict[str, str]]) -> FinalDecisionResult:
    logger.info("在多个深度分析报告中进行最终决策...")
    if not reports:
        raise ValueError("task_final_decision需要至少1份报告进行评估。")
    user_prompt_parts = ["请对以下深度分析报告进行评估、排序，并选出最佳方案："]
    for i, report_data in enumerate(reports):
        user_prompt_parts.append(
            f"\n---\n\n# 候选方案 {i+1}: 【{report_data['platform']}】 - 【{report_data['genre']}】\n{report_data['report']}"
        )
    user_prompt = "".join(user_prompt_parts)
    messages = get_llm_messages(system_prompt=FINAL_DECISION_system_prompt_JSON, user_prompt=user_prompt)
    llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=0.1)
    response_message = await llm_completion(llm_params=llm_params, response_model=FinalDecisionResult)
    decision = response_message.validated_data
    if not decision:
        logger.error("最终决策失败，LLM未返回有效结果。")
        raise ValueError("最终决策失败。")
    logger.success(f"最终决策完成。选择: 【{decision.final_choice.platform}】 - 【{decision.final_choice.genre}】")
    return decision


@task(name="deep_dive_analysis",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_deep_dive_analysis(platform: str, genre: str, platform_profile: str, broad_scan_report: str, opportunity_report: str) -> Optional[str]:
    logger.info(f"对【{platform} - {genre}】启动深度分析...")
    system_prompt = DEEP_DIVE_system_prompt.format(
        platform=platform,
        genre=genre,
        platform_profile=platform_profile,
        broad_scan_report=broad_scan_report,
        opportunity_report=opportunity_report
    )
    user_prompt = f"请开始为【{platform}】平台的【{genre}】题材生成深度分析报告。"
    report = await query_react(
        agent_system_prompt=system_prompt, query_str=user_prompt
    )
    if not report:
        logger.error(f"为【{platform} - {genre}】生成深度分析报告失败。")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{platform.replace(' ', '_')}_{genre.replace(' ', '_')}_{timestamp}_deep_dive.md"
    file_path = data_market_dir / file_name
    file_path.write_text(report, encoding="utf-8")
    logger.success(f"报告已保存为Markdown文件: {file_path}")

    logger.success("深度分析完成！")
    return report


@task(name="generate_opportunities",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_generate_opportunities(market_report: str, genre: str) -> Optional[str]:
    logger.info("启动创意脑暴，生成小说选题...")
    logger.info(f"正在查询【{genre}】相关的历史创意库，避免重复...")
    
    query_engine = get_vector_query_engine(
        vector_store=get_market_vector_store(),
        filters=MetadataFilters(filters=[ExactMatchFilter(key="type", value="novel_concept")]),
        similarity_top_k=10,
        rerank_top_n=5,
    )
    historical_concepts_contents = await index_query(
        query_engine=query_engine,
        questions=[f"{genre} 小说核心创意"],
    )

    if historical_concepts_contents:
        historical_concepts_str = "\n\n---\n\n".join(historical_concepts_contents)
        logger.success(f"查询到 {len(historical_concepts_contents)} 份历史创意，将用于规避重复。")
    else:
        historical_concepts_str = "无相关历史创意可供参考。"
        logger.info("无相关历史创意可供参考。")

    user_prompt = OPPORTUNITY_GENERATION_user_prompt.format(
            market_report=market_report,
            historical_concepts=historical_concepts_str
    )
    messages = get_llm_messages(system_prompt=OPPORTUNITY_GENERATION_system_prompt, user_prompt=user_prompt)
    llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=0.5)
    response_message = await llm_completion(llm_params=llm_params)
    opportunities = response_message.content
    if opportunities:
        logger.success("小说选题生成完毕！")
    else:
        logger.error("生成小说选题失败。")
    return opportunities


@task(name="generate_novel_concept",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
async def task_generate_novel_concept(opportunities_report: str, platform: str, genre: str) -> Optional[str]:
    logger.info("深化选题，生成详细小说创意...")
    logger.info(f"正在查询【{platform} - {genre}】相关的历史成功案例...")

    query_engine = get_vector_query_engine(
        vector_store=get_market_vector_store(),
        filters=MetadataFilters(filters=[
            ExactMatchFilter(key="type", value="novel_concept"),
            ExactMatchFilter(key="platform", value=platform)
        ]),
        similarity_top_k=5,
        rerank_top_n=3,
    )
    historical_success_contents = await index_query(
        query_engine=query_engine,
        questions=[f"{platform} {genre} 爆款成功小说创意案例"],
    )

    if historical_success_contents:
        historical_success_cases_str = "\n\n---\n\n".join(historical_success_contents)
        logger.success(f"查询到 {len(historical_success_contents)} 份成功案例，将用于借鉴。")
    else:
        historical_success_cases_str = "无相关历史成功案例可供参考。"
        logger.info("无相关历史成功案例可供参考。")

    user_prompt = NOVEL_CONCEPT_user_prompt.format(
            selected_opportunity=opportunities_report,
            historical_success_cases=historical_success_cases_str,
    )
    concept = await call_react_agent(
        system_prompt=NOVEL_CONCEPT_system_prompt,
        user_prompt=user_prompt,
        llm_type='reasoning',
        tools=get_market_tools(),
        temperature=0.7
    )
    if not concept:
        logger.error("生成详细小说创意失败。")
        return None
    return concept


@task(
    name="save_markdown",
    persist_result=True,
    result_storage=local_storage,
    result_serializer=readable_json_serializer,
    retries=2,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
def task_save_markdown(platform: str, genre: str, deep_dive_report: str, final_opportunities: str, detailed_concept: str) -> bool:
    logger.info(f"生成【{platform} - {genre}】的汇总 Markdown 文件...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{platform.replace(' ', '_')}_{genre.replace(' ', '_')}_{timestamp}_summary.md"
    file_path = data_market_dir / file_name

    summary_content = f"""
# 【{platform}】平台 - 【{genre}】创意总结报告

## 深度分析报告
{deep_dive_report}

## 小说选题建议
{final_opportunities}

## 详细小说创意
{detailed_concept}
"""
    file_path.write_text(summary_content, encoding="utf-8")
    logger.success(f"已生成汇总 Markdown 文件: {file_path}")
    return True


@flow(name="creative_concept_flow")
async def creative_concept(candidates_to_explore: List[Candidate]):
    if not candidates_to_explore:
        logger.error("未提供任何需要探索的方案，工作流终止。")
        return

    logger.info(f"收到手动指定的 {len(candidates_to_explore)} 个方案，开始第二部分流程: {candidates_to_explore}")
    
    # 注意：下面的任务（task_load_platform_profile, task_platform_briefing, task_new_author_opportunity）
    # 与第一阶段（platform_subject.py）中的任务相同。
    # Prefect的缓存机制（基于result_storage_key）将确保这些任务不会重复执行，而是直接从缓存加载结果，
    # 从而高效地为本流程提供必要的上下文信息。
    platforms_to_process = list(set([c.platform for c in candidates_to_explore]))

    # 重新获取上下文信息 (平台档案、广域扫描报告、新人机会报告)
    # 由于已统一缓存键(result_storage_key)，Prefect的缓存机制将避免重复执行已在第一阶段完成的任务
    profile_futures = task_load_platform_profile.map(platforms_to_process)
    platform_profiles: Dict[str, str] = {}
    for future in profile_futures:
        platform, content = await future.result()
        platform_profiles[platform] = content

    scan_futures = task_platform_briefing.map(platforms_to_process)
    opportunity_futures = task_new_author_opportunity.map(platforms_to_process)

    platform_reports = {}
    opportunity_reports = {}
    for i, platform_name in enumerate(platforms_to_process):
        try:
            report = await scan_futures[i].result()
            platform_reports[platform_name] = report
        except Exception as e:
            logger.error(f"为后续流程获取平台 '{platform_name}' 扫描报告失败: {e}")
            platform_reports[platform_name] = f"## {platform_name} 平台市场动态简报\n\n生成报告时出错: {e}"
        
        try:
            opportunity_report = await opportunity_futures[i].result()
            opportunity_reports[platform_name] = opportunity_report
        except Exception as e:
            logger.error(f"为后续流程获取平台 '{platform_name}' 新人机会报告失败: {e}")
            opportunity_reports[platform_name] = f"## {platform_name} 平台新人机会评估报告\n\n生成报告时出错: {e}"

    # 并行深度钻取
    deep_dive_futures = []
    for candidate in candidates_to_explore:
        future = await task_deep_dive_analysis.submit(
            platform=candidate.platform,
            genre=candidate.genre,
            platform_profile=platform_profiles.get(candidate.platform, "无基础信息"),
            broad_scan_report=platform_reports.get(candidate.platform, "无动态简报"),
            opportunity_report=opportunity_reports.get(candidate.platform, "无新人机会评估报告")
        )
        deep_dive_futures.append((candidate, future))

    deep_dive_reports = []
    for candidate, future in deep_dive_futures:
        try:
            report_content = await future.result()
            if report_content:
                task_save_vector(
                    content=report_content,
                    doc_type="deep_dive_report",
                    platform=candidate.platform,
                    genre=candidate.genre,
                    content_format="markdown"
                )
                deep_dive_reports.append(
                    {
                    "platform": candidate.platform,
                    "genre": candidate.genre,
                    "report": report_content
                    }
                )
                logger.success(f"完成【{candidate.platform} - {candidate.genre}】的深度分析。")
            else:
                logger.error(f"【{candidate.platform} - {candidate.genre}】的深度分析返回空，将忽略此方案。")
        except Exception as e:
            logger.error(f"【{candidate.platform} - {candidate.genre}】的深度分析失败: {e}，将忽略此方案。")

    if not deep_dive_reports:
        logger.error("所有方案的深度分析均失败，工作流终止。")
        return

    # 最终决策
    if len(deep_dive_reports) == 1:
        final_choice_data = deep_dive_reports[0]
        logger.info("只有一个方案成功完成深度分析，直接采纳。")

        # 为保持数据结构一致性，同样创建 FinalDecisionResult 对象
        final_choice_obj = FinalDecision(
            platform=final_choice_data["platform"],
            genre=final_choice_data["genre"],
            reasoning="只有一个方案成功完成深度分析，因此被直接采纳。"
        )
        ranked_concept = RankedConcept(
            rank=1,
            platform=final_choice_data["platform"],
            genre=final_choice_data["genre"],
            brief_reasoning="唯一的成功方案，直接采纳。"
        )
        final_decision_result = FinalDecisionResult(
            final_choice=final_choice_obj,
            ranking=[ranked_concept]
        )

        logger.info("--- 最终市场方向决策 ---")
        logger.info(f"选择: 【{final_choice_obj.platform}】 - 【{final_choice_obj.genre}】")
        logger.info(f"理由: {final_choice_obj.reasoning}")

        task_save_vector(
            content=final_decision_result.model_dump_json(indent=2, ensure_ascii=False),
            doc_type="final_decision_report",
            platform=final_choice_data["platform"],
            genre=final_choice_data["genre"],
            content_format="json"
        )
    else:
        final_decision_result = await task_final_decision(deep_dive_reports)
        final_choice_obj = final_decision_result.final_choice
        final_choice_data = {
            "platform": final_choice_obj.platform,
            "genre": final_choice_obj.genre,
            "report": next((r['report'] for r in deep_dive_reports if r['platform'] == final_choice_obj.platform and r['genre'] == final_choice_obj.genre), None)
        }
        logger.info("--- 最终市场方向决策 ---")
        logger.info(f"选择: 【{final_choice_obj.platform}】 - 【{final_choice_obj.genre}】")
        logger.info(f"理由: {final_choice_obj.reasoning}")
        logger.info(f"完整排名:\n{json.dumps([r.model_dump() for r in final_decision_result.ranking], indent=2, ensure_ascii=False)}")
        task_save_vector(
            content=final_decision_result.model_dump_json(indent=2, ensure_ascii=False),
            doc_type="final_decision_report",
            platform=final_choice_obj.platform,
            genre=final_choice_obj.genre,
            content_format="json"
        )

    chosen_platform = final_choice_data["platform"]
    chosen_genre = final_choice_data["genre"]
    deep_dive_report = final_choice_data["report"]

    if not deep_dive_report:
        logger.error(f"最终选择的方案【{chosen_platform} - {chosen_genre}】没有有效的深度分析报告，工作流终止。")
        return

    # 后续流程 (机会生成, 创意深化)
    logger.info("--- 深度分析报告 (最终选定) ---")
    logger.info(f"\n{deep_dive_report}")

    final_opportunities = await task_generate_opportunities(market_report=deep_dive_report, genre=chosen_genre)
    if not final_opportunities:
        logger.error(f"生成小说选题失败，工作流终止。")
        return

    task_save_vector(
        content=final_opportunities,
        doc_type="opportunity_generation_report",
        platform=chosen_platform,
        genre=chosen_genre,
        content_format="markdown"
    )

    logger.info("--- 小说选题建议 ---")
    logger.info(f"\n{final_opportunities}")

    detailed_concept = await task_generate_novel_concept(opportunities_report=final_opportunities, platform=chosen_platform, genre=chosen_genre)
    if not detailed_concept:
        logger.error(f"深化小说创意失败，工作流终止。")
        return

    task_save_vector(
        content=detailed_concept,
        doc_type="novel_concept",
        platform=chosen_platform,
        genre=chosen_genre,
        content_format="markdown"
    )

    logger.info("--- 详细小说创意 ---")
    logger.info(f"\n{detailed_concept}")

    task_save_markdown(chosen_platform, chosen_genre, deep_dive_report, final_opportunities, detailed_concept)


if __name__ == "__main__":
    # ==================================================================
    # 用户手动配置区域
    # 请根据第一部分流程的输出，在这里填入你想要深度探索的方案
    # 格式: [Candidate(platform="平台名", genre="题材名"), ...]
    # ==================================================================
    candidates_to_run = [
        Candidate(platform="番茄小说", genre="都市脑洞"),
        Candidate(platform="起点中文网", genre="东方玄幻"),
    ]
    # ==================================================================

    flow_run_name = f"creative_concept-{datetime.now().strftime('%Y%m%d')}"
    # analysis_flow = creative_concept.with_options(name=flow_run_name)
    asyncio.run(creative_concept(candidates_to_explore=candidates_to_run))
