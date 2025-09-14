import os
import json
import asyncio
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_community.embeddings.litellm import LiteLLMEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from utils.llm import get_llm_params, get_embedding_params, get_llm_messages, llm_acompletion


load_dotenv()
embedding_params = get_embedding_params(embedding='bge-m3')
embeddings = LiteLLMEmbeddings(**embedding_params)
search_tool = TavilySearchResults(max_results=5)
chroma_db_path = ".chroma_db/story"
Path(chroma_db_path).mkdir(parents=True, exist_ok=True)
vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)


class BroadScanReport(BaseModel):
    platform: str = Field(..., description="平台名称")
    main_tone: str = Field(..., description="一句话总结平台主流风格")
    hot_genres: List[str] = Field(..., description="列出3-5个当前最热门的题材大类")
    official_direction: str = Field(..., description="总结近期的官方征文、激励活动方向，如果没有则为'无'")
    opportunity_assessment: str = Field(..., description="对当前在该平台发展的机会进行一句话评估")

class BestOpportunity(BaseModel):
    platform: str = Field(..., description="选择的平台名称")
    genre: str = Field(..., description="建议从该平台入手的题材大类")
    reason: str = Field(..., description="做出此选择的理由")


# 广域扫描的提示词
BROAD_SCAN_SYSTEM_PROMPT = """
# 角色
你是一位宏观市场策略师，专精于网络小说平台分析。

# 任务
根据提供的【平台名称】和【实时搜索数据】，分析该平台的整体市场环境。

# 输出要求
严格按照Pydantic模型的JSON格式输出。
"""
BROAD_SCAN_USER_PROMPT = """
# 平台名称
{platform}

# 实时搜索数据
{search_results}
"""

# 决策最佳机会的提示词
CHOOSE_BEST_OPPORTUNITY_SYSTEM_PROMPT = """
# 角色
你是一位顶尖的网文市场战略家，拥有敏锐的商业嗅觉。

# 任务
根据提供的【广域扫描对比报告】，分析并选择一个最具潜力的平台和题材方向进行深度钻取。
你的决策应基于报告中的“热门题材”和“机会评估”。

# 输出要求
严格按照Pydantic模型的JSON格式输出。
"""
CHOOSE_BEST_OPPORTUNITY_USER_PROMPT = """
# 广域扫描对比报告
{platform_reports}
"""


# 深度钻取的提示词 (与原版类似，但更聚焦)
DEEP_DIVE_SYSTEM_PROMPT = """
# 角色
你是一位顶尖的网络小说市场分析师，对【{platform}】平台的【{genre}】题材有深入研究。

# 任务
根据提供的【实时搜索数据】（包含热门作品、评论摘要），产出一份关于该细分市场的深度洞察报告。

# 输出要求 (Markdown格式)
## 【{platform}】平台 - 【{genre}】题材深度分析报告

### 1. 核心标签与元素
- [提炼3-5个最关键的标签]

### 2. 核心爽点与读者心理
- [提炼2-3个最受追捧的爽点，并分析背后心理]

### 3. 新兴机会与蓝海方向
- [发现数据中暗示的、尚未饱和的题材组合或创新方向]

### 4. 常见“毒点”与风险规避
- [总结读者最反感的3个情节或设定]
"""
DEEP_DIVE_USER_PROMPT = """
# 实时搜索数据
{search_results}
"""

# 机会生成的提示词
OPPORTUNITY_GENERATION_SYSTEM_PROMPT = """
# 角色
你是一位经验丰富的金牌小说策划人。

# 任务
根据我提供的【市场深度分析报告】，激发创意，构思 3 个具有商业潜力的小说选题。

# 输出要求 (Markdown格式)
为每个选题提供以下信息：
- **选题名称**: [一个吸引人的名字]
- **核心创意**: [一句话概括故事最有趣的点]
- **主角设定**: [简要描述主角的身份和特点]
- **核心冲突**: [故事的主要矛盾是什么]
- **市场评级**: [S/A/B级]
- **评级理由**: [结合分析报告，解释为什么这么评级]

"""
OPPORTUNITY_GENERATION_USER_PROMPT = """
---
# 市场深度分析报告
{market_report}
"""

# 小说创意生成提示词
NOVEL_CONCEPT_SYSTEM_PROMPT = """
# 角色
你是一位顶级小说编辑和策划人，擅长将一个好的点子扩展成一个完整且吸引人的故事概念。

# 任务
根据我提供的【初步选题列表】，请选择其中市场评级最高的那个选题，并将其扩展为一个更详细的【小说创意】。

# 输出要求 (Markdown格式)
## 小说创意：[选题名称]

### 1. 一句话简介 (Logline)
- [用一句话概括故事的核心卖点，使其听起来非常吸引人]

### 2. 详细故事梗概 (Synopsis)
- [用200-300字详细描述故事的起因、发展和核心冲突。主角如何获得能力/机遇，他面临的主要挑战是什么，故事的高潮可能是什么样的？]

### 3. 主角设定 (Character Profile)
- **背景与动机**: [主角的出身、职业、以及他内心最深的渴望或恐惧是什么？]
- **性格与能力**: [主角的性格特点（例如：杀伐果断、苟道至上、幽默腹黑），以及他的核心能力/金手指的具体设定]
- **成长弧光**: [在故事的最后，主角会在思想或能力上获得怎样的成长？]

### 4. 核心看点与卖点 (Key Selling Points)
- [列出3-4个能吸引目标读者的关键元素，例如：创新的系统设定、极致的情绪反转、新颖的世界观、独特的爽点节奏等]

### 5. 开篇章节构思 (Opening Chapter Idea)
- [设计一个抓人眼球的开篇。第一章应该发生什么事？如何快速展现主角的特点、引入核心设定，并留下悬念？]

"""
NOVEL_CONCEPT_USER_PROMPT = """
---
# 初步选题列表
{selected_opportunity}
"""


async def broad_scan_platform(platform: str) -> dict:
    print(f"  - 正在扫描平台: {platform}")
    query = f"{platform}小说热门榜单、新书榜、官方征文活动"
    # TavilySearchResults._arun 返回的是一个字符串
    search_results = await search_tool.ainvoke(query)
    
    messages = get_llm_messages(
        SYSTEM_PROMPT=BROAD_SCAN_SYSTEM_PROMPT,
        USER_PROMPT=BROAD_SCAN_USER_PROMPT,
        context_dict_user={"platform": platform, "search_results": search_results}
    )
    llm_params = get_llm_params(llm='reasoning', temperature=0.3, messages=messages)
    
    try:
        response_message = await llm_acompletion(llm_params, response_model=BroadScanReport)
        return response_message.validated_data.model_dump()
    except Exception as e:
        print(f"  - 警告: {platform} 平台分析报告生成或解析失败: {e}")
        return {"error": f"Failed to get or parse LLM output: {e}", "platform": platform}

async def broad_scan(platforms: list[str]) -> dict:
    print("🚀 启动广域扫描...")
    tasks = [broad_scan_platform(p) for p in platforms]
    results = await asyncio.gather(*tasks)
    
    scan_results = {}
    for res in results:
        # 使用 res.get("platform", "unknown") 来处理错误情况
        platform_name = res.get("platform", "unknown")
        scan_results[platform_name] = res
            
    print("✅ 广域扫描完成！")
    return scan_results

async def choose_best_opportunity(platform_reports: dict) -> Optional[dict]:
    print("\n🚀 决策最佳市场机会...")
    reports_str = json.dumps(platform_reports, indent=2, ensure_ascii=False)
    
    messages = get_llm_messages(
        SYSTEM_PROMPT=CHOOSE_BEST_OPPORTUNITY_SYSTEM_PROMPT,
        USER_PROMPT=CHOOSE_BEST_OPPORTUNITY_USER_PROMPT,
        context_dict_user={"platform_reports": reports_str}
    )
    llm_params = get_llm_params(llm='reasoning', temperature=0.3, messages=messages)

    try:
        response_message = await llm_acompletion(llm_params, response_model=BestOpportunity)
        choice = response_message.validated_data.model_dump()
        print(f"  - ✅ LLM决策完成：选择平台 '{choice['platform']}'，题材 '{choice['genre']}'。")
        print(f"  - 理由: {choice['reason']}")
        return choice
    except Exception as e:
        print(f"  - 警告: 解析LLM决策结果失败: {e}")
        return None

async def deep_dive_analysis(platform: str, genre: str) -> str:
    print(f"\n🚀 对【{platform} - {genre}】启动深度钻取...")
    query = f"{platform}小说 {genre} 题材热门作品、读者评论、故事大纲"
    search_results = await search_tool.ainvoke(query)
    
    messages = get_llm_messages(
        SYSTEM_PROMPT=DEEP_DIVE_SYSTEM_PROMPT,
        USER_PROMPT=DEEP_DIVE_USER_PROMPT,
        context_dict_system={"platform": platform, "genre": genre},
        context_dict_user={"search_results": search_results}
    )
    llm_params = get_llm_params(llm='reasoning', temperature=0.3, messages=messages)
    response_message = await llm_acompletion(llm_params)
    report = response_message.content
    
    # 存入向量数据库
    report_doc = Document(page_content=report, metadata={"platform": platform, "genre": genre})
    await vector_store.aadd_documents([report_doc])
    print("  - ✅ 报告已存入向量数据库。")
    
    # 保存为Markdown文件
    report_dir = Path(".market/story/")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{platform.replace(' ', '_')}_{genre.replace(' ', '_')}_{timestamp}.md" # noqa
    file_path = report_dir / file_name
    await asyncio.to_thread(file_path.write_text, report, encoding="utf-8")
    print(f"  - ✅ 报告已保存为Markdown文件: {file_path}")
    
    print("✅ 深度分析完成！")
    return report

async def generate_opportunities(market_report: str) -> str:
    print("\n🚀 启动创意脑暴，生成小说选题...")
    messages = get_llm_messages(
        SYSTEM_PROMPT=OPPORTUNITY_GENERATION_SYSTEM_PROMPT,
        USER_PROMPT=OPPORTUNITY_GENERATION_USER_PROMPT,
        context_dict_user={"market_report": market_report}
    )
    llm_params = get_llm_params(llm='reasoning', temperature=0.3, messages=messages)
    response_message = await llm_acompletion(llm_params)
    opportunities = response_message.content
    print("✅ 小说选题生成完毕！")
    return opportunities

async def generate_novel_concept(selected_opportunity: str) -> str:
    print("\n🚀 深化选题，生成详细小说创意...")
    messages = get_llm_messages(
        SYSTEM_PROMPT=NOVEL_CONCEPT_SYSTEM_PROMPT,
        USER_PROMPT=NOVEL_CONCEPT_USER_PROMPT,
        context_dict_user={"selected_opportunity": selected_opportunity}
    )
    llm_params = get_llm_params(llm='reasoning', temperature=0.3, messages=messages)
    response_message = await llm_acompletion(llm_params)
    concept = response_message.content
    print("✅ 详细小说创意生成完毕！")
    return concept

async def query_reports(query: str, n_results: int = 1) -> list[Document]:
    print(f"\n🔍 正在查询题材库，问题: '{query}'")
    results = await vector_store.asimilarity_search(query, k=n_results)
    print(f"✅ 查询到 {len(results)} 份相关报告。")
    return results


async def main():
    # 广域扫描
    platforms_to_scan = ["番茄小说", "起点中文网"]
    platform_reports = await broad_scan(platforms_to_scan)
    print("\n--- 广域扫描对比报告 ---")
    print(json.dumps(platform_reports, indent=2, ensure_ascii=False))

    # 由LLM决策最佳机会
    best_opportunity = await choose_best_opportunity(platform_reports)
    if best_opportunity:
        chosen_platform = best_opportunity["platform"]
        chosen_genre = best_opportunity["genre"]

        # 深度钻取
        deep_dive_report = await deep_dive_analysis(chosen_platform, chosen_genre)
        print("\n--- 深度分析报告 ---")
        print(deep_dive_report)

        # 机会生成
        final_opportunities = await generate_opportunities(deep_dive_report)
        print("\n--- 小说选题建议 ---")
        print(final_opportunities)

        # 深化创意
        # 让AI选择上一步生成的选题列表中的最优解进行深化
        if final_opportunities:
            detailed_concept = await generate_novel_concept(final_opportunities)
            print("\n--- 详细小说创意 ---")
            print(detailed_concept)
    else:
        print("\n未能从广域扫描中确定最佳机会，工作流终止。")

if __name__ == "__main__":
    asyncio.run(main())
