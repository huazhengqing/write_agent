import os
import sys
from typing import List, Optional
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])
from utils.llm import call_agent
from utils.prefect_utils import local_storage, readable_json_serializer
from prefect import flow, task
from market_analysis.story.common import output_market_dir
from market_analysis.story.tasks import task_store


PLATFORM_RESEARCH_SYSTEM_PROMPT = """
# 角色
你是一名专业的网络小说行业研究员。

# 任务
为平台【{platform}】生成一份全面、详尽、聚焦于最新现状的平台档案报告。
这份报告的核心用途是为网络小说创作者提供战略决策参考。

# 工作流程
1.  研究: 将“输出结构”中的要点作为研究目标，使用工具进行搜索和网页内容抓取。优先搜索能反映平台最新动态的官方网站、作者论坛、近半年的行业报告和新闻稿等可靠信源。
2.  总结: 在找到所有要点的信息后，将你的发现综合成一份完整的Markdown报告。如果某个要点确实找不到信息，请在该标题下明确指出“未找到相关信息”。

# 输出结构 (Markdown)
- 报告标题: `# {platform} 平台档案`
- 报告分为五大部分，请严格遵循以下二级(##)和三级(###)标题结构:

## 一、平台概览
### 1.1 平台定位与主流风格
- (一句话概括平台的核心定位和内容风格，例如：主打免费阅读的男频脑洞文聚集地)
### 1.2 平台背景与资本方
- (说明平台的母公司、所属集团、主要投资方等，例如：字节跳动旗下、腾讯阅文集团成员)
### 1.3 市场定位与主要竞品
- (分析平台在整个网文市场的生态位，并列出其最主要的1-2个竞争对手)

## 二、商业模式与读者
### 2.1 主要付费模式
- (例如: 免费+广告, 按章付费, 会员畅读, 混合模式)
### 2.2 核心读者画像
- (描述主流读者的年龄段、性别、职业、阅读偏好和消费习惯)

## 三、内容生态
### 3.1 内容题材偏好
- (评估内容丰富度，指出最热门的题材和相对冷门的题材，以及小众题材的生存空间)
### 3.2 作品推荐与曝光机制
- (说明新书/作品如何获得流量，例如：算法推荐为主、编辑推荐为辅、榜单机制、站内推广活动)
### 3.3 IP衍生开发能力
- (评估平台在推动作品进行有声、漫画、影视、游戏等改编方面的能力和成功案例)

## 四、作者核心政策
### 4.1 作者收入与福利
- (详细说明稿费、广告分成、打赏、全勤奖、完本奖、其他福利保障等)
### 4.2 作者收入水平分布
- (估算头部、中腰部、底层作者的大致月收入范围，以及新人作者的平均变现周期)
### 4.3 作者扶持与培训
- (说明平台为作者提供的支持，例如：新人扶持计划、创作激励活动、线上/线下培训课程、编辑指导)
### 4.4 签约流程与要求
- (说明签约难度，对新人的要求(如开篇字数)，申请方式，审核周期等)
### 4.5 版权归属与合同要点
- (说明作品版权归属，电子版权、衍生版权的划分，以及合同中需要特别注意的关键条款)
### 4.6 内容限制与审核红线
- (明确列出平台禁止或严格限制的题材或元素，例如：涉政、色情、暴力描写的具体尺度)
### 4.7 AIGC政策
- (平台对AI生成内容（AIGC）的态度、规定和限制)

## 五、产品与社区
### 5.1 产品功能与技术特点
- (列出App或网站的特色功能，例如：段评、角色卡、世界观设定集、AI朗读等，这些功能如何影响创作或阅读)
### 5.2 社区与作者生态
- (评估读者社区氛围、读者反馈质量、编辑对接效率与专业度、作者交流环境等)

## 六、报告质量自我评估
### 6.1 信息完整度 (1-5分)
- (评估报告中每个要点是否都找到了信息，缺失的信息点会扣分)
### 6.2 信息可靠度 (1-5分)
- (评估信息来源的权威性，例如，来自官方网站的信息得分高于来自论坛帖子的信息)
### 6.3 综合评价
- (用一两句话总结本次报告的优点和局限性)

# 输出要求
- 严格遵循上述Markdown格式和编号。
- 所有结论必须基于你搜索到的数据，为创作者提供客观、可操作的洞察，避免主观臆测。
- 你的最终输出只能是这份Markdown报告，不要包含任何其他内容，如你的思考过程或工具使用日志。
"""


@task(
    name="search_platform",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/search_platform_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=1,
    retry_delay_seconds=10,
    cache_expiration=604800,
)
def search_platform(platform: str) -> Optional[str]:
    logger.info(f"为平台 '{platform}' 生成平台档案报告...")
    system_prompt = PLATFORM_RESEARCH_SYSTEM_PROMPT.format(platform=platform)
    user_prompt = f"请开始为平台 '{platform}' 生成平台基础信息报告。"
    md_content = call_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1
    )
    if not md_content:
        logger.error(f"为平台 '{platform}' 生成报告失败。")
        return None
    logger.success(f"Agent为 '{platform}' 完成了报告生成，报告长度: {len(md_content)}。")
    return md_content

@task(
    name="save_to_md",
    retries=1,
    retry_delay_seconds=10,
)
def save_to_md(platform: str, md_content: Optional[str]) -> Optional[str]:
    if not md_content:
        logger.warning(f"内容为空，跳过为平台 '{platform}' 保存Markdown文件。")
        return None
    platform_filename_md = f"{platform.replace(' ', '_')}.md"
    file_path_md = output_market_dir / platform_filename_md
    file_path_md.write_text(md_content, encoding="utf-8")
    logger.success(f"{platform} 平台基础信息已保存为Markdown文件: {file_path_md}")
    return str(file_path_md.resolve())


@flow(name="search_platform_all")
def search_platform_all(platforms: List[str]):
    logger.info(f"开始更新 {len(platforms)} 个平台的的基础信息...")
    report_futures = search_platform.map(platforms)
    filepath_futures = save_to_md.map(
        platform=platforms,
        md_content=report_futures
    )
    store_futures = task_store.map(
        content=report_futures,
        doc_type="platform_profile",
        content_format="markdown",
        platform=platforms,
        source=filepath_futures
    )
    for future in store_futures:
        future.result()
    logger.info(f"已完成对 {len(platforms)} 个平台基础信息的更新流程。")


if __name__ == "__main__":
    platforms_cn = ["番茄小说", "起点中文网", "飞卢小说网", "晋江文学城", "七猫免费小说", "纵横中文网", "17K小说网", "刺猬猫", "掌阅"]
    platforms_en = ["Wattpad", "RoyalRoad", "AO3", "Webnovel", "Scribble Hub", "Tapas"]
    platforms = ["番茄小说"]
    search_platform_all(platforms)
