import nest_asyncio
nest_asyncio.apply()


import os
import sys
from typing import List, Optional
from datetime import timedelta
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])


from utils.prefect import local_storage, readable_json_serializer
from prefect import task, flow



platform_research_prompt = """
# 任务背景与目标
你的角色是一名专业的网络小说行业研究员。
你的任务是为平台[{platform}]生成一份全面、详尽、聚焦于最新现状的平台档案报告。
这份报告的核心用途是为网络小说创作者提供战略决策参考。
你需要将下方的"研究清单"作为你的行动指南, 逐项利用工具进行搜索和信息整合。

# 研究清单 (Checklist)
你必须按顺序研究并回答以下所有问题, 以收集报告所需的全部信息。
## 平台概览
- 平台定位与主流风格: [{platform}]的核心定位和内容风格是什么? (例如: 主打免费阅读的男频脑洞文聚集地)
- 平台背景与资本方: [{platform}]的母公司、所属集团、主要投资方是谁?
- 市场定位与主要竞品: [{platform}]在网文市场的生态位是什么? 其最主要的1-2个竞争对手是谁?
## 商业模式与读者
- 主要付费模式: [{platform}]的主要付费模式是什么? (例如: 免费+广告, 按章付费, 会员畅读, 混合模式)
- 核心读者画像: [{platform}]主流读者的年龄段、性别、职业、阅读偏好和消费习惯是怎样的?
## 内容生态
- 内容题材偏好: [{platform}]最热门和最冷门的题材分别是什么? 小众题材的生存空间如何?
- 作品推荐与曝光机制: 在[{platform}], 新书/作品如何获得流量? (例如: 算法推荐、编辑推荐、榜单、活动)
- IP衍生开发能力: [{platform}]在推动作品进行有声、漫画、影视、游戏等改编方面的能力和成功案例有哪些?
## 作者核心政策
- 作者收入与福利: [{platform}]的作者收入构成是怎样的? (包括稿费、广告分成、打赏、全勤奖、完本奖等)
- 作者收入水平分布: [{platform}]的头部、中腰部、底层作者的大致月收入范围是多少? 新人作者平均多久能变现?
- 作者扶持与培训: [{platform}]为作者提供了哪些支持? (例如: 新人扶持计划、创作激励、培训课程、编辑指导)
- 签约流程与要求: 在[{platform}]签约的难度如何? 对新人有什么要求(如开篇字数)? 审核周期多久?
- 版权归属与合同要点: [{platform}]的版权归属政策是怎样的? 合同中有哪些关键条款需要注意?
- 内容限制与审核红线: [{platform}]禁止或严格限制哪些题材或元素? (例如: 涉政、色情、暴力的具体尺度)
- AIGC政策: [{platform}]对AI生成内容(AIGC)的态度和规定是什么?
## 产品与社区
- 产品功能与技术特点: [{platform}]的App或网站有哪些影响创作或阅读的特色功能? (例如: 段评、角色卡、AI朗读)
- 社区与作者生态: [{platform}]的读者社区氛围、读者反馈质量、编辑对接效率和作者交流环境如何?
## 报告质量自我评估
- 信息完整度: 完成上述所有要点的研究后, 对信息完整度进行1-5分的自我评估。
- 信息可靠度: 根据信息来源的权威性(官网 > 行业报告 > 论坛帖子), 对信息可靠度进行1-5分的自我评估。
- 综合评价: 总结本次报告的优点和局限性。

# 最终答案格式
当你通过"思考 -> 操作 -> 观察"循环收集完以上所有信息后, 你的最终`答案`必须是且只能是一份严格遵循以下结构的 Markdown 报告。
如果某个要点确实找不到信息, 请在该标题下明确指出"未找到相关信息"。

```markdown
# {platform} 平台档案

## 一、平台概览
### 1.1 平台定位与主流风格
...
### 1.2 平台背景与资本方
...
### 1.3 市场定位与主要竞品
...

## 二、商业模式与读者
### 2.1 主要付费模式
...
### 2.2 核心读者画像
...

## 三、内容生态
... (以此类推, 补全所有部分)

## 六、报告质量自我评估
### 6.1 信息完整度 (1-5分)
...
### 6.2 信息可靠度 (1-5分)
...
### 6.3 综合评价
...
```
"""



@task(
    name="search_platform",
    persist_result=True,
    result_storage=local_storage,
    result_storage_key="story/market/search_platform_{parameters[platform]}.json",
    result_serializer=readable_json_serializer,
    retries=3,
    retry_delay_seconds=60,
    cache_expiration=timedelta(days=7),
)
async def search_platform(platform: str) -> Optional[str]:
    from utils.react_agent import call_react_agent
    md_content = await call_react_agent(
        system_prompt=platform_research_prompt.format(platform=platform),
        user_prompt=f"请开始为平台 '{platform}' 生成平台基础信息报告。" ,
    )
    if not md_content:
        logger.error(f"为平台 '{platform}' 生成报告失败。")
        return None
    return md_content



@flow(name="search_platform_all") 
async def search_platform_all(platforms: List[str]):
    logger.info(f"开始更新 {len(platforms)} 个平台的的基础信息...")
    
    # 1. 并行执行所有平台的搜索任务
    report_futures = search_platform.map(platforms)
    
    # 2. 等待所有搜索任务完成, 并获取报告内容
    reports = []
    for future in report_futures:
        # .result() 对于异步任务会返回一个协程, 但如果结果被缓存, 它会直接返回结果。
        # 我们需要检查返回的是否是协程, 只有在是协程的情况下才 await。
        result_or_coro = future.result()
        reports.append(await result_or_coro if asyncio.iscoroutine(result_or_coro) else result_or_coro)

    # 3. 使用获取到的报告内容, 并行执行保存 Markdown 和存入向量库的任务
    from market_analysis.story.tasks import task_save_markdown
    save_md_futures = task_save_markdown.map(
        filename=platforms,
        content=reports
    )
    from market_analysis.story.tasks import task_save_vector
    save_vector_futures = task_save_vector.map(
        content=reports,
        type="platform_profile",
        platform=platforms,
    )

    # 4. (可选) 等待保存任务完成
    for future in save_vector_futures:
        future.result()
    for future in save_md_futures:
        future.result()

    logger.info(f"已完成对 {len(platforms)} 个平台基础信息的更新流程。")



async def main(platforms_to_run: List[str]):
    await search_platform_all(platforms_to_run)



if __name__ == "__main__":
    platforms1 = ["番茄小说", "起点中文网", "飞卢小说网", "晋江文学城", "七猫免费小说", "纵横中文网", "17K小说网", "刺猬猫", "掌阅"]
    platforms2 = ["Wattpad", "RoyalRoad", "AO3", "Webnovel", "Scribble Hub", "Tapas"]
    platforms = ["番茄小说"]
    import asyncio
    asyncio.run(main(platforms))
