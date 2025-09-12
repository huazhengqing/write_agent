from pydantic import BaseModel, Field
from typing import Literal, get_args


class RouteOutput(BaseModel):
    category: Literal["market", "title", "style", "hierarchy", "aggregate", "review", "character", "system", "concept", "worldview", "plot", "general"] = Field(
        description=f"判断出的任务类型, 必须是 {get_args(Literal['market', 'title', 'style', 'hierarchy', 'aggregate', 'review', 'character', 'system', 'concept', 'worldview', 'plot', 'general'])} 之一"
    )

USER_PROMPT = """
# 角色
你是一个任务分类专家, 负责将设计任务精确路由到最合适的处理模块。

# 任务
根据提供的`任务目标`, 将其分类到最匹配的预定义类别中。

# 分类类别与关键词
- "market": 市场分析, 目标读者, 对标作品, 竞品, 卖点, 差异化, 商业模式。
- "title": 书名, 简介, 标签, 关键词。
- "style": 叙事风格, 美学基调, 核心意象, 语言风格, 文笔基调, 叙事视角/时态, 主题内核。
- "hierarchy": 结构规划, 层级, 划分, 卷, 幕, 章, 场景, 节拍, 分解。
- "aggregate": 整合, 汇总, 聚合, 敲定, 最终方案。
- "review": 验证, 审查, 风险, 预案, 一致性, 压力测试, 整合审查, 批判性分析, 反思。
- "character": 角色, 主角, 人设, 背景, 驱动力, 目标, 性格, 魅力, 成长路线, 关系, 阵营, 配角, 反派。
- "system": 爽点, 成长体系, 反馈循环, 数值化, 阶段化, 核心循环, 演化, 可扩展性, 生命周期, 升级, 天花板, 金手指。
- "concept": 核心概念, 一句话故事, 吸引点, 钩子, 开篇设计, 黄金三章, 快速入局。
- "worldview": 世界观, 基础规则, 核心体系, 社会结构, 势力, 历史, 地理, 文化风俗。
- "plot": 冲突, 情节, 架构, 主线, 支线, 节点, 事件链, 情绪节奏, 悬念, 伏笔, 叙事节奏。
- "general": 无法归入以上任何类别的其他任务。

# 分类原则
- 关键词优先: 优先根据`任务目标`中的核心关键词进行匹配。
- 具体优先: 如果一个任务目标可以匹配多个类别, 选择最具体、最核心的那个。例如, 一个任务是“根据市场定位设计吸引人的书名”, 它的核心是“书名”, 因此应分类为 "title"。

# 任务目标
"{goal}"

# 输出要求
- 格式: 纯JSON对象, 无额外文本。
- 字段: `category` (字符串), 其值必须是 "market", "title", "style", "hierarchy", "aggregate", "review", "character", "system", "concept", "worldview", "plot", "general" 之一。
"""