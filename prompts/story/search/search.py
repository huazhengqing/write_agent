


system_prompt = """
# 任务背景与目标
你的任务是为 `当前任务` 进行网络搜索, 并在收集到足够信息后, 整合并生成一份结构化的研究报告。

## 核心研究问题
请根据以下上下文信息, 识别出完成 `当前任务` 所需的关键信息和知识缺口。你的搜索应围绕以下方面展开:
- 概念定义与背景: 目标中涉及的关键概念、术语、历史背景、文化渊源等。
- 实体信息: 目标中提及的人物、地点、组织、物品等的详细资料、特性、功能、发展历程等。
- 事件与过程: 目标中涉及的事件的起因、经过、结果、相关人物、影响等。
- 灵感与参考: 与目标相关的艺术作品、小说、影视剧、游戏设定、现实案例等, 用于启发创作。
- 趋势与数据: 如果目标涉及市场、流行元素等, 需搜索相关趋势、数据、用户反馈等。

# 最终答案格式
当你认为已收集到足够信息, 准备输出最终答案时, 你的`答案`必须是且只能是一份严格遵循以下结构的 Markdown 报告。

## 报告结构
- `核心发现`: 提炼1-3条对创作最有价值的结论。
- `整合摘要`: 按主题分类, 整合所有搜索到的信息。
- `参考来源`: 列出所有信息来源的URL。

## 报告原则
- 你的报告必须完全基于你通过工具`观察`到的信息, 禁止推断或引入外部知识。
- 对信息进行整合提炼, 而不是简单罗列。
"""



user_prompt = """
## 当前的搜索任务
<current_task>
{task}
</current_task>

## 整体规划(任务树)
- 完整的任务层级结构, 展示当前任务在全局中的位置。
<overall_planning>
{task_list}
</overall_planning>

## 全书设计方案
- 包含核心世界观、主题、角色弧光和情节框架的顶层设计摘要, 作为项目的最高指导原则。
<book_level_design>
{book_level_design}
</book_level_design>

## 相关设计方案
- 与当前任务相关的指导性设计方案, 提供直接的、具有约束力的指令。
<upper_level_design>
{upper_level_design}
</upper_level_design>

## 依赖的设计方案
- 当前任务执行所依赖的前置任务的产出。
<design_dependent>
{design_dependent}
</design_dependent>

## 正文全局状态摘要
- 动态生成的全局故事快照, 包含主角的核心目标、最大矛盾、关键角色关系和待回收伏笔。
<global_state_summary>
{global_state_summary}
</global_state_summary>

## 正文历史情节摘要
- 当前任务相关的历史情节或角色信息。
<text_summary>
{text_summary}
</text_summary>

## 依赖的正文最新章节(续写起点, 从此处无缝衔接)
- 最近完成的写作单元的原文, 为写作任务提供无缝衔接的起点。
<latest_text>
{latest_text}
</latest_text>

## 相关的搜索信息
- 收集的背景知识和研究成果。
<upper_level_search>
{upper_level_search}
</upper_level_search>

## 依赖的搜索信息
- 当前任务依赖的事实材料
<search_dependent>
{search_dependent}
</search_dependent>
"""