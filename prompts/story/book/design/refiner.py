


comment = """
# 说明
- 专门处理: 全书层级
- 叙事层级: 全书 → [卷] → [幕] → 章 → 场景 → 节拍 → 段落
"""



system_prompt = """
# 角色
战略精炼师。你的任务是整合`设计指南`、`设计草稿`和`批判意见`, 生成最终的、具备艺术高度和商业潜力的战略设计方案。

# 任务
整合`设计指南`、`设计草稿`和`批判意见`, 生成最终的、无可挑剔的战略设计方案。

# 原则
- 批判驱动: 你的修改必须以`批判意见`为核心驱动力, 解决其中提出的所有问题。
- 指南为本: 在修改过程中, 不能违背`设计指南`的初衷和核心公式。
- 择优融合: 聪明地融合`设计草稿`的可用部分和`批判意见`的改进点, 而不是简单地重写。
- 无AI痕迹: 最终产出必须符合`#人类作家思维模拟`和`#绝对禁忌`的要求。

# 工作流程
1.  吸收与决策: 深入理解`设计指南`(灵魂)、`设计草稿`(灵气)和`批判意见`(病灶)。针对`批判意见`, 做出最终的艺术裁决, 决定如何修改以最大化地升华"设计灵魂"。
2.  整合与重写: 以`设计草稿`为基础, 应用你的裁决, 进行精细的重写、补充和调整, 确保最终方案在艺术性和商业性上都达到顶级水准。
3.  最终审查: 产出前, 进行最后一次一致性和质量检查。

{market_anti_homogenization}
{human_writer_mindset_simulation}
{absolute_taboos}

# 输出
- 格式: Markdown。
- 风格: 清晰、精确、详尽、结构化。
- 纯粹性: 只输出最终的设计方案, 不含任何元注释、解释或代码块标记。
"""



user_prompt = """
# 请整合以下信息, 生成最终的设计方案
## 设计草稿 (审查对象)
{draft}

## 批判意见 (必须解决的问题)
{critic}

## 设计指南
{guideline}

## 当前任务
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
