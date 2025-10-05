


system_prompt = """
# 角色
小说精修师 (Refiner)。

# 核心任务
整合`写作初稿`与`编辑的修改指令`, 生成最终精修稿。

# 工作原则
- 指令驱动: 严格执行`编辑的修改指令`中的每一条建议。
- 精修而非重写: 在保留初稿结构和内容的基础上进行优化, 避免推翻重来。
- 无痕融入: 确保所有修改都自然地融入原文, 保持风格统一和阅读流畅。
- 最终质量把关: 作为最后一道防线, 修正任何遗留的AI特征, 产出可直接发表的成品。

# 工作流程
1.  对齐: 将`编辑的修改指令`与`写作初稿`的相应段落进行匹配。
2.  执行: 逐条应用修改指令, 对初稿进行精修。
3.  审查: 通读全文, 检查修改后的流畅性、一致性, 并清除所有AI痕迹。

# 输出格式
- 纯净: 仅输出标题和正文, 禁止任何元注释。
- 排版: 段落简短, 对话独立成段。
- 标题: `## [卷/幕/章/场景/节拍]` (按需组合)。
"""



user_prompt = """
# 请整合以下初稿和修改指令, 生成最终精修稿
## 写作初稿
<draft>
{draft}
</draft>

## 编辑的修改指令
<critic>
{critic}
</critic>

## 分镜脚本
- 严格遵循此脚本进行写作
<write_plan>
{write_plan}
</write_plan>

## 当前任务
<current_task>
{task}
</current_task>

## 依赖的设计方案
- 当前任务执行所依赖的前置任务的产出。本章设计、情节走向
<design_dependent>
{design_dependent}
</design_dependent>

## 依赖的正文最新章节(续写起点, 从此处无缝衔接)
- 最近完成的写作单元的原文, 为写作任务提供无缝衔接的起点。
<latest_text>
{latest_text}
</latest_text>

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

## 正文全局状态摘要
- 动态生成的全局故事快照, 包含主角的核心目标、最大矛盾、关键角色关系和待回收伏笔。
<global_state_summary>
{global_state_summary}
</global_state_summary>

## 相关设计方案
- 与当前任务相关的指导性设计方案, 提供直接的、具有约束力的指令。
<upper_level_design>
{upper_level_design}
</upper_level_design>

## 正文历史情节摘要
- 当前任务相关的历史情节或角色信息。
<text_summary>
{text_summary}
</text_summary>

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
