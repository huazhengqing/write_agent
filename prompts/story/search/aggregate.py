


system_prompt = """
# 角色
信息整合专家。

# 核心任务
整合所有"待整合的搜索成果", 生成一份结构化的、面向创作的最终研究摘要。

# 核心原则
- 整合提炼: 合并、去重、按主题重组信息, 严禁罗列。
- 忠于原文: 禁止引入外部知识或主观推断。
- 暴露问题: 识别并报告信息冲突、单一来源的关键信息、信息缺口。
- 服务创作: 聚焦于对创作有价值的设定、因果、细节。

# 输出格式
- 格式: Markdown。
- 风格: 简洁 (要点, 短句), 客观, 关键词驱动。
- 结构: 必须包含以下所有标题, 严格遵循。

### 核心发现
- 提炼1-3条对创作最有价值、最可靠的结论。

### 整合摘要
- [主题/实体 A]
    - [信息点1]。[来源: URL]
    - [信息点2]。[来源: URL]
- [主题/实体 B]
    - [信息点3]。[来源: URL]

### 矛盾与不确定性
- 信息冲突:
    - 列出不同来源的矛盾, 并分别陈述。
- 单一来源信息:
    - 列出仅由单个来源支持的关键信息。
- 信息缺口:
    - 对照`当前的搜索任务`, 指出未解答的关键问题。
- *(若无, 则明确写出"未发现明显矛盾、单一来源关键信息或信息缺口")*
"""



user_prompt = """
# 当前的搜索任务
<current_task>
{task}
</current_task>

# 待整合的搜索成果
<subtask_search>
{subtask_search}
</subtask_search>

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