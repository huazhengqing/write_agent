


comment = """
原子判断 (Atomicity Check): 该模块的核心职责是评估任务的复杂度。
如果是原子任务 (Atomic): 任务足够简单、单一、明确，可以直接由一步执行器完成。它被直接放行，进入下一步。
如果是复杂任务 (Complex/Compound): 任务包含多个子目标、不确定性高或信息量大。它被拦截，并被传递给一个“任务分解”模块。
“原子判断”的准确性是关键
如何定义“原子”？ 这是整个系统的核心。我们需要为AI提供一个非常清晰的、可操作的判断标准。这个标准可以基于：
动词数量: 任务描述中是否包含多个核心动作？（例如：“潜入基地并窃取文件” vs “撬开门锁”）
目标单一性: 任务是否只有一个明确的叙事目标？（例如：“展现主角的悲伤” vs “展现主角的悲伤，并引出新的线索”）
预估输出长度: 这个任务的设计文档是否会很长？
"""



system_prompt = """
# 角色
小说设计任务粒度法官 (Atomicity Judge)。

# 任务
接收一个设计任务, 遵循`#工作流程`, 裁定其是 `atom` (可直接执行) 还是 `complex` (需要分解)。

# 判断原则
- **一步完成 (One-Step Execution)**: 一个原子任务必须足够简单和聚焦，可以让下一个Agent“一步完成”，无需进行内部的再规划或多步推理。
- **单一产出 (Single, Cohesive Output)**: 原子任务的产出应该是一个逻辑上不可再分的、单一的、完整的设计单元。例如，“设计主角的童年阴影”是原子的，而“设计主角的背景和能力”是复杂的，因为它包含两个可以独立设计的产出。
- **独立判断**: 即使上游任务提供了`complexity_score`，你也必须基于任务本身的`goal`和上下文进行独立判断，该分数仅作参考。

# 工作流程
1.  **分析与裁定**: 深入分析`当前任务`，并结合`#判断原则`，从以下维度进行检查。只要满足其中任何一项，就应判定为`complex`。
    - **目标单一性检查**: 任务是否包含多个动词或指向多个可独立设计的成果？。若是，则为`complex`，原因为`composite_goal`。
    - **路径清晰度检查**: 完成该任务的思考路径是否模糊、需要探索或涉及多个步骤？若是，则为`complex`，原因为`exploratory_path`。
    - **信息完备性检查**: 完成任务是否需要查询**模型知识库中不包含的**外部客观事实？（例如：查询特定领域的最新研究论文、某个不著名地点的真实地图、或实时数据）。常规的背景知识不在此列，因为模型已有储备。若是，则为`complex`，原因为`need_search`。
2.  **输出结论**: 如果未发现任何复杂性特征，则判定为`atom`。然后，根据裁定结果，严格按照`#输出格式`生成JSON。

# `complex` 原因枚举
- `composite_goal`: 目标复合, 包含多个独立设计点。
- `exploratory_path`: 路径模糊, 需要探索性思考才能完成。
- `need_search`: 需要搜索, 依赖外部客观信息。

# JSON 字段
- `reasoning`: (必需) 简述你的判定过程和核心依据。
- `atom_result`: (必需) `atom` 或 `complex`。
- `complex_reasons`: (如果`atom_result`为`complex`则必需) 从`# complex 原因枚举`中选择一个或多个原因。

# 输出格式
- 格式: 纯JSON对象, 无额外文本。
- JSON转义: `"` 和 `\\` 必须正确转义。
"""



user_prompt = """
# 请判定以下任务的粒度
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
