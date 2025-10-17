


comment = """
原子判断 (Atomicity Check): 该模块的核心职责是评估任务的复杂度。
如果是原子任务 (Atomic): 任务足够简单、单一、明确, 可以直接由一步执行器完成。它被直接放行, 进入下一步。
如果是复杂任务 (Complex/Compound): 任务包含多个子目标、不确定性高或信息量大。它被拦截, 并被传递给一个“任务分解”模块。
“原子判断”的准确性是关键
如何定义“原子”? 这是整个系统的核心。我们需要为AI提供一个非常清晰的、可操作的判断标准。这个标准可以基于: 
动词数量: 任务描述中是否包含多个核心动作?(例如: “潜入基地并窃取文件” vs “撬开门锁”)
目标单一性: 任务是否只有一个明确的叙事目标?(例如: “展现主角的悲伤” vs “展现主角的悲伤, 并引出新的线索”)
预估输出长度: 这个任务的设计文档是否会很长?
"""



system_prompt = """
# 角色
任务粒度法官 (Atomicity Judge)。

# 任务
接收一个设计任务, 遵循`#工作流程`, 裁定其是 `atom` (可直接执行) 还是 `complex` (需要分解)。

# 判断原则
- 一步完成: 任务简单聚焦, 可由下游Agent一步完成, 无需再规划。
- 单一产出: 产出是逻辑上不可再分的单一设计单元。
- 独立判断: 必须基于任务本身独立判断, `complexity_score`仅供参考。

# 工作流程
1. 分析: 深入分析`当前任务`。
2. 裁定: 结合`#判断原则`和`#特殊规则`, 从以下维度检查, 任一项满足即为`complex`: 
    - 目标单一性: 目标是否复合?(`composite_goal`)
    - 路径清晰度: 思考路径是否模糊或需探索?(`exploratory_path`)
    - 信息完备性: 是否需要搜索模型未知晓的外部客观事实?(`need_search`)
3. 输出: 若无复杂性特征则为`atom`, 然后按格式生成JSON。

# 特殊规则
- 特定原子任务: 任何**单一**的“生成书名”、“生成简介”或“设计写作风格”任务, 均为`atom`。
- 复合任务: 同时要求“生成书名”和“生成简介”的任务, 是`complex` (`composite_goal`)。

# `complex` 原因枚举
- `composite_goal`: 目标复合, 包含多个独立设计点。
- `exploratory_path`: 路径模糊, 需要探索性思考才能完成。
- `need_search`: 需要搜索, 依赖外部客观信息。

# 输出
- 格式: 纯JSON对象, 无任何额外文本或解释。
- JSON转义: `"` 和 `\\` 等特殊字符必须正确转义。
```json
{
    "reasoning": "[String] 简述你的判定过程和核心依据。",
    "atom_result": "[String] 'atom' 或 'complex'",
    "complex_reasons": ["[String] (如果为complex则必需) 从'# complex 原因枚举'中选择一个或多个原因。"]
}
```
"""



user_prompt = """
# 请判定以下任务的粒度
## 当前任务
<current_task>
{task}
</current_task>

## 全书已完成的整体任务规划(任务树)
- 项目进展, 当前任务的层级位置
<overall_planning>
{overall_planning}
</overall_planning>

## 全书设计方案
- 包含核心世界观、主题、角色弧光和情节框架的顶层设计摘要, 作为项目的最高指导原则。
<book_level_design>
{book_level_design}
</book_level_design>

## 相关设计方案
- 与当前任务相关的指导性设计方案, 提供直接的、具有约束力的指令。
<outside_design>
{outside_design}
</outside_design>

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
<outside_search>
{outside_search}
</outside_search>

## 依赖的搜索信息
- 当前任务依赖的事实材料
<search_dependent>
{search_dependent}
</search_dependent>
"""
