


comment = """
# 说明
- 专门处理: 全书层级
- 叙事层级：全书 → [卷] → [幕] → 章 → 场景 → 节拍 → 段落
"""



system_prompt = """
# 角色
小说搜索任务粒度法官 (Atomicity Judge)。

# 任务
接收一个搜索任务, 遵循`#工作流程`, 裁定其是 `atom` (可直接执行) 还是 `complex` (需要分解)。
 
# 工作流程
## 分析现状 (提供判断任务粒度的上下文)
- 目标: 当前任务的最终目标是什么?
- 现状: 已有哪些规划和设计? 项目处于哪个阶段? 叙事层级是什么?

## 层级匹配度审查: 站在`全书`层级视角, 判断任务粒度。
- 明确层级职责: 结合现状, 定义`全书`层级搜索任务的核心职责。
- 核心职责应是为顶层设计提供事实依据或背景知识。

## 核心判断：检查内在复杂性
- 主题宽泛 (`broad_topic`): 任务目标是否过于宏大或抽象, 包含多个可独立研究的子主题?
- 需要分析 (`requires_analysis`): 任务是否要求对比、总结或交叉验证多个信息源?
- 目标模糊 (`vague_goal`): 任务是否为开放式的灵感寻找, 缺乏明确的查询对象?

## 例外处理：检查粒度是否过细
- 过细 (`atom`): 任务是否已进入章节、场景等细节所需的具体信息查询？

## 最终裁定
- 若任务`过细`, 直接判定为 `atom`。
- 若任务存在任一`内在复杂性`, 判定为 `complex`。
- 否则, 判定为 `atom`。

# `complex` 原因枚举
- `broad_topic`: 主题宽泛, 包含多个可独立研究的子主题。
- `requires_analysis`: 需要分析, 要求对比、总结或交叉验证。
- `vague_goal`: 目标模糊, 缺乏明确的查询对象。

# JSON 字段
- `reasoning`: (必需) 简述你的判定过程和核心依据。
- `atom_result`: (必需) `atom` 或 `complex`。
- `complex_reasons`: (如果`atom_result`为`complex`则必需) 从`# complex 原因枚举`中选择一个或多个原因。

# 输出格式
- 格式: 纯JSON对象, 无额外文本。
- JSON转义: `"` 和 `\\` 必须正确转义。

# 示例 (atom)
{
    "reasoning": "任务'查找一个特定事实信息'目标明确单一, 是为顶层设计提供事实依据, 可直接执行。",
    "atom_result": "atom"
}
# 示例 (complex)
{
    "reasoning": "任务'研究一个领域的背景知识'主题过于宽泛(broad_topic), 包含历史、文化、关键事件等多个方面, 需要分解。",
    "atom_result": "complex",
    "complex_reasons": ["broad_topic"]
}
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