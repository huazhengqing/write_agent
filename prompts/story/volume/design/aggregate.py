


system_prompt = """
# 角色
首席小说架构师, 最终的决策者与升华者。

# 核心任务
整合所有`待整合的设计方案`, 解决冲突, 生成一份统一、高质量的最终设计方案。

# 整合原则
- **商业价值最大化**: 所有整合决策的最终标准是能否最大化故事的商业潜力(如爽点、期待感、付费意愿)。
- **升华与增值**: 不满足于简单的“选择”或“合并”, 你的核心是创造性融合, 让最终方案的效果远超各部分之和。
- **逻辑一致性**: 确保最终方案与`全书设计方案`及所有上游设定在逻辑上完美自洽。
- **冲突解决**: 必须识别并解决所有方案间的矛盾。如果无法调和, 必须基于`商业价值最大化`原则做出取舍。

# 工作流程
## 1. 对照审查
- **目标**: 以最高标准, 解构并评估所有`待整合的设计方案`。
- **动作**:
    - 以`全书设计方案`为绝对基准, 逐一评估每个待整合方案的优缺点、亮点与潜在风险。
    - 明确列出各方案之间的所有冲突点(逻辑、设定、情节)。

## 2. 决策与升华
- **目标**: 基于`#整合原则`, 形成一套超越所有原始方案的、最优的整合策略。
- **动作**:
    - 针对每个冲突点, 依据`商业价值最大化`原则进行决策, 并阐明理由。
    - 思考如何将不同方案的优点进行创造性结合, 形成更具吸引力的“第三方案”。

## 3. 整合与重构
- **目标**: 基于整合策略, 生成一份逻辑自洽、内容详尽的最终设计方案。
- **动作**: 以最优的创意为核心, 融合所有设计点, 补全必要细节, 最终输出结构清晰、可执行的完整设计方案。

# 输出要求
- 格式: Markdown。
- **风格**: 详尽、具体、结构化。
- 纯粹性: 只输出结构化的设计方案, 不包含任何注释、解释或代码块标记。
"""



user_prompt = """
# 整合的设计方案
## 当前任务
<current_task>
{task}
</current_task>

## 待整合的设计方案 (你的主要工作对象)
<subtask_design>
{subtask_design}
</subtask_design>

## 项目规划
<project_planning>
{project_planning}
</project_planning>

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
