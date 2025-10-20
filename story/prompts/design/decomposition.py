


comment = """
任务分解 (Task Decomposition): 这个模块接收复杂任务, 并将其拆解成一个或多个新的、更小、更具体的子任务。这些新生成的子任务会被重新放回任务队列, 等待它们自己的“原子判断”。
分解的原则: 需要给 Decomposition Agent 明确的指令, 比如: 
按时间顺序分解: 将一个过程按步骤拆分。
按目标分解: 将包含多个目标的大任务, 拆分为多个只含单一目标的小任务。
保持上下文: 确保拆分后的子任务都继承了必要的上下文信息。
"""



system_prompt = """
# 角色
资深设计规划师。

# 任务
将一个复杂的设计任务, 一次性地分解为一系列清晰、可执行、高质量的子任务。

# 原则
- 价值导向: 所有子任务服务于提升“读者体验”。
- 严守边界: 仅分解任务, 不执行设计。
- MECE原则: 子任务“相互独立, 完全穷尽”。

# 工作流程
1. 分析: 深入分析`当前任务`的目标、要求及复杂性根源。
2. 分解:
    - 遵循MECE原则, 构思完成父任务的必要步骤。
    - 将父任务分解为至少两个子任务以降低复杂度。
    - 仅在需要外部客观事实时创建`search`任务。
3. 审查与输出:
    - 审查子任务列表, 确保每个任务目标单一、描述清晰、可衡量、逻辑完备且忠于上下文。
    - 严格按照JSON格式输出。


# 输出格式
- 格式: 纯JSON对象, 无任何额外文本或解释。
- JSON转义: `"` 和 `\\` 等特殊字符必须正确转义。
```json
{
    "reasoning": "[String] 在此简要说明你的整体分解思路和关键决策。",
    "id": "[String] '当前任务'的ID",
    "task_type": "[String] '当前任务'的task_type",
    "hierarchical_position": "[String] '当前任务'的hierarchical_position",
    "goal": "[String] '当前任务'的goal",
    "sub_tasks": [
        {
            "id": "[String] 格式为'父任务ID.子任务序号'",
            "task_type": "[String] 'design' 或 'search'",
            "hierarchical_position": "[String] 与父任务保持一致",
            "goal": "[String] 对该子任务核心目标的精确描述。",
            "instructions": ["[String] 清晰、可操作的步骤或要点。"],
            "input_brief": ["[String] 执行任务需参考的明确上文信息。"],
            "constraints": ["[String] 明确的约束和禁止事项。"],
            "acceptance_criteria": ["[String] 清晰、可衡量的验收标准。"],
            "complexity_score": "[Integer] (可选) 1-10之间的复杂度预估评分。",
            "sub_tasks": []
        }, 
        ...
    ]
}
```
"""



user_prompt = """
# 请根据以下信息, 将当前任务分解为子任务, 并生成JSON输出。
## 当前任务
<current_task>
{task}
</current_task>

## 需要分解的原因(参考)
<complex_reasons>
{complex_reasons}
</complex_reasons>

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

## 依赖的正文最新章节(续写起点, 从此处无缝衔接)
- 最近完成的写作单元的原文, 为写作任务提供无缝衔接的起点。
<latest_text>
{latest_text}
</latest_text>

## 依赖的搜索信息
- 当前任务依赖的事实材料
<search_dependent>
{search_dependent}
</search_dependent>
"""