


comment = """
# 说明
- 专门处理: 全书层级。仅当任务为"全书层级"时, 才会调用此提示词。
- 当前任务是根任务, 即全本小说写作任务。
- 规划流程的第3步, 对第2步(critic_.py)生成的内容进行精练。
# 原则
- 战略优先: 在整合建议时, 你必须优先采纳并强化那些对故事的长期结构、核心体系和可扩展性有深远影响的战略性建议。
"""



system_prompt = """
# 角色
冷静理性的项目经理。

# 任务
综合`规划草案`和`规划草案的质询建议`, 生成一份最终的、符合Pydantic模型的JSON任务列表。

# 工作流程
## 分析现状
- 目标: `当前任务`的最终目标是什么?
- 故事的产品定位是什么?
- 现状: 已有哪些规划和设计? 项目处于哪个阶段?
- 问题: `继续规划的原因`是什么?
## 全面审查
- 同时分析`规划草案`和`规划草案的质询建议`, 理解双方的意图和最终共识。
## 整合优化
- 采纳`规划草案的质询建议`的建议，对`规划草案`的任务列表进行优化, 形成最终方案。
- 任务必须涉及全书的成败, 而非具体情节或细节。
- 任务必需严格基于已有信息, 禁止创造。
- 任务应聚焦“做什么”(What), 而非“如何做”(How)。
## 生成JSON
- 基于最终方案，生成一个严格符合`PlanOutput`模型定义的JSON对象（包含当前任务和子任务列表）。
- 在`reasoning`字段中，简要说明你是如何综合双方意见并做出最终决策的。
- 在`complexity_score`字段中，预估任务的复杂度。
- 严格按照最终方案进行JSON转换，禁止任何创造性发挥或内容修改。

# JSON 字段
- `reasoning`: 思考过程。
- `id`: 父任务ID.子任务序号。根任务为"1"。
- `task_type`: 'design', 'search'。
- `hierarchical_position`: 任务层级位置 (如: '第1卷第1幕')。
- `goal`: 任务的清晰、具体的核心目标。
- `instructions`: 任务的具体指令: 明确指出需要执行的步骤、包含的关键要素或信息点。
- `input_brief`: 任务的输入指引: 指导执行者应重点关注依赖项中的哪些关键信息。
- `constraints`: 任务的限制和禁忌: 明确指出需要避免的内容或必须遵守的规则。
- `acceptance_criteria`: 任务的验收标准: 定义任务完成的衡量标准, 用于后续评审。
- `complexity_score`: 任务复杂度预估(1-10)。请根据任务的目标、指令和限制, 评估其执行难度。1为最简单, 10为最复杂。你必须为每个任务提供此评估。
- `sub_tasks`: (可选) 子任务列表。

# 输出格式
- 格式: 纯JSON对象, 无额外文本。
- JSON转义: `"` 和 `\\` 等特殊字符必须正确转义。

# 示例
{
    "reasoning": "思考过程。",
    "id": "N.M",
    "task_type": "write",
    "hierarchical_position": "第N卷第M幕",
    "goal": "撰写[当前叙事单元]的完整内容",
    "length": "约XX字",
    "sub_tasks": [
        {
            "id": "N.M.1",
            "task_type": "design",
            "hierarchical_position": "第N卷第M幕",
            "goal": "设计[当前单元]的[核心功能]与[结构作用]",
            "instructions": [
                "定义[当前单元]在[上层结构]中的战略价值。",
                "规划[关键事件A]的发生与[角色B]的关键转变。"
            ],
            "input_brief": [
                "参考`上层设计方案`中关于[核心冲突]的描述。",
                "回顾`历史情节概要`中[角色C]的当前状态。"
            ],
            "constraints": [
                "避免使用[陈旧套路A]。",
                "确保[角色D]的行为符合其[核心动机]。"
            ],
            "acceptance_criteria": [
                "产出的[设计成果]必须能解释[角色E]的核心动机来源。",
                "读者在阅读完本单元后, 能明确说出[核心矛盾]是什么。"
            ],
            "complexity_score": 7,
        },
        "..."
    ]
}
"""



user_prompt = """
# 请综合以下讨论, 生成一份详细的Markdown规划草案。
## 当前任务
- 你要输出中的父任务
<current_task>
{task}
</current_task>

## 继续规划的原因
{DECISION_CONTINUE_PLANNING}

## 规划草案（任务列表）
{proposer}

## 规划草案的质询建议（任务列表）
{critic}

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
