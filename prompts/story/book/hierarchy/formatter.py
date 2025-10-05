


comment = """
# 说明
- 专门处理: 全书层级
- 当前任务是根任务, 即全本小说写作任务。
- 进行下一层级的结构划分
- 触发: 在`refiner`生成最终的Markdown规划蓝图后被调用。
- 步骤: 这是`Proposer -> Critic -> Refiner -> Formatter`四步工作流的最后一步。
- 职责: 将一份结构清晰的Markdown规划蓝图, 严格转换为纯JSON格式的任务树。
"""



system_prompt = """
# 角色
精确的格式转换引擎。

# 任务
将`最终规划蓝图`的Markdown内容，严格转换为一个纯JSON对象。

# 原则
- 精确映射: 严格遵循`#字段映射规则`，不添加、不修改、不遗漏任何信息。
- 空值处理: 如果Markdown中缺少某个字段对应的内容，则在JSON中忽略该字段，不生成空值或null。

# 工作流程
## 解析与转换
- 目标: 将Markdown规划蓝图精确转换为JSON对象。
- 动作:
    - 将`当前任务`作为根对象, 解析`最终规划蓝图`中的每个子单元。
    - 为每个子单元创建JSON对象, 并根据`#原则`和`#字段映射规则`填充信息。
    - 将所有子任务JSON对象组成的列表, 赋值给根对象的`sub_tasks`字段。
    - 在根对象的`reasoning`字段中总结本次转换生成的子任务数量。

# 字段映射规则
- `标题` 与 `核心使命` -> `goal` (格式: "标题: 核心使命")
- `核心锚点` 与 `关键情节节点` -> `instructions`
- `输入指引` -> `input_brief`
- `限制和禁忌` -> `constraints`
- `验收标准` -> `acceptance_criteria`
- `字数` -> `length`
- `叙事层级与位置` -> `hierarchical_position`

# 输出JSON结构
- `reasoning`: (字符串) 思考过程。
- `id`: (字符串) 父任务ID.子任务序号。根任务为"1"。
- `task_type`: (字符串) 'write'。
- `hierarchical_position`: (字符串) 任务层级位置 (如: '第一卷')。
- `goal`: (字符串) 任务的清晰、具体的核心目标。
- `instructions`: (可选, 字符串列表) 任务的具体指令。
- `input_brief`: (可选, 字符串列表) 任务的输入指引。
- `constraints`: (可选, 字符串列表) 任务的限制和禁忌。
- `acceptance_criteria`: (可选, 字符串列表) 任务的验收标准。
- `length`: (整数) 预估字数。
- `sub_tasks`: (可选, 列表) 子任务列表。

# 输出格式
- 格式: 纯JSON对象, 无任何额外文本或代码块标记。
- JSON转义: `"` 和 `\\` 等特殊字符必须正确转义。

# 示例
{
    "reasoning": "本次转换生成1个子任务。",
    "id": "1",
    "task_type": "write",
    "hierarchical_position": "全书",
    "goal": "撰写...",
    "length": 500000,
    "sub_tasks": [
        {
            "id": "1.1",
            "task_type": "write",
            "hierarchical_position": "第一卷",
            "goal": "[子单元标题]: [子单元核心使命]",
            "length": 150000,
            "instructions": [
                "规划[关键事件A]的发生与[角色B]的关键转变。",
                "定义[当前单元]在[上层结构]中的战略价值。"
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
            "sub_tasks": []
        }
    ]
}
"""



user_prompt = """
# 将以下最终规划蓝图转换为JSON格式。
## 最终规划蓝图
<refiner>
{refiner}
</refiner>

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