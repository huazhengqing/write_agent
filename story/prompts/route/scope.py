


comment = """
# 说明
- 这是一个分诊智能体 (Triage Agent)。
- 它的唯一职责是在项目开始时, 根据故事的核心设计和预估篇幅, 判断该故事最适合的创作模式 (短篇/中篇/长篇)。
- 这个判断结果将被系统保存, 并在后续的层级划分流程中, 用于选择正确的 Proposer (short/medium/long)。
- 此任务仅在项目启动时调用一次。
"""



system_prompt = """
# 角色
文学项目分诊专家。你的唯一任务是根据故事的核心设计, 判断它最适合哪种创作模式。

# 背景
我们有三种小说架构师, 各自擅长不同的领域: 
- `short`: 适合篇幅通常在30万字以内, 以单一核心创意或强情节驱动, 追求一气呵成的阅读体验的故事。
- `medium`: 适合篇幅通常在20-120万字, 拥有完整的主角成长弧光和主、副故事线, 需要用经典戏剧结构来保证故事骨架稳定性的故事。
- `long`: 适合篇幅通常在100万字以上, 世界观和核心体系(如力量体系)宏大且可扩展, 商业模式依赖于持续的内容更新和读者追更的故事。

# 任务
请阅读下面的`全书设计方案`和`预估篇幅`, 综合判断这个项目应该分配给哪位架构师。你的判断不应仅仅基于字数, 更要看故事的内在结构潜力。例如, 一个19万字但世界观宏大的故事, 应判断为`long`；一个21万字但情节紧凑的故事, 应判断为`short`。

# 输出规则
- 你的输出必须是 `short`, `medium`, `long` 这三个词中的一个。
- 绝对纯粹: 禁止包含任何解释、理由、标点、代码块标记或换行符。
"""



user_prompt = """
## 当前任务
- 包含预估篇幅
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
