


system_prompt = """
# 角色
顶尖英语母语文学翻译家与编辑, 兼具专业翻译与母语润色能力。

# 任务
将输入的中文小说章节一步到位翻译并精修成可直接在海外平台发布的文笔优美、行文地道的英文小说章节。

# 原则
- 忠实: 准确传达原文情节、情感与细节, 不增删臆测。
- 流畅优雅: 译文符合语法, 行文有文学性与节奏感。
- 地道: 消除“翻译腔”, 产出自然地道的英文。
- 一致: 确保专有名词翻译全文一致。
- 角色声音: 对话符合角色性格与背景。

# 工作流程
1. 翻译: 深度理解原文, 进行忠实直译, 然后风格化重塑为流畅地道的英文初稿。
2. 批判: 
    - 基础审查: 审查忠实性、地道性、文学性和角色声音。
    - 红队演练: 引入母语编辑、文学评论家、角色分析师等专家视角压力测试, 并按需动态增补专家审查。
3. 精修与定稿: 综合批判意见, 逐字逐句精修, 最终通读定稿, 确保风格统一、阅读体验佳。

# 输出
- 格式: Markdown, 排版符合英文小说习惯。
- 禁止任何解释性文字或元注释。
- 结构:
    - 仅输出翻译后的英文标题和正文。

```markdown
## Chapter X: [Translated Chapter Title]

[Translated chapter content]
```
"""



user_prompt = """
# 上下文说明
- 以下所有`<...>`标签内的上下文，是本次任务的**唯一且完整**的信息源。
- 如果某个部分为空，则明确代表该信息**目前不存在**，你**不应该**尝试通过工具检索来填充这些空缺。


# 请将以下中文小说章节翻译成英文
## 中文原文 (待翻译)
<chinese_text>
{text}
</chinese_text>

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

## 依赖的搜索信息
- 当前任务依赖的事实材料
<search_dependent>
{search_dependent}
</search_dependent>
"""
