#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()
import json


@prompt_register.register_module()
class ReportWriter(PromptTemplate):
    def __init__(self) -> None:
        system_message = "".strip()
        
        content_template = """
The collaborative report-writing requirement to be completed:  
**{to_run_root_question}**  

It has been decomposed into several parts (writing tasks), as shown below. The `You Need To Write` is the part you should write.
```
{to_run_global_writing_task}
```

Based on the existing report analysis conclusions and the requirements, continue writing the report. You need to continue writing:  
**{to_run_task}**

---
**The existing report analysis conclusions and search results are as follows, you should use it**:s 
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

---
今天是 {today_date}。你是一位专业的报告撰写人, 与其他作者合作, 根据用户的要求完成一份专业的报告。

# 要求: 

* 无缝衔接: 从上一节结束的地方开始写作, 保持相同的写作风格、词汇和整体基调。自然地完成你的部分(章节), 不要重复或重新解释已经陈述过的细节或信息。

* 专注于现有的分析和搜索结果: 
\t* 密切关注先前分析和搜索任务的结论和发现, 以指导你的写作
\t* 搜索结果在 <web_pages_short_summary></<web_pages_short_summary> 中提供, 并已标记其来源索引。
\t* 需要根据与问题的相关性筛选和过滤搜索结果
\t* 注意, 并非所有搜索结果都是相关和有用的, 你应该仔细辨别。
\t* 绝不杜撰内容
\t* 不要简单地堆砌证据和事实；相反, 要将事实、证据和观点有机地整合起来, 使其成为叙述和论证的一部分。

* 数据准确性和引文支持: 
\t* 在适当的句子末尾使用 [reference:X] 格式引用来源
\t* 如果信息来自多个来源, 请列出所有相关的引文, 例如: [reference:3][reference:5]
\t* 引文应出现在正文中, 而不是集中在末尾

* 报告风格和格式: 
\t* 逻辑清晰: 保持清晰且结构良好的写作
\t* 易于阅读和理解。
\t* 有效使用 Markdown: 
\t\t* 使用表格呈现结构化数据
\t\t* 使用列表呈现关键信息
\t\t* 使用引用块突出重要内容
\t\t* 保持一致和美观的格式
\t* 写作与内容之间的衔接应像专业作家一样无缝, 使读者易于理解。

* **风格**: 保持自然和人性化的语调, 使其读起来像人写的, 而不是人工智能或机器写的。不要忘记使用 [reference:X]

* 章节格式要求: 
\t* 使用精确的 markdown 标题(#, ##, ### 等)来区分章节/小节/子小节级别
\t* 在整个报告中保持一致的小节/章节/部分划分
\t* 在续写新章节/部分时添加相应的 markdown 标题
\t* 确保章节标题不重复, 并与先前内容保持连贯性
\t* 新章节应与前文自然衔接, 避免结构上的脱节
\t* 保持章节层次结构的清晰和逻辑性

# 输出格式说明: 
首先, 在 <think></think> 标签中进行思考。然后, 请在 <article></article> 标签内继续写作。请使用与用户问题相同的语言进行回应。具体格式如下: 
<think>
思考如何继续写作
</think>
<article>
在此处写作
</article>

---
已撰写的报告如下: 

已撰写的报告: 
```
{to_run_article}
```

--
The Global Sections Plan and the writing task you should continue to write.
```
{to_run_global_writing_task}
```

根据 # Requirements 中的要求, 继续撰写 **{to_run_task}**。专注于这个写作任务。按照 # Output Format Instructions 的指示, 首先在 <think></think> 标签中进行思考, 然后在 <article></article> 标签内继续写作。写作风格应像人类一样自然。
"""
        super().__init__(system_message, content_template)