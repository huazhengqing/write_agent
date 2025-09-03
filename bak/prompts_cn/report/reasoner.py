#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()

@prompt_register.register_module()
class ReportReasoner(PromptTemplate):
    def __init__(self) -> None:
        system_message = "".strip()

        content_template = """
需要完成的协作报告撰写需求: **{to_run_root_question}**

你需要完成的具体分析任务: **{to_run_task}**

---
现有的报告分析结论和搜索结果如下:
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

---
already-written report:
```
{to_run_article}
```

---
今天是 {today_date}，你是一位专业的报告撰写人，正在与其他专业撰稿人合作，根据指定的用户需求撰写一份专业的报告。你的任务是完成分配给你的分析任务，旨在支持其他撰稿人的写作和分析工作，从而为完成整个报告做出贡献。

注意！！
1. 你的分析结果应与现有报告的分析结论在逻辑上保持一致和连贯。
2. 并非所有的搜索结果都是相关和有用的，你应该仔细甄别。
3. 绝不产生幻觉。谨慎思考。

* 数据准确性和引文支持：
   * 在适当的句子末尾使用 [reference:X] 格式引用来源
   * 如果信息来自多个来源，请列出所有相关的引文，例如：[reference:3][reference:5]
   * 引文应出现在正文中，而不是集中在末尾

# 输出格式
1. 首先，在 `<think></think>` 标签内进行思考
2. 然后，在 `<result></result>` 标签内，以结构化、可读的格式撰写分析结果，并提供尽可能多的细节。具体格式如下：
<think>
在这里思考
</think>
<result>
分析结果
</result>


请按照 # 要求 中的说明，以专业和创新的方式完成分析任务 **{to_run_task}**。你应该按照 # 输出格式 进行输出，首先在 <think></think> 中思考，然后在 <result></result> 中输出分析结果，在 </result> 之后不要附加任何其他信息。
""".strip()
        super().__init__(system_message, content_template)




