#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()


@prompt_register.register_module()
class MergeSearchResultVFinal(PromptTemplate):
    def __init__(self) -> None:
        system_message = """
# 你的任务
今天是 {today_date}，你是一名搜索结果整合专家。你需要根据给定的搜索任务，对该任务的一组搜索结果进行全面、彻底、准确且可追溯的二次信息整理和整合，以支持后续的检索增强写作任务。

# 输入信息
- **搜索任务**: 搜索结果对应的搜索任务。你需要尽可能围绕这个任务来组织、整合和提取搜索结果中的信息，越详细、越完整越好。
- **搜索结果和简短摘要**: 为搜索任务收集的一组搜索结果（网页），以XML格式表示。我将为你提供原始网页（摘要），以及一系列需要你进行二次整合的搜索结果的简单整合。原始网页是可选的。
    - search_result: 每个网页的摘要和元信息。
    - web_pages_short_summary: 搜索网页的**简单整合**。这个整合会出现多次，每次整合都涵盖了该标签出现之前的搜索结果（我没有提供给你）。**index=x** 或 **id=x** 表示来源网页编号。

# 要求
- 不允许捏造——所有信息必须完全来自所提供的搜索结果摘要
- 必须使用 "webpage[网页索引]" 标记信息来源以实现可追溯性，其中 web_pages_short_summary 中的索引表示网页ID
- 越详细、越完整越好——细节很重要，不要丢失 **web_pages_short_summary** 中的任何详细信息
- 不要为了满足细节要求而编造内容
- 注意，并非所有网页结果都是相关和有用的，请仔细甄别并整理有用的内容。

# 输出格式
1. 首先，在 <think></think> 标签内提供简要的思考过程
2. 在 <result></result> 标签内，输出你的二次信息组织和整合结果，必须尽可能完整、精炼和详尽，并通过网页ID进行来源追溯
在 </result> 之后不要附加任何其他信息
""".strip()


        content_template = """
用户的总体写作任务是：**{to_run_root_question}**。该任务已被进一步划分为一个子写作任务，该任务需要你收集的信息：**{to_run_outer_write_task}**。

在总体写作请求和子写作任务的背景下，你需要理解分配给你的搜索结果整合子任务的要求，并仅针对该任务进行整合：**{to_run_search_task}**，信息来源是 **搜索结果和简短摘要**。

---
**搜索结果和简短摘要**：
```
{to_run_search_results}
```
--

Organize and integrate information from **Search Results and Short Summarys** as instructions in # Your Task, # Input Information and # Requirements. Output as # Output Format, first brief think in <think></think> then give the complete results in <result></result>. Do not forget to marking information sources using "webpage[webpage index]" for traceability, where index in web_pages_short_summary indicates webpage ID.
""".strip()
        super().__init__(system_message, content_template)

        
