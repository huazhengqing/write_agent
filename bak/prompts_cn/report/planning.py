#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime

now = datetime.now()

@prompt_register.register_module()
class ReportPlanning(PromptTemplate):
    def __init__(self) -> None:
        system_message = ""
        content_template = """
Writing tasks that require further planning:
{to_run_task}

Reference planning:
{to_run_candidate_plan}

Reference thinking:
{to_run_candidate_think}
---

Overall plan:
```
{to_run_full_plan}
```
---

Results of analysis tasks completed:
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```
---


Already-written report content:
```
{to_run_article}
```
---
# 总体介绍
你是一个递归的专业报告撰写和信息搜集规划专家，专门根据深入的研究、搜索和分析来规划专业报告的撰写。一个为用户知识问题解决需求量身定制的高层计划已经存在，你的任务是在这个框架内进一步递归地规划指定的撰写子任务。通过你的规划，最终的报告将严格遵守用户要求，并在分析、逻辑和内容深度上达到完美。

1. 继续对指定的专业报告撰写子任务进行递归规划。根据研究和分析理论，将报告撰写的组织和分析任务的结果分解为更细粒度的撰写子任务，明确其范围和具体的撰写内容。
2. 根据需要规划分析子任务和搜索子任务，以辅助和支持具体的撰写工作。分析子任务可以包括设计大纲、详细大纲、数据分析、信息组织、逻辑结构构建和关键论点确定等任务，以支持实际的撰写。搜索子任务负责从互联网上收集必要的信息和数据。
3. 为每个任务规划一个子任务的有向无环图（DAG），其中边表示同一层DAG内搜索和分析任务之间的依赖关系。递归地规划每个子任务，直到所有子任务都成为原子任务。

# 任务类型
## 撰写 (核心，实际撰写)
- **功能**: 按照计划顺序执行实际的报告撰写任务。根据具体的撰写要求和已撰写的内容，结合分析任务和搜索任务的结论继续撰写。
- **所有撰写任务都是续写任务**: 在规划时确保与前面内容的连续性和逻辑一致性。撰写任务之间应该流畅无缝地衔接，保持报告的整体连贯性和统一性。
- **可分解任务**: 撰写、分析、搜索
- 除非必要，每个撰写子任务应大于500字。

## 分析
- **功能**: 分析和设计实际报告撰写之外的任何需求。这包括但不限于研究计划设计、设计大纲、详细大纲、数据分析、信息组织、逻辑结构构建、关键论点确定等，以支持实际的撰写。
- **可分解任务**: 分析、搜索

## 搜索
- **功能**: 执行信息收集任务，包括从互联网上收集必要的数据、材料和信息，以支持分析和撰写任务。
- **可分解任务**: 搜索

# 规划技巧
1. 从一个撰写任务派生出的最后一个子任务必须总是一个撰写任务。
2. 合理控制DAG每一层子任务的数量，**通常为3-5个子任务**。如果任务数量超过这个范围，应旨在进行递归规划。**不要在同一层规划超过8个子任务。**
3. **分析任务**和**搜索任务**可以作为**撰写任务的子任务**，应生成更多的分析子任务和搜索子任务以提高撰写质量。
4. 使用 `dependency` 列出同一层DAG内分析任务和搜索任务的ID。尽可能全面地列出所有潜在的依赖关系。
5. **当一个分析子任务涉及设计具体的写作结构时，后续依赖的写作任务不应被扁平化，而应进行递归。例如，根据该结构撰写xxx。** 永远不要忘记这一点。
6. **不要重复规划 `总体计划` 中已涵盖的任务，或重复 `已撰写报告内容` 和先前分析任务中已存在的内容。**
7. 遵循分析任务和搜索任务的结果。
8. 搜索任务的目标只指定信息需求，不指定来源或如何搜索。
**9**. 除非用户指定，每个撰写任务的长度应大于500字。

# 任务属性 (必需)
1. **id**: 子任务的唯一标识符，表明其层级和任务编号。
2. **goal**: 对子任务目标的精确、完整的字符串格式描述。
3. **dependency**: 一个ID列表，包含该任务所依赖的同一层DAG内的搜索和分析任务。尽可能全面地列出所有潜在的依赖关系。如果没有依赖的子任务，此项应为空。
4. **task_type**: 一个表示任务类型的字符串。撰写任务标记为 `write`，分析任务标记为 `think`，搜索任务标记为 `search`。
5. **length**: 对于撰写任务，此属性指定范围。撰写任务需要此属性。分析任务和搜索任务不需要此属性。
6. **sub_tasks**: 一个表示子任务DAG的JSON列表。列表中的每个元素都是一个代表任务的JSON对象。

# 示例
<示例 index=1>
用户给定的撰写任务:
{{
    "id": "",
    "task_type": "write",
    "goal": "生成一份详细的商业传记，记录 DeepSeek 的崛起",
    "length": "8600 字"
}}

提供一个部分完成的递归全局计划作为参考，以递归嵌套的JSON结构表示。`sub_tasks` 字段表示任务规划的DAG（有向无环图）。如果 `sub_tasks` 为空，则表示它是一个原子任务或尚未进一步规划的任务：

{{"id":"root","task_type":"write","goal":"Generate a detailed business biography to document DeepSeek's rise","dependency":[],"length":"8600 words","sub_tasks":[{{"id":"1","task_type":"search","goal":"Briefly collect DeepSeek's company information, including: founding team background, establishment time, financing history, product development history, technological breakthroughs, market performance and other key information, to determine the overall article structure","dependency":[],"sub_tasks":[]}},{{"id":"2","task_type":"think","goal":"Analyze DeepSeek's development trajectory and success factors, identify key milestone events, design the overall structure and key content of the biography","dependency":["1"],"sub_tasks":[]}},{{"id":"3","task_type":"write","goal":"Write biography content based on search results and designed overall structure and key content","length":"8600 words","dependency":["1","2"],"sub_tasks":[{{"id":"3.1","task_type":"write","goal":"Write the founder and team background chapter, focusing on Liang Wenfeng's quantitative investment experience and team characteristics","length":"1200 words","dependency":[],"sub_tasks":[{{"id":"3.1.1","task_type":"search","goal":"Collect detailed information about Liang Wenfeng's experience at Ubiquant, including entrepreneurial process, quantitative investment achievements, technical accumulation, etc.","dependency":[]}},{{"id":"3.1.2","task_type":"search","goal":"Collect detailed background information of DeepSeek's founding team, collect Ubiquant's AI technology reserve information, especially details of the 'Firefly' series supercomputing platform","dependency":[]}},{{"id":"3.1.3","task_type":"write","goal":"Complete the writing of founder background and team characteristics sections, highlighting Liang Wenfeng's quantitative investment achievements and AI layout, as well as the young team composition and technical strength","length":"1200 words","dependency":["3.1.1","3.1.2"]}}]}},{{"id":"3.2","task_type":"write","goal":"Write the company founding and initial vision chapter, describing the 2023 entrepreneurial background and positioning","length":"1000 words","dependency":[],"sub_tasks":[{{"id":"3.2.1","task_type":"search","goal":"Collect 2023 AI industry background materials, Search for deep reasons why Liang Wenfeng chose the AI track, especially DeepSeek's differentiated positioning","dependency":[],"sub_tasks":[]}},{{"id":"3.2.1","task_type":"write","goal":"Write about entrepreneurial background and era opportunities, as well as initial strategic positioning and technical route choices, especially the deep reasons for Liang Wenfeng choosing the AI track, and DeepSeek's differentiated positioning","length":"1000 words","dependency":["3.2.1"],"sub_tasks":[]}}]}},{{"id":"3.3","task_type":"write","goal":"Write key development nodes chapter, detailing the release and impact of three important products: V2, V3, and R1","length":"1800 words","dependency":[],"sub_tasks":[{{"id":"3.3.1","task_type":"search","goal":"Collect detailed information about DeepSeek V2, V3 and R1 releases, and their impact on the industry","dependency":[]}},{{"id":"3.3.2","task_type":"think","goal":"Analyze the technical progress path of the three products and their impact on the industry","dependency":["3.3.1"]}},{{"id":"3.3.3","task_type":"write","goal":"Write the chapter, including three sections: V2 triggering price war, V3's shocking release and R1's inference breakthrough","length":"1800 words","dependency":["3.3.1","3.3.2"],"sub_tasks":[]}}]}},{{"id":"3.4","task_type":"write","goal":"Based on the written releases and impacts of V2, V3, and R1, further write core technology and product advantages chapter, analyzing sources of competitiveness","length":"1500 words","dependency":[],"sub_tasks":[{{"id":"3.4.1","task_type":"search","goal":"Collect information about DeepSeek's technical innovations, computing power optimization solutions and engineering innovations","dependency":[],"sub_tasks":[]}},{{"id":"3.4.2","task_type":"write","goal":"Based on collected materials and analysis conclusions, write about model architecture innovation, hardware-software coordination optimization, and model optimization and distillation strategies","length":"1500 words","dependency":["3.4.1"],"sub_tasks":[]}}]}},{{"id":"3.5","task_type":"write","goal":"Write market competition pattern and business strategy chapter, analyzing the game with domestic and foreign competitors","length":"1200 words","dependency":[],"sub_tasks":[{{"id":"3.5.1","task_type":"search","goal":"Collect product strategies and market performance of major domestic and foreign large model companies (Baidu, Alibaba, etc.)","dependency":[],"sub_tasks":[]}},{{"id":"3.5.2","task_type":"search","goal":"Collect and analysis DeepSeek's differentiated competition strategy compared with other large model companies","dependency":["3.5.1","3.5.2"],"sub_tasks":[]}},{{"id":"3.5.3","task_type":"write","goal":"Based on collected materials and analysis conclusions, write about domestic competition pattern, international competitiveness and influence analysis, and business strategy innovation analysis","length":"1200 words","dependency":["3.5.1","3.5.2"],"sub_tasks":[]}}]}},{{"id":"3.6","task_type":"write","goal":"Further write industry influence and external response chapter, summarizing DeepSeek's social influence","length":"1000 words","dependency":[],"sub_tasks":[]}},{{"id":"3.7","task_type":"write","goal":"Write future outlook chapter, predicting DeepSeek's development direction and challenges","length":"900 words","dependency":[],"sub_tasks":[{{"id":"3.7.1","task_type":"search","goal":"Collect future development plans and goals revealed by DeepSeek officially","dependency":[],"sub_tasks":[]}},{{"id":"3.7.2","task_type":"write","goal":"Based on collected materials and analysis conclusions, write future outlook chapter, including future plans, technology innovation outlook, ecosystem building outlook, talent strategy outlook and internationalization outlook","length":"900 words","dependency":["3.7.1"],"sub_tasks":[]}}]}}]}}]}}
</example>

# 输出格式
1. 首先，在 `<think></think>` 标签中进行深入和全面的思考。
2. 在 `<result></result>` 标签中，按照示例的 JSON 格式输出规划结果。顶层对象应代表给定的任务，其 `sub_tasks` 是规划的结果。具体格式如下：
<think>
思考如何继续写作
</think>
<result>
在此处写入
</result>

---
需要进一步规划的写作任务，请遵循之前的要求，并按照 # 输出格式 进行输出，先在 <think></think> 中思考，然后直接在 <result></result> 中给出格式化的结果，不要忘记递归规划：
**{to_run_task}**
"""
        super().__init__(system_message, content_template)