#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()

@prompt_register.register_module()
class ReportAtomWithUpdate(PromptTemplate):
    def __init__(self) -> None:
        system_message = ""
        
        content_template = """
results of search and analysis tasks completed:
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

already-written report:
```
{to_run_article}
```

overall plan
```
{to_run_full_plan}
```

The writing task you need to evaluate:
```
{to_run_task}
```

---
# 总结与介绍
今天是 {today_date}，你是一个递归式专业报告写作规划系统中的目标更新和原子写作任务判定智能体：

1. **目标更新**：基于总体计划、已写报告、现有搜索结果和分析结论，根据需要更新、修正或修订当前的写作任务要求，使其更符合需求、更具体。例如，根据搜索结果和设计结论提供更详细的要求，或删除已写报告中的冗余内容。

2. **原子写作任务判定**：在总体计划和已写报告的背景下，评估给定的写作任务是否是原子任务，即不需要进一步规划。根据研究和分析理论，一个写作任务可以进一步分解为更细粒度的写作子任务、搜索子任务和分析子任务。写作任务涉及具体文本部分的实际创作，而分析子任务可以包括设计大纲、详细大纲、数据分析、信息组织、逻辑结构构建和关键论点确定等任务，以支持实际写作；搜索子任务负责从互联网收集必要的信息和数据。

# 目标更新技巧
- 当任务目标中的引用可以使用搜索或分析任务的结果来解决时，更新任务目标。
- 当结合搜索任务或分析任务的结果可以使任务目标更具体时，更新任务目标。
- 仔细审查依赖的搜索或分析任务的结果。如果根据这些结果，当前的任务目标不合适或包含错误，则更新任务目标。
- 直接输出更新后的目标。如果不需要更新，则输出原始目标。

# 原子任务判定规则
按顺序独立判断以下三类子任务是否需要拆分：

1. **分析子任务**：如果写作需要某些设计来支持，而这些设计要求没有由**依赖的设计任务**或**已完成的报告内容**提供，那么就需要规划一个设计子任务。

2. **搜索子任务**：如果写作需要外部信息（如文献、学术成果、行业数据、政策文件、在线资源等），而这些信息没有由**依赖的任务**或**已完成的报告内容**提供，那么就需要规划一个搜索子任务。

3. **写作子任务**：当且仅当需要大量文本输出时，至少 > 1000字，可能需要分解为多个写作子任务，以减轻一次性创作的负担。**当 > 1500字时，必须分解。**

如果需要创建分析子任务、搜索子任务或写作子任务中的任何一个，该任务就被认为是复杂任务。

# 报告要求（可通过分析任务和搜索任务实现）
- **数据准确性和证据支持**：
\t- **详细数据**：报告必须依赖于来自权威来源的全面而准确的数据。
\t- **可靠证据**：每个论点都必须有可靠的数据或文献支持。
- **系统性论证**：确保报告包含系统和透彻的推理
\t- 超越表面观察的深刻见解
\t- 对多种观点的批判性评估
\t- 有证据支持的、论证充分的结论
\t- 原创且发人深省的解读
- **信息整合与多角度分析**：
\t- **全面整合**：结合来自多个角度和来源的信息和数据。
\t- **彻底验证**：总结、比较和验证各种论点，以确保全面深入的分析。

# 输出格式要求
1. 首先，在 `<think></think>` 中思考目标更新。然后，根据原子任务判定规则，深入全面地评估是否需要分解分析、搜索和写作子任务。这决定了该任务是原子任务还是复杂任务。

2. 然后，在 `<result></result>` 中输出结果。在 `<goal_updating></goal_updating>` 中，直接输出更新后的目标；如果不需要更新，则输出原始目标。在 `<atomic_task_determination></atomic_task_determination>` 中，输出任务是原子任务还是复杂任务。

具体格式如下：
<think>
思考目标更新；然后根据原子任务判定规则进行思考。
</think>
<result>
<goal_updating>
[更新后的目标]
</goal_updating>
<atomic_task_determination>
atomic/complex
</atomic_task_determination>
</result>

===
你需要评估的写作任务：
```
{to_run_task}
```
Complete the goal-updating and atomic writing task determination job as requirements in # Summary and Introduction, # Goal Updating Tips, # Atomic Task Determination Rules and # Report Requirements. Output follow the # Output Format Requirement, think in <think></think> and output the result in <result></result>
""".strip()
        super().__init__(system_message, content_template)
        
        
@prompt_register.register_module()
class ReportAtom(PromptTemplate):
    def __init__(self) -> None:
        system_message = ""

        content_template = """
results of search and analysis tasks completed: 
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

already-written report:
```
{to_run_article}
```

overall plan
```
{to_run_full_plan}
```

The writing task you need to evaluate:
```
{to_run_task}
```

---
# 摘要与介绍
你是一个递归式专业报告写作规划系统中的原子写作任务判定智能体：

在整体计划和已完成报告的背景下，评估给定的写作任务是否为原子任务，即不需要进一步规划。根据研究和分析理论，一个写作任务可以进一步分解为更细粒度的写作子任务、搜索子任务和分析子任务。写作任务涉及具体文本内容的实际创作，而分析子任务可以包括设计大纲、详细大纲、数据分析、信息组织、逻辑结构构建和关键论点确定等任务，以支持实际写作；搜索子任务负责从互联网上收集必要的信息和数据。

# 原子任务判定规则
按顺序独立判断以下三类子任务是否需要拆分：

1. **分析子任务**：如果写作需要某些设计来支持，而这些设计要求没有被**依赖的设计任务**或**已完成的报告内容**提供，那么就需要规划一个分析子任务。

2. **搜索子任务**：如果写作需要外部信息（如文献、学术成果、行业数据、政策文件、网络资源等），而这些信息没有被**依赖的任务**或**已完成的报告内容**提供，那么就需要规划一个搜索子任务。

3. **写作子任务**：当且仅当需要大量文本输出时，至少 > 1000字，可能需要将其分解为多个写作子任务，以减轻一次性创作的负担。**当 > 1500字时，必须进行分解。**

如果需要创建分析子任务、搜索子任务或写作子任务中的任何一个，该任务就被认为是复杂任务。

# 报告要求（可通过分析任务和搜索任务实现）
- **数据准确性与证据支持**：
\t- **详实数据**：报告必须依赖于来自权威来源的全面、准确的数据。
\t- **可靠证据**：每个论点都必须有可靠的数据或文献支持。
- **系统性论证**：确保报告包含系统和深入的推理
\t- 超越表层观察的深刻见解
\t- 对多种观点的批判性评估
\t- 有证据支持的、论证充分的结论
\t- 独到且发人深省的解读
- **信息整合与多角度分析**：
\t- **全面整合**：结合来自多个角度和来源的信息与数据。
\t- **充分验证**：总结、比较和验证各种论点，以确保分析的全面性和深度。

# 输出格式
1. 首先，在 `<think></think>` 标签中，遵循原子任务判定规则，依次评估是否需要拆分分析、搜索和写作子任务。这将决定该任务是原子任务还是复杂任务。

2. 然后，在 `<result><atomic_task_determination></atomic_task_determination</result>` 标签中输出结果。

具体格式如下：
<think>
根据原子任务判定规则进行思考。
</think>
<result>
<atomic_task_determination>
原子/复杂
</atomic_task_determination>
</result>

===
你需要评估的写作任务：
```
{to_run_task}
```
Complete the atomic writing task determination job as requirements in # Summary and Introduction, # Atomic Task Determination Rules and # Report Requirements. Output follow the # Output Format Requirement, think in <think></think> and output the result in <result></result>
""".strip()


        super().__init__(system_message, content_template)