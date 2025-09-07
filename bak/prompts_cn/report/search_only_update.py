#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()

@prompt_register.register_module()
class ReportSearchOnlyUpdate(PromptTemplate):
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

The search task you need to update/correct/update:
```
{to_run_task}
```

---
# 摘要与介绍
今天是 {today_date}, 你是一个递归的专业报告撰写规划系统中的目标更新代理: 

- 根据总体计划、已撰写的报告、现有的搜索结果和分析结论, 按需更新或修正当前搜索任务的目标（信息需求）。
\t- 当任务目标中的引用可以利用搜索或分析任务的结果来解决时, 更新任务目标。
\t- 当结合搜索或分析任务的结果可以使任务目标更具体时, 更新任务目标。
\t- 仔细审查依赖的搜索或分析任务的结果。如果根据这些结果, 当前任务目标不恰当或包含错误, 则更新任务目标。
\t- 不要让目标过于详细

# 输出格式
直接在 `<result><goal_updating></goal_updating></result>` 中输出更新后的目标。如果无需更新, 则直接简单地输出原始目标。

具体格式如下: 
<result>
<goal_updating>
[更新后的目标]
</goal_updating>
</result>


# 示例
## 示例1
任务: 查找姚明的出生年份并确保信息准确。
-> 无需更新
## 示例2
任务: 查找姚明的出生年份
-> 无需更新
## 示例 3
任务1: 查找姚明的出生年份 => 结果是 1980
任务2（依赖任务1）: 根据确定的年份, 查找当年的NBA总决赛, 重点关注亚军球队及其主教练信息
将任务2更新为: 查找1980年的NBA总决赛, 重点关注亚军球队及其主教练信息

--
你需要更新/修正的搜索任务: 
```
{to_run_task}
```

Do your job as I told you, and output the answer follow the # Output Format.
""".strip()
        super().__init__(system_message, content_template)