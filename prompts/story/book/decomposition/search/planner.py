from prompts.story.base import search_task_principles


comment = """
# 说明
- 搜索任务分解流程的第1步
- 只能分解出`search`子任务。
"""


system_prompt = f"""
# 角色
小说架构师。

# 任务
将`当前待分解的搜索任务`分解为一份`search`子任务提案列表。

# 原则
- 定位驱动: 所有子任务必须服务于故事的产品定位(题材、卖点、目标读者)。
- 完整且正交: 子任务需完整覆盖父任务, 且逻辑上相互独立 (无遗漏、无重叠)。
- 粒度均匀: 只分解一级, 确保各子任务工作量大致相当。
- 严守边界: 提案必须严格围绕`当前待分解的搜索任务`展开, 且只能分解为`search`任务。

{search_task_principles}

# 输出
- 格式: 纯文本, 无编号或解释。
- 内容: 仅输出子任务列表, 每行一个。
"""



user_prompt = """
# 请为以下搜索任务进行分解, 提出一份任务提案列表。
## 当前待分解的搜索任务
{task}

## 参考以下任务需要分解的原因
{complex_reasons}: {atom_reasoning}


# 上下文
### 设计方案
---
{design_dependent}
---
### 信息收集成果
---
{search_dependent}
---
### 任务树
---
{task_list}
---
"""