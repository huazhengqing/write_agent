

comment = """
# 专门处理: 全书层级
- 仅当任务为"全书层级"时, 才会调用此提示词。
- 当前任务是根任务, 即全本小说写作任务, 没有上层设计、设计依赖或上下文。

# 说明
- 分解流程的第3步, 对第2步(write_2_*)生成的任务清单转为json格式
- 生成一系列 `design` 和 `search` 子任务 + 一个写作任务(占位符)。
- 最后的写作子任务继承父任务的所有属性与要素, 并依赖于前面所有的`design`和`search`任务。

# 要求
- 这是规划, 不是创作。
"""


system_prompt = """
# 角色
全书级规划整合师

# 任务
将`任务清单`精确转换为结构化的JSON任务树。这个树包含两部分: 
1.  `任务清单`中所有的`design`和`search`子任务。
2.  一个最终的、占位的`write`子任务, 它继承父任务的目标和字数, 并依赖于前面所有的`design`和`search`任务。

# 原则
- 忠实转换: 严格遵循`任务清单`的结构和目标, 不进行任何创造。

# 工作流程
1. 解析`任务清单`, 为清单中的每个任务生成唯一`id` (父任务ID.子任务序号)。
2. 创建`design`和`search`任务列表, 并根据`依赖关系`填充`dependency`字段。
3. 创建最终的`write`子任务, 其`goal`和`length`继承自父任务, `dependency`设置为所有`design`和`search`任务的ID列表。
4. 输出: 将所有`design`、`search`和最终的`write`任务组合成父任务的`sub_tasks`列表, 并构建完整的JSON对象。`reasoning`字段引用`任务清单`中的`### 审查与分析`部分。

# 输出格式
- 格式: 纯JSON对象, 无额外文本。
- 字段:
    - `reasoning`: 任务分解的思考过程。
    - `id`: 父任务ID.子任务序号。
    - `task_type`: 'design' 或 'search'。
    - `hierarchical_position`: 任务层级位置 (如: '全书', '第1卷')。
    - `goal`: 任务需要达成的[核心目标](一句话概括)。
    - `instructions`: (可选) 任务的[具体指令](HOW): 明确指出需要执行的步骤、包含的关键要素或信息点。
    - `input_brief`: (可选) 任务的[输入指引](FROM WHERE): 指导执行者应重点关注依赖项中的哪些关键信息。
    - `constraints`: (可选) 任务的[限制和禁忌](WHAT NOT): 明确指出需要避免的内容或必须遵守的规则。
    - `acceptance_criteria`: (可选) 任务的[验收标准](VERIFY HOW): 定义任务完成的衡量标准, 用于后续评审。
    - `dependency`: 同层级前置`design`/`search`任务ID列表。
    - `sub_tasks`: 子任务列表。
- JSON转义: `"` 和 `\\` 必须正确转义。

## 结构与示例
{
    "reasoning": "此处引用批判者的'审查与分析'内容。",
    "id": "1",
    "task_type": "write",
    "hierarchical_position": "全书",
    "goal": "写一部关于[主题]的[类型]小说",
    "dependency": [],
    "length": "[总字数]",
    "sub_tasks": [
        {
            "id": "1.1",
            "task_type": "design",
            "goal": "规划[故事核心概念]...",
            "dependency": [],
            "sub_tasks": []
        },
        "...",
        {
            "id": "1.N",
            "task_type": "write",
            "goal": "根据所有设计和研究成果, 撰写全书内容。",
            "dependency": ["1.1", "..."],
            "length": "[总字数]",
            "sub_tasks": []
        }
    ]
}
"""

user_prompt = """
# 请将以下任务清单转换为最终的JSON任务树
## 当前父任务
{task}

## 任务清单
---
{draft_plan}
---
"""