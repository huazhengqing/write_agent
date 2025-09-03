#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime

now = datetime.now()

@prompt_register.register_module()
class StoryWrtingNLReasonerEN(PromptTemplate):
    def __init__(self) -> None:
        system_message = ""

        content_template = """
The collaborative story-writing requirement to be completed: **{to_run_root_question}**

Your specific story design task to complete: **{to_run_task}**

---
The existing novel design conclusions are as follows:
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

---
already-written novel:
```
{to_run_article}
```

---
你是一位富有创新精神的专业作家，正在与其他专业作家合作，共同创作一个满足特定用户需求的创意故事。你的任务是完成分配给你的故事设计任务，旨在创新性地支持其他作家的写作和设计工作，从而为整部小说的完成做出贡献。

注意！！你的设计成果应与现有的小说设计结论在逻辑上保持一致和连贯。

# 设计提示
- **结构**：你叙事的整体架构，包括情节发展、节奏和叙事弧（阐述、上升、高潮、下降、解决）。
- **角色发展**：角色在整个故事中如何被引入、塑造和演变。
- **视角**：故事叙述的视角（第一人称、第三人称有限视角、全知视角等）。
- **背景设定**：时间和地点的发展，包括世界观构建元素。
- **主题**：所探讨的潜在信息或中心思想。
- **基调与氛围**：在整个作品中创造并维持的情感氛围。
- **对话**：角色如何说话和进行口头互动。
- **写作风格**：你独特的声音，包括句子结构、词语选择和修辞手法。
- **叙事技巧**：如伏笔、闪回、象征和反讽等工具。
- **场景构建**：单个场景如何构建，包括它们之间的过渡。

# 输出格式
1. 首先，在 `<think></think>` 标签内进行思考。
2. 然后，在 `<result></result>` 标签内，以结构化、可读的格式撰写设计结果，并提供尽可能多的细节。

请根据要求，以合理且创新的方式完成故事设计任务 **{to_run_task}**。
""".strip()
        super().__init__(system_message, content_template)

 
@prompt_register.register_module()
class StoryWritingReasonerFinalAggregate(PromptTemplate):
    def __init__(self) -> None:
        system_message = ""


# , **providing details which are useful for writing, do not provide too much un-useful details**.

        content_template = """
The collaborative story-writing requirement to be completed: **{to_run_root_question}**

Your specific story design task to complete: **{to_run_task}**

---
The existing novel design conclusions are as follows:
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

---
already-written novel:
```
{to_run_article}
```

---
**The novel design conclusions you need to integrate and refine**, to give the final story design task: **{to_run_task}**:
```
{to_run_final_aggregate}
```

---
你是一位富有创新精神的专业作家，与其他专业作家合作，创作一个满足用户特定需求的创意故事。你的任务是**整合和完善**多位小说设计师提供的故事设计成果，并完成分配给你的故事设计任务，确保最终的设计具有**创新性**、逻辑一致性和连贯性。你需要解决潜在的冲突，增强元素之间的联系，并在必要时填补空白，以产生一个统一且引人入胜的故事等。

注意！！你的设计成果应与小说设计提供的结论保持**逻辑一致性**和**连贯性**，同时提升整体小说设计的质量。

# 整合与完善要求
- **整合**：
  - 结合并综合来自多位小说设计师的输入，确保所有元素（例如，情节、角色、主题）统一并连贯成一个整体。
  - 识别并解决不同推理器（Reasoner）结果之间的逻辑不一致或矛盾。
  - 确保没有遗漏关键设计元素，并且所有方面都有助于故事的进展和深度。

- **完善**：
  - 增强整合后设计的清晰度、深度和情感共鸣。
  - 填补空白或详细阐述缺乏足够细节或发展的领域。
  - 确保故事的基调、节奏和风格始终保持一致。

- **创新与影响**：
  - 验证整体故事设计保持原创性并避免陈词滥调。
  - 深化普世或深刻的主题，确保它们能与读者产生共鸣。
  - 引入微妙的改进或创造性的增强，以提升故事的整体影响力。

  
# 设计提示
- **结构**：你叙事的整体架构，包括情节发展、节奏和叙事弧（阐述、上升、高潮、下降、解决）。
- **角色发展**：角色在整个故事中如何被引入、塑造和演变。
- **视角**：故事叙述的视角（第一人称、第三人称有限视角、全知视角等）。
- **背景设定**：时间和地点的发展，包括世界观构建元素。
- **主题**：所探讨的潜在信息或中心思想。
- **基调与氛围**：在整个作品中创造和维持的情感氛围。
- **对话**：角色如何说话和进行口头互动。
- **写作风格**：你独特的声音，包括句子结构、词语选择和比喻性语言。
- **叙事技巧**：如伏笔、闪回、象征和反讽等工具。
- **场景构建**：单个场景如何构建，包括它们之间的过渡。


# 输出格式
1. 首先，在 `<think></think>` 标签内进行深入思考，在思考过程中考虑各种可能性。确保审查和评估所有推理器（Reasoner）的输入，解决不一致之处，并确定需要完善或增强的领域。
2. 然后，在 `<result></result>` 标签内，以**结构化且可读的格式**编写**最终**的设计结果。


请根据要求，以合理且创新的方式完成故事设计任务 **{to_run_task}**。
""".strip()
        super().__init__(system_message, content_template)