#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()
import json

@prompt_register.register_module()
class StoryWrtingNLWriterEN(PromptTemplate):
    def __init__(self) -> None:
        system_message = """
你是一位专业且富有创新精神的作家, 与其他作家合作, 共同创作用户要求的小说。

### 要求: 
- 从故事的上一个结尾开始, 与现有文本的写作风格、词汇和整体氛围相匹配。根据写作要求自然地完成你的部分, 不要重新解释或重新描述已经涵盖的细节或事件。
- 密切关注现有的小说设计结论。
- 使用修辞、语言和文学手法（例如, 模糊、头韵）来创造引人入胜的效果。
- 避免平淡或重复的短语（除非有意用于创造叙事、主题或语言效果）。
- 使用多样化和丰富的语言: 变化句子结构、词语选择和词汇。
- 除非绝对必要, 否则避免总结性、解释性或说明性的内容或句子。
- 确保情节或描述中没有脱节感或突兀感。你可以写一些过渡性内容, 以保持与现有材料的完全连续性。

### 指示: 
首先, 在 `<think></think>` 中反思任务。然后, 在 `<article></article>` 中继续故事的创作。
""".strip()

        
        content_template = """
The collaborative story-writing requirement to be completed:  
**{to_run_root_question}**  

Based on the existing novel design conclusions and the requirements, continue writing the story. You need to continue writing:  
**{to_run_task}**

---
The existing novel design conclusions are as follows, you should obey it:  
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

---
already-written novel:
```
{to_run_article}
```

Based on the requirements, continue writing **{to_run_task}**.
"""
        super().__init__(system_message, content_template)