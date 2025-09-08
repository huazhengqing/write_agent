#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime

now = datetime.now()


fewshot = """
<example index=1>
User-given writing task:
{
    "id": "",
    "task_type": "write",
    "goal": "Create an excellent suspense story set in the colony on Europa (Jupiter’s moon), incorporating the real environmental characteristics of Europa. Write the story directly, skillfully integrating the background setting, characters, character relationships, and plot conflicts into the story to make it gripping. Character motivations and plot development must be logical and free of contradictions.",
    "length": "4000 words"
}

A partially complete recursive global plan is provided as a reference, represented in a recursively nested JSON structure. The `sub_tasks` field represents the DAG (Directed Acyclic Graph) of the task planning. If `sub_tasks` is empty, it indicates an atomic task or one that has not yet been further planned:

{"id":"","task_type":"write","goal":"Create an excellent suspense story set in the colony on Europa (Jupiter’s moon), incorporating the real environmental characteristics of Europa. Write the story directly, skillfully integrating the background setting, characters, character relationships, and plot conflicts into the story to make it gripping. Character motivations and plot development must be logical and free of contradictions.","dependency":[],"length":"4000 words","sub_tasks":[{"id":"1","task_type":"think","goal":"Design the main characters and core suspense elements of the story. This includes the cause, progression, and resolution of the suspense event, as well as determining the theme and ideas the story aims to express. This includes the truth behind the case, a misleading clue system, and the pacing of the truth reveal.","dependency":[],"sub_tasks":[{"id":"1.1","task_type":"think","goal":"Design the truth behind the case: determine the nature of the event, the specific individuals involved, their motivations, methods, and timeline.","dependency":[]},{"id":"1.2","task_type":"think","goal":"Design a misleading clue system: include false suspects, misleading evidence, and coincidental events.","dependency":["1.1"]},{"id":"1.3","task_type":"think","goal":"Design the pacing of the truth reveal: plan the order of key clues, methods for eliminating misleading clues, and the gradual progression of the truth.","dependency":["1.1","1.2"]},{"id":"1.4","task_type":"think","goal":"Determine the theme and ideas of the story, exploring deeper issues such as humanity, technology, and ethics.","dependency":["1.1","1.2","1.3"]}]},{"id":"2","task_type":"think","goal":"Based on the results of Task 2, further refine the character design. Create detailed settings for the main characters, including their backgrounds, personalities, motivations, and relationships with one another. This includes public relationships, hidden relationships, conflicts of interest, emotional bonds, and their psychological changes and behavioral evolution throughout the events.","dependency":["1"],"sub_tasks":[{"id":"2.1","task_type":"think","goal":"Create detailed backgrounds for the main characters, including their life experiences, professional skills, and personal goals.","dependency":[]},{"id":"2.2","task_type":"think","goal":"Detail the characters' personality traits and behavior patterns to create vivid characterizations.","dependency":["2.1"]},{"id":"2.3","task_type":"think","goal":"Design the character relationship network, including public relationships, hidden relationships, conflicts of interest, and emotional bonds.","dependency":["2.1","2.2"]},{"id":"2.4","task_type":"think","goal":"Based on the designed suspense elements and plot, plan the psychological changes and behavioral evolution of the characters during the events, reflecting their growth or downfall.","dependency":["2.1","2.2","2.3"]}]},{"id":"3","task_type":"think","goal":"Integrate the design elements and refine the story framework. This includes designing the story structure, plotting the development of events, setting up turning points, pacing the clues, and designing key scenes and atmospheres.","dependency":["1","2"],"sub_tasks":[{"id":"3.1","task_type":"think","goal":"Design the story structure: plan the main content for the beginning, development, climax, and ending.","dependency":[]},{"id":"3.2","task_type":"think","goal":"Plot the development of events, arranging the order and pacing of key events.","dependency":["3.1"]},{"id":"3.3","task_type":"think","goal":"Set up turning points and the climax to ensure the story's tension and appeal.","dependency":["3.1","3.2"]},{"id":"3.4","task_type":"think","goal":"Plan the layout and reveal of clues to maintain the suspense.","dependency":["3.1","3.2","3.3"]},{"id":"3.5","task_type":"think","goal":"Design key scenes and atmospheres to highlight the sci-fi and suspense features of the story.","dependency":["3.1","3.2","3.3","3.4"]}]},{"id":"4","task_type":"write","goal":"Based on the previous designs, write the complete story, including the opening, development, climax, and ending. Skillfully integrate Europa’s environmental characteristics and suspense elements to make it gripping.","dependency":["1","2","3"],"length":"4000 words","sub_tasks":[{"id":"4.1","task_type":"write","goal":"Write the opening part of the story. Introduce the colony environment and main characters through a specific scene and lay the groundwork for suspense.","dependency":[],"length":"800 words","sub_tasks":[{"id":"4.1.1","task_type":"think","goal":"Conceive the specific scene for the story's opening, selecting one that showcases the Europa colony's environment and main characters.","dependency":[]},{"id":"4.1.2","task_type":"think","goal":"Determine the suspenseful hints to be laid in the opening, considering how to subtly introduce suspense elements.","dependency":["4.1.1"]},{"id":"4.1.3","task_type":"write","goal":"Write the opening part of the story, incorporating the designed scene and suspenseful hints.","dependency":["4.1.1","4.1.2"],"length":"800 words"}]},{"id":"4.2","task_type":"write","goal":"Write the event outbreak and initial investigation part, describing the occurrence of the suspenseful event and the characters’ initial reactions and investigation.","dependency":[],"length":"1200 words","sub_tasks":[{"id":"4.2.1","task_type":"think","goal":"Further detail the process of the suspenseful event's occurrence, including time, location, and method, ensuring the plot is logical and engaging.","dependency":[]},{"id":"4.2.2","task_type":"think","goal":"Based on the settings, further consider the main characters' reactions and initial investigative actions after the event, reflecting their personalities and relationships.","dependency":["4.2.1"]},{"id":"4.2.3","task_type":"write","goal":"Write the event outbreak and initial investigation part, showcasing the suspenseful event and characters' reactions to advance the plot.","dependency":["4.2.1","4.2.2"],"length":"1200 words"}]},{"id":"4.3","task_type":"write","goal":"Write the in-depth investigation section, revealing character relationships through the investigation process and increasing the story's suspense.","dependency":[],"length":"1000 words","sub_tasks":[{"id":"4.3.1","task_type":"think","goal":"Based on the settings, further detail the clues and misleading information that appear during the in-depth investigation, adding complexity and suspense to the story.","dependency":[]},{"id":"4.3.2","task_type":"think","goal":"Based on the settings, further consider the interactions between the main characters during the investigation, revealing their backgrounds and hidden relationships.","dependency":["4.3.1"]},{"id":"4.3.3","task_type":"write","goal":"Write the in-depth investigation section, integrating the designed clues and character interactions to enhance the story's suspense.","dependency":["4.3.1","4.3.2"],"length":"1000 words"}]},{"id":"4.4","task_type":"write","goal":"Write the climax and ending section, resolving the mystery and showcasing the characters' fates and the story's theme.","dependency":[],"sub_tasks":[],"length":"1000 words"}]}]}
</example>
"""


@prompt_register.register_module()
class StoryWritingNLPlanningEN(PromptTemplate):
    def __init__(self) -> None:
        system_message = """
# 总体介绍
你是一位递归的专业小说写作规划专家, 擅长根据叙事理论规划专业的小说写作。一个为用户小说写作需求量身定制的高层计划已经制定好, 你的任务是在这个框架内进一步递归地规划指定的写作子任务。通过你的规划, 最终的小说将严格遵循用户要求, 并在情节、创意(想法、主题和话题)和发展方面达到完美。

1. 继续对指定的专业小说写作子任务进行递归规划。根据叙事理论、故事写作的组织方式以及设计任务的结果, 将任务分解为更细粒度的写作子任务, 明确它们的范围和具体的写作内容。
2. 根据需要规划设计子任务, 以协助和支持具体的写作。设计子任务用于设计包括大纲、角色、写作风格、叙事技巧、视角、背景、主题、基调和场景构建等元素, 以支持实际的写作。
3. 为每个任务规划一个子任务的有向无环图(DAG), 其中边表示同一层DAG内设计任务之间的依赖关系。递归地规划每个子任务, 直到所有子任务都是原子任务。

# 任务类型
## 写作(核心, 实际写作)
- **功能**: 按照计划顺序执行实际的小说写作任务。根据具体的写作要求和已写内容, 结合设计任务的结论继续写作。
- **所有写作任务都是续写任务**: 在规划时确保与前面内容的连续性。写作任务之间应该流畅无缝地衔接。
- **可分解的任务**: 写作、设计
- 除非必要, 每个写作子任务应超过500字。不要将少于500字的写作任务分解为子写作任务。

## 设计
- **功能**: 分析和设计除实际写作之外的任何小说写作需求。这可能包括大纲、角色、写作风格、叙事技巧、视角、背景、主题、基调和场景构建等, 以支持实际的写作。
- **可分解的任务**: 设计

# 提供给你的信息
- **`已写的小说内容`**: 先前写作任务中已经写好的内容。
- **`总体计划`**: 整体写作计划, 通过 `is_current_to_plan_task` 键指定了你需要规划的任务。
- **`上层任务中已完成的设计任务的结果`**
- **`同一层DAG任务所依赖的设计任务的结果`**
- **`需要进一步规划的写作任务`**
- **`参考规划`**: 提供一个规划示例, 你可以谨慎参考。

# 规划技巧
1. 从写作任务派生出的最后一个子任务必须始终是写作任务。
2. 合理控制DAG每一层子任务的数量, 通常为 **2到5** 个子任务。如果任务数量超过这个范围, 请进行递归规划。
3. **设计任务** 可以作为 **写作任务的子任务**, 并且应尽可能多地生成设计子任务以提高写作质量。
4. 使用 `dependency` 列出同一层DAG内设计任务的ID。尽可能全面地列出所有潜在的依赖关系。
5. 当一个设计子任务涉及设计特定的写作结构(例如, 情节设计)时, 后续的依赖写作任务不应平铺展开, 而应等待后续轮次的递归规划。
6. **不要重复规划 `总体计划` 中已涵盖的任务, 或重复 `已写的小说内容` 和之前设计任务中已有的内容。**
7. 写作任务应流畅无缝, 确保叙事的连续性。
8. 遵循设计任务的结果
**9**. 除非用户特别指定, 每个写作任务的长度应大于500字。不要将少于500字的写作任务分解为子写作任务。

# 任务属性
1. **id**: 子任务的唯一标识符, 表示其层级和任务编号。
2. **goal**: 以字符串格式对子任务目标进行精确而完整的描述。
3. **dependency**: 该任务所依赖的同一层DAG中的设计任务ID列表。尽可能全面地列出所有潜在的依赖关系。如果没有依赖的子任务, 此项应为空。
4. **task_type**: 一个表示任务类型的字符串。写作任务标记为 `write`, 设计任务标记为 `think`。
5. **length**: 对于写作任务, 此属性指定范围, 是写作任务的必需属性。设计任务不需要此属性。
6. **sub_tasks**: 一个表示子任务DAG的JSON列表。列表中的每个元素都是一个代表任务的JSON对象。


# 示例
{}

# 输出格式
1. 首先, 在 `<think></think>` 中进行深入和全面的思考。
2. 然后, 在 `<result></result>` 中, 按照示例所示的JSON格式输出规划结果。顶层对象应代表给定的任务, 其 `sub_tasks` 是规划的结果。
""".strip().format(fewshot)
        
        content_template = """
Writing tasks that require further planning:
{to_run_task}

Reference planning: 
{to_run_candidate_plan}

Reference thinking:
{to_run_candidate_think}
---

Already-written novel content: 
```
{to_run_article}
```

Overall plan: 
```
{to_run_full_plan}
```

Results of design tasks completed in higher-level tasks: 
```
{to_run_outer_graph_dependent}
```

Results of design tasks dependent on the same-layer DAG tasks: 
```
{to_run_same_graph_dependent}
```

Plan the writing task according to the aforementioned requirements and examples.  
""".strip()
        super().__init__(system_message, content_template)
    
