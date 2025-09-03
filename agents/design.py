import litellm
from loguru import logger
from ..util.models import Task
from ..util.llm import get_llm_params
from ..memory import memory
from ..prompts.story.design_cn import SYSTEM_PROMPT, USER_PROMPT


"""


分析、审查当前文件的代码，找出bug并改正， 指出可以优化的地方。


根据以上分析，改进建议， 请直接修改 文件，并提供diff。



"""


###############################################################################


async def design(task: Task) -> Task:
    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")
    if task.task_type != "design":
        raise ValueError("Task type must be 'design'.")
    context_dict = await memory.get_context(task)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(**context_dict)}
    ]
    llm_params = get_llm_params(messages, temperature=0.75)
    logger.info(f"{llm_params}")
    response = await litellm.acompletion(**llm_params)
    if not response.choices or not response.choices[0].message:
        raise ValueError("LLM API 调用失败，没有返回任何 choices 或 message。")
    message = response.choices[0].message
    reason = message.get("reasoning_content") or message.get("reasoning", "")
    content = message.content
    if not content:
        raise ValueError("LLM API 调用失败，没有返回任何 content。")
    updated_task = task.model_copy(deep=True)
    updated_task.results = {
        "result": content,
        "reasoning": reason,
    }
    logger.info(f"{updated_task}")
    return updated_task




