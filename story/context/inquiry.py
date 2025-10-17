import importlib
from typing import Literal
from loguru import logger
from story.models.inquiry import InquiryOutput
from utils import call_llm
from utils.llm import get_llm_messages, get_llm_params
from utils.models import Task
from utils.sqlite_task import get_task_db


async def inquiry(
    inquiry_type: Literal['search', 'design', 'summary'], 
    task: Task,
    book_level_design: str,
    global_state_summary: str,
    design_dependent: str,
    search_dependent: str,
    latest_text: str,
    overall_planning: str,
) -> InquiryOutput:
    task_db = get_task_db(task.run_id)
    task_data = task_db.get_task_by_id(task.id)
    field_name = f"inquiry_{inquiry_type}"
    if task_data and (existing_inquiry := task_data.get(field_name)):
        try:
            return InquiryOutput.model_validate_json(existing_inquiry)
        except Exception as e:
            logger.warning(f"解析任务 {task.id} 的 {field_name} 失败: {e}。将重新调用 LLM。")

    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "overall_planning": overall_planning,
    }
    module = importlib.import_module(f"story.prompts.inquiry.{inquiry_type}")
    messages = get_llm_messages(module.system_prompt, module.user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.1)
    response = await call_llm.completion(llm_params, output_cls=InquiryOutput)
    
    inquiry_result = response.validated_data
    if inquiry_result:
        inquiry_content = inquiry_result.model_dump_json(indent=2, ensure_ascii=False)
        task_db.update_task_inquiry(task.id, inquiry_type, inquiry_content)

    return inquiry_result
