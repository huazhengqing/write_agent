from typing import Literal
from story.prompts.models.inquiry import InquiryOutput
from utils.llm import get_llm_messages, get_llm_params, llm_completion
from utils.loader import load_prompts
from utils.models import Task



async def inquiry(
    inquiry_type: Literal['search', 'design', 'summary'], 
    task: Task,
    book_level_design: str,
    global_state_summary: str,
    design_dependent: str,
    search_dependent: str,
    latest_text: str,
    task_list: str,
) -> InquiryOutput:
    context = {
        "task": task.to_context(),
        "book_level_design": book_level_design,
        "global_state_summary": global_state_summary,
        "design_dependent": design_dependent,
        "search_dependent": search_dependent,
        "latest_text": latest_text,
        "task_list": task_list,
    }
    system_prompt, user_prompt = load_prompts(f"{task.category}.prompts.inquiry.{inquiry_type}", "system_prompt", "user_prompt")
    messages = get_llm_messages(system_prompt, user_prompt, None, context)
    llm_params = get_llm_params(llm_group="summary", messages=messages, temperature=0.1)
    llm_message = await llm_completion(llm_params, response_model=InquiryOutput)
    return llm_message.validated_data
