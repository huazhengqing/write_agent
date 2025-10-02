from typing import List, Optional
from pydantic import BaseModel, Field


class RefineOutput(BaseModel):
    reasoning: Optional[str] = Field(None, description="推理过程。")
    refine_goal: Optional[str] = Field(None, description="在分析了任务后, 对原始[核心目标]的优化或澄清。如果LLM认为不需要修改, 则此字段必需要省略。")
    refine_instructions: Optional[List[str]] = Field(None, description="对[具体指令]的优化或补充。")
    refine_input_brief: Optional[List[str]] = Field(None, description="对[输入指引]的优化或补充。")
    refine_constraints: Optional[List[str]] = Field(None, description="对[限制和禁忌]的优化或补充。")
    refine_acceptance_criteria: Optional[List[str]] = Field(None, description="对[验收标准]的优化或补充。")

