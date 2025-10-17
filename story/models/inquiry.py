from pydantic import BaseModel, Field
from typing import List


class InquiryOutput(BaseModel):
    reasoning: str = Field(..., description="简要说明识别出的核心信息缺口, 以及提问的总体策略。")
    causality_probing: List[str] = Field(..., description="按重要性降序排列的因果探询问题列表。")
    setting_probing: List[str] = Field(..., description="按重要性降序排列的设定探询问题列表。")
    state_probing: List[str] = Field(..., description="按重要性降序排列的状态探询问题列表。")


