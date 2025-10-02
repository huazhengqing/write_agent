from typing import Optional, Literal
from pydantic import BaseModel, Field


class FlowDecisionOutput(BaseModel):
    reasoning: Optional[str] = Field(None, description="决策的推理过程。当 'decision' 为 'DECISION_CONTINUE_PLANNING' 时, 此字段必须说明设计不完备的具体原因。")
    decision: Literal['DECISION_CONTINUE_PLANNING', 'DECISION_DIVIDE_HIERARCHY', 'DECISION_PROCEED_TO_WRITE'] = Field(description="最终的流程决策指令。")

