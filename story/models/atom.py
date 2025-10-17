from typing import Optional, Literal
from pydantic import BaseModel, Field, conlist


ComplexReason = Literal[
    # write
    "design_insufficient",
    "length_excessive",
    # design
    "dependency_insufficient",
    "need_search",
    "composite_goal",
    # search
    "broad_topic",
    "requires_analysis",
    "vague_goal"
]

class AtomOutput(BaseModel):
    reasoning: Optional[str] = Field(None, description="关于任务是原子还是复杂的推理过程。")
    atom_result: Literal['atom', 'complex'] = Field(description="判断任务是否为原子任务的结果, 值必须是 'atom' 或 'complex'。")
    complex_reasons: Optional[conlist(item_type=ComplexReason, min_length=1)] = Field(None, description="当任务被判定为 'complex' 时, 此字段列出具体原因。")
