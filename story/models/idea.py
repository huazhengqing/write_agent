from pydantic import BaseModel, Field


class IdeaOutput(BaseModel):
    name: str = Field(description="一个吸引人的书名。")
    length: str = Field(description="故事的预计字数")
    goal: str = Field(description="项目的核心目标或故事的一句话简介。")
    instructions: str = Field(description="关于故事风格、基调、节奏或关键元素的具体指令。")
    input_brief: str = Field(description="关于故事背景、主角、世界观、开篇情节等关键设定的输入指引。")
    constraints: str = Field(description="创作时需要避免的内容, 例如特定的情节、元素或主题。")
    acceptance_criteria: str = Field(description="衡量项目或故事成功的可验证标准, 例如必须达成的核心体验、必须塑造的角色弧光等。")
