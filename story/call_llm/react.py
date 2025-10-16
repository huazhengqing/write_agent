from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import AsyncFunctionTool
from story.rag import query
from story.rag.base import get_vector
from rag.vector_query import get_vector_query_engine
from utils.llm import llm_group_type



async def react(
    run_id: str,
    llm_group: llm_group_type = 'reasoning',
    temperature = 0.1, 
    system_header: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: str = "",
    output_cls: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:
    design_tool = AsyncFunctionTool.from_defaults(
        fn=lambda q: query.design(run_id, [q]),
        name="query_story_design",
        description="查询小说的核心世界观和官方设定。当需要获取角色（如'主角的童年经历'）、地点（如'首都的建筑风格'）、物品（如'圣剑的来历'）、概念（如'魔法系统的等级划分'）等权威设定信息时使用。适用于回答 '是什么' 或 '设定是怎样' 的问题。"
    )

    write_summary_tool = AsyncFunctionTool.from_defaults(
        fn=lambda q: query.write(run_id, [q]),
        name="query_plot_history",
        description="查询小说中已经发生过的故事情节、事件摘要和人物动态。当需要回忆过去的剧情（如'主角上次见到反派是什么时候'）、确认人物关系（如'A和B在第一卷的关系'）或事件结果（如'上次战争的结果'）时使用，以确保故事的连续性。适用于回答 '发生了什么' 的问题。"
    )

    search_engine = get_vector_query_engine(get_vector(run_id, "search"), top_n=50)
    search_tool = QueryEngineTool.from_defaults(
        query_engine=search_engine,
        name="search_external_references",
        description="用于查找外部世界的参考资料和事实。只有当你自己的知识库无法回答，且问题不涉及故事内部设定或已发生情节时，才使用此工具获取非虚构的背景知识或事实依据（例如：'中世纪城堡的结构'、'某种植物的特性'）。"
    )

    from utils import call_llm
    result = await call_llm.react(
        llm_group=llm_group,
        temperature = temperature, 
        system_header=system_header,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=[design_tool, write_summary_tool, search_tool],
        output_cls=output_cls
    )
    return result
