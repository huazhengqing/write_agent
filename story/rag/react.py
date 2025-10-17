from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import AsyncFunctionTool
import story
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
        fn=lambda q: story.rag.query.design(run_id, [q]),
        name="query_story_design",
        description="查询小说的核心世界观和官方设定。当需要获取角色(如'主角的童年经历')、地点(如'首都的建筑风格')、物品(如'圣剑的来历')、概念(如'魔法系统的等级划分')等权威设定信息时使用。适用于回答 '是什么' 或 '设定是怎样' 的问题。"
    )

    write_summary_tool = AsyncFunctionTool.from_defaults(
        fn=lambda q: story.rag.query.write(run_id, [q]),
        name="query_plot_history",
        description="查询小说中已经发生过的故事情节、事件摘要和人物动态。当需要回忆过去的剧情(如'主角上次见到反派是什么时候')、确认人物关系(如'A和B在第一卷的关系')或事件结果(如'上次战争的结果')时使用, 以确保故事的连续性。适用于回答 '发生了什么' 的问题。"
    )

    search_engine = get_vector_query_engine(story.rag.base.get_vector(run_id, "search"), top_n=50)
    search_tool = QueryEngineTool.from_defaults(
        query_engine=search_engine,
        name="search_external_references",
        description="查询一个**离线的、预先存档**的外部参考资料库。当你需要获取非虚构的背景知识或事实依据(例如: '中世纪城堡的结构'、'某种植物的特性'), 且这些信息可能已提前收集时使用。注意: 此工具**无法访问实时互联网**。如果查询后仍无结果, 但你确信需要实时网络信息, 你应该在最终思考阶段决策创建一个`search`类型的任务。"
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
