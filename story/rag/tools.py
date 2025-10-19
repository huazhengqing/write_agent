from llama_index.core.tools import AsyncFunctionTool, QueryEngineTool
import story
from rag.vector_query import get_vector_query_engine



def get_design_tool(run_id: str) -> AsyncFunctionTool:
    return AsyncFunctionTool.from_defaults(
        fn=lambda q: story.rag.query.design(run_id, [q]),
        name="query_story_design",
        description="查询所有官方设计文档, 包括世界观、角色设定, 以及**未来的情节大纲和章节规划**。当需要了解作者的'创作意图'或'未来规划'时使用。此工具回答'故事规划是怎样的?'"
    )



def get_plot_history_tool(run_id: str) -> AsyncFunctionTool:
    return AsyncFunctionTool.from_defaults(
        fn=lambda q: story.rag.query.write(run_id, [q]),
        name="query_plot_history",
        description="查询**已经写入正文并完成**的故事情节、事件摘要。当需要回忆'过去已发生'的剧情以确保连续性时使用。此工具是故事的'历史记录', 不包含任何未来的规划。它回答'过去发生了什么?'"
    )



def get_search_tool(run_id: str, top_n: int = 50) -> QueryEngineTool:
    search_engine = get_vector_query_engine(story.rag.base.get_vector(run_id, "search"), top_n=top_n)
    return QueryEngineTool.from_defaults(
        query_engine=search_engine,
        name="search_external_references",
        description="查询一个**离线的、预先存档**的外部参考资料库。当你需要获取非虚构的背景知识或事实依据, 且这些信息可能已提前收集时使用。注意: 此工具**无法访问实时互联网**。"
    )
