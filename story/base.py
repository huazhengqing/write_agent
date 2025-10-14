from functools import lru_cache
from loguru import logger
from typing import List, Any, Literal, Optional, Type, Union
from pydantic import BaseModel
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.vector_stores import MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore
from llama_index.core.tools import QueryEngineTool
from utils.file import data_dir
from utils.react_agent import call_react_agent
from rag.vector_query import get_vector_query_engine
from rag.kg import get_kg_query_engine
from rag.hybrid_query import hybrid_query_batch


@lru_cache(maxsize=None)
def get_story_vector_store(run_id: str, content_type: str) -> ChromaVectorStore:
    chroma_path = data_dir / run_id / content_type
    
    # ChromaDB 的 collection 名称有严格限制, 不能包含中文等特殊字符。
    # 我们在这里专门为 collection 名称进行一次清理, 而不影响包含中文的 run_id。
    import re
    sanitized_run_id = re.sub(r'[^a-zA-Z0-9._-]', '_', run_id).strip('._')
    collection_name = f"{sanitized_run_id}_{content_type}"

    from rag.vector import get_vector_store
    vector_store = get_vector_store(db_path=str(chroma_path), collection_name=collection_name)
    return vector_store



@lru_cache(maxsize=None)
def get_story_kg_store(run_id: str, content_type: str) -> KuzuPropertyGraphStore:
    kuzu_db_path = data_dir / run_id / content_type / "kuzu_db"
    from rag.kg import get_kg_store
    graph_store = get_kg_store(db_path=str(kuzu_db_path))
    return graph_store


###############################################################################


async def _hybrid_query_base(
    run_id: str,
    questions: list[str],
    vector_content_type: str,
    kg_content_type: str,
    vector_query_engine_params: dict,
    kg_query_engine_params: dict,
    vector_filters: MetadataFilters | None = None
) -> str:
    """混合查询的基础函数, 结合了向量查询和知识图谱查询。"""
    logger.info(f"开始在 '{vector_content_type}'(向量) 和 '{kg_content_type}'(图) 库中对 {len(questions)} 个问题执行混合查询...")
    vector_store = get_story_vector_store(run_id, vector_content_type)
    kg_store = get_story_kg_store(run_id, kg_content_type)

    vector_query_engine = get_vector_query_engine(
        vector_store=vector_store,
        filters=vector_filters,
        **vector_query_engine_params
    )
    kg_query_engine = get_kg_query_engine(
        kg_store=kg_store,
        **kg_query_engine_params
    )

    results = await hybrid_query_batch(
        vector_query_engine=vector_query_engine,
        kg_query_engine=kg_query_engine,
        questions=questions,
    )
    return "\n\n---\n\n".join(results)


async def hybrid_query_design(
    run_id: str,
    questions: list[str],
    vector_filters: MetadataFilters | None = None
) -> str:
    """在 'design' 库中执行混合查询。"""
    return await _hybrid_query_base(
        run_id=run_id,
        questions=questions,
        vector_content_type="design",
        kg_content_type="design",
        vector_query_engine_params={"similarity_top_k": 150, "top_n": 30},
        kg_query_engine_params={"kg_similarity_top_k": 600, "top_n": 100},
        vector_filters=vector_filters
    )


async def hybrid_query_write(
    run_id: str,
    questions: list[str],
    vector_filters: MetadataFilters | None = None
) -> str:
    """从摘要库(向量)和正文库(知识图谱)中执行混合查询。"""
    return await _hybrid_query_base(
        run_id=run_id,
        questions=questions,
        vector_content_type="summary",
        kg_content_type="write",
        vector_query_engine_params={"similarity_top_k": 300, "top_n": 50},
        kg_query_engine_params={"kg_similarity_top_k": 600, "top_n": 100},
        vector_filters=vector_filters
    )


###############################################################################


async def hybrid_query_react(
    run_id: str,
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[BaseModel]] = None
) -> Optional[Union[BaseModel, str]]:

    # 1. 设计库工具
    from llama_index.core.tools import AsyncFunctionTool
    design_tool = AsyncFunctionTool.from_defaults(
        fn=lambda q: hybrid_query_design(run_id, [q]),
        name="design_library_search",
        description="用于查询故事的核心设定，如角色背景、世界观、物品道具、关键概念定义等。当你需要了解 '是什么' 或 '设定是怎样' 时使用。"
    )

    # 2. 正文与摘要库工具
    write_summary_tool = AsyncFunctionTool.from_defaults(
        fn=lambda q: hybrid_query_write(run_id, [q]),
        name="story_content_search",
        description="用于查询已经发生的故事情节、事件经过、人物关系演变等。当你需要了解 '发生了什么' 或 '谁和谁是什么关系' 时使用。"
    )

    # 3. 搜索库工具
    search_engine = get_vector_query_engine(get_story_vector_store(run_id, "search"), top_n=50)
    search_tool = QueryEngineTool.from_defaults(
        query_engine=search_engine,
        name="reference_material_search",
        description="用于查找创作过程中收集的外部参考资料、研究笔记或灵感片段。当你需要寻找背景资料或事实依据时使用。"
    )

    result = await call_react_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=[design_tool, write_summary_tool, search_tool],
        response_model=response_model
    )

    logger.success(f"基于 ReAct 的混合查询完成。")
    return result
