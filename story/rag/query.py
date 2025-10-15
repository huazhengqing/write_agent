from loguru import logger
from llama_index.core.vector_stores import MetadataFilters
from story.rag.base import get_kg, get_vector
from rag.vector_query import get_vector_query_engine
from rag.kg import get_kg_query_engine
from rag.hybrid_query import hybrid_query_batch



async def query_base(
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
    vector_store = get_vector(run_id, vector_content_type)
    kg_store = get_kg(run_id, kg_content_type)

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



async def design(
    run_id: str,
    questions: list[str],
    vector_filters: MetadataFilters | None = None
) -> str:
    """在 'design' 库中执行混合查询。"""
    return await query_base(
        run_id=run_id,
        questions=questions,
        vector_content_type="design",
        kg_content_type="design",
        vector_query_engine_params={"similarity_top_k": 150, "top_n": 30},
        kg_query_engine_params={"kg_similarity_top_k": 600, "top_n": 100},
        vector_filters=vector_filters
    )



async def write(
    run_id: str,
    questions: list[str],
    vector_filters: MetadataFilters | None = None
) -> str:
    """从摘要库(向量)和正文库(知识图谱)中执行混合查询。"""
    return await query_base(
        run_id=run_id,
        questions=questions,
        vector_content_type="summary",
        kg_content_type="write",
        vector_query_engine_params={"similarity_top_k": 300, "top_n": 50},
        kg_query_engine_params={"kg_similarity_top_k": 600, "top_n": 100},
        vector_filters=vector_filters
    )


