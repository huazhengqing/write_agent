import os
from loguru import logger
from functools import lru_cache
from typing import List, Optional
from llama_index.llms.litellm import LiteLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataInfo
from llama_index.core.vector_stores import VectorStoreInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo
from utils.llm import llm_temperatures, get_llm_params
from rag.vector_prompts import vector_store_query_prompt
from rag.vector import get_synthesizer



@lru_cache(maxsize=None)
def get_vector_store_info_default() -> VectorStoreInfo:
    metadata_field_info = [
        MetadataInfo(
            name="source",
            type="str",
            description="文档来源的标识符, 例如 'test_doc_1' 或文件名。",
        ),
        MetadataInfo(
            name="type",
            type="str",
            description="文档的类型, 例如 'platform_profile', 'character_relation'。用于区分不同种类的内容。",
        ),
        MetadataInfo(
            name="platform",
            type="str",
            description="内容相关的平台名称, 例如 '知乎', 'B站', '起点中文网'。",
        ),
        MetadataInfo(
            name="date",
            type="str",
            description="内容的创建或关联日期, 格式为 'YYYY-MM-DD'。",
        ),
        MetadataInfo(
            name="word_count",
            type="int",
            description="文档的字数统计",

        ),
    ]
    return VectorStoreInfo(
        content_info="关于故事、书籍、报告、市场分析等的文本片段。",
        metadata_info=metadata_field_info,
    )



def _create_auto_retriever_engine(
    index: VectorStoreIndex,
    vector_store_info: 'VectorStoreInfo',
    similarity_top_k: int,
    node_postprocessors: List,
) -> BaseQueryEngine:
    logger.info("正在创建 Auto-Retriever 查询引擎...")
    reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)
    from llama_index.core.retrievers import VectorIndexAutoRetriever
    retriever = VectorIndexAutoRetriever(
        index=index,
        vector_store_info=vector_store_info,
        llm=reasoning_llm,
        prompt_template_str=vector_store_query_prompt, 
        similarity_top_k=similarity_top_k,
    )
    from llama_index.core.query_engine import RetrieverQueryEngine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_synthesizer(),
        node_postprocessors=node_postprocessors,
    )
    logger.success("Auto-Retriever 查询引擎创建成功。")
    return query_engine



def get_vector_query_engine(
    vector_store: VectorStore,
    filters: Optional[MetadataFilters] = None,
    similarity_top_k: int = 50,
    top_n: int = 10,
    use_auto_retriever: bool = False,
    vector_store_info: Optional[VectorStoreInfo] = None,
) -> BaseQueryEngine:
    
    logger.info("创建 VectorStoreIndex ...")
    index = VectorStoreIndex.from_vector_store(vector_store)
    logger.info("创建 VectorStoreIndex 完成。")

    reranker = None
    if top_n and top_n > 0:
        from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank
        reranker = SiliconFlowRerank(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_n=top_n,
        )
    node_postprocessors = [reranker] if reranker else []

    if use_auto_retriever:
        effective_vector_store_info = vector_store_info or get_vector_store_info_default()
        query_engine = _create_auto_retriever_engine(
            index=index,
            vector_store_info=effective_vector_store_info,
            similarity_top_k=similarity_top_k,
            node_postprocessors=node_postprocessors,
        )
    else:
        logger.info("正在创建标准查询引擎...")
        query_engine = index.as_query_engine(
            response_synthesizer=get_synthesizer(), 
            filters=filters, 
            similarity_top_k=similarity_top_k,
            node_postprocessors=node_postprocessors, 
        )
        logger.success("标准查询引擎创建成功。")
    
    return query_engine



async def index_query(query_engine: BaseQueryEngine, question: str) -> str:
    if not question:
        return ""
    logger.info(f"向量索引查询={question}")
    result = await query_engine.aquery(question)
    answer = str(getattr(result, "response", "")).strip()
    source_nodes = getattr(result, "source_nodes", [])
    if not source_nodes or not answer or answer == "Empty Response" or "无法回答" in answer or "无法回答该问题" in answer:
        logger.warning(f"未检索到任何源节点或有效响应, 返回空回答。")
        answer = ""
    else:
        logger.info(f"检索到 {len(source_nodes)} 个源节点。")
        for i, node in enumerate(source_nodes):
            logger.info(f"\n    - 源节点 {i+1} (ID: {node.node_id}, 分数: {node.score:.4f}):\n{node.get_content()}")
    logger.info(f"\n问题=\n{question}\n\n回答=\n{answer}")
    return answer



async def index_query_batch(query_engine: BaseQueryEngine, questions: List[str]) -> List[str]:
    if not questions:
        return []

    import asyncio
    sem = asyncio.Semaphore(3)
    async def safe_query(question: str) -> str:
        async with sem:
            try:
                return await index_query(query_engine, question)
            except Exception as e:
                logger.error("批量查询中, 问题 '{}' 失败: {}", question, e, exc_info=True)
                return ""

    tasks = [safe_query(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return results
