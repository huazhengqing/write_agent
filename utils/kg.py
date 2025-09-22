import os
import json
from typing import Any, Dict, List, Literal, Optional
import hashlib
import time
import kuzu
import llama_index.graph_stores.kuzu.utils as kuzu_utils
from loguru import logger

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.graph_stores.types import (
    ChunkNode, EntityNode, LabelledNode
)
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.indices.property_graph.base import PropertyGraphIndex
from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore
from llama_index.llms.litellm import LiteLLM 
from llama_index.core.node_parser import SentenceSplitter, NodeParser, MarkdownElementNodeParser, \
    SimpleNodeParser
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.postprocessor.siliconflow_rerank.base import SiliconFlowRerank

from utils.config import llm_temperatures, get_llm_params
from utils.vector import synthesizer
from utils.kg_prompts import (
    kg_extraction_prompt,
)


###############################################################################


llm_params_for_extraction = get_llm_params(llm_group="summary", temperature=llm_temperatures["classification"])
llm_for_extraction = LiteLLM(**llm_params_for_extraction)


reasoning_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["reasoning"])
llm_for_reasoning = LiteLLM(**reasoning_llm_params)


def get_kg_store(db_path: str) -> KuzuPropertyGraphStore:
    logger.info(f"正在访问或创建知识图谱存储: path='{db_path}'")
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    db = kuzu.Database(db_path)
    logger.debug(f"Kuzu 数据库实例已创建: {db_path}")

    conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE IF NOT EXISTS __Document__(doc_id STRING, content_hash STRING, PRIMARY KEY (doc_id))")
    logger.debug("确保 __Document__ 表存在。")

    kg_store = KuzuPropertyGraphStore(
        db,
        embed_model=Settings.embed_model,
    )
    logger.success(f"成功获取知识图谱存储。")
    return kg_store


###############################################################################


def _is_content_unchanged(
    kg_store: KuzuPropertyGraphStore, doc_id: str, new_content_hash: str
) -> bool:
    logger.debug(f"正在为 doc_id '{doc_id}' 检查内容哈希值...")
    hash_check_query = "MATCH (d:__Document__ {doc_id: $doc_id}) RETURN d.content_hash AS old_hash"
    query_result = kg_store.structured_query(hash_check_query, param_map={"doc_id": doc_id})
    
    if not query_result:
        logger.debug(f"未在 __Document__ 表中找到 doc_id '{doc_id}' 的记录，视为新内容。")
        return False
        
    old_hash = query_result[0].get('old_hash')
    if old_hash == new_content_hash:
        logger.debug(f"内容哈希值匹配 (old: {old_hash[:8]}..., new: {new_content_hash[:8]}...), 内容未改变。")
        return True
    else:
        logger.debug(f"内容哈希值不匹配 (old: {old_hash[:8]}..., new: {new_content_hash[:8]}...), 内容已改变。")
        return False


def _get_kg_node_parser(
    content_format: Literal["md", "txt", "json"],
    content_length: int,
    chunk_size: int,
    chunk_overlap: int,
) -> NodeParser:
    logger.debug(f"为 '{content_format}' (长度: {content_length}) 选择知识图谱节点解析器。")
    if content_length < chunk_size:
        logger.debug("内容较短，使用 SimpleNodeParser。")
        return SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if content_format == "json":
        logger.debug("内容格式为 JSON，使用 SimpleNodeParser。")
        parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif content_format == "md":
        parser = MarkdownElementNodeParser(
            llm=None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True,
        )
    else:
        logger.debug("内容格式为 TXT，使用 SentenceSplitter。")
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return parser


def _update_document_hash(
    kg_store: KuzuPropertyGraphStore, 
    doc_id: str, 
    content_hash: str
):
    logger.debug(f"正在更新 doc_id '{doc_id}' 的内容哈希值为 '{content_hash[:8]}...'")
    hash_update_query = """
    MERGE (d:__Document__ {doc_id: $doc_id})
    SET d.content_hash = $content_hash
    """
    kg_store.structured_query(hash_update_query, param_map={"doc_id": doc_id, "content_hash": content_hash})


def kg_add(
    kg_store: KuzuPropertyGraphStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    content_format: Literal["md", "txt", "json"] = "md",
    chars_per_triplet: int = 120,
    kg_extraction_prompt: str = kg_extraction_prompt,
    chunk_size: int = 2048,
    chunk_overlap: int = 400,
) -> None:
    logger.info(f"开始向知识图谱添加内容, doc_id='{doc_id}', format='{content_format}'...")

    # kg_store.clear_schema_cache()
    
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    if _is_content_unchanged(kg_store, doc_id, new_content_hash):
        logger.info(f"内容 (doc_id: {doc_id}) 未发生变化，跳过知识图谱更新。")
        return

    logger.info(f"内容 (doc_id: {doc_id}) 已更新，正在删除旧索引并重新构建...")
    delete_query = "MATCH (c:Chunk {ref_doc_id: $doc_id}) DETACH DELETE c"
    kg_store.structured_query(delete_query, param_map={"doc_id": doc_id})
    logger.debug(f"已从知识图谱中删除 doc_id '{doc_id}' 的旧 Chunk 节点。")

    doc = Document(id_=doc_id, text=content, metadata=metadata)
    kg_node_parser = _get_kg_node_parser(
        content_format, len(content), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # chunk_size 直接从函数参数获取，确保一致性
    max_triplets_per_chunk = max(1, round(chunk_size / chars_per_triplet))
    logger.info(f"根据 chars_per_triplet={chars_per_triplet} 和 chunk_size={chunk_size}，动态设置 max_triplets_per_chunk={max_triplets_per_chunk}")
    
    path_extractor = SimpleLLMPathExtractor(
        llm=llm_for_extraction,
        extract_prompt=kg_extraction_prompt,
        max_paths_per_chunk=max_triplets_per_chunk,
    )

    logger.debug("KG 路径提取器 (SimpleLLMPathExtractor) 创建完成。")

    logger.info(f"开始为 doc_id '{doc_id}' 构建知识图谱索引...")
    step_start_time = time.time()
    PropertyGraphIndex.from_documents(
        [doc],
        llm=llm_for_extraction,
        property_graph_store=kg_store,
        transformations=[kg_node_parser],
        kg_extractors=[path_extractor],
        embed_kg_nodes=True,
        embed_model=Settings.embed_model,
        show_progress=False,
    )
    logger.info(f"PropertyGraphIndex 核心处理完成，耗时: {time.time() - step_start_time:.2f}s")

    _update_document_hash(kg_store, doc_id, new_content_hash)

    logger.success(f"成功处理内容 (doc_id: {doc_id}) 到知识图谱和向量库。")


###############################################################################


def get_kg_query_engine(
    kg_store: KuzuPropertyGraphStore,
    kg_similarity_top_k: int = 300,
    kg_rerank_top_n: int = 100,
) -> BaseQueryEngine:
    logger.info("开始构建知识图谱查询引擎...")
    logger.debug(
        f"参数: kg_similarity_top_k={kg_similarity_top_k}, kg_rerank_top_n={kg_rerank_top_n}"
    )

    step_time = time.time()
    kg_index = PropertyGraphIndex.from_existing(
        property_graph_store=kg_store,
        llm=llm_for_reasoning,
        embed_kg_nodes=True,
        embed_model=Settings.embed_model,
    )
    logger.debug(f"从现有存储加载PropertyGraphIndex完成，耗时: {time.time() - step_time:.2f}s")

    step_time = time.time()
    postprocessors = []
    if kg_rerank_top_n > 0:
        reranker = SiliconFlowRerank(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_n=kg_rerank_top_n,
        )
        postprocessors.append(reranker)
        logger.debug(f"Reranker创建完成，耗时: {time.time() - step_time:.2f}s")
    else:
        logger.debug("kg_rerank_top_n <= 0, 不使用 Reranker。")

    step_time = time.time()
    from llama_index.core.indices.knowledge_graph.retrievers import (
        KGRetrieverMode,
        KGTableRetriever,
    )
    query_engine = kg_index.as_query_engine(
        retriever_mode = KGRetrieverMode.HYBRID, 
        similarity_top_k=kg_similarity_top_k,
        response_synthesizer=synthesizer,
        node_postprocessors=postprocessors,
        llm=llm_for_reasoning,
        include_text=True,
    )
    logger.debug(f"as_query_engine调用完成，耗时: {time.time() - step_time:.2f}s")

    logger.success("知识图谱查询引擎构建成功。")
    return query_engine
