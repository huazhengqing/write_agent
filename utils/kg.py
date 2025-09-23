import os
from typing import Any, Dict, Literal
import hashlib
from pathlib import Path
import kuzu
from diskcache import Cache
from loguru import logger

from llama_index.core import Document, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.property_graph.base import PropertyGraphIndex
from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore
from llama_index.llms.litellm import LiteLLM 
from llama_index.core.node_parser import SentenceSplitter, NodeParser, MarkdownElementNodeParser, SimpleNodeParser
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.postprocessor.siliconflow_rerank.base import SiliconFlowRerank

from utils.config import llm_temperatures, get_llm_params
from utils.vector import synthesizer
from utils.kg_prompts import kg_extraction_prompt


KuzuPropertyGraphStore.model_config['extra'] = 'allow'
if hasattr(KuzuPropertyGraphStore, 'model_rebuild'):
    KuzuPropertyGraphStore.model_rebuild(force=True)


llm_params_for_extraction = get_llm_params(llm_group="summary", temperature=llm_temperatures["classification"])
llm_for_extraction = LiteLLM(**llm_params_for_extraction)


reasoning_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["reasoning"])
llm_for_reasoning = LiteLLM(**reasoning_llm_params)


def get_kg_store(db_path: str) -> KuzuPropertyGraphStore:
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    db = kuzu.Database(str(db_path_obj))
    kg_store = KuzuPropertyGraphStore(db, embed_model=Settings.embed_model)

    store_cache_path = Path(f"{db_path_obj}.cache.db")
    kg_store.cache = Cache(str(store_cache_path), size_limit=int(32 * (1024**2)))

    return kg_store


###############################################################################


def _get_kg_node_parser(
    content_format: Literal["md", "txt", "json"],
    content_length: int,
    chunk_size: int,
    chunk_overlap: int,
) -> NodeParser:
    if content_length < chunk_size:
        return SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if content_format == "json":
        parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif content_format == "md":
        parser = MarkdownElementNodeParser(
            llm=None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True,
        )
    else:
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return parser


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

    if not content or not content.strip():
        logger.warning(f"内容 (doc_id: {doc_id}) 为空或仅包含空白字符, 跳过处理。")
        return
    
    doc_cache = getattr(kg_store, "cache", None)
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    if doc_cache and doc_cache.get(new_content_hash):
        return

    doc = Document(id_=doc_id, text=content, metadata=metadata)
    kg_node_parser = _get_kg_node_parser(content_format, len(content), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    max_triplets_per_chunk = max(1, round(chunk_size / chars_per_triplet))
    path_extractor = SimpleLLMPathExtractor(
        llm=llm_for_extraction,
        extract_prompt=kg_extraction_prompt,
        max_paths_per_chunk=max_triplets_per_chunk,
    )

    logger.info(f"开始为 doc_id '{doc_id}' 构建知识图谱索引...")
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

    if doc_cache:
        doc_cache.set(new_content_hash, True)

    logger.success(f"成功处理内容 (doc_id: {doc_id}) 到知识图谱和向量库。")


###############################################################################


def get_kg_query_engine(
    kg_store: KuzuPropertyGraphStore,
    kg_similarity_top_k: int = 300,
    top_n: int = 100,
) -> BaseQueryEngine:
    logger.info("开始构建知识图谱查询引擎...")

    kg_index = PropertyGraphIndex.from_existing(
        property_graph_store=kg_store,
        llm=llm_for_reasoning,
        embed_kg_nodes=True,
        embed_model=Settings.embed_model,
    )

    postprocessors = []
    if top_n > 0:
        reranker = SiliconFlowRerank(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_n=top_n,
        )
        postprocessors.append(reranker)

    from llama_index.core.indices.knowledge_graph.retrievers import KGRetrieverMode
    query_engine = kg_index.as_query_engine(
        retriever_mode = KGRetrieverMode.HYBRID, 
        similarity_top_k=kg_similarity_top_k,
        response_synthesizer=synthesizer,
        node_postprocessors=postprocessors,
        llm=llm_for_reasoning,
        include_text=True,
    )

    logger.success("知识图谱查询引擎构建成功。")
    return query_engine
