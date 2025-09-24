import os
from typing import Any, Dict, Literal
import hashlib
from pathlib import Path
import kuzu
from diskcache import Cache
from loguru import logger

from llama_index.core import Document, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore
from llama_index.llms.litellm import LiteLLM
from llama_index.core.node_parser import NodeParser

from utils.llm_api import llm_temperatures, get_llm_params
from utils.vector import synthesizer
from utils.kg_prompts import kg_extraction_prompt


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
) -> NodeParser:
    if content_length > 0 and content_length < 512:
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(
            chunk_size=512, 
            chunk_overlap=100,
        )
    if content_format == "json":
        from llama_index.core.node_parser import JSONNodeParser
        return JSONNodeParser(
            include_metadata=True,
            max_depth=5,
            levels_to_keep=2
        )
    elif content_format == "md":
        from llama_index.core.node_parser import MarkdownNodeParser
        return MarkdownNodeParser(
            include_metadata=True,
        )
    from llama_index.core.node_parser import SentenceSplitter
    return SentenceSplitter(
        chunk_size=512, 
        chunk_overlap=100,
    )


def kg_add(
    kg_store: KuzuPropertyGraphStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    content_format: Literal["md", "txt", "json"] = "md",
    kg_extraction_prompt: str = kg_extraction_prompt,
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
    kg_node_parser = _get_kg_node_parser(content_format, len(content))
    from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
    path_extractor = SimpleLLMPathExtractor(
        llm=llm_for_extraction,
        extract_prompt=kg_extraction_prompt,
        max_paths_per_chunk=150,
    )

    logger.info(f"开始为 doc_id '{doc_id}' 构建知识图谱索引...")
    from llama_index.core.indices.property_graph.base import PropertyGraphIndex
    try:
        index = PropertyGraphIndex(
            nodes=[],
            index_struct=None,
            llm=llm_for_extraction,
            property_graph_store=kg_store,
            kg_extractors=[path_extractor],
            embed_kg_nodes=True,
            embed_model=Settings.embed_model,
            show_progress=False,
        )
        index.insert([doc], transformations=[kg_node_parser])
    except RuntimeError as e:
        if "Cannot set property vec in table embeddings because it is used in one or more indexes" in str(e):
            logger.warning(f"尝试更新已索引的 'vec' 属性失败，可能文档 '{doc_id}' 或其内部实体已存在且已处理。跳过。错误: {e}")
            # 如果此异常发生，我们假设数据已经成功存入，因此可以跳过。
        else:
            raise # 重新抛出其他运行时错误

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
    from llama_index.core.indices.property_graph.base import PropertyGraphIndex
    kg_index = PropertyGraphIndex.from_existing(
        property_graph_store=kg_store,
        llm=llm_for_reasoning,
        embed_kg_nodes=True,
        embed_model=Settings.embed_model,
    )

    postprocessors = []
    if top_n > 0:
        from llama_index.postprocessor.siliconflow_rerank.base import SiliconFlowRerank
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
