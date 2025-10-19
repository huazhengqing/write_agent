import os
from functools import lru_cache
from typing import Any, Dict, Literal, Optional
from loguru import logger
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.llms.litellm import LiteLLM
from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore


from rag.vector import init_llama_settings
init_llama_settings()


llm_for_extraction = LiteLLM(
    model="openai/summary",
    temperature=0.0,
    max_tokens=None,
    max_retries=10,
    api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
    api_base=os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
)


llm_for_reasoning = LiteLLM(
    model="openai/summary",
    temperature=0.1,
    max_tokens=None,
    max_retries=10,
    api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
    api_base=os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
)

@lru_cache(maxsize=None)
def get_kg_store(db_path: str) -> KuzuPropertyGraphStore:
    from pathlib import Path
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    import kuzu
    db = kuzu.Database(str(db_path_obj), max_db_size=1024*1024*1024*32)
    from llama_index.core import Settings
    kg_store = KuzuPropertyGraphStore(db, embed_model=Settings.embed_model)

    store_cache_path = Path(f"{db_path_obj}.cache.db")
    from diskcache import Cache
    kg_store.cache = Cache(str(store_cache_path), size_limit=int(32 * (1024**2)))

    return kg_store



###############################################################################



# 关键设置: 必须为 False。
# 原因: 当向知识图谱中添加新文档时, 如果文档中包含已存在的实体(例如"龙傲天"), 
# 程序会尝试"更新插入(Upsert)"该实体节点。若 embed_kg_nodes=True, 程序会尝试更新该节点的向量, 
# 这会与 Kuzu "不允许直接更新(SET)已索引属性"的规则冲突, 导致运行时错误。
# 设置为 False 后, 向量信息将仅存储在 ChunkNode 上, 实体节点本身不含向量, 从而避免此问题, 同时保证混合检索能力。
embed_kg_nodes = False



def kg_add(
    kg_store: KuzuPropertyGraphStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    content_format: Literal["md", "txt", "json"] = "md",
    extract_prompt: Optional[str] = None
) -> None:
    logger.info(f"开始向知识图谱添加内容, doc_id='{doc_id}', format='{content_format}'...")

    if not content or not content.strip():
        logger.warning(f"内容 (doc_id: {doc_id}) 为空或仅包含空白字符, 跳过处理。")
        return
    
    doc_cache = getattr(kg_store, "cache", None)
    import hashlib
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    if doc_cache and doc_cache.get(new_content_hash):
        return

    logger.info(f"为 doc_id '{doc_id}' 执行更新操作, 将首先删除旧的图谱数据。")
    # 1. 删除与 doc_id 关联的 Chunk 节点及其关系
    kg_store.delete(properties={"ref_doc_id": doc_id})
    # 2. 删除不再被任何 Chunk 提及的孤立实体节点
    kg_store.structured_query(
        """
        MATCH (e:Entity) WHERE NOT EXISTS ((:Chunk)-[:MENTIONS]->(e)) DETACH DELETE e
        """
    )

    from llama_index.core import Document
    doc = Document(id_=doc_id, text=content, metadata=metadata)
    
    from rag.splitter import get_vector_node_parser
    kg_node_parser = get_vector_node_parser(content_format, len(content))

    from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
    path_extractor = SimpleLLMPathExtractor(
        llm=llm_for_extraction,
        extract_prompt=extract_prompt,
        max_paths_per_chunk=50,
    )

    logger.info(f"开始为 doc_id '{doc_id}' 构建知识图谱索引...")
    from llama_index.core.indices.property_graph.base import PropertyGraphIndex
    from llama_index.core import Settings
    index = PropertyGraphIndex(
        nodes=[],
        index_struct=None,
        llm=llm_for_extraction,
        property_graph_store=kg_store,
        transformations=[kg_node_parser],
        kg_extractors=[path_extractor],
        embed_kg_nodes=embed_kg_nodes,
        embed_model=Settings.embed_model,
        show_progress=False,
    )
    index.insert(doc)

    if doc_cache:
        doc_cache.set(new_content_hash, True)

    logger.success(f"成功处理内容 (doc_id: {doc_id}) 到知识图谱和向量库。")



###############################################################################



def get_kg_query_engine(
    kg_store: KuzuPropertyGraphStore,
    kg_similarity_top_k: int = 500, 
    top_n: int = 100,
) -> BaseQueryEngine:
    logger.info("开始构建知识图谱查询引擎...")
    from llama_index.core.indices.property_graph.base import PropertyGraphIndex
    from llama_index.core import Settings
    kg_index = PropertyGraphIndex.from_existing(
        property_graph_store=kg_store,
        llm=llm_for_reasoning,
        embed_kg_nodes=embed_kg_nodes,
        embed_model=Settings.embed_model,
    )

    postprocessors = []
    if top_n > 0:
        from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank
        reranker = SiliconFlowRerank(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_n=top_n,
        )
        postprocessors.append(reranker)

    from llama_index.core.indices.knowledge_graph.retrievers import KGRetrieverMode
    from rag.vector import get_synthesizer
    query_engine = kg_index.as_query_engine(
        retriever_mode = KGRetrieverMode.HYBRID, 
        similarity_top_k=kg_similarity_top_k,
        response_synthesizer=get_synthesizer(),
        node_postprocessors=postprocessors,
        llm=llm_for_reasoning,
        include_text=True, 
        use_llm=True,
    )

    logger.success("知识图谱查询引擎构建成功。")
    return query_engine
