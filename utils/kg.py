import os
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
import hashlib
import time
from collections import defaultdict
import kuzu
from loguru import logger

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.vector_stores.types import VectorStore
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from llama_index.llms.litellm import LiteLLM 
from llama_index.core.node_parser import SentenceSplitter, NodeParser, MarkdownElementNodeParser, SimpleNodeParser
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
    DynamicLLMPathExtractor,
    TextToCypherRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

from utils.config import llm_temperatures, get_llm_params
from utils.vector import synthesizer
from utils.kg_prompts import (
    entities,
    relations,
    validation_schema,
    kg_extraction_prompt,
    response_template,
    summarization_template,
    kg_gen_cypher_prompt,
)


# ==============================================================================
#  模块级实例以提高性能
# ==============================================================================

# 用于知识提取的LLM实例，在模块加载时创建一次，以供复用
llm_params_for_extraction = get_llm_params(llm_group="summary", temperature=llm_temperatures["classification"])
llm_for_extraction = LiteLLM(**llm_params_for_extraction)

# 用于Cypher生成和最终答案合成的LLM实例
reasoning_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["reasoning"])
llm_for_reasoning = LiteLLM(**reasoning_llm_params)


class PatchedKuzuPropertyGraphStore(KuzuPropertyGraphStore):
    @property
    def supports_structured_queries(self) -> bool:
        """指明此图谱存储支持结构化 (Cypher) 查询。"""
        return True

    def get_schema(self, refresh: bool = False) -> str:
        """
        获取图存储的模式。
        这是一个补丁，用于处理上游 `KuzuPropertyGraphStore` 的 `get_schema` 方法
        与 `PropertyGraphStore` 接口不一致的问题 (缺少 `refresh` 参数)。
        """
        if refresh and hasattr(self, "_schema"):
            self._schema = None

        # 调用父类的 get_schema，它不接受 refresh 参数
        return super().get_schema()


def get_kg_store(db_path: str) -> PatchedKuzuPropertyGraphStore:
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    db = kuzu.Database(db_path)

    conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE IF NOT EXISTS __Document__(doc_id STRING, content_hash STRING, PRIMARY KEY (doc_id))")

    kg_store = PatchedKuzuPropertyGraphStore(
        db,
        relationship_schema=validation_schema,
        has_structured_schema=True,
        embed_model=Settings.embed_model,
    )
    return kg_store


###############################################################################


def _is_content_unchanged(
    kg_store: PatchedKuzuPropertyGraphStore, doc_id: str, new_content_hash: str
) -> bool:
    """检查内容哈希，如果内容未变则返回True。"""
    hash_check_query = "MATCH (d:__Document__ {doc_id: $doc_id}) RETURN d.content_hash AS old_hash"
    query_result = kg_store.structured_query(hash_check_query, param_map={"doc_id": doc_id})
    return bool(query_result and query_result[0].get('old_hash') == new_content_hash)


def _get_kg_node_parser(content_format: Literal["md", "txt", "json"], content_length: int) -> NodeParser:
    """根据内容格式和长度获取合适的节点解析器。"""
    if content_length < 512:
        logger.info(f"内容长度 ({content_length}) < 512，使用 SimpleNodeParser。")
        return SimpleNodeParser()

    if content_format == "json":
        parser = SimpleNodeParser()
        logger.info("使用 JSON 节点解析策略。")
    elif content_format == "md":
        parser = MarkdownElementNodeParser(
            llm=None, 
            chunk_size=2048,
            chunk_overlap=400,
            include_metadata=True,
        )
        logger.info(f"使用 Markdown 元素解析策略 (无LLM摘要)，内部 chunk_size=2048")
    else:  # txt
        parser = SentenceSplitter(chunk_size=2048, chunk_overlap=400)
        logger.info(f"使用句子分割策略，chunk_size={parser.chunk_size}")
    return parser


def _update_document_hash(
    kg_store: PatchedKuzuPropertyGraphStore, doc_id: str, content_hash: str
):
    """在知识图谱中更新文档的内容哈希记录。"""
    hash_update_query = """
    MERGE (d:__Document__ {doc_id: $doc_id})
    SET d.content_hash = $content_hash
    """
    kg_store.structured_query(hash_update_query, param_map={"doc_id": doc_id, "content_hash": content_hash})
    logger.info(f"已更新 doc_id '{doc_id}' 的内容哈希。")


def kg_add(
    kg_store: PatchedKuzuPropertyGraphStore,
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    content_format: Literal["md", "txt", "json"] = "md",
    chars_per_triplet: int = 120,
    kg_extraction_prompt: str = kg_extraction_prompt
) -> None:
    start_time = time.time()
    logger.info(f"开始处理 doc_id: {doc_id} (格式: {content_format})...")

    # 1. 内容哈希检查
    step_start_time = time.time()
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    if _is_content_unchanged(kg_store, doc_id, new_content_hash):
        logger.info(f"内容 (doc_id: {doc_id}) 未发生变化，跳过更新。")
        return
    logger.debug(f"步骤1: 内容哈希检查完成，耗时: {time.time() - step_start_time:.2f}s")

    # 2. 删除旧节点
    step_start_time = time.time()
    vector_store.delete(ref_doc_id=doc_id)
    logger.debug(f"步骤2: 从向量库中删除旧节点完成，耗时: {time.time() - step_start_time:.2f}s")

    # 3. 准备文档和节点解析器
    step_start_time = time.time()
    doc = Document(id_=doc_id, text=content, metadata=metadata)
    kg_node_parser = _get_kg_node_parser(content_format, len(content))
    logger.debug(f"步骤3: 文档和节点解析器准备完成，耗时: {time.time() - step_start_time:.2f}s")

    # 4. 动态计算三元组数量
    step_start_time = time.time()
    chunk_size = getattr(kg_node_parser, 'chunk_size', 2048)
    max_triplets_per_chunk = max(1, round(chunk_size / chars_per_triplet))
    logger.info(f"根据 chars_per_triplet={chars_per_triplet} 和 chunk_size={chunk_size}，动态设置 max_triplets_per_chunk={max_triplets_per_chunk}")
    logger.debug(f"步骤4: 动态计算三元组数量完成，耗时: {time.time() - step_start_time:.2f}s")

    # 5. 准备存储上下文和路径提取器
    step_start_time = time.time()
    storage_context = StorageContext.from_defaults(graph_store=kg_store, vector_store=vector_store)
    logger.debug(f"  - StorageContext 创建完成，耗时: {time.time() - step_start_time:.2f}s")
    
    inner_step_time = time.time()
    path_extractor = SchemaLLMPathExtractor(
        llm=llm_for_extraction,
        extract_prompt=kg_extraction_prompt,
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=validation_schema,
        max_triplets_per_chunk=max_triplets_per_chunk,
    )
    logger.debug(f"  - SchemaLLMPathExtractor 创建完成，耗时: {time.time() - inner_step_time:.2f}s")
    logger.debug(f"步骤5: 存储上下文和提取器准备完成，总耗时: {time.time() - step_start_time:.2f}s")

    # 6. 核心处理：构建图谱
    step_start_time = time.time()
    PropertyGraphIndex.from_documents(
        [doc],
        llm=llm_for_extraction,
        storage_context=storage_context,
        transformations=[kg_node_parser],
        kg_extractors=[path_extractor],
        embed_kg_nodes=True,
        embed_model=Settings.embed_model,
        show_progress=False,
    )
    logger.info(f"步骤6: PropertyGraphIndex 核心处理完成，耗时: {time.time() - step_start_time:.2f}s")

    # 7. 更新文档哈希
    step_start_time = time.time()
    _update_document_hash(kg_store, doc_id, new_content_hash)
    logger.debug(f"步骤7: 文档哈希更新完成，耗时: {time.time() - step_start_time:.2f}s")

    logger.success(f"成功处理内容 (doc_id: {doc_id}) 到知识图谱和向量库。总耗时: {time.time() - start_time:.2f}s")


###############################################################################


def get_kg_query_engine(
    kg_store: PatchedKuzuPropertyGraphStore,
    kg_vector_store: VectorStore,
    kg_similarity_top_k: int = 20,
    kg_rerank_top_n: int = 10,
    text_to_cypher_template: str = kg_gen_cypher_prompt,
) -> BaseQueryEngine:
    start_time = time.time()
    logger.info("开始创建知识图谱混合查询引擎...")
    logger.debug(f"参数: kg_similarity_top_k={kg_similarity_top_k}, kg_rerank_top_n={kg_rerank_top_n}")

    step_time = time.time()
    logger.debug(f"使用模块级推理LLM。")

    step_time = time.time()
    kg_index = PropertyGraphIndex.from_existing(
        property_graph_store=kg_store,
        vector_store=kg_vector_store,
        llm=llm_for_reasoning,
        embed_kg_nodes=True,
        embed_model=Settings.embed_model,
    )
    logger.debug(f"从现有存储加载PropertyGraphIndex完成，耗时: {time.time() - step_time:.2f}s")

    # --- 为混合搜索定义子检索器 ---

    step_time = time.time()
    # 1. Text-to-Cypher 检索器，用于结构化查询
    text2cypher_retriever = TextToCypherRetriever(
        graph_store=kg_store,
        llm=llm_for_reasoning,
        text_to_cypher_template=PromptTemplate(text_to_cypher_template),
        response_template=response_template,
        summarize_response=True,
        summarization_template=summarization_template,
    )

    # 2. 向量检索器，用于在KG节点上进行语义搜索
    vector_retriever = VectorContextRetriever(
        graph_store=kg_store,
        include_text=True,
        vector_store=kg_vector_store,
        embed_model=Settings.embed_model,
        similarity_top_k=kg_similarity_top_k,
        limit=kg_similarity_top_k, 
        path_depth=2,
        similarity_score=0
    )

    # 3. 同义词检索器，用于基于关键词的扩展
    synonym_retriever = LLMSynonymRetriever(
        graph_store=kg_store,
        llm=llm_for_reasoning,
        max_keywords=10,
        path_depth=2,
        limit=kg_similarity_top_k, 
    )
    logger.debug(f"子检索器创建完成，耗时: {time.time() - step_time:.2f}s")

    # --- 定义后处理器 ---
    step_time = time.time()
    reranker = SiliconFlowRerank(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        top_n=kg_rerank_top_n,
    )
    logger.debug(f"Reranker创建完成，耗时: {time.time() - step_time:.2f}s")

    # --- 使用现代API创建查询引擎 ---
    step_time = time.time()
    query_engine = kg_index.as_query_engine(
        similarity_top_k=kg_similarity_top_k,
        sub_retrievers=[text2cypher_retriever, vector_retriever, synonym_retriever],
        response_synthesizer=synthesizer,
        node_postprocessors=[reranker] if kg_rerank_top_n > 0 else [],
        llm=llm_for_reasoning,  # 用于合成最终答案的LLM
        include_text=True,
    )
    logger.debug(f"as_query_engine调用完成，耗时: {time.time() - step_time:.2f}s")

    logger.success(f"知识图谱混合查询引擎创建成功。总耗时: {time.time() - start_time:.2f}s")
    return query_engine
