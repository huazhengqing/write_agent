import os
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple
import threading
import kuzu
from loguru import logger
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    Response,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer, CompactAndRefine
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import VectorStore
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import LLM_TEMPERATURES, get_llm_messages, get_llm_params, llm_completion
from utils.models import natural_sort_key
from utils.vector import get_embed_model
from utils.log import init_logger


_kg_stores: Dict[str, KuzuGraphStore] = {}
_kg_store_lock = threading.Lock()
def get_kg_store(db_path: str) -> KuzuGraphStore:
    with _kg_store_lock:
        if db_path in _kg_stores:
            return _kg_stores[db_path]
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        db = kuzu.Database(db_path)
        graph_store = KuzuGraphStore(db)
        _kg_stores[db_path] = graph_store
        return graph_store


def kg_add(
    storage_context: StorageContext,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    kg_extraction_prompt: str,
    content_format: Literal["markdown", "text", "json"] = "markdown",
    max_triplets_per_chunk: int = 15,
) -> None:

    doc = Document(id_=doc_id, text=content, metadata=metadata)

    transformations = []
    if content_format == "markdown":
        transformations.append(MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True))
    elif content_format == "text":
        transformations.append(SentenceSplitter(chunk_size=512, chunk_overlap=100, include_metadata=True, include_prev_next_rel=True))
    elif content_format == "json":
        pass
    else:
        raise ValueError("格式错误")

    llm_extract_params = get_llm_params(llm="fast", temperature=LLM_TEMPERATURES["summarization"])
    llm = LiteLLM(**llm_extract_params)

    KnowledgeGraphIndex.from_documents(
        [doc],
        storage_context=storage_context,
        llm=llm,
        embed_model=get_embed_model(),
        kg_extraction_prompt=PromptTemplate(kg_extraction_prompt),
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=True,
        transformations=transformations,
    )


def _format_response_with_sorting(response: Response, sort_by: Literal["time", "narrative", "relevance"]) -> str:
    """
    sort_by (Literal): 排序策略: 'time' (时间倒序), 'narrative' (章节顺序), 'relevance' (相关性)。
    """
    if not response.source_nodes:
        return f"未找到相关来源信息, 但综合回答是: \n{str(response)}"

    if sort_by == "narrative":
        sorted_nodes = sorted(
            list(response.source_nodes),
            key=lambda n: natural_sort_key(n.metadata.get("task_id", "")),
            reverse=False,  # 正序排列
        )
        sort_description = "按小说章节顺序排列 (从前到后)"
    elif sort_by == "time":
        sorted_nodes = sorted(
            list(response.source_nodes),
            key=lambda n: n.metadata.get("created_at", "1970-01-01T00:00:00"),
            reverse=True,  # 倒序排列, 最新的在前
        )
        sort_description = "按时间倒序排列 (最新的在前)"
    else:  # 'relevance' 或其他默认情况
        sort_description = "按相关性排序"
        sorted_nodes = list(response.source_nodes)

    source_details = []
    for node in sorted_nodes:
        timestamp = node.metadata.get("created_at", "未知时间")
        task_id = node.metadata.get("task_id", "未知章节")
        score = node.get_score()
        score_str = f"{score:.4f}" if score is not None else "N/A"
        content = re.sub(r"\s+", " ", node.get_content()).strip()
        source_details.append(
            f"来源信息 (章节: {task_id}, 时间: {timestamp}, 相关性: {score_str}):\n---\n{content}\n---"
        )

    formatted_sources = "\n\n".join(source_details)

    final_output = (
        f"综合回答:\n{str(response)}\n\n详细来源 ({sort_description}):\n{formatted_sources}"
    )
    return final_output


def hybrid_query(
    vector_store: VectorStore,
    graph_store: KuzuGraphStore,
    retrieval_query_text: str,
    synthesis_query_text: str,
    synthesis_system_prompt: str,
    synthesis_user_prompt: str,
    kg_nl2graphquery_prompt: Optional[PromptTemplate] = None,
    vector_filters: Optional[MetadataFilters] = None,
    vector_similarity_top_k: int = 150,
    vector_rerank_top_n: int = 50,
    kg_similarity_top_k: int = 300,
    kg_rerank_top_n: int = 100,
    vector_sort_by: Literal["time", "narrative", "relevance"] = "relevance",
    kg_sort_by: Literal["time", "narrative", "relevance"] = "relevance",
) -> str:
    """
    retrieval_query_text (str): 用于向量和图谱检索的查询文本。
    synthesis_query_text (str): 用于最终LLM综合的、更详细的代理查询文本。
    synthesis_system_prompt (str): 综合阶段的系统提示。
    synthesis_user_prompt (str): 综合阶段的用户提示模板。
    vector_store (VectorStore): 用于向量检索的向量存储。
    graph_store (KuzuGraphStore): 用于知识图谱检索的图存储。
    kg_nl2graphquery_prompt (Optional[PromptTemplate], optional): KG中NL2GraphQuery的提示。 Defaults to None.
    vector_filters (Optional[MetadataFilters]): 应用于向量检索的元数据过滤器。
    vector_similarity_top_k (int): 向量检索的top_k。
    vector_rerank_top_n (int): 向量检索后LLM重排的top_n。
    kg_similarity_top_k (int): KG混合检索中向量部分的top_k。
    kg_rerank_top_n (int): KG检索后LLM重排的top_n。
    vector_sort_by (Literal): 向量结果的排序方式。
    kg_sort_by (Literal): 知识图谱结果的排序方式。
    """

    embed_model = get_embed_model()

    reasoning_llm_params = get_llm_params(llm="reasoning", temperature=LLM_TEMPERATURES["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    synthesis_llm_params = get_llm_params(llm="reasoning", temperature=LLM_TEMPERATURES["synthesis"])
    synthesis_llm = LiteLLM(**synthesis_llm_params)

    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 4096),
            num_output=synthesis_llm_params.get('max_tokens', 512),
            chunk_overlap_ratio=0.2
        )
    )

    logger.info("构建向量查询引擎...")
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    vector_query_engine = vector_index.as_query_engine(
        filters=vector_filters,
        llm=reasoning_llm,
        response_synthesizer=response_synthesizer,
        similarity_top_k=vector_similarity_top_k,
        node_postprocessors=[
            LLMRerank(llm=reasoning_llm, top_n=vector_rerank_top_n)
        ]
    )
    logger.info(f"正在执行向量查询: '{retrieval_query_text}'")
    vector_response = vector_query_engine.query(retrieval_query_text)
    formatted_vector_str = _format_response_with_sorting(vector_response, vector_sort_by)
    logger.info(f"向量查询完成, 检索到 {len(vector_response.source_nodes)} 个节点。")

    logger.info("构建知识图谱查询引擎...")
    kg_storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex.from_documents(
        [],
        storage_context=kg_storage_context,
        llm=reasoning_llm,
        include_embeddings=True,
        embed_model=embed_model
    )
    kg_retriever = kg_index.as_retriever(
        retriever_mode="hybrid",
        similarity_top_k=kg_similarity_top_k,
        with_nl2graphquery=True,
        graph_traversal_depth=2,
        nl2graphquery_prompt=kg_nl2graphquery_prompt,
    )
    kg_query_engine = RetrieverQueryEngine(
        retriever=kg_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            LLMRerank(llm=reasoning_llm, top_n=kg_rerank_top_n)
        ]
    )
    logger.info(f"正在执行知识图谱查询: '{retrieval_query_text}'")
    kg_response = kg_query_engine.query(retrieval_query_text)
    formatted_kg_str = _format_response_with_sorting(kg_response, kg_sort_by)
    logger.info(f"知识图谱查询完成, 检索到 {len(kg_response.source_nodes)} 个节点。")

    context_dict_user = {
        "query_text": synthesis_query_text,
        "formatted_vector_str": formatted_vector_str,
        "formatted_kg_str": formatted_kg_str,
    }
    messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)

    final_llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=LLM_TEMPERATURES["synthesis"])

    final_message = llm_completion(final_llm_params)
    result = final_message.content

    return result
