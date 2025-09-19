import os
import re
import sys
import numpy as np
import threading
import asyncio
from datetime import datetime
from pathlib import Path
import json
import chromadb
from diskcache import Cache
from typing import cast
from loguru import logger
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from pydantic import Field

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import MarkdownElementNodeParser, JSONNodeParser, SentenceSplitter
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo, MetadataInfo, MetadataFilter
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import llm_temperatures, get_llm_params, get_embedding_params
from utils.file import cache_dir


cache_query_dir = cache_dir / "query"
cache_query_dir.mkdir(parents=True, exist_ok=True)
cache_query = Cache(str(cache_query_dir), size_limit=int(32 * (1024**2)))


###############################################################################


default_text_qa_prompt_tmpl_cn = """
上下文信息如下。
---------------------
{context_str}
---------------------
请严格根据上下文信息而不是你的先验知识，回答问题。
如果上下文中没有足够的信息来回答问题，请不要编造答案，你的回答必须是且只能是一个空字符串，不包含任何其他文字。
问题: {query_str}
回答: 
"""

default_text_qa_prompt_cn = PromptTemplate(default_text_qa_prompt_tmpl_cn)


default_refine_prompt_tmpl_cn = """
原始问题如下: {query_str}
我们已经有了一个回答: {existing_answer}
我们有机会通过下面的更多上下文来优化已有的回答(仅在需要时)。
------------
{context_str}
------------
根据新的上下文，优化原始回答以更好地回答问题。
如果上下文没有用，请返回原始回答。
优化后的回答: 
"""

default_refine_prompt_cn = PromptTemplate(default_refine_prompt_tmpl_cn)


synthesis_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["synthesis"])

response_synthesizer_default = CompactAndRefine(
    llm=LiteLLM(**synthesis_llm_params),
    text_qa_template=default_text_qa_prompt_cn,
    refine_template=default_refine_prompt_cn,
    prompt_helper = PromptHelper(
        context_window=synthesis_llm_params.get('context_window', 8192),
        num_output=synthesis_llm_params.get('max_tokens', 2048),
        chunk_overlap_ratio=0.2,
    )
)


###############################################################################


def init_llama_settings():
    llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["summarization"])
    Settings.llm = LiteLLM(**llm_params)
    
    Settings.prompt_helper = PromptHelper(
        context_window=llm_params.get('context_window', 8192),
        num_output=llm_params.get('max_tokens', 2048),
        chunk_overlap_ratio=0.2,
    )

    embedding_params = get_embedding_params()
    embed_model_name = embedding_params.pop('model')
    Settings.embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)

init_llama_settings()


###############################################################################


_vector_stores: Dict[Tuple[str, str], ChromaVectorStore] = {}
_vector_store_lock = threading.Lock()

def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    with _vector_store_lock:
        cache_key = (db_path, collection_name)
        if cache_key in _vector_stores:
            return _vector_stores[cache_key]
        os.makedirs(db_path, exist_ok=True)
        db = chromadb.PersistentClient(path=db_path)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        _vector_stores[cache_key] = vector_store
        return vector_store


_vector_indices: Dict[int, VectorStoreIndex] = {}
_vector_index_lock = threading.Lock()

def clear_vector_index_cache(vector_store: Optional[VectorStore] = None):
    with _vector_index_lock:
        if vector_store:
            cache_key = id(vector_store)
            if cache_key in _vector_indices:
                del _vector_indices[cache_key]
        else:
            _vector_indices.clear()


###############################################################################


def get_node_parser(content_format: Literal["markdown", "text", "json"]) -> NodeParser:
    if content_format == "json":
        return JSONNodeParser(
            include_metadata=True,
            max_depth=3, 
            levels_to_keep=0
        )
    elif content_format == "text":
        return SentenceSplitter(
            chunk_size=256, 
            chunk_overlap=50,
        )
    return MarkdownElementNodeParser(
        llm=Settings.llm,
        chunk_size = 256, 
        chunk_overlap = 50, 
        # num_workers=3,
        include_metadata=True,
        show_progress=False,
    )


###############################################################################


def _filter_invalid_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """过滤掉无效的节点（内容为空或仅包含空白/非词汇字符）。"""
    valid_nodes = []
    for node in nodes:
        if node.text.strip() and re.search(r'\w', node.text):
            valid_nodes.append(node)
    return valid_nodes


def default_file_metadata(file_path_str: str) -> dict:
    file_path = Path(file_path_str)
    stat = file_path.stat()
    creation_time = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
    modification_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "file_name": file_path.name,
        "file_path": file_path_str,
        "creation_date": creation_time,
        "modification_date": modification_time,
    }


def vector_add_from_dir(
    vector_store: VectorStore,
    input_dir: str,
    file_metadata_func: Optional[Callable[[str], dict]] = None,
) -> bool:
    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            del _vector_indices[cache_key]

    metadata_func = file_metadata_func or default_file_metadata

    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        required_exts=[".md", ".txt", ".json"],
        file_metadata=metadata_func,
        recursive=True,
        exclude_hidden=False
    )

    documents = reader.load_data()
    if not documents:
        logger.warning(f"🤷 在 '{input_dir}' 目录中未找到任何符合要求的文件。")
        return False

    logger.info(f"🔍 找到 {len(documents)} 个文件，开始解析并构建节点...")

    # 按内容格式对文档进行分组，以便批量处理
    docs_by_format: Dict[str, List[Document]] = {"markdown": [], "text": [], "json": []}
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", doc.id_))
        if not doc.text or not doc.text.strip():
            logger.warning(f"⚠️ 文件 '{file_path.name}' 内容为空，已跳过。")
            continue
        
        file_extension = file_path.suffix.lstrip('.')
        content_format_map = {"md": "markdown", "txt": "text", "json": "json"}
        content_format = content_format_map.get(file_extension, "text")
        docs_by_format[content_format].append(doc)

    all_nodes = []
    for content_format, format_docs in docs_by_format.items():
        if not format_docs:
            continue
        
        logger.info(f"正在为 {len(format_docs)} 个 '{content_format}' 文件批量解析节点...")
        node_parser = get_node_parser(content_format)
        parsed_nodes = node_parser.get_nodes_from_documents(format_docs, show_progress=False)
        
        nodes_for_format = _filter_invalid_nodes(parsed_nodes)
        # 过滤掉仅包含分隔符或空白等非文本内容的无效节点 (已移至 _filter_invalid_nodes)
        logger.info(f"  - 从 '{content_format}' 文件中成功解析出 {len(nodes_for_format)} 个节点。")
        all_nodes.extend(nodes_for_format)

    if not all_nodes:
        logger.warning("🤷‍♀️ 没有从文件中解析出任何可索引的节点。")
        return False

    unique_nodes = []
    seen_ids = set()
    for node in all_nodes:
        if node.id_ not in seen_ids:
            unique_nodes.append(node)
            seen_ids.add(node.id_)
        else:
            logger.warning(f"发现并移除了重复的节点ID: {node.id_}。这可能由包含多个表格的Markdown文件引起。")

    pipeline = IngestionPipeline(vector_store=vector_store)
    pipeline.run(nodes=unique_nodes)

    logger.success(f"成功从目录 '{input_dir}' 添加 {len(unique_nodes)} 个节点到向量库。")
    return True


def vector_add(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["markdown", "text", "json"] = "markdown",
    doc_id: Optional[str] = None,
) -> bool:
    if not content or not content.strip() or "生成报告时出错" in content:
        logger.warning(f"🤷 内容为空或包含错误，跳过存入向量库。元数据: {metadata}")
        return False
    
    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            del _vector_indices[cache_key]
    
    # 相似度搜索去重
    query_embedding = Settings.embed_model.get_text_embedding(content)
    logger.trace(f"为 doc_id '{doc_id}' 生成的嵌入向量 (前10维): {query_embedding[:10]}")
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=1,
        filters=None,
    )
    query_result = vector_store.query(vector_store_query)
    if query_result.nodes:
        # 如果找到一个相似度极高的节点，我们有理由相信这是重复内容。
        # 之前的实现试图比较完整内容和节点内容，这是不准确的，因为节点只是文档的一部分。
        # 仅基于高相似度分数进行判断是更简单且鲁棒的做法。
        if query_result.similarities[0] > 0.995:
            logger.warning(f"发现与 doc_id '{doc_id}' 内容高度相似 (相似度: {query_result.similarities[0]:.4f}) 的文档，跳过添加。")
            return False

    if doc_id:
        logger.info(f"正在从向量库中删除 doc_id '{doc_id}' 的旧节点...")
        vector_store.delete(ref_doc_id=doc_id)
        logger.info(f"已删除 doc_id '{doc_id}' 的旧节点。")

    final_metadata = metadata.copy()
    if "date" not in final_metadata:
        final_metadata["date"] = datetime.now().strftime("%Y-%m-%d")

    doc = Document(text=content, metadata=final_metadata, id_=doc_id)
    node_parser = get_node_parser(content_format)
    parsed_nodes = node_parser.get_nodes_from_documents([doc], show_progress=False)
    nodes_to_insert = _filter_invalid_nodes(parsed_nodes)
    
    if not nodes_to_insert:
        logger.warning(f"内容 (doc_id: {doc_id}) 未解析出任何有效节点，跳过添加。")
        return False
    logger.debug(f"为 doc_id '{doc_id}' 创建的节点内容: {[n.get_content(metadata_mode='all') for n in nodes_to_insert]}")

    pipeline = IngestionPipeline(vector_store=vector_store)
    pipeline.run(nodes=nodes_to_insert)

    logger.success(f"成功将内容 (doc_id: {doc_id}, {len(nodes_to_insert)}个节点) 添加到向量库。")
    return True


###############################################################################


def get_default_vector_store_info() -> VectorStoreInfo:
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
            description="内容的创建或关联日期，格式为 'YYYY-MM-DD'。",
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


def get_vector_query_engine(
    vector_store: VectorStore,
    filters: Optional[MetadataFilters] = None,
    similarity_top_k: int = 15,
    rerank_top_n: Optional[int] = 3,
    use_auto_retriever: bool = False,
    vector_store_info: Optional[VectorStoreInfo] = None,
    similarity_cutoff: Optional[float] = None,
) -> BaseQueryEngine:
    
    logger.debug(
        f"参数: similarity_top_k={similarity_top_k}, rerank_top_n={rerank_top_n}, "
        f"use_auto_retriever={use_auto_retriever}, filters={filters}, "
        f"similarity_cutoff={similarity_cutoff}"
    )

    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            index = _vector_indices[cache_key]
        else:
            index = VectorStoreIndex.from_vector_store(vector_store)
            _vector_indices[cache_key] = index

    postprocessors = []
    if rerank_top_n and rerank_top_n > 0:
        reranker = SiliconFlowRerank(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_n=rerank_top_n,
        )
        postprocessors.append(reranker)

    reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)
    
    if use_auto_retriever:
        logger.info("使用 VectorIndexAutoRetriever 模式创建查询引擎。")
        
        final_vector_store_info = vector_store_info or get_default_vector_store_info()
        
        retriever = VectorIndexAutoRetriever(
            index,
            vector_store_info=final_vector_store_info,
            similarity_top_k=similarity_top_k,
            llm=reasoning_llm,
            verbose=True,
            similarity_cutoff=similarity_cutoff,
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer_default,
            node_postprocessors=postprocessors,
        )
        logger.success("自动检索查询引擎创建成功。")
        return query_engine
    else:
        logger.info("使用标准 as_query_engine 模式创建查询引擎。")
        retriever_kwargs = {}
        if similarity_cutoff is not None:
            retriever_kwargs["similarity_cutoff"] = similarity_cutoff

        query_engine = index.as_query_engine(
            llm=reasoning_llm,
            response_synthesizer=response_synthesizer_default,
            filters=filters,
            similarity_top_k=similarity_top_k,
            node_postprocessors=postprocessors,
            **retriever_kwargs,
        )
        logger.success("标准查询引擎创建成功。")
        return query_engine


###############################################################################


async def index_query(query_engine: BaseQueryEngine, question: str) -> str:
    if not question:
        return ""

    cache_key = None
    retriever = getattr(query_engine, "retriever", getattr(query_engine, "_retriever", None))
    # 注意：下面的缓存键生成方式依赖于 llama-index 和 chromadb 的内部实现细节（如 `_vector_store`, `_path`）。
    # 这在库版本更新时可能会失效。更稳妥的方案是显式传递数据库路径和集合名称来构建缓存键。
    # 此处使用 getattr 进行安全访问以增加代码韧性。
    vector_store = getattr(retriever, '_vector_store', None)
    if isinstance(vector_store, ChromaVectorStore):
        collection = getattr(vector_store, 'collection', None)
        client = getattr(vector_store, 'client', None)
        collection_name = getattr(collection, 'name', None)
        db_path = getattr(client, '_path', None)
        
        if db_path and collection_name:
            cache_key = f"index_query:{db_path}:{collection_name}:{question}"

    if cache_key:
        cached_result = cache_query.get(cache_key)
        if cached_result is not None:
            logger.info(f"从缓存中获取查询 '{question}' 的结果。")
            return cached_result

    logger.info(f"开始执行索引查询: '{question}'")
    result = await query_engine.aquery(question)

    answer = str(getattr(result, "response", "")).strip()
    if not result or not getattr(result, "source_nodes", []) or not answer or answer == "Empty Response":
        logger.warning(f"查询 '{question}' 未检索到任何源节点或有效响应，返回空回答。")
        answer = ""

    logger.debug(f"问题 '{question}' 的回答: {answer}")

    if cache_key:
        cache_query.set(cache_key, answer)

    return answer


async def index_query_batch(query_engine: BaseQueryEngine, questions: List[str]) -> List[str]:
    if not questions:
        return []

    logger.info(f"接收到 {len(questions)} 个索引查询问题。")
    logger.debug(f"问题列表: \n{questions}")

    # 使用 Semaphore 限制并发量为3，防止对LLM API造成过大压力。
    sem = asyncio.Semaphore(3)

    async def safe_query(question: str) -> str:
        async with sem:
            return await index_query(query_engine, question)

    tasks = [safe_query(q) for q in questions]
    results = await asyncio.gather(*tasks)

    logger.success(f"批量查询完成。")
    return results


###############################################################################
