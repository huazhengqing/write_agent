import os
import sys
import re
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import chromadb
from loguru import logger
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, get_leaf_nodes
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings_api.litellm import LiteLLMEmbedding
from llama_index.llms_api.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import call_react_agent, llm_temperatures, get_embedding_params, get_llm_params


_embed_model: Optional[LiteLLMEmbedding] = None
_embed_model_lock = threading.Lock()
def get_embed_model() -> LiteLLMEmbedding:
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            if _embed_model is None:
                embedding_params = get_embedding_params()
                embed_model_name = embedding_params.pop('model')
                _embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)
    return _embed_model


_vector_stores: Dict[Tuple[str, str], ChromaVectorStore] = {}
_vector_store_lock = threading.Lock()
def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    with _vector_store_lock:
        if (db_path, collection_name) in _vector_stores:
            return _vector_stores[(db_path, collection_name)]
        os.makedirs(db_path, exist_ok=True)
        db = chromadb.PersistentClient(path=db_path)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        _vector_stores[(db_path, collection_name)] = vector_store
        return vector_store


def _default_file_metadata(file_path_str: str) -> dict:
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


def get_nodes_from_document(doc: Document) -> List[Document]:
    """
    使用混合策略将文档解析为细粒度节点的辅助函数。
    - 结构化解析器：优先使用Markdown解析器保持文档结构。
    - 细粒度解析器：对结构化块进行二次切分，确保能检索到小片段信息。
    """
    structural_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
    fine_grained_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=50,
        include_metadata=True,
        include_prev_next_rel=True
    )
    # 步骤 1: 结构化分块
    structural_nodes = structural_parser.get_nodes_from_documents([doc])
    # 步骤 2: 对大块进行细粒度切分
    fine_grained_nodes = fine_grained_parser.get_nodes_from_documents(structural_nodes)
    # 使用叶子节点进行索引
    return get_leaf_nodes(fine_grained_nodes)


def vector_add_from_dir(
    vector_store: VectorStore,
    input_dir: str,
    file_metadata_func: Optional[Callable[[str], dict]] = None,
) -> bool:
    metadata_func = file_metadata_func or _default_file_metadata

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

    all_nodes = []
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", doc.id_))
        if not doc.text.strip():
            logger.warning(f"⚠️ 文件 '{file_path.name}' 内容为空，已跳过。")
            continue
        
        nodes = []
        if file_path.suffix == ".json":
            nodes = [doc]
        else:
            # 对于 .md 和 .txt, 使用辅助函数进行分块
            nodes = get_nodes_from_document(doc)

        logger.info(f"  - 文件 '{file_path.name}' 被解析成 {len(nodes)} 个节点。")
        all_nodes.extend(nodes)

    if not all_nodes:
        logger.warning("🤷‍♀️ 没有从文件中解析出任何可索引的节点。")
        return False

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())
    index.insert_nodes(all_nodes, show_progress=True)
    logger.success(f"成功从目录 '{input_dir}' 添加 {len(all_nodes)} 个节点到向量库。")
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
    
    final_metadata = metadata.copy()
    if "date" not in final_metadata:
        final_metadata["date"] = datetime.now().strftime("%Y-%m-%d")

    doc = Document(text=content, metadata=final_metadata, id_=doc_id)
    nodes = []
    if content_format == "json":
        nodes = [doc]
    else:
        nodes = get_nodes_from_document(doc)

    if not nodes:
        logger.warning(f"内容 (doc_id: {doc_id}) 未解析出任何节点，跳过添加。")
        return False

    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=get_embed_model()
    )
    index.insert_nodes(nodes)
    logger.success(f"成功将内容 (doc_id: {doc_id}, {len(nodes)}个节点) 添加到向量库。")
    return True


def get_vector_query_engine(
    vector_store: VectorStore,
    filters: Optional[MetadataFilters] = None,
    similarity_top_k: int = 15,
    rerank_top_n: Optional[int] = 3,
    use_auto_retriever: bool = False,
    vector_store_info: Optional[VectorStoreInfo] = None,
) -> BaseQueryEngine:
    logger.info("正在创建向量查询引擎...")
    logger.debug(
        f"参数: similarity_top_k={similarity_top_k}, rerank_top_n={rerank_top_n}, "
        f"use_auto_retriever={use_auto_retriever}, filters={filters}"
    )
    
    reasoning_llm_params = get_llm_params(llm="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    synthesis_llm_params = get_llm_params(llm="reasoning", temperature=llm_temperatures["synthesis"])
    synthesis_llm = LiteLLM(**synthesis_llm_params)

    rerank_llm_params = get_llm_params(llm="fast", temperature=0.0)
    rerank_llm = LiteLLM(**rerank_llm_params)

    postprocessors = []
    if rerank_top_n and rerank_top_n > 0:
        logger.info(f"配置 LLMRerank 后处理器, top_n={rerank_top_n}")
        reranker = LLMRerank(choice_batch_size=5, top_n=rerank_top_n, llm=rerank_llm)
        postprocessors.append(reranker)

    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 4096),
            num_output=synthesis_llm_params.get('max_tokens', 512),
            chunk_overlap_ratio=0.2
        )
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=get_embed_model()
    )

    if use_auto_retriever:
        logger.info("使用 VectorIndexAutoRetriever 模式。")
        if not vector_store_info:
            raise ValueError("使用自动检索器时, 必须提供 vector_store_info。")
        
        retriever = VectorIndexAutoRetriever(
            index,
            vector_store_info=vector_store_info,
            similarity_top_k=similarity_top_k,
            verbose=True
        )
        
        logger.success("自动检索查询引擎创建成功。")
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
            use_async=True,
        )
    else:
        logger.info("使用标准 as_query_engine 模式。")
        logger.success("标准查询引擎创建成功。")
        return index.as_query_engine(
            llm=reasoning_llm,
            response_synthesizer=response_synthesizer,
            filters=filters,
            similarity_top_k=similarity_top_k,
            node_postprocessors=postprocessors,
            use_async=True,
        )


async def index_query(
    query_engine: BaseQueryEngine,
    questions: List[str],
) -> List[str]:
    if not questions:
        return []

    logger.info(f"接收到 {len(questions)} 个索引查询问题。")
    logger.debug(f"问题列表: \n{questions}")

    all_nodes: Dict[str, NodeWithScore] = {}

    tasks = []
    for q in questions:
        query_text = f"{q}\n# 请使用中文回复"
        tasks.append(query_engine.aquery(query_text))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            logger.warning(f"查询 '{question}' 时出错: {result}")
            continue
        if result and result.source_nodes:
            for node in result.source_nodes:
                all_nodes[node.node.id_] = node

    if not all_nodes:
        logger.info("所有查询均未找到相关的源节点。")
        return []

    nodes_in_order = list(all_nodes.values())
    final_content = [re.sub(r"\s+", " ", node.get_content()).strip() for node in nodes_in_order]
    logger.info(f"查询完成，共聚合了 {len(final_content)} 个独特的知识片段。")
    logger.debug(f"返回的知识片段内容: \n{final_content}")
    return final_content


async def index_query_react(
    query_engine: BaseQueryEngine,
    query_str: str,
    agent_system_prompt: Optional[str] = None,
) -> str:
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="vector_search",
        description="用于查找设定、摘要等语义相似的内容 (例如: 角色背景, 世界观设定, 物品描述)。当问题比较复杂时, 你可以多次调用此工具来回答问题的不同部分, 然后综合答案。"
    )
    result = await call_react_agent(
        system_prompt=agent_system_prompt,
        user_prompt=query_str,
        tools=[vector_tool],
        llm_type="reasoning",
        temperature=llm_temperatures["reasoning"]
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)
    return result
