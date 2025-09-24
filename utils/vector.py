import os
from pathlib import Path
from loguru import logger
from typing import Any, Callable, Dict, List, Literal, Optional

from llama_index.core import Document
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.schema import BaseNode
from llama_index.llms.litellm import LiteLLM


###############################################################################


from llama_index.vector_stores.chroma import ChromaVectorStore
ChromaVectorStore.model_config['extra'] = 'allow'
if hasattr(ChromaVectorStore, 'model_rebuild'):
    ChromaVectorStore.model_rebuild(force=True)


def init_llama_settings():
    from utils.llm_api import llm_temperatures, get_llm_params, get_embedding_params
    llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["summarization"])
    Settings.llm = LiteLLM(**llm_params)
    
    Settings.prompt_helper = PromptHelper(
        context_window=llm_params.get('context_window', 8192),
        num_output=llm_params.get('max_tokens', 2048),
        chunk_overlap_ratio=0.2,
    )

    embedding_params = get_embedding_params()
    embed_model_name = embedding_params.pop('model')
    from llama_index.embeddings.litellm import LiteLLMEmbedding
    Settings.embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)

init_llama_settings()


def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    db_path_obj = Path(db_path)
    db_path_obj.mkdir(parents=True, exist_ok=True)

    import chromadb
    db = chromadb.PersistentClient(path=str(db_path_obj))
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    store_cache_path = db_path_obj / f"{collection_name}.cache.db"
    from diskcache import Cache  
    vector_store.cache = Cache(str(store_cache_path), size_limit=int(32 * (1024**2)))
    
    return vector_store


def get_synthesizer():
    from utils.llm_api import llm_temperatures, get_llm_params
    synthesis_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["synthesis"])
    from llama_index.core.response_synthesizers import TreeSummarize
    from utils.vector_prompts import tree_summary_prompt
    synthesizer = TreeSummarize(
        llm=LiteLLM(**synthesis_llm_params),
        summary_template=PromptTemplate(tree_summary_prompt),
        prompt_helper = PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 8192),
            num_output=synthesis_llm_params.get('max_tokens', 2048),
            chunk_overlap_ratio=0.2,
        ),
        use_async=True,
    )
    return synthesizer


###############################################################################


def filter_invalid_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    valid_nodes = []
    initial_count = len(nodes)
    for node in nodes:
        import re
        if node.text.strip() and re.search(r'\w', node.text):
            valid_nodes.append(node)
    
    removed_count = initial_count - len(valid_nodes)
    if removed_count > 0:
        logger.debug(f"过滤掉 {removed_count} 个无效或空节点。")
    return valid_nodes


def _parse_content_to_nodes(
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"],
    doc_id: Optional[str] = None,
) -> List[BaseNode]:
    logger.info(f"开始为 doc_id '{doc_id}' 解析内容为节点 (格式: {content_format})...")
    doc = Document(text=content, metadata=metadata, id_=doc_id)
    from utils.vector_splitter import get_vector_node_parser
    node_parser = get_vector_node_parser(content_format, content_length=len(content))
    nodes = filter_invalid_nodes(node_parser.get_nodes_from_documents([doc], show_progress=False))
    logger.info(f"为 doc_id '{doc_id}' 解析出 {len(nodes)} 个节点。")
    return nodes


def vector_add(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"] = "md",
    doc_id: Optional[str] = None,
) -> bool:
    logger.info(f"开始向向量库添加内容, doc_id='{doc_id}', format='{content_format}'...")
    if not content or not content.strip() or "生成报告时出错" in content:
        logger.warning(f"🤷 内容为空或包含错误, 跳过存入向量库。元数据: {metadata}")
        return False

    import hashlib
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    doc_cache = getattr(vector_store, "cache", None)
    if doc_cache and doc_cache.get(new_content_hash):
        return True

    effective_doc_id = doc_id or new_content_hash

    nodes_to_insert = _parse_content_to_nodes(content, metadata, content_format, effective_doc_id)
    if not nodes_to_insert:
        logger.warning(f"内容 (id: {effective_doc_id}) 未解析出任何有效节点, 跳过添加。")
        return False

    if doc_id:
        logger.info(f"为 doc_id '{doc_id}' 执行更新操作, 将首先删除旧节点。")
        vector_store.delete(doc_id)

    from llama_index.core.ingestion import IngestionPipeline
    pipeline = IngestionPipeline(vector_store=vector_store, transformations=[Settings.embed_model])
    pipeline.run(nodes=nodes_to_insert)

    if doc_cache:
        doc_cache.set(new_content_hash, True)

    logger.success(f"成功将内容 (id: {effective_doc_id}, {len(nodes_to_insert)}个节点) 添加到向量库。")
    return True


###############################################################################


def get_vector_store_info_default() -> 'VectorStoreInfo':
    from llama_index.core.vector_stores import VectorStoreInfo
    from llama_index.core.vector_stores import MetadataInfo
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
    from utils.llm_api import llm_temperatures, get_llm_params
    from utils.vector_prompts import vector_store_query_prompt
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
    filters: Optional['MetadataFilters'] = None,
    similarity_top_k: int = 50,
    top_n: int = 10,
    use_auto_retriever: bool = False,
    vector_store_info: Optional['VectorStoreInfo'] = None,
) -> BaseQueryEngine:
    from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo

    logger.info("开始构建向量查询引擎...")

    index = VectorStoreIndex.from_vector_store(vector_store)
    logger.debug("从 VectorStore 创建 VectorStoreIndex 完成。")

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
    
    logger.success("向量查询引擎构建成功。")
    return query_engine


###############################################################################


async def index_query(query_engine: BaseQueryEngine, question: str) -> str:
    if not question:
        return ""
    
    logger.info(f"开始执行向量索引查询: '{question}'")
    result = await query_engine.aquery(question)

    answer = str(getattr(result, "response", "")).strip()
    source_nodes = getattr(result, "source_nodes", [])

    if not source_nodes or not answer or answer == "Empty Response" or "无法回答" in answer:
        logger.warning(f"查询 '{question}' 未检索到任何源节点或有效响应, 返回空回答。")
        answer = ""
    else:
        logger.info(f"查询 '{question}' 检索到 {len(source_nodes)} 个源节点。")
        for i, node in enumerate(source_nodes):
            logger.debug(
                f"  - 源节点 {i+1} (ID: {node.node_id}, 分数: {node.score:.4f}):\n"
                f"{node.get_content()[:200]}..."
            )
        logger.success(f"成功完成对 '{question}' 的查询, 生成回答长度: {len(answer)}")

    logger.debug(f"问题 '{question}' 的回答:\n{answer}")

    return answer


async def index_query_batch(query_engine: BaseQueryEngine, questions: List[str]) -> List[str]:
    if not questions:
        return []

    logger.info(f"开始执行 {len(questions)} 个问题的批量向量查询...")
    
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

    logger.info(f"收到批量回答: \n{results}")
    return results
