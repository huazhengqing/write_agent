import os
import re
import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
import chromadb
from loguru import logger
from diskcache import Cache
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import JSONNodeParser, SentenceSplitter
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo, MetadataInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

from utils.config import llm_temperatures, get_llm_params, get_embedding_params
from utils.file import cache_dir
from utils.vector_prompts import (
    summary_query_str,
    # text_qa_prompt,
    # refine_prompt,
    tree_summary_prompt,
    mermaid_summary_prompt,
    vector_store_query_prompt
)


ChromaVectorStore.model_config['extra'] = 'allow'
if hasattr(ChromaVectorStore, 'model_rebuild'):
    ChromaVectorStore.model_rebuild(force=True)


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


def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    db_path_obj = Path(db_path)
    db_path_obj.mkdir(parents=True, exist_ok=True)

    db = chromadb.PersistentClient(path=str(db_path_obj))
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    store_cache_path = db_path_obj / f"{collection_name}.cache.db"
    vector_store.cache = Cache(str(store_cache_path), size_limit=int(32 * (1024**2)))
    
    return vector_store


synthesis_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["synthesis"])

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


###############################################################################


def file_metadata_default(file_path_str: str) -> dict:
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


def _load_and_filter_documents(
    input_dir: str,
    metadata_func: Callable[[str], dict]
) -> List[Document]:
    logger.info(f"开始从目录 '{input_dir}' 加载和过滤文档...")
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
        return []
    
    logger.debug(f"从 '{input_dir}' 初始加载了 {len(documents)} 个文档。")
    valid_docs = []
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", doc.id_))
        if not doc.text or not doc.text.strip():
            logger.warning(f"⚠️ 文件 '{file_path.name}' 内容为空, 已跳过。")
            continue
        valid_docs.append(doc)
    
    logger.success(f"完成文档加载和过滤, 共获得 {len(valid_docs)} 个有效文档。")
    return valid_docs


class MermaidExtractor:
    def __init__(self, llm: LiteLLM, summary_prompt_str: str):
        self._llm = llm
        self._summary_prompt = PromptTemplate(summary_prompt_str)

    def get_nodes(self, mermaid_code: str, metadata: dict) -> List[BaseNode]:
        if not mermaid_code.strip():
            return []

        logger.debug("正在为 Mermaid 图表生成摘要...")
        summary_response = self._llm.predict(self._summary_prompt, mermaid_code=mermaid_code)
        logger.debug(f"Mermaid 图表摘要生成完毕, 长度: {len(summary_response)}")

        summary_node = TextNode(
            text=f"Mermaid图表摘要:\n{summary_response}",
            metadata=metadata,
        )
        code_node = TextNode(
            text=f"```mermaid\n{mermaid_code}\n```", metadata=metadata
        )
        logger.debug(f"创建了 Mermaid 摘要节点 (ID: {summary_node.id_}) 和代码节点 (ID: {code_node.id_})。")

        summary_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=code_node.id_)
        code_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=summary_node.id_)
        logger.debug("已在摘要节点和代码节点之间建立双向关系。")
        return [summary_node, code_node]


class CustomMarkdownNodeParser(MarkdownElementNodeParser):
    _mermaid_extractor: MermaidExtractor = PrivateAttr()

    def __init__(self, llm: LiteLLM, summary_query_str: str, mermaid_summary_prompt: str, **kwargs: Any):
        super().__init__(llm=llm, summary_query_str=summary_query_str, **kwargs)
        self._mermaid_extractor = MermaidExtractor(llm=llm, summary_prompt_str=mermaid_summary_prompt)

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        logger.debug(f"CustomMarkdownNodeParser: 开始从节点 (ID: {node.id_}) 提取子节点...")
        text = node.get_content()
        parts = re.split(r"(```mermaid\n.*?\n```)", text, flags=re.DOTALL)

        final_nodes: List[BaseNode] = []
        for part in parts:
            if not part.strip():
                continue
            
            if part.startswith("```mermaid"):
                logger.debug("在 Markdown 中检测到 Mermaid 图表, 正在提取...")
                mermaid_code = part.removeprefix("```mermaid\n").removesuffix("\n```")
                mermaid_nodes = self._mermaid_extractor.get_nodes(mermaid_code, node.metadata)
                logger.debug(f"  - Mermaid 图表部分提取了 {len(mermaid_nodes)} 个节点。")
                final_nodes.extend(mermaid_nodes)
            else:
                logger.debug("在 Markdown 中检测到常规文本部分, 正在使用父解析器处理...")
                temp_node = Document(text=part, metadata=node.metadata)
                regular_nodes = super().get_nodes_from_node(temp_node)
                logger.debug(f"  - 常规文本部分解析出 {len(regular_nodes)} 个节点。")
                final_nodes.extend(regular_nodes)

        logger.debug(f"CustomMarkdownNodeParser 完成处理, 共生成 {len(final_nodes)} 个子节点。")
        return final_nodes


def _get_node_parser(content_format: Literal["md", "txt", "json"], content_length: int = 0) -> NodeParser:
    if content_length > 20000:
        chunk_size = 1024
        chunk_overlap = 200
    elif content_length > 5000:
        chunk_size = 512
        chunk_overlap = 128
    else:
        chunk_size = 256
        chunk_overlap = 64

    logger.debug(f"为 '{content_format}' (长度: {content_length}) 选择节点解析器。")

    if content_format == "json":
        logger.debug("使用 JSONNodeParser。")
        return JSONNodeParser(
            include_metadata=True,
            max_depth=5, 
            levels_to_keep=2
        )
    elif content_format == "txt":
        logger.debug(f"使用 SentenceSplitter, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}。")
        return SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
    logger.debug("使用 CustomMarkdownNodeParser。")
    return CustomMarkdownNodeParser(
        llm=Settings.llm,
        summary_query_str=summary_query_str,
        mermaid_summary_prompt=mermaid_summary_prompt
    )


def filter_invalid_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    valid_nodes = []
    initial_count = len(nodes)
    for node in nodes:
        if node.text.strip() and re.search(r'\w', node.text):
            valid_nodes.append(node)
    
    removed_count = initial_count - len(valid_nodes)
    if removed_count > 0:
        logger.debug(f"过滤掉 {removed_count} 个无效或空节点。")
    return valid_nodes


def _parse_docs_to_nodes_by_format(documents: List[Document]) -> List[BaseNode]:
    logger.info("开始按文件格式解析文档为节点...")
    docs_by_format: Dict[str, List[Document]] = {
        "md": [], 
        "txt": [], 
        "json": []
    }
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", doc.id_))
        file_extension = file_path.suffix.lstrip('.')
        if file_extension in docs_by_format:
            docs_by_format[file_extension].append(doc)
        else:
            logger.warning(f"检测到未支持的文件扩展名 '{file_extension}', 将忽略。")

    all_nodes = []
    for content_format, format_docs in docs_by_format.items():
        if not format_docs:
            continue
        
        logger.info(f"正在处理 {len(format_docs)} 个 '{content_format}' 文件...")
        nodes_for_format = []
        for doc in format_docs:
            node_parser = _get_node_parser(content_format, content_length=len(doc.text))
            parsed_nodes = node_parser.get_nodes_from_documents([doc], show_progress=False)
            nodes_for_format.extend(filter_invalid_nodes(parsed_nodes))
        logger.info(f"  - 从 '{content_format}' 文件中成功解析出 {len(nodes_for_format)} 个节点。")
        all_nodes.extend(nodes_for_format)
    
    logger.success(f"文档解析完成, 总共生成 {len(all_nodes)} 个节点。")
    return all_nodes


def vector_add_from_dir(
    vector_store: VectorStore,
    input_dir: str,
    metadata_func: Callable[[str], dict] = file_metadata_default,
) -> bool:
    logger.info(f"开始从目录 '{input_dir}' 添加内容到向量库...")
    documents = _load_and_filter_documents(input_dir, metadata_func)
    if not documents:
        return False

    all_nodes = _parse_docs_to_nodes_by_format(documents)
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

    logger.info(f"准备将 {len(unique_nodes)} 个唯一节点注入 IngestionPipeline...")
    pipeline = IngestionPipeline(vector_store=vector_store)
    pipeline.run(nodes=unique_nodes)

    logger.success(f"成功从目录 '{input_dir}' 添加 {len(unique_nodes)} 个节点到向量库。")
    return True


def _parse_content_to_nodes(
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"],
    doc_id: Optional[str] = None,
) -> List[BaseNode]:
    logger.info(f"开始为 doc_id '{doc_id}' 解析内容为节点 (格式: {content_format})...")
    doc = Document(text=content, metadata=metadata, id_=doc_id)
    node_parser = _get_node_parser(content_format, content_length=len(content))
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

    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    doc_cache = getattr(vector_store, "cache", None)
    if doc_cache and doc_cache.get(new_content_hash):
        logger.info(f"内容 (hash: {new_content_hash[:8]}...) 已存在, 跳过重复添加。")
        return True

    effective_doc_id = doc_id or new_content_hash

    nodes_to_insert = _parse_content_to_nodes(content, metadata, content_format, effective_doc_id)
    if not nodes_to_insert:
        logger.warning(f"内容 (id: {effective_doc_id}) 未解析出任何有效节点, 跳过添加。")
        return False

    logger.info(f"准备将 {len(nodes_to_insert)} 个节点 (id: {effective_doc_id}) 注入 IngestionPipeline...")
    pipeline = IngestionPipeline(vector_store=vector_store)
    pipeline.run(nodes=nodes_to_insert)

    if doc_cache:
        doc_cache.set(new_content_hash, True)

    logger.success(f"成功将内容 (id: {effective_doc_id}, {len(nodes_to_insert)}个节点) 添加到向量库。")
    return True


###############################################################################


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
    vector_store_info: VectorStoreInfo,
    similarity_top_k: int,
    node_postprocessors: List,
) -> BaseQueryEngine:
    logger.info("正在创建 Auto-Retriever 查询引擎...")
    reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)
    retriever = VectorIndexAutoRetriever(
        index=index,
        vector_store_info=vector_store_info,
        llm=reasoning_llm,
        prompt_template_str=vector_store_query_prompt, 
        similarity_top_k=similarity_top_k,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=node_postprocessors,
    )
    logger.success("Auto-Retriever 查询引擎创建成功。")
    return query_engine


def get_vector_query_engine(
    vector_store: VectorStore,
    filters: Optional[MetadataFilters] = None,
    similarity_top_k: int = 25,
    top_n: int = 5,
    use_auto_retriever: bool = False,
    vector_store_info: VectorStoreInfo = get_vector_store_info_default(),
) -> BaseQueryEngine:
    logger.info("开始构建向量查询引擎...")
    logger.debug(
        f"参数: similarity_top_k={similarity_top_k}, top_n={top_n}, "
        f"use_auto_retriever={use_auto_retriever}, filters={filters}, "
    )

    index = VectorStoreIndex.from_vector_store(vector_store)
    logger.debug("从 VectorStore 创建 VectorStoreIndex 完成。")

    reranker = None
    if top_n and top_n > 0:
        logger.debug(f"正在创建 Reranker, top_n={top_n}")
        reranker = SiliconFlowRerank(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_n=top_n,
        )
    node_postprocessors = [reranker] if reranker else []

    if use_auto_retriever:
        query_engine = _create_auto_retriever_engine(
            index=index,
            vector_store_info=vector_store_info,
            similarity_top_k=similarity_top_k,
            node_postprocessors=node_postprocessors,
        )
    else:
        logger.info("正在创建标准查询引擎...")
        query_engine = index.as_query_engine(
            response_synthesizer=synthesizer, 
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

    if not source_nodes or not answer or answer == "Empty Response":
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

    logger.success(f"批量向量查询完成, 成功处理 {len(results)} 个问题。")
    return results
