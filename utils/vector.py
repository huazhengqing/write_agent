import os
import re
import asyncio
from datetime import datetime
from pathlib import Path
import chromadb
from loguru import logger
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

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
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo, MetadataInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

from utils.config import llm_temperatures, get_llm_params, get_embedding_params
from utils.vector_prompts import (
    summary_query_str,
    text_qa_prompt,
    refine_prompt,
    mermaid_summary_prompt,
)


###############################################################################


synthesis_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["synthesis"])

synthesizer = CompactAndRefine(
    llm=LiteLLM(**synthesis_llm_params),
    text_qa_template=PromptTemplate(text_qa_prompt),
    refine_template=PromptTemplate(refine_prompt),
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


def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    os.makedirs(db_path, exist_ok=True)
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


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
    """从目录加载文档并过滤掉空文件。"""
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

    logger.info(f"🔍 找到 {len(documents)} 个文件，开始过滤和解析...")
    
    valid_docs = []
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", doc.id_))
        if not doc.text or not doc.text.strip():
            logger.warning(f"⚠️ 文件 '{file_path.name}' 内容为空，已跳过。")
            continue
        valid_docs.append(doc)
    
    return valid_docs


class MermaidExtractor:
    """
    一个辅助类，专门用于解析Mermaid图表。
    它会为图表生成一个自然语言摘要，并将摘要和原始代码作为独立的节点返回。
    """

    def __init__(self, llm: LiteLLM, summary_prompt_str: str):
        self._llm = llm
        self._summary_prompt = PromptTemplate(summary_prompt_str)

    def get_nodes(self, mermaid_code: str, metadata: dict) -> List[BaseNode]:
        if not mermaid_code.strip():
            return []

        summary_response = self._llm.predict(self._summary_prompt, mermaid_code=mermaid_code)

        summary_node = TextNode(
            text=f"Mermaid图表摘要:\n{summary_response}",
            metadata=metadata,
        )
        code_node = TextNode(
            text=f"```mermaid\n{mermaid_code}\n```", metadata=metadata
        )

        summary_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=code_node.id_)
        code_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=summary_node.id_)
        return [summary_node, code_node]


class CustomMarkdownNodeParser(MarkdownElementNodeParser):
    """
    一个自定义的Markdown节点解析器。
    它首先分离出Mermaid图表进行特殊处理（生成摘要），
    然后将其余的Markdown内容交给内置的MarkdownElementNodeParser处理。
    """
    def __init__(self, llm: LiteLLM, summary_query_str: str, mermaid_summary_prompt: str, **kwargs: Any):
        # 将llm和summary_query_str传递给父类，以确保表格摘要功能正常工作
        super().__init__(llm=llm, summary_query_str=summary_query_str, **kwargs)
        self.mermaid_extractor = MermaidExtractor(llm=llm, summary_prompt_str=mermaid_summary_prompt)

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """重写此方法以实现对Mermaid图的特殊处理。"""
        text = node.get_content()
        # 使用正则表达式分割文本，保留Mermaid代码块作为独立部分
        parts = re.split(r"(```mermaid\n.*?\n```)", text, flags=re.DOTALL)
        
        final_nodes: List[BaseNode] = []
        for part in parts:
            if not part.strip():
                continue
            
            if part.startswith("```mermaid"):
                # 这是Mermaid图表部分，使用自定义提取器处理
                mermaid_code = part.removeprefix("```mermaid\n").removesuffix("\n```")
                final_nodes.extend(self.mermaid_extractor.get_nodes(mermaid_code, node.metadata))
            else:
                # 这是普通Markdown部分，创建临时文档并调用父类的方法处理
                # 这样可以完美复用父类对表格、标题等的原生解析能力
                temp_node = Document(text=part, metadata=node.metadata)
                # 直接调用父类的 get_nodes_from_node 方法，以避免无限递归
                final_nodes.extend(super().get_nodes_from_node(temp_node))
                
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

    if content_format == "json":
        return JSONNodeParser(
            include_metadata=True,
            max_depth=5, 
            levels_to_keep=2
        )
    elif content_format == "txt":
        return SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
    # 使用自定义的Markdown解析器，以支持Mermaid图表和自定义中文表格摘要
    return CustomMarkdownNodeParser(
        llm=Settings.llm,
        summary_query_str=summary_query_str,
        mermaid_summary_prompt=mermaid_summary_prompt
    )


def filter_invalid_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """过滤掉无效的节点（内容为空或仅包含空白/非词汇字符）。"""
    valid_nodes = []
    for node in nodes:
        if node.text.strip() and re.search(r'\w', node.text):
            valid_nodes.append(node)
    return valid_nodes


def _parse_docs_to_nodes_by_format(documents: List[Document]) -> List[BaseNode]:
    """根据文件格式将文档解析为节点。"""
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
            logger.warning(f"检测到未支持的文件扩展名 '{file_extension}'，将忽略。")

    all_nodes = []
    for content_format, format_docs in docs_by_format.items():
        if not format_docs:
            continue
        
        logger.info(f"正在为 {len(format_docs)} 个 '{content_format}' 文件动态解析节点...")
        nodes_for_format = []
        for doc in format_docs:
            node_parser = _get_node_parser(content_format, content_length=len(doc.text))
            parsed_nodes = node_parser.get_nodes_from_documents([doc], show_progress=False)
            nodes_for_format.extend(filter_invalid_nodes(parsed_nodes))
        logger.info(f"  - 从 '{content_format}' 文件中成功解析出 {len(nodes_for_format)} 个节点。")
        all_nodes.extend(nodes_for_format)
    
    return all_nodes


def vector_add_from_dir(
    vector_store: VectorStore,
    input_dir: str,
    metadata_func: Callable[[str], dict] = file_metadata_default,
) -> bool:
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

    pipeline = IngestionPipeline(vector_store=vector_store)
    pipeline.run(nodes=unique_nodes)

    logger.success(f"成功从目录 '{input_dir}' 添加 {len(unique_nodes)} 个节点到向量库。")
    return True


def _is_content_too_similar(
    vector_store: VectorStore,
    content: str,
    threshold: float,
    doc_id: Optional[str] = None
) -> bool:
    """检查内容是否与向量库中现有文档过于相似。"""
    query_embedding = Settings.embed_model.get_text_embedding(content)
    logger.trace(f"为 doc_id '{doc_id}' 生成的嵌入向量 (前10维): {query_embedding[:10]}")
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=1, filters=None
    )
    query_result = vector_store.query(vector_store_query)
    if query_result.nodes and query_result.similarities:
        is_updating_itself = doc_id and query_result.nodes[0].ref_doc_id == doc_id
        if not is_updating_itself and query_result.similarities[0] > threshold:
            logger.warning(f"发现与 doc_id '{doc_id}' 内容高度相似 (相似度: {query_result.similarities[0]:.4f}) 的文档 (ID: '{query_result.nodes[0].ref_doc_id}'), 跳过添加。")
            return True
    return False


def _parse_content_to_nodes(
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"],
    doc_id: Optional[str] = None,
) -> List[BaseNode]:
    """将单个内容字符串解析为节点列表。"""
    final_metadata = metadata.copy()
    if "date" not in final_metadata:
        final_metadata["date"] = datetime.now().strftime("%Y-%m-%d")
    doc = Document(text=content, metadata=final_metadata, id_=doc_id)
    node_parser = _get_node_parser(content_format, content_length=len(content))
    return filter_invalid_nodes(node_parser.get_nodes_from_documents([doc], show_progress=False))


def vector_add(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"] = "md",
    doc_id: Optional[str] = None,
    check_similarity: bool = False,
    similarity_threshold: float = 0.999,
) -> bool:
    if not content or not content.strip() or "生成报告时出错" in content:
        logger.warning(f"🤷 内容为空或包含错误，跳过存入向量库。元数据: {metadata}")
        return False
    
    if check_similarity and _is_content_too_similar(vector_store, content, similarity_threshold, doc_id):
        return False

    if doc_id:
        logger.info(f"正在从向量库中删除 doc_id '{doc_id}' 的旧节点...")
        vector_store.delete(ref_doc_id=doc_id)

    nodes_to_insert = _parse_content_to_nodes(content, metadata, content_format, doc_id)
    if not nodes_to_insert:
        logger.warning(f"内容 (doc_id: {doc_id}) 未解析出任何有效节点，跳过添加。")
        return False

    pipeline = IngestionPipeline(vector_store=vector_store)
    pipeline.run(nodes=nodes_to_insert)

    logger.success(f"成功将内容 (doc_id: {doc_id}, {len(nodes_to_insert)}个节点) 添加到向量库。")
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


def _create_reranker(rerank_top_n: int) -> Optional[SiliconFlowRerank]:
    if rerank_top_n and rerank_top_n > 0:
        return SiliconFlowRerank(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_n=rerank_top_n,
        )
    return None


def _create_auto_retriever_engine(
    index: VectorStoreIndex,
    vector_store_info: VectorStoreInfo,
    similarity_top_k: int,
    similarity_cutoff: float,
    postprocessors: List,
) -> BaseQueryEngine:
    logger.info("使用 VectorIndexAutoRetriever 模式创建查询引擎。")
    reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)
    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
        similarity_top_k=similarity_top_k,
        llm=reasoning_llm,
        verbose=True,
        similarity_cutoff=similarity_cutoff,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=postprocessors,
    )
    logger.success("自动检索查询引擎创建成功。")
    return query_engine


def _create_standard_query_engine(
    index: VectorStoreIndex,
    filters: Optional[MetadataFilters],
    similarity_top_k: int,
    similarity_cutoff: float,
    postprocessors: List,
) -> BaseQueryEngine:
    logger.info("使用标准 as_query_engine 模式创建查询引擎。")
    return index.as_query_engine(
        response_synthesizer=synthesizer, filters=filters, similarity_top_k=similarity_top_k,
        node_postprocessors=postprocessors, similarity_cutoff=similarity_cutoff
    )


def get_vector_query_engine(
    vector_store: VectorStore,
    filters: Optional[MetadataFilters] = None,
    similarity_top_k: int = 25,
    rerank_top_n: int = 5,
    similarity_cutoff: float = 0,
    use_auto_retriever: bool = False,
    vector_store_info: VectorStoreInfo = get_vector_store_info_default(),
) -> BaseQueryEngine:
    
    logger.debug(
        f"参数: similarity_top_k={similarity_top_k}, rerank_top_n={rerank_top_n}, "
        f"use_auto_retriever={use_auto_retriever}, filters={filters}, "
        f"similarity_cutoff={similarity_cutoff}"
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    reranker = _create_reranker(rerank_top_n)
    postprocessors = [reranker] if reranker else []

    if use_auto_retriever:
        return _create_auto_retriever_engine(
            index=index,
            vector_store_info=vector_store_info,
            similarity_top_k=similarity_top_k,
            similarity_cutoff=similarity_cutoff,
            postprocessors=postprocessors,
        )
    else:
        return _create_standard_query_engine(
            index=index,
            filters=filters,
            similarity_top_k=similarity_top_k,
            similarity_cutoff=similarity_cutoff,
            postprocessors=postprocessors,
        )


###############################################################################


async def index_query(query_engine: BaseQueryEngine, question: str) -> str:
    if not question:
        return ""

    logger.info(f"开始执行索引查询: '{question}'")
    result = await query_engine.aquery(question)

    answer = str(getattr(result, "response", "")).strip()
    if not result or not getattr(result, "source_nodes", []) or not answer or answer == "Empty Response":
        logger.warning(f"查询 '{question}' 未检索到任何源节点或有效响应，返回空回答。")
        answer = ""

    logger.debug(f"问题 '{question}' 的回答: {answer}")

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
            try:
                return await index_query(query_engine, question)
            except Exception as e:
                logger.error("批量查询中，问题 '{}' 失败: {}", question, e, exc_info=True)
                return ""

    tasks = [safe_query(q) for q in questions]
    results = await asyncio.gather(*tasks)

    logger.success(f"批量查询完成。")
    return results
