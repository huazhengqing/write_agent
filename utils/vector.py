import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import chromadb
from loguru import logger
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core import PromptTemplate
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm import get_embedding_params, get_llm_params


_embed_model: Optional[LiteLLMEmbedding] = None
def get_embed_model() -> LiteLLMEmbedding:
    global _embed_model
    if _embed_model is None:
        embedding_params = get_embedding_params()
        embed_model_name = embedding_params.pop('model')
        _embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)
    return _embed_model


def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    os.makedirs(db_path, exist_ok=True)
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def vector_query(
    vector_store: VectorStore,
    query_text: str,
    filters: Optional[MetadataFilters] = None,
    similarity_top_k: int = 15,
    rerank_top_n: Optional[int] = 3,
) -> Tuple[Optional[str], Optional[List[NodeWithScore]]]:
    
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())

    postprocessors = []
    if rerank_top_n and rerank_top_n > 0:
        rerank_llm_params = get_llm_params(llm="fast")
        reranker = LLMRerank(choice_batch_size=5, top_n=rerank_top_n, llm=LiteLLM(**rerank_llm_params))
        postprocessors.append(reranker)

    synthesis_llm_params = get_llm_params(llm="reasoning")
    synthesis_llm = LiteLLM(**synthesis_llm_params)

    # 定义中文提示词
    TEXT_QA_TEMPLATE_CN = PromptTemplate(
        """你是一个问答机器人。
        你将根据以下上下文回答问题。
        ---------------------
        {context_str}
        ---------------------
        基于以上上下文，请回答以下问题：{query_str}
        """
    )

    REFINE_TEMPLATE_CN = PromptTemplate(
        """你是一个问答机器人，你正在改进一个已有的答案。
        你已经提供了一个答案：{existing_answer}
        你现在有更多的上下文信息：
        ---------------------
        {context_msg}
        ---------------------
        请根据新的上下文信息改进你的答案。
        如果你不能改进你的答案，请直接返回已有的答案。
        """
    )

    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params['context_window'],
            num_output=synthesis_llm_params['max_tokens'],
            chunk_overlap_ratio=0.2
        ),
        text_qa_template=TEXT_QA_TEMPLATE_CN,
        refine_template=REFINE_TEMPLATE_CN
    )

    query_engine = index.as_query_engine(
        llm=synthesis_llm,
        response_synthesizer=response_synthesizer,
        filters=filters,
        similarity_top_k=similarity_top_k,
        node_postprocessors=postprocessors
    )

    response = query_engine.query(query_text)
    if not response.response or not response.source_nodes:
        logger.warning("🤷 未能生成答案或找到相关文档。")
        return None, None

    return response.response, response.source_nodes


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

    md_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
    txt_parser = SentenceSplitter(chunk_size=600, chunk_overlap=120, include_metadata=True, include_prev_next_rel=True)
    all_nodes = []
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", doc.id_))
        if not doc.text.strip():
            logger.warning(f"⚠️ 文件 '{file_path.name}' 内容为空，已跳过。")
            continue
        if file_path.suffix == ".md":
            nodes = md_parser.get_nodes_from_documents([doc])
        elif file_path.suffix == ".json":
            nodes = [doc]
        else:
            nodes = txt_parser.get_nodes_from_documents([doc])
        logger.info(f"  - 文件 '{file_path.name}' 被解析成 {len(nodes)} 个节点。")
        all_nodes.extend(nodes)

    if not all_nodes:
        logger.warning("🤷‍♀️ 没有从文件中解析出任何可索引的节点。")
        return False

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())
    index.insert_nodes(all_nodes, show_progress=True)
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

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())

    if content_format == "markdown":
        parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
        nodes = parser.get_nodes_from_documents([doc])
        index.insert_nodes(nodes)
    elif content_format == "json":
        index.insert(doc)
    else:
        parser = SentenceSplitter(chunk_size=600, chunk_overlap=120, include_metadata=True, include_prev_next_rel=True)
        nodes = parser.get_nodes_from_documents([doc])
        index.insert_nodes(nodes)

    return True


