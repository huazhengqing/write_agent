import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
from litellm.exceptions import RateLimitError
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
        logger.info("首次访问，正在初始化嵌入模型...")
        embedding_params = get_embedding_params()
        embed_model_name = embedding_params.pop('model')
        _embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)
        logger.success("嵌入模型初始化完成。")
    return _embed_model

def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    logger.info(f"正在访问ChromaDB: path='{db_path}', collection='{collection_name}'")
    os.makedirs(db_path, exist_ok=True)
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    logger.success(f"ChromaDB向量存储 '{collection_name}' 已准备就绪。")
    return vector_store

def vector_query(
    vector_store: VectorStore,
    query_text: str,
    filters: Optional[MetadataFilters] = None,
    similarity_top_k: int = 15,
    rerank_top_n: Optional[int] = 3,
) -> Tuple[Optional[str], Optional[List[NodeWithScore]]]:
    logger.info(f"🚀 开始执行向量查询与合成: '{query_text}'")

    logger.info("从向量存储和嵌入模型初始化向量索引...")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())

    postprocessors = []
    if rerank_top_n and rerank_top_n > 0:
        logger.info(f"正在配置LLM重排序器 (top_n={rerank_top_n})...")
        rerank_llm_params = get_llm_params(llm="fast")
        reranker = LLMRerank(choice_batch_size=5, top_n=rerank_top_n, llm=LiteLLM(**rerank_llm_params))
        postprocessors.append(reranker)
        log_message = f"正在执行查询：初步检索 {similarity_top_k} 个文档，重排并选出前 {rerank_top_n} 个用于合成答案..."
    else:
        log_message = f"正在执行查询：检索 {similarity_top_k} 个文档用于合成答案 (无重排)..."

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

    # response_synthesizer = get_response_synthesizer(
    #     llm=synthesis_llm,
    #     text_qa_template=TEXT_QA_TEMPLATE_CN,
    #     refine_template=REFINE_TEMPLATE_CN
    # )
    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params['context_window'],
            num_output=synthesis_llm_params['max_tokens'],
            chunk_overlap_ratio=0.2
        )
        ,text_qa_template=TEXT_QA_TEMPLATE_CN,
        refine_template=REFINE_TEMPLATE_CN
    )

    query_engine = index.as_query_engine(
        llm=synthesis_llm,
        response_synthesizer=response_synthesizer,
        filters=filters,
        similarity_top_k=similarity_top_k,
        node_postprocessors=postprocessors
    )

    logger.info(log_message)
    response = query_engine.query(query_text)
    if not response.response or not response.source_nodes:
        logger.warning("🤷 未能生成答案或找到相关文档。")
        return None, None
    else:
        logger.success("✅ 查询成功完成。")
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
    """
    从指定目录加载、解析文件，并将内容存入向量数据库。
    Args:
        vector_store (VectorStore): 目标向量存储。
        input_dir (str): 输入目录的路径。
        file_metadata_func (Optional[Callable[[str], dict]]): 用于从文件名生成元数据的函数。如果为None，则使用默认函数提取文件名和时间戳。
    """
    logger.info(f"📂 开始从目录 '{input_dir}' 摄取文件...")
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
            # JSON文件作为一个整体节点，不进行分割
            nodes = [doc]
        else:
            nodes = txt_parser.get_nodes_from_documents([doc])
        logger.info(f"  - 文件 '{file_path.name}' 被解析成 {len(nodes)} 个节点。")
        all_nodes.extend(nodes)

    if all_nodes:
        logger.info(f"⚙️ 节点构建完成 ({len(all_nodes)}个节点)，准备将数据存入向量数据库...")
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())
        index.insert_nodes(all_nodes, show_progress=True)
        logger.success(f"✅ 成功处理 {len(documents)} 个文件并存入向量数据库。")
        return True
    else:
        logger.warning("🤷‍♀️ 没有从文件中解析出任何可索引的节点。")
        return False


def vector_add(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    content_format: str = "text",
    doc_id: Optional[str] = None,
) -> bool:
    """
    Args:
        vector_store (VectorStore): 目标向量存储。
        content (str): 要存储的文本内容。
        metadata (Dict[str, Any]): 与内容关联的元数据字典。
        content_format (str): 内容格式，支持 "text", "markdown", "json"。
        doc_id (Optional[str]): 文档的唯一ID。如果为None，则自动生成。
    """
    if not content or not content.strip() or "生成报告时出错" in content:
        logger.warning(f"🤷 内容为空或包含错误，跳过存入向量库。元数据: {metadata}")
        return False

    logger.info(f"✍️ 正在将类型为 '{metadata.get('type', 'N/A')}' (格式: {content_format}) 的内容存入向量库...")

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
        # 对于JSON字符串，通常作为单个文档直接插入
        index.insert(doc)
    else:  # "text"
        parser = SentenceSplitter(chunk_size=600, chunk_overlap=120, include_metadata=True, include_prev_next_rel=True)
        nodes = parser.get_nodes_from_documents([doc])
        index.insert_nodes(nodes)

    logger.success(f"✅ 内容已成功存入向量库。元数据: {final_metadata}")
    return True


if __name__ == "__main__":
    from utils.log import init_logger
    init_logger(os.path.splitext(os.path.basename(__file__))[0])

    test_db_path = "./.test_chroma_db_vector"
    test_collection_name = "test_collection_vector"
    vector_store = get_vector_store(db_path=test_db_path, collection_name=test_collection_name)
    
    doc_id_1 = "single_doc_001"
    metadata_1 = {"type": "test_doc", "author": "tester1"}
    content_1 = "这是一个关于人工智能如何改变软件工程的测试文档。"
    vector_store(
        vector_store=vector_store,
        content=content_1,
        metadata=metadata_1,
        content_format="text",
        doc_id=doc_id_1
    )

    test_input_dir = "./.test_input_dir_vector"
    os.makedirs(test_input_dir, exist_ok=True)
    
    with open(os.path.join(test_input_dir, "test1.txt"), "w", encoding="utf-8") as f:
        f.write("这是第一个文本文件，内容是关于机器学习的基础知识。")
        
    with open(os.path.join(test_input_dir, "test2.md"), "w", encoding="utf-8") as f:
        f.write("# Markdown 测试\n\n这是一个 Markdown 文件，讨论了大型语言模型（LLM）的应用。")

    vector_add_from_dir(
        vector_store=vector_store,
        input_dir=test_input_dir,
        required_exts=[".txt", ".md"]
    )

    query_text = "大型语言模型有什么用？"
    answer, source_nodes = vector_query(
        vector_store=vector_store,
        query_text=query_text,
        similarity_top_k=5,
        rerank_top_n=2
    )

    if answer and source_nodes:
        logger.success("\n--- ✅ 最终答案 ---")
        logger.info(answer)
        
        logger.info("\n--- 答案来源 (经重排序) ---")
        for i, node in enumerate(source_nodes):
            score = node.score if node.score is not None else 'N/A'
            score_str = f"{score:.4f}" if isinstance(score, float) else score
            logger.info(f"\n📄 文档 {i+1}: (相关性得分: {score_str})")
            logger.info(f"  - 元数据: {node.metadata}")
            logger.info(f"  - 内容:\n{node.get_content()}")
    else:
        logger.error("查询失败，未返回任何结果。")

    logger.info("\n--- 测试完成 ---")
    logger.info(f"你可以检查以下目录来验证结果: '{test_db_path}', '{test_input_dir}'")
