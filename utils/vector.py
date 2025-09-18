import os
import sys
import re
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from pydantic import Field
import chromadb
from loguru import logger
from llama_index.core import Settings
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo, MetadataInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from litellm import arerank, rerank
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import llm_temperatures, get_embedding_params, get_llm_params, get_rerank_params
from utils.agent import call_react_agent


def setup_global_settings():
    if getattr(Settings, '_llm', None) is None:
        default_llm_params = get_llm_params(llm_group="fast", temperature=llm_temperatures["summarization"])
        Settings.llm = LiteLLM(**default_llm_params)

setup_global_settings()


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
                if getattr(Settings, '_embed_model', None) is None:
                    Settings.embed_model = _embed_model
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


_vector_indices: Dict[int, VectorStoreIndex] = {}
_vector_index_lock = threading.Lock()


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
    summary_llm_params = get_llm_params(llm_group="fast", temperature=llm_temperatures["summarization"])
    summary_llm = LiteLLM(**summary_llm_params)
    parser = MarkdownElementNodeParser(
        llm=summary_llm,
        chunk_size=256,
        chunk_overlap=50
    )
    nodes = parser.get_nodes_from_documents([doc])
    return nodes


def vector_add_from_dir(
    vector_store: VectorStore,
    input_dir: str,
    file_metadata_func: Optional[Callable[[str], dict]] = None,
) -> bool:
    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            logger.info(f"向量库内容变更, 使缓存的 VectorStoreIndex 失效 (key: {cache_key})。")
            del _vector_indices[cache_key]

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
    
    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            logger.info(f"向量库内容变更, 使缓存的 VectorStoreIndex 失效 (key: {cache_key})。")
            del _vector_indices[cache_key]
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=get_embed_model()
    )

    if doc_id:
        try:
            logger.info(f"正在从向量库中删除 doc_id '{doc_id}' 的旧节点...")
            index.delete_ref_doc(doc_id, delete_from_docstore=True)
            logger.info(f"已删除 doc_id '{doc_id}' 的旧节点。")
        except Exception as e:
            logger.warning(f"删除 doc_id '{doc_id}' 的旧节点时出错 (可能是首次添加): {e}")

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

    index.insert_nodes(nodes)
    logger.success(f"成功将内容 (doc_id: {doc_id}, {len(nodes)}个节点) 添加到向量库。")
    return True


class LiteLLMReranker(BaseNodePostprocessor):
    top_n: int = 3
    rerank_params: Dict[str, Any] = Field(default_factory=dict)
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("必须提供查询信息 (QueryBundle) 才能进行重排。")
        if not nodes:
            return []

        query_str = query_bundle.query_str
        documents = [node.get_content() for node in nodes]

        rerank_request_params = self.rerank_params.copy()
        rerank_request_params.update({
            "query": query_str,
            "documents": documents,
            "top_n": self.top_n,
        })
        
        logger.debug(f"向 LiteLLM Reranker 发送同步请求: model={rerank_request_params.get('model')}, top_n={self.top_n}, num_docs={len(documents)}")
        
        response = rerank(**rerank_request_params)

        new_nodes_with_scores = []
        for result in response.results:
            original_node = nodes[result.index]
            original_node.score = result.relevance_score
            new_nodes_with_scores.append(original_node)
        
        logger.debug(f"重排后返回 {len(new_nodes_with_scores)} 个节点。")
        return new_nodes_with_scores


    async def _aprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("必须提供查询信息 (QueryBundle) 才能进行重排。")
        if not nodes:
            return []

        query_str = query_bundle.query_str
        documents = [node.get_content() for node in nodes]

        rerank_request_params = self.rerank_params.copy()
        rerank_request_params.update({
            "query": query_str,
            "documents": documents,
            "top_n": self.top_n,
        })
        
        logger.debug(f"向 LiteLLM Reranker 发送异步请求: model={rerank_request_params.get('model')}, top_n={self.top_n}, num_docs={len(documents)}")
        
        response = await arerank(**rerank_request_params)

        new_nodes_with_scores = []
        for result in response.results:
            original_node = nodes[result.index]
            original_node.score = result.relevance_score
            new_nodes_with_scores.append(original_node)
        
        logger.debug(f"重排后返回 {len(new_nodes_with_scores)} 个节点。")
        return new_nodes_with_scores


def get_default_vector_store_info() -> VectorStoreInfo:
    """
    为项目创建一个默认的 VectorStoreInfo, 定义了常见的元数据字段。
    这使得自动检索器 (AutoRetriever) 能够理解元数据结构并生成过滤查询。
    """
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
) -> BaseQueryEngine:
    logger.info("正在创建向量查询引擎...")
    logger.debug(
        f"参数: similarity_top_k={similarity_top_k}, rerank_top_n={rerank_top_n}, "
        f"use_auto_retriever={use_auto_retriever}, filters={filters}"
    )
    
    reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    synthesis_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["synthesis"])
    synthesis_llm = LiteLLM(**synthesis_llm_params)

    postprocessors = []
    if rerank_top_n and rerank_top_n > 0:
        logger.info(f"配置 LiteLLM Reranker 后处理器, top_n={rerank_top_n}")
        rerank_params = get_rerank_params()
        reranker = LiteLLMReranker(top_n=rerank_top_n, rerank_params=rerank_params)
        postprocessors.append(reranker)

    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 4096),
            num_output=synthesis_llm_params.get('max_tokens', 512),
            chunk_overlap_ratio=0.2
        )
    )

    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            logger.info(f"从缓存中获取 VectorStoreIndex (key: {cache_key})。")
            index = _vector_indices[cache_key]
        else:
            logger.info(f"缓存中未找到 VectorStoreIndex, 正在创建并缓存 (key: {cache_key})。")
            index = VectorStoreIndex.from_vector_store(
                vector_store, 
                embed_model=get_embed_model()
            )
            _vector_indices[cache_key] = index

    if use_auto_retriever:
        logger.info("使用 VectorIndexAutoRetriever 模式。")
        # 如果用户没有提供 vector_store_info, 则使用默认的。
        # 这使得自动元数据过滤功能开箱即用。
        final_vector_store_info = vector_store_info or get_default_vector_store_info()
        
        retriever = VectorIndexAutoRetriever(
            index,
            vector_store_info=final_vector_store_info,
            similarity_top_k=similarity_top_k,
            llm=reasoning_llm,
            verbose=True
        )
        
        logger.success("自动检索查询引擎创建成功。")
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
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
        llm_group="reasoning",
        temperature=llm_temperatures["reasoning"]
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)
    return result


###############################################################################


if __name__ == '__main__':
    import asyncio
    import tempfile
    import shutil
    from pathlib import Path
    from utils.log import init_logger

    # 1. 初始化日志和临时目录
    init_logger("vector_test")
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "chroma_db")
    input_dir = os.path.join(test_dir, "input_data")
    os.makedirs(input_dir, exist_ok=True)
    logger.info(f"测试目录已创建: {test_dir}")

    # 2. 准备测试数据
    (Path(input_dir) / "doc1.md").write_text("# 角色：龙傲天\n龙傲天是一名来自异世界的穿越者。", encoding='utf-8')
    (Path(input_dir) / "doc2.txt").write_text("世界树是宇宙的中心，连接着九大王国。", encoding='utf-8')
    logger.info(f"测试文件已写入: {input_dir}")

    # 3. 测试 get_vector_store
    logger.info("--- 测试 get_vector_store ---")
    vector_store = get_vector_store(db_path=db_path, collection_name="test_collection")
    logger.info(f"成功获取 VectorStore: {vector_store}")

    # 4. 测试 vector_add_from_dir
    logger.info("--- 测试 vector_add_from_dir ---")
    vector_add_from_dir(vector_store, input_dir)

    # 5. 测试 vector_add
    logger.info("--- 测试 vector_add ---")
    vector_add(vector_store, "虚空之石是一个神秘物品。", {"category": "item"}, doc_id="item_void_stone")

    # 6. 测试 get_vector_query_engine
    logger.info("--- 测试 get_vector_query_engine ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=2)
    logger.info(f"成功创建查询引擎: {type(query_engine)}")

    async def run_queries():
        # 7. 测试 index_query
        logger.info("--- 测试 index_query ---")
        questions = ["龙傲天是谁？", "世界树有什么用？"]
        results = await index_query(query_engine, questions)
        logger.info(f"index_query 查询结果:\n{results}")

        # 8. 测试 index_query_react
        logger.info("--- 测试 index_query_react ---")
        react_question = "请详细介绍一下龙傲天。"
        react_result = await index_query_react(query_engine, react_question, "你是一个小说设定助手。")
        logger.info(f"index_query_react 查询结果:\n{react_result}")

    try:
        asyncio.run(run_queries())
    finally:
        # 9. 清理
        shutil.rmtree(test_dir)
        logger.info(f"测试目录已删除: {test_dir}")
