import os
import sys
import re
import threading
import asyncio
from datetime import datetime
from pathlib import Path
import json
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
                logger.info("正在创建并缓存 LiteLLMEmbedding 模型...")
                embedding_params = get_embedding_params()
                embed_model_name = embedding_params.pop('model')
                _embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)
                if getattr(Settings, '_embed_model', None) is None:
                    Settings.embed_model = _embed_model
                logger.success("LiteLLMEmbedding 模型创建成功。")
    return _embed_model


_vector_stores: Dict[Tuple[str, str], ChromaVectorStore] = {}
_vector_store_lock = threading.Lock()
def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    with _vector_store_lock:
        cache_key = (db_path, collection_name)
        if cache_key in _vector_stores:
            return _vector_stores[cache_key]
        logger.info(f"创建并缓存 ChromaDB 向量库: path='{db_path}', collection='{collection_name}'")
        os.makedirs(db_path, exist_ok=True)
        db = chromadb.PersistentClient(path=db_path)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        _vector_stores[cache_key] = vector_store
        logger.success("ChromaDB 向量库创建成功。")
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
    logger.debug(
        f"参数: similarity_top_k={similarity_top_k}, rerank_top_n={rerank_top_n}, "
        f"use_auto_retriever={use_auto_retriever}, filters={filters}"
    )

    # 步骤 1: 获取或创建 VectorStoreIndex
    # 这是所有查询模式共享的基础。
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

    # 步骤 2: 配置后处理器 (Reranker)
    # Reranker 对两种查询模式都适用。
    postprocessors = []
    if rerank_top_n and rerank_top_n > 0:
        logger.info(f"配置 LiteLLM Reranker 后处理器, top_n={rerank_top_n}")
        rerank_params = get_rerank_params()
        reranker = LiteLLMReranker(top_n=rerank_top_n, rerank_params=rerank_params)
        postprocessors.append(reranker)

    # 步骤 3: 配置响应合成器
    # 响应合成器也对两种模式都适用，它负责将检索到的节点整合成最终答案。
    synthesis_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["synthesis"])
    synthesis_llm = LiteLLM(**synthesis_llm_params)
    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 4096),
            num_output=synthesis_llm_params.get('max_tokens', 512),
            chunk_overlap_ratio=0.2
        )
    )

    # 步骤 4: 根据模式创建并返回具体的查询引擎
    if use_auto_retriever:
        # 自动检索模式: 使用 LLM 动态生成元数据过滤器。
        logger.info("使用 VectorIndexAutoRetriever 模式创建查询引擎。")
        
        # 此模式需要一个 "reasoning" LLM 来解析自然语言并生成过滤器。
        reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
        reasoning_llm = LiteLLM(**reasoning_llm_params)
        
        final_vector_store_info = vector_store_info or get_default_vector_store_info()
        
        retriever = VectorIndexAutoRetriever(
            index,
            vector_store_info=final_vector_store_info,
            similarity_top_k=similarity_top_k,
            llm=reasoning_llm,
            verbose=True
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )
        logger.success("自动检索查询引擎创建成功。")
        return query_engine
    else:
        # 标准模式: 使用固定的过滤器进行检索。
        logger.info("使用标准 as_query_engine 模式创建查询引擎。")
        query_engine = index.as_query_engine(
            # 在标准模式下, as_query_engine 内部创建的 RetrieverQueryEngine
            # 会使用此 LLM 进行响应合成。我们传入专用的 synthesis_llm。
            llm=synthesis_llm,
            response_synthesizer=response_synthesizer,
            filters=filters,
            similarity_top_k=similarity_top_k,
            node_postprocessors=postprocessors,
        )
        logger.success("标准查询引擎创建成功。")
        return query_engine


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


###############################################################################


if __name__ == '__main__':
    import asyncio
    import tempfile
    import shutil
    from pathlib import Path
    import json
    from utils.log import init_logger
    from llama_index.core.vector_stores import MetadataFilters

    init_logger("vector_test")

    # 1. 初始化临时目录
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "chroma_db")
    input_dir = os.path.join(test_dir, "input_data")
    os.makedirs(input_dir, exist_ok=True)
    logger.info(f"测试目录已创建: {test_dir}")

    async def main():
        # 2. 准备测试数据
        (Path(input_dir) / "doc1.md").write_text("# 角色：龙傲天\n龙傲天是一名来自异世界的穿越者。", encoding='utf-8')
        (Path(input_dir) / "doc2.txt").write_text("世界树是宇宙的中心，连接着九大王国。", encoding='utf-8')
        (Path(input_dir) / "doc3.md").write_text(
            "# 势力成员表\n\n| 姓名 | 门派 | 职位 |\n|---|---|---|\n| 萧炎 | 炎盟 | 盟主 |\n| 林动 | 武境 | 武祖 |\n\n## 功法清单\n- 焚决\n- 大荒芜经",
            encoding='utf-8'
        )
        (Path(input_dir) / "doc4.json").write_text(
            json.dumps({"character": "药尘", "alias": "药老", "occupation": "炼药师", "specialty": "异火"}, ensure_ascii=False),
            encoding='utf-8'
        )
        (Path(input_dir) / "empty.txt").write_text("", encoding='utf-8')
        logger.info(f"测试文件已写入: {input_dir}")

        # 3. 测试 get_vector_store
        logger.info("--- 3. 测试 get_vector_store ---")
        vector_store = get_vector_store(db_path=db_path, collection_name="test_collection")
        logger.info(f"成功获取 VectorStore: {vector_store}")

        # 4. 测试 vector_add_from_dir
        logger.info("--- 4. 测试 vector_add_from_dir ---")
        vector_add_from_dir(vector_store, input_dir, _default_file_metadata)

        # 5. 测试 vector_add (首次添加)
        logger.info("--- 5. 测试 vector_add (各种场景) ---")
        logger.info("--- 5.1. 首次添加 ---")
        vector_add(
            vector_store, 
            "虚空之石是一个神秘物品。", 
            {"type": "item", "source": "manual_add_1"}, 
            doc_id="item_void_stone"
        )

        logger.info("--- 5.2. 更新文档 ---")
        vector_add(
            vector_store, 
            "虚空之石是一个极其稀有的神秘物品，据说蕴含着宇宙初开的力量。", 
            {"type": "item", "source": "manual_add_2"}, 
            doc_id="item_void_stone"
        )

        logger.info("--- 5.3. 添加 JSON 内容 ---")
        json_content = json.dumps({"event": "双帝之战", "protagonist": ["萧炎", "魂天帝"]}, ensure_ascii=False)
        vector_add(
            vector_store,
            content=json_content,
            metadata={"type": "event", "source": "manual_json"},
            content_format="json",
            doc_id="event_doudi"
        )

        logger.info("--- 5.4. 添加空内容 (应跳过) ---")
        added = vector_add(
            vector_store,
            content="  ",
            metadata={"type": "empty"},
            doc_id="empty_content"
        )
        assert not added

        # 6. 测试 get_vector_query_engine (标准模式)
        logger.info("--- 6. 测试 get_vector_query_engine (标准模式) ---")
        query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=2)
        logger.info(f"成功创建标准查询引擎: {type(query_engine)}")
        
        questions1 = ["龙傲天是谁？", "虚空之石有什么用？", "萧炎是什么门派的？", "药老是谁？", "双帝之战的主角是谁？"]
        results1 = await index_query(query_engine, questions1)
        logger.info(f"标准查询结果:\n{results1}")
        assert any("龙傲天" in r for r in results1)
        assert any("虚空之石" in r for r in results1)
        assert any("萧炎" in r and "炎盟" in r for r in results1)
        assert any("药尘" in r for r in results1)
        assert any("萧炎" in r and "魂天帝" in r for r in results1)

        # 7. 测试 get_vector_query_engine (带固定过滤器)
        logger.info("--- 7. 测试 get_vector_query_engine (带固定过滤器) ---")
        filters = MetadataFilters(filters=[MetadataFilters.ExactMatch(key="type", value="item")])
        query_engine_filtered = get_vector_query_engine(vector_store, filters=filters)
        questions2 = ["介绍一下那个石头。"]
        results2 = await index_query(query_engine_filtered, questions2)
        logger.info(f"带过滤器的查询结果:\n{results2}")
        assert len(results2) > 0 and "虚空之石" in results2[0]
        
        questions3 = ["龙傲天是谁？"]  # 这个查询应该被过滤器挡住
        results3 = await index_query(query_engine_filtered, questions3)
        logger.info(f"被过滤器阻挡的查询结果:\n{results3}")
        assert len(results3) == 0

        # 8. 测试无重排器和同步查询
        logger.info("--- 8. 测试无重排器和同步查询 ---")
        query_engine_no_rerank = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=0)
        sync_question = "林动的功法是什么？"
        # 使用 .query() 来测试同步路径
        sync_response = query_engine_no_rerank.query(sync_question)
        logger.info(f"同步查询 (无重排器) 结果:\n{sync_response}")
        assert "大荒芜经" in str(sync_response)

        # 9. 测试 get_vector_query_engine (自动检索模式)
        logger.info("--- 9. 测试 get_vector_query_engine (自动检索模式) ---")
        query_engine_auto = get_vector_query_engine(vector_store, use_auto_retriever=True, similarity_top_k=5, rerank_top_n=2)
        logger.info(f"成功创建自动检索查询引擎: {type(query_engine_auto)}")
        
        # 这个查询应该能被 AutoRetriever 解析为针对 metadata 'type'='item' 的过滤
        auto_question = "请根据类型为 'item' 的文档，介绍一下那个物品。"
        auto_results = await index_query(query_engine_auto, [auto_question])
        logger.info(f"自动检索查询结果:\n{auto_results}")
        assert len(auto_results) > 0 and "虚空之石" in auto_results[0]

        # 10. 测试空查询
        logger.info("--- 10. 测试空查询 ---")
        empty_results = await index_query(query_engine, ["一个不存在的概念xyz"])
        logger.info(f"空查询结果: {empty_results}")
        assert len(empty_results) == 0

    try:
        asyncio.run(main())
        logger.success("所有 vector.py 测试用例通过！")
    finally:
        # 清理
        shutil.rmtree(test_dir)
        logger.info(f"测试目录已删除: {test_dir}")
