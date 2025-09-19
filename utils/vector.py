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
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo, MetadataInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
from llama_index.postprocessors.siliconflow_rerank import SiliconFlowRerank

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import llm_temperatures, get_llm_params, get_embedding_params


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
    llm_params = get_llm_params(llm_group="fast", temperature=llm_temperatures["summarization"])
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
    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
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
        parsed_nodes = node_parser.get_nodes_from_documents(format_docs, show_progress=True)
        
        # 过滤掉仅包含分隔符或空白等非文本内容的无效节点
        nodes_for_format = [node for node in parsed_nodes if node.text.strip() and re.search(r'\w', node.text)]
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
        similarity_score = query_result.similarities[0]
        if similarity_score > 0.99:
            most_similar_node_content = query_result.nodes[0].get_content()
            if most_similar_node_content == content:
                logger.warning(f"发现与 doc_id '{doc_id}' 内容完全相同 (相似度: {similarity_score:.4f}) 的文档，跳过添加。")
                return False
            else:
                logger.critical(
                    f"检测到向量碰撞 (相似度: {similarity_score:.4f})！"
                    f"不同的内容产生了相同的向量，这通常是嵌入模型存在严重问题的迹象。"
                    f"Doc ID: '{doc_id}', 新内容: '{content[:500]}...', "
                    f"已存在内容: '{most_similar_node_content[:500]}...'"
                )

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
    nodes_to_insert = [node for node in parsed_nodes if node.text.strip() and re.search(r'\w', node.text)]
    
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


async def index_query(query_engine: BaseQueryEngine, questions: List[str]) -> List[str]:
    if not questions:
        return []

    logger.info(f"接收到 {len(questions)} 个索引查询问题。")
    logger.debug(f"问题列表: \n{questions}")

    tasks = []
    for q in questions:
        query_text = f"{q}\n# 请使用中文回复"
        tasks.append(query_engine.aquery(query_text))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_answers = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            logger.warning(f"查询 '{question}' 时出错: {result}")
            final_answers.append("")
            continue

        answer = str(getattr(result, "response", "")).strip()
        if (
            not result
            or not getattr(result, "source_nodes", [])
            or not answer
            or answer == "Empty Response"
        ):
            logger.warning(f"查询 '{question}' 未检索到任何源节点或有效响应，返回空回答。")
            final_answers.append("")
            continue

        final_answers.append(answer)
        logger.debug(f"问题 '{question}' 的回答: {answer}")

    logger.success(f"批量查询完成，共返回 {len(final_answers)} 个回答。")
    return final_answers


###############################################################################


async def _test_embedding_model():
    """专门测试嵌入模型的功能和正确性。"""
    logger.info("--- 3. 测试嵌入模型 (Embedding Model) ---")
    embed_model = Settings.embed_model

    # 1. 测试不同文本是否产生不同向量
    logger.info("--- 3.1. 测试不同文本的向量差异性 ---")
    text1 = "这是一个关于人工智能的句子。"
    text2 = "这是一个关于自然语言处理的句子。"
    
    try:
        embedding1_list = await embed_model.aget_text_embedding(text1)
        embedding2_list = await embed_model.aget_text_embedding(text2)
        embedding1 = np.array(embedding1_list)
        embedding2 = np.array(embedding2_list)

        logger.debug(f"文本1的向量 (前5维): {embedding1[:5]}")
        logger.debug(f"文本2的向量 (前5维): {embedding2[:5]}")

        # 检查向量是否全为零
        assert np.any(embedding1 != 0), "嵌入向量1不应为全零向量，这表明嵌入模型可能未正确工作。"
        assert np.any(embedding2 != 0), "嵌入向量2不应为全零向量，这表明嵌入模型可能未正确工作。"
        logger.info("向量非零检查通过。")

        # 检查向量是否相同
        are_equal = np.array_equal(embedding1, embedding2)
        assert not are_equal, "不同文本不应产生完全相同的嵌入向量。如果相同，说明嵌入模型存在严重问题（向量碰撞）。"
        logger.info("不同文本的向量不同，检查通过。")

        # 检查向量相似度
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        assert norm1 > 0 and norm2 > 0, "向量模长不能为零。"
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        logger.info(f"两个不同但相关句子的余弦相似度: {similarity:.4f}")
        assert 0.5 < similarity < 0.999, "相关句子的相似度应在合理范围内 (大于0.5，小于1)。"
        logger.info("相关句子相似度检查通过。")

    except Exception as e:
        logger.error(f"获取嵌入向量时出错: {e}", exc_info=True)
        assert False, "嵌入模型调用失败，请检查API密钥、网络连接或模型配置。"

    # 2. 测试相同文本是否产生相同向量
    logger.info("--- 3.2. 测试相同文本的向量一致性 ---")
    try:
        embedding1_again_list = await embed_model.aget_text_embedding(text1)
        embedding1_again = np.array(embedding1_again_list)
        np.testing.assert_allclose(embedding1, embedding1_again, rtol=1e-5)
        logger.info("相同文本的向量相同，检查通过。")
    except Exception as e:
        logger.error(f"测试相同文本向量时出错: {e}", exc_info=True)
        assert False, "相同文本向量一致性测试失败。"

    # 3. 测试批量嵌入
    logger.info("--- 3.3. 测试批量嵌入 ---")
    try:
        texts_batch = [text1, text2, "第三个完全不同的句子。"]
        embeddings_batch = await embed_model.aget_text_embedding_batch(texts_batch)
        assert len(embeddings_batch) == 3, f"批量嵌入应返回3个向量，但返回了{len(embeddings_batch)}个。"
        logger.info("批量嵌入返回了正确数量的向量。")
        np.testing.assert_allclose(np.array(embeddings_batch[0]), embedding1, rtol=1e-5)
        logger.info("批量嵌入的第一个结果与单个嵌入结果一致，检查通过。")
    except Exception as e:
        logger.error(f"批量嵌入测试失败: {e}", exc_info=True)
        assert False, "批量嵌入测试失败。"
    logger.success("--- 嵌入模型测试通过 ---")

async def _test_reranker():
    """专门测试重排服务的功能和正确性。"""
    logger.info("--- 测试重排服务 (Reranker) ---")
    
    query = "哪部作品是关于一个男孩发现自己是巫师的故事？"
    documents = [
        "《沙丘》是一部关于星际政治和巨型沙虫的史诗科幻小说。", # low relevance
        "《哈利·波特与魔法石》讲述了一个名叫哈利·波特的年轻男孩，他发现自己是一个巫师，并被霍格沃茨魔法学校录取。", # high relevance
        "《魔戒》讲述了霍比特人佛罗多·巴金斯摧毁至尊魔戒的旅程。", # medium relevance
        "《神经漫游者》是一部赛博朋克小说，探讨了人工智能和虚拟现实。", # low relevance
        "一个男孩在魔法学校学习的故事，他最好的朋友是一个红发男孩和一个聪明的女孩。", # high relevance, but less specific
    ]
    
    reranker = SiliconFlowRerank(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        top_n=3,
    )
    
    nodes = [NodeWithScore(node=Document(text=d), score=1.0) for d in documents]
    query_bundle = QueryBundle(query_str=query)
    
    try:
        reranked_nodes = await reranker.aprocess_nodes(nodes, query_bundle=query_bundle)
        
        assert len(reranked_nodes) <= 3, f"重排后应返回最多 3 个节点, 但返回了 {len(reranked_nodes)} 个。"
        logger.info(f"重排后返回 {len(reranked_nodes)} 个节点，数量正确。")
        
        assert len(reranked_nodes) > 0, "Reranker 返回了空列表，服务可能未正常工作。"

        reranked_texts = [node.get_content() for node in reranked_nodes]
        reranked_scores = [node.score for node in reranked_nodes]
        logger.info(f"重排后的文档顺序及分数: {list(zip(reranked_texts, reranked_scores))}")
        
        assert "哈利·波特" in reranked_texts[0], "最相关的文档没有排在第一位。"
        logger.info("最相关的文档排序正确。")
        
        for i in range(len(reranked_scores) - 1):
            assert reranked_scores[i] >= reranked_scores[i+1], f"重排后分数没有递减: {reranked_scores}"
        logger.info("重排后分数递减，检查通过。")

    except Exception as e:
        logger.error(f"重排服务测试失败: {e}", exc_info=True)
        assert False, "重排服务测试失败，请检查API或配置。"
        
    logger.success("--- 重排服务测试通过 ---")

def _prepare_test_data(input_dir: str):
    """准备所有用于测试的输入文件。"""
    logger.info(f"--- 2. 准备多样化的测试文件 ---")
    # 短文本
    (Path(input_dir) / "doc1.md").write_text("# 角色：龙傲天\n龙傲天是一名来自异世界的穿越者。", encoding='utf-8')
    (Path(input_dir) / "doc2.txt").write_text("世界树是宇宙的中心，连接着九大王国。", encoding='utf-8')
    # 表格和简单列表
    (Path(input_dir) / "doc3.md").write_text(
        "# 势力成员表\n\n| 姓名 | 门派 | 职位 |\n|---|---|---|\n| 萧炎 | 炎盟 | 盟主 |\n| 林动 | 武境 | 武祖 |\n\n## 功法清单\n- 焚决\n- 大荒芜经",
        encoding='utf-8'
    )
    # JSON
    (Path(input_dir) / "doc4.json").write_text(
        json.dumps({"character": "药尘", "alias": "药老", "occupation": "炼药师", "specialty": "异火"}, ensure_ascii=False),
        encoding='utf-8'
    )
    # 空文件
    (Path(input_dir) / "empty.txt").write_text("", encoding='utf-8')
    # 长文本段落
    (Path(input_dir) / "long_text.md").write_text(
        "# 设定：九天世界\n\n九天世界是一个广阔无垠的修炼宇宙，由九重天界层叠构成。每一重天界都拥有独特的法则和能量体系，居住着形态各异的生灵。从最低的第一重天到至高的第九重天，灵气浓度呈指数级增长，修炼环境也愈发严苛。传说中，第九重天之上，是触及永恒的彼岸。世界的中心是“建木”，一棵贯穿九天、连接万界的通天神树，其枝叶延伸至无数个下位面，是宇宙能量流转的枢纽。武道、仙道、魔道、妖道等千百种修炼体系在此并存，共同谱写着一曲波澜壮阔的史诗。无数天骄人杰为了争夺有限的资源、追求更高的境界，展开了永无休止的争斗与探索。",
        encoding='utf-8'
    )
    # 包含Mermaid图
    (Path(input_dir) / "diagram.md").write_text(
        '# 关系图：主角团\n\n```mermaid\ngraph TD\n    A[龙傲天] -->|师徒| B(风清扬)\n    A -->|宿敌| C(叶良辰)\n    A -->|挚友| D(赵日天)\n    C -->|同门| E(魔尊重楼)\n    B -->|曾属于| F(华山剑派)\n```\n\n上图展示了主角龙傲天与主要角色的关系网络。',
        encoding='utf-8'
    )
    # 复杂嵌套列表
    (Path(input_dir) / "complex_list.md").write_text(
        "# 物品清单\n\n- **神兵利器**\n  1. 赤霄剑: 龙傲天的佩剑，削铁如泥。\n  2. 诛仙四剑: 上古遗留的杀伐至宝，分为四柄。\n     - 诛仙剑\n     - 戮仙剑\n     - 陷仙剑\n     - 绝仙剑\n- **灵丹妙药**\n  - 九转还魂丹: 可活死人，肉白骨。\n  - 菩提子: 辅助悟道，提升心境。",
        encoding='utf-8'
    )
    # 复合设计文档，模拟真实场景
    (Path(input_dir) / "composite_design_doc.md").write_text(
        """# 卷一：东海风云 - 章节设计

本卷主要围绕主角龙傲天初入江湖，在东海区域结识盟友、遭遇宿敌，并最终揭开“苍龙七宿”秘密一角的序幕。

> **创作笔记**: 本卷的重点是快节奏的奇遇和人物关系的建立，为后续更宏大的世界观铺垫。

![东海地图](./images/donghai_map.png)

## 章节大纲

### 流程图：龙傲天成长路径
```mermaid
graph LR
    A[初入江湖] --> B{遭遇危机}
    B --> C{获得奇遇}
    C --> D[实力提升]
    D --> A
```

| 章节 | 标题 | 核心事件 | 出场角色 | 关键场景/物品 | 备注 |
|---|---|---|---|---|---|
| 1.1 | 孤舟少年 | 龙傲天乘孤舟抵达临海镇，初遇赵日天。 | - **龙傲天** (主角)<br>- 赵日天 (挚友) | 临海镇码头、海鲜酒楼 | 奠定本卷轻松诙谐的基调。 |
| 1.2 | 不打不相识 | 龙傲天与赵日天因误会大打出手，结为兄弟。 | - 龙傲天<br>- 赵日天 | 镇外乱石岗 | 展示龙傲天的剑法和赵日天的拳法。 |
| 1.3 | 黑风寨之危 | 黑风寨山贼袭扰临海镇，掳走镇长之女。 | - 龙傲天<br>- 赵日天<br>- 黑风寨主 (反派) | 临海镇、黑风寨 | 引入第一个小冲突，主角团首次合作。 |
| 1.4 | 夜探黑风寨 | 龙傲天与赵日天潜入黑风寨，发现其与北冥魔殿有关。 | - 龙傲天<br>- 赵日天 | 黑风寨地牢 | 获得关键物品：**北冥令牌**。 |
| 1.5 | 决战黑风寨 | 主角团与黑风寨决战，救出人质，叶良辰首次登场。 | - 龙傲天<br>- 赵日天<br>- **叶良辰** (宿敌) | 黑风寨聚义厅 | 叶良辰以压倒性实力击败黑风寨主，带走令牌，与龙傲天结下梁子。 |

## 核心设定：苍龙七宿

“苍龙七宿”是流传于东海之上的古老传说，与七件上古神器及星辰之力有关。

- **设定细节**:
  - **东方七宿**: 角、亢、氐、房、心、尾、箕。
  - **对应神器**: 每宿对应一件神器，如“角宿”对应“苍龙角”。
  - **力量体系**:
    ```json
    {
      "system_name": "星宿之力",
      "activation": "集齐七件神器，于特定时辰在特定地点（东海之眼）举行仪式。",
      "effect": "可号令四海，引动星辰之力，拥有毁天灭地的威能。"
    }
    ```
- **剧情关联**: 北冥魔殿和主角团都在寻找这七件神器。

### 关键情节线索
- **北冥令牌**: 叶良辰从黑风寨夺走的令牌，是寻找北冥魔殿分舵的关键。
- **龙傲天的身世**: 主角的身世之谜，可能与某个隐世家族有关。
- **赵日天的背景**: 挚友赵日天看似憨厚，但其拳法路数不凡，背后或有故事。
""",
        encoding='utf-8'
    )
    # 包含特殊字符和不同语言代码块的文档
    (Path(input_dir) / "special_chars_and_code.md").write_text(
        """# 特殊内容测试

这是一段包含各种特殊字符的文本： `!@#$%^&*()_+-=[]{};':"\\|,.<>/?~`

## Python 代码示例

下面是一个 Python 函数，用于计算斐波那契数列。

```python
def fibonacci(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
```""",
        encoding='utf-8'
    )
    logger.info(f"测试文件已写入目录: {input_dir}")


async def _test_data_ingestion(vector_store: VectorStore, input_dir: str, test_dir: str):
    """测试从目录和单个内容添加向量，包括各种边缘情况。"""
    # 4. 测试从目录添加入库
    logger.info("--- 4. 测试 vector_add_from_dir (常规) ---")
    vector_add_from_dir(vector_store, input_dir, _default_file_metadata)

    # 5. 测试 vector_add (各种场景)
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
    added_empty = vector_add(
        vector_store,
        content="  ",
        metadata={"type": "empty"},
        doc_id="empty_content"
    )
    assert not added_empty

    logger.info("--- 5.5. 添加包含错误信息的内容 (应跳过) ---")
    added_error = vector_add(
        vector_store,
        content="这是一个包含错误信息的报告: 生成报告时出错。",
        metadata={"type": "error"},
        doc_id="error_content"
    )
    assert not added_error
    logger.info("包含错误信息的内容未被添加，验证通过。")

    logger.info("--- 5.6. 添加无法解析出节点的内容 (应跳过) ---")
    added_no_nodes = vector_add(
        vector_store,
        content="---\n\n---\n",  # 仅包含 Markdown 分割线
        metadata={"type": "no_nodes"},
        doc_id="no_nodes_content"
    )
    assert not added_no_nodes
    logger.info("无法解析出节点的内容未被添加，验证通过。")

    # 6. 测试从无效目录添加
    logger.info("--- 6. 测试 vector_add_from_dir (空目录或仅含无效文件) ---")
    empty_input_dir = os.path.join(test_dir, "empty_input_data")
    os.makedirs(empty_input_dir, exist_ok=True)
    (Path(empty_input_dir) / "unsupported.log").write_text("some log data", encoding='utf-8')
    (Path(empty_input_dir) / "another_empty.txt").write_text("   ", encoding='utf-8')
    added_from_empty = vector_add_from_dir(vector_store, empty_input_dir)
    assert not added_from_empty
    logger.info("从仅包含无效文件的目录添加，返回False，验证通过。")


async def _test_node_deletion(vector_store: VectorStore):
    """测试节点的显式删除功能。"""
    logger.info("--- 7. 测试显式删除 ---")
    doc_id_to_delete = "to_be_deleted"
    content_to_delete = "这是一个唯一的、即将被删除的节点XYZ123。"
    vector_add(
        vector_store,
        content_to_delete,
        {"type": "disposable", "source": "delete_test"},
        doc_id=doc_id_to_delete
    )
    await asyncio.sleep(2)

    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            del _vector_indices[cache_key]
    
    filters = MetadataFilters(filters=[ExactMatchFilter(key="ref_doc_id", value=doc_id_to_delete)])
    query_engine_for_check = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=1, rerank_top_n=0)
    response_before = await query_engine_for_check.aquery("any")
    retrieved_nodes_before = response_before.source_nodes

    assert retrieved_nodes_before and content_to_delete in retrieved_nodes_before[0].get_content()
    logger.info("删除前节点存在，验证通过。")

    vector_store.delete(ref_doc_id=doc_id_to_delete)
    logger.info("已调用删除方法。")

    with _vector_index_lock:
        cache_key = id(vector_store)
        if cache_key in _vector_indices:
            del _vector_indices[cache_key]
    logger.info("已使向量索引缓存失效以反映删除操作。")

    query_engine_after_delete = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=1, rerank_top_n=0)
    response_after = await query_engine_after_delete.aquery("any")
    retrieved_nodes_after = response_after.source_nodes
    assert not retrieved_nodes_after
    logger.success("--- 节点删除测试通过 ---")


async def _test_node_update(vector_store: VectorStore):
    """测试节点的更新操作（通过覆盖doc_id）。"""
    logger.info("--- 8. 测试更新操作 ---")
    doc_id_to_update = "to_be_updated"
    content_v1 = "这是文档的初始版本 V1，用于测试更新功能。"
    content_v2 = "这是文档更新后的版本 V2，旧内容应被覆盖。"

    vector_add(
        vector_store,
        content_v1,
        {"type": "update_test", "version": 1},
        doc_id=doc_id_to_update
    )
    await asyncio.sleep(2)

    filters_update = MetadataFilters(filters=[ExactMatchFilter(key="ref_doc_id", value=doc_id_to_update)])
    query_engine_v1 = get_vector_query_engine(vector_store, filters=filters_update, similarity_top_k=1)
    response_v1 = await query_engine_v1.aquery("any")
    retrieved_v1 = response_v1.source_nodes
    assert retrieved_v1 and "V1" in retrieved_v1[0].get_content()
    logger.info("更新前，版本 V1 存在，验证通过。")

    vector_add(
        vector_store,
        content_v2,
        {"type": "update_test", "version": 2},
        doc_id=doc_id_to_update
    )
    await asyncio.sleep(2)

    query_engine_v2 = get_vector_query_engine(vector_store, filters=filters_update, similarity_top_k=1)
    response_v2 = await query_engine_v2.aquery("any")
    retrieved_v2 = response_v2.source_nodes
    assert retrieved_v2 and "V2" in retrieved_v2[0].get_content() and "V1" not in retrieved_v2[0].get_content()
    logger.success("--- 节点更新测试通过 ---")


async def _test_standard_query(vector_store: VectorStore):
    """测试标准查询模式。"""
    logger.info("--- 9. 测试 get_vector_query_engine (标准模式) ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=2)
    logger.info(f"成功创建标准查询引擎: {type(query_engine)}")

    questions = [
        "龙傲天是谁？",
        "虚空之石有什么用？",
        "萧炎是什么门派的？",
        "药老是谁？",
        "双帝之战的主角是谁？",
        "九天世界的中心是什么？",
        "龙傲天和叶良辰是什么关系？",
        "诛仙四剑包括哪些？",
        "黑风寨发生了什么事？",
        "苍龙七宿是什么？",
        "龙傲天的成长路径是怎样的？",
        "北冥令牌有什么用？",
        "如何用python计算斐波那契数列？"
    ]
    results = await index_query(query_engine, questions)
    logger.info(f"标准查询结果:\n{results}")
    assert any("龙傲天" in r for r in results)
    assert any("虚空之石" in r for r in results)
    assert any("萧炎" in r and "炎盟" in r for r in results)
    assert any("药尘" in r for r in results)
    assert any("萧炎" in r and "魂天帝" in r for r in results)
    assert any("建木" in r for r in results)
    assert any("宿敌" in r for r in results)
    assert any("戮仙剑" in r and "绝仙剑" in r for r in results)
    assert any("黑风寨" in r and "北冥魔殿" in r for r in results)
    assert any("苍龙七宿" in r and "星宿之力" in r for r in results)
    assert any("初入江湖" in r and "实力提升" in r for r in results)
    assert any("北冥魔殿分舵" in r for r in results)
    assert any("fibonacci" in r and "def" in r for r in results)
    assert not any("错误信息" in r for r in results)
    assert not any("即将被删除" in r for r in results)
    logger.success("--- 标准查询测试通过 ---")


async def _test_filtered_query(vector_store: VectorStore):
    """测试带固定元数据过滤器的查询。"""
    logger.info("--- 10. 测试 get_vector_query_engine (带固定过滤器) ---")
    filters = MetadataFilters(filters=[ExactMatchFilter(key="type", value="item")])
    query_engine_filtered = get_vector_query_engine(vector_store, filters=filters)
    
    results_hit = await index_query(query_engine_filtered, ["介绍一下那个石头。"])
    logger.info(f"带过滤器的查询结果 (应命中):\n{results_hit}")
    assert len(results_hit) > 0 and "虚空之石" in results_hit[0]

    results_miss = await index_query(query_engine_filtered, ["龙傲天是谁？"])
    logger.info(f"被过滤器阻挡的查询结果 (应未命中):\n{results_miss}")
    assert not results_miss[0]
    logger.success("--- 带固定过滤器的查询测试通过 ---")


async def _test_no_reranker_sync_query(vector_store: VectorStore):
    """测试无重排器和同步查询模式。"""
    logger.info("--- 11. 测试无重排器和同步查询 ---")
    query_engine_no_rerank = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=0)
    sync_question = "林动的功法是什么？"
    sync_response = query_engine_no_rerank.query(sync_question)
    logger.info(f"同步查询 (无重排器) 结果:\n{sync_response}")
    assert "大荒芜经" in str(sync_response)
    logger.success("--- 无重排器和同步查询测试通过 ---")


async def _test_auto_retriever_query(vector_store: VectorStore):
    """测试自动检索（AutoRetriever）模式。"""
    logger.info("--- 12. 测试 get_vector_query_engine (自动检索模式) ---")
    query_engine_auto = get_vector_query_engine(vector_store, use_auto_retriever=True, similarity_top_k=5, rerank_top_n=2)
    logger.info(f"成功创建自动检索查询引擎: {type(query_engine_auto)}")

    auto_question = "请根据类型为 'item' 的文档，介绍一下那个物品。"
    auto_results = await index_query(query_engine_auto, [auto_question])
    logger.info(f"自动检索查询结果:\n{auto_results}")
    assert len(auto_results) > 0 and "虚空之石" in auto_results[0]
    logger.success("--- 自动检索查询测试通过 ---")


async def _test_empty_query(vector_store: VectorStore):
    """测试对无结果查询的处理。"""
    logger.info("--- 13. 测试空查询 ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=2)
    empty_results = await index_query(query_engine, ["一个不存在的概念xyz"])
    logger.info(f"空查询结果: {empty_results}")
    assert not empty_results[0]
    logger.success("--- 空查询测试通过 ---")


if __name__ == '__main__':
    import asyncio
    import shutil
    from pathlib import Path
    import json
    from utils.log import init_logger
    from utils.file import project_root
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    import nest_asyncio

    init_logger("vector_test")

    nest_asyncio.apply()

    import logging
    logging.getLogger("litellm").setLevel(logging.WARNING)

    test_dir = project_root / ".test" / "vector_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    db_path = os.path.join(test_dir, "chroma_db")
    input_dir = os.path.join(test_dir, "input_data")
    os.makedirs(input_dir, exist_ok=True)

    async def main():
        collection_name = "test_collection"
        _prepare_test_data(input_dir)
        
        await _test_embedding_model()
        await _test_reranker()

        vector_store = get_vector_store(db_path=db_path, collection_name=collection_name)

        await _test_data_ingestion(vector_store, input_dir, str(test_dir))

        # 原 _test_query_and_delete 已拆分为以下独立测试
        await _test_node_deletion(vector_store)
        await _test_node_update(vector_store)
        await _test_standard_query(vector_store)
        await _test_filtered_query(vector_store)
        await _test_no_reranker_sync_query(vector_store)
        await _test_auto_retriever_query(vector_store)
        await _test_empty_query(vector_store)

    try:
        asyncio.run(main())
        logger.success("所有 vector.py 测试用例通过！")
    finally:
        logger.info(f"测试完成。测试数据保留在: {test_dir}")
