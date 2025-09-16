import os
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple

import kuzu
from loguru import logger
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    Response,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer, CompactAndRefine
from llama_index.core.vector_stores import MetadataFilters, VectorStore
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms.litellm import LiteLLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .llm import LLM_TEMPERATURES, get_llm_messages, get_llm_params, llm_completion
from .models import natural_sort_key
from .vector import get_embed_model
from utils.log import init_logger


def get_kuzu_graph_store(db_path: str) -> KuzuGraphStore:
    logger.info(f"正在访问 Kùzu 图数据库: path='{db_path}'")
    os.makedirs(db_path, exist_ok=True)
    db = kuzu.Database(db_path)
    graph_store = KuzuGraphStore(db)
    logger.success(f"Kùzu 图数据库已在路径 '{db_path}' 准备就绪。")
    return graph_store


def store(
    storage_context: StorageContext,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    kg_extraction_prompt: str,
    content_format: Literal["markdown", "text", "json"] = "markdown",
    max_triplets_per_chunk: int = 15,
) -> None:
    """
    将内容存储到知识图谱中。

    Args:
        storage_context (StorageContext): LlamaIndex的存储上下文。
        content (str): 要存储的文本内容。
        metadata (Dict[str, Any]): 与文档关联的元数据。
        doc_id (str): 文档的唯一ID。
        kg_extraction_prompt (str): 用于知识图谱提取的提示模板。
        content_format (Literal["markdown", "text", "json"], optional): 内容格式。 Defaults to "markdown".
        max_triplets_per_chunk (int, optional): 每个块最多提取的三元组数量。 Defaults to 15.
    """
    logger.info(f"开始为文档 '{doc_id}' (格式: {content_format}) 构建知识图谱...")

    doc = Document(id_=doc_id, text=content, metadata=metadata)

    transformations = []
    if content_format == "markdown":
        transformations.append(MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True))
    elif content_format == "text":
        transformations.append(SentenceSplitter(chunk_size=512, chunk_overlap=100, include_metadata=True, include_prev_next_rel=True))
    elif content_format == "json":
        logger.info("内容格式为 'json'，将整个文档作为一个节点处理。")
    else:
        transformations.append(SentenceSplitter(chunk_size=512, chunk_overlap=100, include_metadata=True, include_prev_next_rel=True))

    llm_extract_params = get_llm_params(llm="fast", temperature=LLM_TEMPERATURES["summarization"])
    llm = LiteLLM(**llm_extract_params)

    KnowledgeGraphIndex.from_documents(
        [doc],
        storage_context=storage_context,
        llm=llm,
        embed_model=get_embed_model(),
        kg_extraction_prompt=PromptTemplate(kg_extraction_prompt),
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=True,
        transformations=transformations,
    )
    logger.success(f"文档 '{doc_id}' 的知识图谱构建完成。")



def _format_response_with_sorting(response: Response, sort_by: Literal["time", "narrative", "relevance"]) -> str:
    """
    sort_by (Literal): 排序策略: 'time' (时间倒序), 'narrative' (章节顺序), 'relevance' (相关性)。
    """
    if not response.source_nodes:
        return f"未找到相关来源信息, 但综合回答是: \n{str(response)}"

    if sort_by == "narrative":
        sorted_nodes = sorted(
            list(response.source_nodes),
            key=lambda n: natural_sort_key(n.metadata.get("task_id", "")),
            reverse=False,  # 正序排列
        )
        sort_description = "按小说章节顺序排列 (从前到后)"
    elif sort_by == "time":
        sorted_nodes = sorted(
            list(response.source_nodes),
            key=lambda n: n.metadata.get("created_at", "1970-01-01T00:00:00"),
            reverse=True,  # 倒序排列, 最新的在前
        )
        sort_description = "按时间倒序排列 (最新的在前)"
    else:  # 'relevance' 或其他默认情况
        sort_description = "按相关性排序"
        sorted_nodes = list(response.source_nodes)

    source_details = []
    for node in sorted_nodes:
        timestamp = node.metadata.get("created_at", "未知时间")
        task_id = node.metadata.get("task_id", "未知章节")
        score = node.get_score()
        score_str = f"{score:.4f}" if score is not None else "N/A"
        content = re.sub(r"\s+", " ", node.get_content()).strip()
        source_details.append(
            f"来源信息 (章节: {task_id}, 时间: {timestamp}, 相关性: {score_str}):\n---\n{content}\n---"
        )

    formatted_sources = "\n\n".join(source_details)

    final_output = (
        f"综合回答:\n{str(response)}\n\n详细来源 ({sort_description}):\n{formatted_sources}"
    )
    return final_output


def hybrid_query(
    vector_store: VectorStore,
    graph_store: KuzuGraphStore,
    retrieval_query_text: str,
    synthesis_query_text: str,
    synthesis_system_prompt: str,
    synthesis_user_prompt: str,
    kg_nl2graphquery_prompt: Optional[PromptTemplate] = None,
    vector_filters: Optional[MetadataFilters] = None,
    vector_similarity_top_k: int = 150,
    vector_rerank_top_n: int = 50,
    kg_similarity_top_k: int = 300,
    kg_rerank_top_n: int = 100,
    vector_sort_by: Literal["time", "narrative", "relevance"] = "relevance",
    kg_sort_by: Literal["time", "narrative", "relevance"] = "relevance",
) -> str:
    """
    执行一个完整的混合查询流程：向量检索 -> 知识图谱检索 -> LLM综合。
    此函数封装了从创建查询引擎到最终答案合成的所有步骤。

    Args:
        retrieval_query_text (str): 用于向量和图谱检索的查询文本。
        synthesis_query_text (str): 用于最终LLM综合的、更详细的代理查询文本。
        synthesis_system_prompt (str): 综合阶段的系统提示。
        synthesis_user_prompt (str): 综合阶段的用户提示模板。
        vector_store (VectorStore): 用于向量检索的向量存储。
        graph_store (KuzuGraphStore): 用于知识图谱检索的图存储。
        kg_nl2graphquery_prompt (Optional[PromptTemplate], optional): KG中NL2GraphQuery的提示。 Defaults to None.
        vector_filters (Optional[MetadataFilters]): 应用于向量检索的元数据过滤器。
        vector_similarity_top_k (int): 向量检索的top_k。
        vector_rerank_top_n (int): 向量检索后LLM重排的top_n。
        kg_similarity_top_k (int): KG混合检索中向量部分的top_k。
        kg_rerank_top_n (int): KG检索后LLM重排的top_n。
        vector_sort_by (Literal): 向量结果的排序方式。
        kg_sort_by (Literal): 知识图谱结果的排序方式。

    Returns:
        str: LLM综合后的最终答案。
    """
    logger.info("🚀 开始执行混合查询（向量 + 知识图谱）...")

    embed_model = get_embed_model()

    reasoning_llm_params = get_llm_params(llm="reasoning", temperature=LLM_TEMPERATURES["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    synthesis_llm_params = get_llm_params(llm="reasoning", temperature=LLM_TEMPERATURES["synthesis"])
    synthesis_llm = LiteLLM(**synthesis_llm_params)

    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 4096),
            num_output=synthesis_llm_params.get('max_tokens', 512),
            chunk_overlap_ratio=0.2
        )
    )

    # --- 向量查询 ---
    logger.info("构建向量查询引擎...")
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    vector_query_engine = vector_index.as_query_engine(
        filters=vector_filters,
        llm=reasoning_llm,
        response_synthesizer=response_synthesizer,
        similarity_top_k=vector_similarity_top_k,
        node_postprocessors=[
            LLMRerank(llm=reasoning_llm, top_n=vector_rerank_top_n)
        ]
    )
    logger.info(f"正在执行向量查询: '{retrieval_query_text}'")
    vector_response = vector_query_engine.query(retrieval_query_text)
    formatted_vector_str = _format_response_with_sorting(vector_response, vector_sort_by)
    logger.info(f"向量查询完成, 检索到 {len(vector_response.source_nodes)} 个节点。")

    # --- 知识图谱查询 ---
    logger.info("构建知识图谱查询引擎...")
    kg_storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex.from_documents(
        [],
        storage_context=kg_storage_context,
        llm=reasoning_llm,
        include_embeddings=True,
        embed_model=embed_model
    )
    kg_retriever = kg_index.as_retriever(
        retriever_mode="hybrid",
        similarity_top_k=kg_similarity_top_k,
        with_nl2graphquery=True,
        graph_traversal_depth=2,
        nl2graphquery_prompt=kg_nl2graphquery_prompt,
    )
    kg_query_engine = RetrieverQueryEngine(
        retriever=kg_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            LLMRerank(llm=reasoning_llm, top_n=kg_rerank_top_n)
        ]
    )
    logger.info(f"正在执行知识图谱查询: '{retrieval_query_text}'")
    kg_response = kg_query_engine.query(retrieval_query_text)
    formatted_kg_str = _format_response_with_sorting(kg_response, kg_sort_by)
    logger.info(f"知识图谱查询完成, 检索到 {len(kg_response.source_nodes)} 个节点。")

    logger.info("正在整合向量和知识图谱的查询结果...")
    context_dict_user = {
        "query_text": synthesis_query_text,
        "formatted_vector_str": formatted_vector_str,
        "formatted_kg_str": formatted_kg_str,
    }
    messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)

    final_llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=LLM_TEMPERATURES["synthesis"])

    final_message = llm_completion(final_llm_params)
    result = final_message.content
    logger.success("✅ 混合查询及结果整合完成。")

    return result


if __name__ == "__main__":
    from datetime import datetime
    from utils.vector import get_chroma_vector_store

    init_logger(os.path.splitext(os.path.basename(__file__))[0])

    test_db_path = "./.test_chroma_db_graph"
    test_kuzu_path = "./.test_kuzu_db_graph"
    test_collection_name = "test_collection_graph"
    vector_store = get_chroma_vector_store(db_path=test_db_path, collection_name=test_collection_name)
    graph_store = get_kuzu_graph_store(db_path=test_kuzu_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        graph_store=graph_store
    )

    doc_id = "test_story_001"
    metadata = {
        "author": "测试员", 
        "task_id": "第一章", 
        "created_at": datetime.now().isoformat()
    }
    content = """
    在一个阳光明媚的下午，小明在村庄后面的小溪边玩耍。
    他无意间踢到了一块闪闪发光的石头。这块石头不同寻常，
    它通体呈深蓝色，表面刻有古老的符文，并且散发着微弱的暖意。
    小明好奇地捡起了它，感觉一股奇妙的能量涌入身体。
    这块石头，就是传说中的“苍穹之石”，据说拥有连接天空与大地的力量。
    村里的长老曾说过，只有心灵纯洁的人才能唤醒它。
    """
    # 用于知识图谱三元组提取的提示
    kg_extraction_prompt = """
    从以下文本中提取知识三元组。三元组应为 (主语, 谓语, 宾语) 格式。
    请专注于实体及其之间的关系。
    例如:
    文本: "小明发现了一块蓝色的石头。"
    三元组: (小明, 发现, 蓝色石头)
    ---
    文本:
    {text}
    ---
    提取的三元组:
    """

    # 4. 调用 store 函数将内容存入知识图谱和向量存储
    logger.info("\n--- 步骤1: 开始存储内容 ---")
    store(
        storage_context=storage_context,
        content=content,
        metadata=metadata,
        doc_id=doc_id,
        kg_extraction_prompt=kg_extraction_prompt,
        content_format="text",
    )
    logger.success("--- 内容存储完成 ---")

    # 5. 准备查询
    logger.info("\n--- 步骤2: 开始混合查询 ---")
    retrieval_query_text = "小明和苍穹之石有什么关系？"
    synthesis_query_text = f"请详细总结一下关于'{retrieval_query_text}'的所有信息。"

    synthesis_system_prompt = "你是一个小说分析助手。请根据下面提供的“向量检索信息”和“知识图谱信息”，整合并详细回答用户的问题。请优先使用提供的信息，并以流畅、连贯的语言组织答案。"
    synthesis_user_prompt = """
    [用户信息]
    问题: {query_text}

    [向量检索信息]
    {formatted_vector_str}

    [知识图谱信息]
    {formatted_kg_str}

    [你的任务]
    请综合以上所有信息，给出最终的详细回答。
    """

    final_answer = hybrid_query(
        retrieval_query_text=retrieval_query_text,
        synthesis_query_text=synthesis_query_text,
        synthesis_system_prompt=synthesis_system_prompt,
        synthesis_user_prompt=synthesis_user_prompt,
        vector_store=vector_store,
        graph_store=graph_store,
    )
    logger.success("\n--- 最终综合回答 ---")
    logger.info(f"\n{final_answer}")

    logger.info("\n--- 测试完成 ---")
    logger.info(f"你可以检查以下目录来验证结果: '{test_db_path}', '{test_kuzu_path}'")