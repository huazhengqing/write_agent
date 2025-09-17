import os
import sys
from typing import Any, Dict, List, Literal, Optional
import threading
import kuzu
from loguru import logger
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores.types import VectorStore
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms_api.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import llm_temperatures, get_llm_params, call_react_agent
from utils.vector import get_embed_model, get_nodes_from_document
from utils.log import init_logger


kg_gen_cypher_prompt = """
# 角色
你是一位精通 Cypher 的图数据库查询专家。

# 任务
根据用户提供的自然语言问题和图谱 Schema, 生成一条精确、高效、且符合所有规则的 Cypher 查询语句。

# 上下文
- 用户问题: '{query_str}'
- 图谱 Schema:
---
{schema}
---

# 核心规则 (必须严格遵守)
1.  强制过滤 (最重要!):
    - 查询必须包含 `WHERE` 子句。
    - `WHERE` 子句必须对查询路径中的 每一个节点 都应用以下所有属性过滤条件。
    - 假设一个节点变量是 `n`, 那么过滤条件必须是: `n.status = 'active'`
2.  Schema遵从: 仅使用 Schema 中定义的节点标签和关系类型。
3.  字符串安全: 在Cypher查询中, 所有字符串值都必须是有效的。如果从用户问题中提取的实体名称包含双引号(`"`), 必须用反斜杠(`\`)进行转义(例如, `\"`)以防止语法错误。
4.  单行输出: Cypher 查询必须是单行文本, 无换行。
5.  效率优先: 生成的查询应尽可能高效。
6.  无效处理: 若问题无法基于 Schema 回答, 固定返回字符串 "INVALID_QUERY"。

# 示例
- 用户问题: '角色"龙傲天"和"赵日天"是什么关系?'
- Cypher 查询: MATCH (a:角色 {{name: "龙傲天"}})-[r]-(b:角色 {{name: "赵日天"}}) WHERE a.status = 'active' AND b.status = 'active' RETURN type(r)

# 指令
现在, 请为上述用户问题生成 Cypher 查询语句。
"""


_kg_stores: Dict[str, KuzuGraphStore] = {}
_kg_store_lock = threading.Lock()
def get_kg_store(db_path: str) -> KuzuGraphStore:
    with _kg_store_lock:
        if db_path in _kg_stores:
            return _kg_stores[db_path]
        logger.info(f"创建并缓存 KuzuGraphStore: path='{db_path}'")
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        db = kuzu.Database(db_path)
        kg_store = KuzuGraphStore(db)
        _kg_stores[db_path] = kg_store
        return kg_store


def kg_add(
    kg_store: KuzuGraphStore,
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    kg_extraction_prompt: str,
    content_format: Literal["markdown", "text", "json"] = "markdown",
    max_triplets_per_chunk: int = 15,
) -> None:
    
    doc = Document(id_=doc_id, text=content, metadata=metadata)
    
    nodes = []
    if content_format == "json":
        nodes = [doc]
    else:
        nodes = get_nodes_from_document(doc)
    if not nodes:
        logger.warning(f"内容 (doc_id: {doc_id}) 未解析出任何节点，跳过添加。")
        return

    logger.info(f"内容被解析成 {len(nodes)} 个节点, max_triplets_per_chunk={max_triplets_per_chunk}")

    storage_context = StorageContext.from_defaults(
        kg_store=kg_store, 
        vector_store=vector_store
    )

    llm_extract_params = get_llm_params(llm="fast", temperature=llm_temperatures["summarization"])
    llm = LiteLLM(**llm_extract_params)

    KnowledgeGraphIndex.from_documents(
        documents=nodes,
        storage_context=storage_context,
        llm=llm,
        embed_model=get_embed_model(),
        kg_extraction_prompt=PromptTemplate(kg_extraction_prompt),
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=True,
        show_progress=True,
    )
    logger.success(f"成功将内容 (doc_id: {doc_id}) 添加到知识图谱。")


def get_kg_query_engine(
    kg_store: KuzuGraphStore,
    kg_vector_store: VectorStore,
    kg_similarity_top_k: int = 300,
    kg_rerank_top_n: int = 100,
    kg_nl2graphquery_prompt: Optional[str] = kg_gen_cypher_prompt,
) -> BaseQueryEngine:
    
    reasoning_llm_params = get_llm_params(llm="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    synthesis_llm_params = get_llm_params(llm="reasoning", temperature=llm_temperatures["synthesis"])
    synthesis_llm = LiteLLM(**synthesis_llm_params)

    rerank_llm_params = get_llm_params(llm="fast", temperature=0.0)
    rerank_llm = LiteLLM(**rerank_llm_params)

    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 4096),
            num_output=synthesis_llm_params.get('max_tokens', 512),
            chunk_overlap_ratio=0.2
        )
    )

    kg_storage_context = StorageContext.from_defaults(
        graph_store=kg_store, 
        vector_store=kg_vector_store
    )

    kg_index = KnowledgeGraphIndex.from_documents(
        [], 
        storage_context=kg_storage_context, 
        llm=reasoning_llm,
        include_embeddings=True, 
        embed_model=get_embed_model()
    )

    kg_retriever = kg_index.as_retriever(
        retriever_mode="hybrid", 
        similarity_top_k=kg_similarity_top_k,
        with_nl2graphquery=True, 
        graph_traversal_depth=2,
        nl2graphquery_prompt=PromptTemplate(kg_nl2graphquery_prompt) if kg_nl2graphquery_prompt else None,
    )

    return RetrieverQueryEngine(
        retriever=kg_retriever, 
        response_synthesizer=response_synthesizer,
        node_postprocessors=[LLMRerank(llm=rerank_llm, top_n=kg_rerank_top_n, choice_batch_size=10)],
        use_async=True
    )


async def kg_query_react(
    kg_query_engine: BaseQueryEngine,
    query_str: str,
    agent_system_prompt: Optional[str] = None,
) -> str:
    kg_tool = QueryEngineTool.from_defaults(
        query_engine=kg_query_engine,
        name="knowledge_graph_search",
        description="用于探索实体及其关系 (例如: 角色A和角色B是什么关系? 事件C导致了什么后果?)。当问题比较复杂时, 你可以多次调用此工具来回答问题的不同部分, 然后综合答案。"
    )
    result = await call_react_agent(
        system_prompt=agent_system_prompt,
        user_prompt=query_str,
        tools=[kg_tool],
        llm_type="reasoning",
        temperature=llm_temperatures["reasoning"]
    )
    if not isinstance(result, str):
        logger.warning(f"Agent 返回了非字符串类型, 将其强制转换为字符串: {type(result)}")
        result = str(result)
    return result


if __name__ == '__main__':
    import asyncio
    import tempfile
    import shutil
    from pathlib import Path
    from utils.log import init_logger
    from utils.vector import get_vector_store

    init_logger("kg_test")

    # 1. 初始化临时目录
    test_dir = tempfile.mkdtemp()
    kg_db_path = os.path.join(test_dir, "kuzu_db")
    vector_db_path = os.path.join(test_dir, "chroma_for_kg")
    logger.info(f"测试目录已创建: {test_dir}")

    # 2. 准备测试数据和配置
    kg_extraction_prompt_test = """
    从以下文本中提取知识三元组 (主语, 谓语, 宾语)。
    - 主语和宾语应该是实体。
    - 谓语应该是它们之间的关系。
    - 仅提取与角色、地点、事件、物品及其关系相关的信息。
    - 忽略不重要的信息。
    - 如果文本中没有可提取的信息, 返回空列表。
    文本:
    ---
    {text}
    ---
    """
    content_to_add = """
    龙傲天是青云宗的首席大弟子。青云宗位于东海之滨的苍梧山。
    龙傲天有一个宿敌，名叫叶良辰。叶良辰来自北冥魔殿。
    龙傲天使用的武器是'赤霄剑'。
    """
    metadata = {"source": "test_doc_1"}
    doc_id = "test_doc_1"

    async def main():
        # 3. 测试 get_kg_store 和 get_vector_store
        logger.info("--- 测试 get_kg_store 和 get_vector_store ---")
        kg_store = get_kg_store(db_path=kg_db_path)
        vector_store = get_vector_store(db_path=vector_db_path, collection_name="kg_hybrid")
        logger.info(f"成功获取 KuzuGraphStore: {kg_store}")
        logger.info(f"成功获取 ChromaVectorStore for KG: {vector_store}")

        # 4. 测试 kg_add
        logger.info("--- 测试 kg_add ---")
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_to_add,
            metadata=metadata,
            doc_id=doc_id,
            kg_extraction_prompt=kg_extraction_prompt_test,
            max_triplets_per_chunk=10
        )
        logger.info("kg_add 调用完成")

        # 5. 测试 get_kg_query_engine 和 kg_query_react
        logger.info("--- 测试 get_kg_query_engine 和 kg_query_react ---")
        kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)
        question = "龙傲天的宿敌是谁？他来自哪里？"
        answer = await kg_query_react(kg_query_engine=kg_query_engine, query_str=question)
        logger.info(f"对于问题 '{question}', kg_query_react 的回答是:\n{answer}")

    try:
        asyncio.run(main())
    finally:
        # 6. 清理
        shutil.rmtree(test_dir)
        logger.info(f"测试目录已删除: {test_dir}")
