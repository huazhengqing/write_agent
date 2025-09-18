import os
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple
import threading
import kuzu
from loguru import logger
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import llm_temperatures, get_llm_params, get_rerank_params
from utils.agent import call_react_agent
from utils.vector import LiteLLMReranker, get_embed_model, get_nodes_from_document
from utils.log import init_logger


kg_extraction_prompt_default = """
# 角色
你是一位高度精确的知识图谱工程师。

# 任务
从提供的文本中，以 (主语, 关系, 宾语) 的形式，提取高质量、信息丰富的知识三元组。

# 核心原则
1.  **实体唯一性**: 主语和宾语必须是明确的命名实体，如人名、地名、组织、物品、概念等。避免使用代词。
2.  **关系明确性**: 关系（谓语）应该是描述实体间具体联系的动词或动词短语。优先使用具体的动词，而不是模糊的“是”、“有”。
3.  **指代消解**: 在提取前，必须解析文本中的代词（如“他”、“她”、“它”、“他们”），并用其指代的具体实体名称替换。
4.  **事实为本**: 仅提取文本中明确陈述或可直接推断的事实。禁止创造信息。
5.  **信息合并**: 将关于同一关系的多条信息合并。例如，如果文本说“A是B的父亲”和“A是C的父亲”，则可以提取 `(A, '是父亲', B)` 和 `(A, '是父亲', C)`。

# 提取步骤
1.  通读文本，识别所有关键的命名实体。
2.  分析实体之间的关系，确定描述该关系的最精确的动词或短语。
3.  构建 `(主语, 关系, 宾语)` 格式的三元组。
4.  检查并确保所有代词都已被替换为具体的实体。

# 优质示例
- **文本**: "龙傲天是青云宗的首席大弟子。他使用的武器是'赤霄剑'，他的宿敌是来自北冥魔殿的叶良辰。"
- **优质三元组**:
  - ("龙傲天", "属于", "青云宗")
  - ("龙傲天", "职位是", "首席大弟子")
  - ("龙傲天", "使用", "赤霄剑")
  - ("龙傲天", "宿敌是", "叶良辰")
  - ("叶良辰", "来自", "北冥魔殿")

# 劣质示例
- **文本**: "龙傲天是青云宗的首席大弟子。"
- **劣质三元组**:
  - ("龙傲天", "是", "首席大弟子")  # '是' 关系太模糊
  - ("他", "使用", "赤霄剑")       # 未解决代词 '他'
  - ("龙傲天", "来自", "青云宗")    # '来自' 是过度推断，原文是'属于'

# 指令
现在，请从以下文本中提取知识三元组。如果文本中没有可提取的信息, 返回空列表。
文本:
---
{text}
---
"""


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
    - `WHERE` 子句必须对查询路径中的 **每一个节点** 都应用属性过滤: `n.status = 'active'` (假设节点变量是 `n`)。
2.  Schema遵从: 仅使用 Schema 中定义的节点标签、关系类型和属性。仔细检查 Schema 中是否存在与时间、地点相关的属性 (如 `date`, `location`) 或节点类型 (如 `Event`, `Location`)。
3.  字符串安全: 在Cypher查询中, 所有字符串值都必须是有效的。如果从用户问题中提取的实体名称包含双引号(`"`), 必须用反斜杠(`\\`)进行转义(例如, `\\"`)以防止语法错误。
4.  单行输出: Cypher 查询必须是单行文本, 无换行。
5.  效率优先: 生成的查询应尽可能高效。
6.  无效处理: 若问题无法基于 Schema 回答, 固定返回字符串 "INVALID_QUERY"。

# 高级查询能力
1.  **上下文查询 (时间/地点)**:
    - 当问题包含时间信息 (如 "2023年", "最近"), 优先寻找带有时间属性 (如 `date`, `timestamp`) 的关系或节点, 并使用 `WHERE` 子句进行过滤。
    - 当问题包含地点信息 (如 "在苍梧山"), 优先通过与地点实体的关系 (如 `LOCATED_IN`, `HAPPENED_AT`) 进行匹配。
2.  **聚合 (Aggregation)**: 当问题涉及计数(如 "有多少个?")或列表(如 "列出所有...")时, 使用 `COUNT()` 或 `COLLECT()`。
3.  **排序与限制 (Sorting & Limiting)**: 当问题涉及排名(如 "最常见的"、"最重要的")或数量限制(如 "前5个")时, 使用 `ORDER BY` 和 `LIMIT`。

# 示例
- **基础查询**:
  - 用户问题: '实体"龙傲天"和"赵日天"是什么关系?'
  - Cypher 查询: MATCH (a:__Entity__ {{name: "龙傲天"}})-[r]-(b:__Entity__ {{name: "赵日天"}}) WHERE a.status = 'active' AND b.status = 'active' RETURN type(r)
- **聚合查询**:
  - 用户问题: '青云宗有多少个弟子?'
  - Cypher 查询: MATCH (p:__Entity__)-[:属于]->(s:__Entity__ {{name: "青云宗"}}) WHERE p.status = 'active' AND s.status = 'active' RETURN count(p)
- **排序与限制查询**:
  - 用户问题: '列出与龙傲天关系最多的前3个实体。'
  - Cypher 查询: MATCH (a:__Entity__ {{name:"龙傲天"}})-[r]-(b:__Entity__) WHERE a.status = 'active' AND b.status = 'active' RETURN b.name, count(r) AS relationship_count ORDER BY relationship_count DESC LIMIT 3
- **上下文查询 (地点)**:
  - 用户问题: '苍梧山上发生了什么事件?'
  - Cypher 查询: MATCH (e:Event)-[:LOCATED_IN]->(l:__Entity__ {{name: "苍梧山"}}) WHERE e.status = 'active' AND l.status = 'active' RETURN e.name
- **上下文查询 (时间)**:
  - 用户问题: '2023年龙傲天参与了哪些事件?'
  - Cypher 查询: MATCH (p:__Entity__ {{name: "龙傲天"}})-[r:PARTICIPATED_IN]->(e:Event) WHERE p.status = 'active' AND e.status = 'active' AND r.date STARTS WITH '2023' RETURN e.name

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


_kg_indices: Dict[Tuple[int, int], KnowledgeGraphIndex] = {}
_kg_index_lock = threading.Lock()


def kg_add(
    kg_store: KuzuGraphStore,
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    kg_extraction_prompt: Optional[str] = None,
    content_format: Literal["markdown", "text", "json"] = "markdown",
    max_triplets_per_chunk: int = 15,
) -> None:
    
    final_kg_extraction_prompt = kg_extraction_prompt or kg_extraction_prompt_default

    with _kg_index_lock: 
        cache_key = (id(kg_store), id(vector_store))
        if cache_key in _kg_indices:
            del _kg_indices[cache_key]

    logger.info(f"开始为 doc_id '{doc_id}' 更新知识图谱, 首先将旧数据标记为非活动状态。")
    try:
        update_query = """
        MATCH (n)
        WHERE n.doc_id = $doc_id
        SET n.status = 'inactive'
        """
        kg_store.query(update_query, params={"doc_id": doc_id})
        logger.info(f"已将 doc_id '{doc_id}' 的旧节点标记为 inactive。")
    except Exception as e:
        logger.warning(f"标记旧节点为 inactive 时出错 (可能是首次添加): {e}")

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

    vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())

    try:
        logger.info(f"正在从知识图谱的向量库中删除 doc_id '{doc_id}' 的旧节点...")
        vector_index.delete_ref_doc(doc_id, delete_from_docstore=True)
        logger.info(f"已删除 doc_id '{doc_id}' 的旧节点。")
    except Exception as e:
        logger.warning(f"从知识图谱的向量库中删除 doc_id '{doc_id}' 的旧节点时出错 (可能是首次添加): {e}")

    vector_index.insert_nodes(nodes)
    logger.info(f"已将 {len(nodes)} 个文本节点存入向量库。")

    logger.info("正在提取知识三元组...")
    temp_storage_context = StorageContext.from_defaults(graph_store=SimpleGraphStore())
    llm_extract_params = get_llm_params(llm_group="fast", temperature=llm_temperatures["summarization"])
    llm = LiteLLM(**llm_extract_params)

    temp_index = KnowledgeGraphIndex(
        nodes=nodes,
        storage_context=temp_storage_context,
        llm=llm,
        embed_model=get_embed_model(),
        kg_extraction_prompt=PromptTemplate(final_kg_extraction_prompt),
        max_triplets_per_chunk=max_triplets_per_chunk, # type: ignore
        include_embeddings=False,  # 不需要在临时索引中创建嵌入
        show_progress=True,
    )

    # 从临时图存储中提取三元组。
    graph_store = temp_index.graph_store
    all_subjects = list(graph_store.get_rel_map(depth=1).keys())
    triplets = []
    for subj in all_subjects:
        rel_objs = graph_store.get(subj)
        for rel, obj in rel_objs:
            triplets.append((subj, rel, obj))

    if not triplets:
        logger.warning(f"未能从内容 (doc_id: {doc_id}) 中提取任何三元组。")
        logger.success(f"成功将内容 (doc_id: {doc_id}) 添加到向量库, 但未提取到知识图谱三元组。")
        return

    logger.info(f"提取了 {len(triplets)} 个三元组。正在写入主知识图谱...")
    for subj, rel, obj in triplets:
        # Cypher 不支持参数化关系类型, 因此我们对其进行清理以防止注入,
        # 同时保留反引号以处理特殊字符。
        safe_rel = rel.replace('`', '')
        
        # 使用参数化查询来防止 Cypher 注入, 并确保所有值都得到正确转义。
        # 这也修复了一个bug, 即错误地将 s.status 应用于对象 o。
        query = f"""
        MERGE (s:__Entity__ {{name: $subj}}) SET s.doc_id = $doc_id, s.status = 'active'
        MERGE (o:__Entity__ {{name: $obj}}) SET o.doc_id = $doc_id, o.status = 'active'
        MERGE (s)-[:`{safe_rel}`]->(o)
        """
        params = {"subj": subj, "obj": obj, "doc_id": doc_id}
        try:
            kg_store.query(query, params=params)
        except Exception as e:
            logger.error(f"写入三元组 ('{subj}', '{rel}', '{obj}') 时出错: {e}")
            continue

    logger.success(f"成功将内容 (doc_id: {doc_id}) 添加到知识图谱和向量库。")


def get_kg_query_engine(
    kg_store: KuzuGraphStore,
    kg_vector_store: VectorStore,
    kg_similarity_top_k: int = 600,
    kg_rerank_top_n: int = 100,
    kg_nl2graphquery_prompt: Optional[str] = kg_gen_cypher_prompt,
) -> BaseQueryEngine:
    logger.debug(f"参数: kg_similarity_top_k={kg_similarity_top_k}, kg_rerank_top_n={kg_rerank_top_n}")

    # 步骤 1: 初始化 LLM
    # reasoning_llm 用于查询解析和Cypher生成, synthesis_llm 用于最终答案的合成。
    reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    synthesis_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["synthesis"])
    synthesis_llm = LiteLLM(**synthesis_llm_params)

    # 步骤 2: 获取或创建 KnowledgeGraphIndex
    # 这是查询引擎的基础, 包含了图谱和向量存储的上下文。
    with _kg_index_lock:
        cache_key = (id(kg_store), id(kg_vector_store))
        if cache_key in _kg_indices:
            logger.info(f"从缓存中获取 KnowledgeGraphIndex (key: {cache_key})。")
            kg_index = _kg_indices[cache_key]
        else:
            logger.info(f"缓存中未找到 KnowledgeGraphIndex, 正在创建并缓存 (key: {cache_key})。")
            kg_storage_context = StorageContext.from_defaults(
                graph_store=kg_store, 
                vector_store=kg_vector_store
            )
            # 从空的 documents 列表创建索引, 因为数据已经存在于存储中
            kg_index = KnowledgeGraphIndex.from_documents(
                [], 
                storage_context=kg_storage_context, 
                llm=reasoning_llm,
                include_embeddings=True, 
                embed_model=get_embed_model()
            )
            _kg_indices[cache_key] = kg_index

    # 步骤 3: 配置后处理器 (Reranker)
    # 用于对检索到的文本节点进行重排, 提高相关性。
    logger.info(f"配置 LiteLLM Reranker 后处理器, top_n={kg_rerank_top_n}")
    rerank_params = get_rerank_params()
    reranker = LiteLLMReranker(top_n=kg_rerank_top_n, rerank_params=rerank_params)
    postprocessors = [reranker]

    # 步骤 4: 配置响应合成器
    # 负责将从图谱和向量库中检索到的信息整合成流畅的答案。
    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 4096),
            num_output=synthesis_llm_params.get('max_tokens', 512),
            chunk_overlap_ratio=0.2
        )
    )

    # 步骤 5: 创建并返回混合查询引擎
    # 组装所有组件, 创建一个能够进行混合检索 (关键词+向量+图谱) 的查询引擎。
    logger.info("正在创建知识图谱混合查询引擎...")
    query_engine = kg_index.as_query_engine(
        llm=reasoning_llm,
        retriever_mode="hybrid", 
        similarity_top_k=kg_similarity_top_k,
        with_nl2graphquery=True, 
        graph_traversal_depth=2,
        nl2graphquery_prompt=PromptTemplate(kg_nl2graphquery_prompt),
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
        synonym_degree=2,
    )
    logger.success("知识图谱混合查询引擎创建成功。")
    return query_engine


###############################################################################


if __name__ == '__main__':
    import asyncio
    import json
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

    async def main():
        # 2. 初始化 Store
        logger.info("--- 2. 初始化 Store ---")
        kg_store = get_kg_store(db_path=kg_db_path)
        vector_store = get_vector_store(db_path=vector_db_path, collection_name="kg_hybrid")
        logger.info(f"成功获取 KuzuGraphStore: {kg_store}")
        logger.info(f"成功获取 ChromaVectorStore for KG: {vector_store}")

        # 3. 测试 kg_add (首次添加)
        logger.info("--- 3. 测试 kg_add (首次添加) ---")
        content_v1 = """
        龙傲天是青云宗的首席大弟子。青云宗位于东海之滨的苍梧山。
        龙傲天有一个宿敌，名叫叶良辰。叶良辰来自北冥魔殿。
        龙傲天使用的武器是'赤霄剑'。
        """
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v1,
            metadata={"source": "test_doc_1", "version": 1},
            doc_id="test_doc_1",
            max_triplets_per_chunk=10
        )
        # 验证: 查询一个节点
        res_v1 = kg_store.query("MATCH (n:__Entity__ {name: '龙傲天'}) RETURN n.status, n.doc_id")
        assert res_v1[0] == ['active', 'test_doc_1']
        logger.info("首次添加验证成功。")

        # 4. 测试 kg_add (更新文档)
        logger.info("--- 4. 测试 kg_add (更新文档) ---")
        content_v2 = """
        龙傲天叛逃了青云宗，加入了合欢派。他的新武器是'玄阴十二剑'。
        """
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v2,
            metadata={"source": "test_doc_1", "version": 2},
            doc_id="test_doc_1",
            max_triplets_per_chunk=10
        )
        # 验证: 旧关系中的实体 '青云宗' 应该被标记为 inactive
        res_v2_old = kg_store.query("MATCH (n:__Entity__ {name: '青云宗'}) RETURN n.status")
        assert res_v2_old[0] == ['inactive']
        # 验证: 新关系中的实体 '合欢派' 应该是 active
        res_v2_new = kg_store.query("MATCH (n:__Entity__ {name: '合欢派'}) RETURN n.status")
        assert res_v2_new[0] == ['active']
        logger.info("更新文档验证成功，旧节点已标记为 inactive。")

        # 5. 测试 kg_add (复杂 Markdown 内容)
        logger.info("--- 5. 测试 kg_add (复杂 Markdown 内容) ---")
        content_v3 = """
        # 势力成员表

        | 姓名 | 门派 | 职位 |
        |---|---|---|
        | 赵日天 | 天机阁 | 阁主 |
        | 龙傲天 | 合欢派 | 荣誉长老 |

        ## 物品清单
        - '天机算盘' (法宝): 赵日天的标志性法宝。

        赵日天与龙傲天在苍梧山之巅有过一次对决。
        """
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v3,
            metadata={"source": "test_doc_3"},
            doc_id="test_doc_3"
        )
        res_v3 = kg_store.query("MATCH (n:__Entity__ {name: '赵日天'})-[r:属于]->(m:__Entity__ {name: '天机阁'}) RETURN count(r)")
        assert res_v3[0][0] > 0
        logger.info("复杂 Markdown 内容添加测试成功。")

        # 6. 测试 kg_add (JSON 内容)
        logger.info("--- 6. 测试 kg_add (JSON 内容) ---")
        content_v4_json = json.dumps({
            "event": "苍梧山之巅对决",
            "participants": [
                {"name": "龙傲天", "role": "挑战者"},
                {"name": "赵日天", "role": "应战者"}
            ],
            "location": "苍梧山之巅",
            "outcome": "龙傲天胜"
        }, ensure_ascii=False)
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v4_json,
            metadata={"source": "test_doc_4"},
            doc_id="test_doc_4",
            content_format="json"
        )
        res_v4 = kg_store.query("MATCH (n:__Entity__ {name: '苍梧山之巅对决'}) RETURN n.status")
        assert res_v4[0] == ['active']
        logger.info("JSON 内容添加测试成功。")

        # 7. 测试 kg_add (无三元组内容)
        logger.info("--- 7. 测试 kg_add (无三元组内容) ---")
        content_no_triplets = "这是一段没有实体关系的普通描述性文字。"
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_no_triplets,
            metadata={"source": "test_doc_2"},
            doc_id="test_doc_2"
        )
        res_no_triplets = kg_store.query("MATCH (n) WHERE n.doc_id = 'test_doc_2' RETURN count(n)")
        assert res_no_triplets[0] == [0]
        logger.info("无三元组内容添加测试成功。")

        # 8. 测试 get_kg_query_engine 和查询
        logger.info("--- 8. 测试 get_kg_query_engine 和查询 ---")
        kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)
        
        logger.info("--- 8.1. 查询更新后的数据 ---")
        question1 = "龙傲天现在属于哪个门派？"
        response1 = await kg_query_engine.aquery(question1)
        logger.info(f"Q: {question1}\nA: {response1}")
        assert "合欢派" in str(response1)
        assert "青云宗" not in str(response1)

        logger.info("--- 8.2. 查询复杂 Markdown 数据 ---")
        question2 = "赵日天和龙傲天在哪里对决过？"
        response2 = await kg_query_engine.aquery(question2)
        logger.info(f"Q: {question2}\nA: {response2}")
        assert "苍梧山" in str(response2)

        logger.info("--- 8.3. 查询 JSON 数据 ---")
        question3 = "苍梧山之巅对决的结果是什么？"
        response3 = await kg_query_engine.aquery(question3)
        logger.info(f"Q: {question3}\nA: {response3}")
        assert "龙傲天胜" in str(response3)

    try:
        asyncio.run(main())
        logger.success("所有 kg.py 测试用例通过！")
    finally:
        # 清理
        shutil.rmtree(test_dir)
        logger.info(f"测试目录已删除: {test_dir}")
