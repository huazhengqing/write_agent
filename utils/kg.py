import os
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
import hashlib
import time
from collections import defaultdict
import kuzu
from loguru import logger

from llama_index.core import Document, Settings, KnowledgeGraphIndex, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms.litellm import LiteLLM
from llama_index.core.node_parser import SentenceSplitter, MarkdownElementNodeParser

from utils.config import llm_temperatures, get_llm_params
from utils.vector import synthesizer


kg_extraction_prompt = """
# 角色
你是一位专攻知识图谱构建的专家级信息提取AI。

# 任务
从给定的文本中，以 `(主语, 关系, 宾语)` 的形式，提取所有高质量、信息丰富、事实准确的知识三元组。

# 核心原则 (必须严格遵守)
1.  实体规范化:
    - 实体定义: 主语和宾语必须是明确的实体。这包括：命名实体（人名、地名、组织）、具体事物（物品、功法）、技术术语（函数名、API、协议）、以及关键抽象概念（市场规模、增长率）。
    - 唯一性与消歧: 尽可能将同一实体的不同称谓（如“龙傲天”、“他”、“主角”）归一到最完整的实体名称上。
    - 完整性: 实体名称应保持完整，例如“青云宗”而非“青云”。

2.  关系精确性:
    - 具体动词: 关系（谓语）应使用描述实体间具体联系的动词或动词短语。优先使用具体的动词（如“击败”、“创立”、“位于”），避免使用模糊的“是”、“有”、“关于”。
    - 属性即关系: 将实体的属性也视为一种关系。例如，“龙傲天是炼气期”应提取为 `("龙傲天", "修为是", "炼气期")`。

3.  事实为本:
    - 忠于原文: 仅提取文本中明确陈述的事实。禁止进行主观推断或引入外部知识。
    - 忽略常识: 不要提取普遍的、非特定于文本的常识性信息（例如，`("天空", "是", "蓝色")`）。

4.  信息密度:
    - 避免冗余: 如果多个句子描述同一事实，只提取一次。
    - 合并信息: 将关于同一主语和关系的多条信息合并。

# 提取流程
1.  识别实体: 通读文本，识别出所有符合实体定义的关键实体。
2.  解析关系: 分析实体之间的关系，包括动作、属性、从属等，并确定最精确的动词短语作为关系。
3.  指代消解: 在构建三元组前， mentally 解析所有代词（如“他”、“她”、“它”、“他们”），并用其指代的具体实体名称替换。
4.  构建三元组: 按照 `(主语, 关系, 宾语)` 的格式构建三元组列表。

# 示例分析

## 示例1: 小说/叙事文本
- 文本: "黄昏时分，在临海镇的'听潮轩'酒楼，龙傲天展开了一张指向黑松林的羊皮卷地图。这张地图是他在青云宗的师父风清扬所赠。"
- 最终三元组:
  - ("龙傲天", "位于", "听潮轩")
  - ("听潮轩", "位于", "临海镇")
  - ("龙傲天", "展开", "羊皮卷地图")
  - ("羊皮卷地图", "指向", "黑松林")
  - ("风清扬", "赠送给", "龙傲天")
  - ("风清扬", "是师父", "龙傲天")
  - ("龙傲天", "属于", "青云宗")
  - ("风清扬", "属于", "青云宗")

## 示例2: 报告/分析文本
- 文本: "根据艾瑞咨询2023年的报告，中国AIGC市场的规模达到了200亿元人民币，并预测将在2025年增长至700亿元。"
- 最终三元组:
  - ("中国AIGC市场", "规模是", "200人民币")
  - ("200人民币", "统计年份", "2023年")
  - ("中国AIGC市场", "预测规模", "700亿元")
  - ("700亿元", "预测年份", "2025年")
  - ("中国AIGC市场", "数据来源", "艾瑞咨询报告")

## 示例3: 技术手册/工具书
- 文本: "在React中，`useState` Hook是一个函数，它允许你在函数组件中添加和管理状态(state)。它返回一个状态值和一个更新该值的函数。"
- 最终三元组:
  - ("useState Hook", "属于", "React")
  - ("useState Hook", "类型是", "函数")
  - ("useState Hook", "允许", "在函数组件中管理状态")
  - ("useState Hook", "返回", "状态值")
  - ("useState Hook", "返回", "更新函数")

## 劣质示例
- 文本: "他很强大。"
- 劣质三元组:
  - ("他", "是", "强大")  # 错误: 使用了代词"他"；"强大"是形容词而非实体；"是"关系模糊。
- 文本: "龙傲天去了青云宗。"
- 劣质三元组:
  - ("龙傲天", "去", "地方") # 错误: "地方"不是命名实体，信息密度太低。应为 `("龙傲天", "前往", "青云宗")`。

# 输出要求
- 格式: 必须返回一个 Python 的元组列表 `List[Tuple[str, str, str]]`。
- 空结果: 如果文本中没有可提取的有效信息, 必须返回一个空列表 `[]`。
- 无额外内容: 除了三元组列表，不要包含任何解释、注释或代码块标记。

# 指令
现在，请严格遵循以上所有规则，从以下文本中提取知识三元组。
文本:
---
{text}
---
"""


kg_gen_cypher_prompt = """
# 角色
你是一位顶级的图数据库工程师，精通 Cypher 查询语言。

# 任务
根据`用户问题`和`图谱 Schema`，生成一条精确、高效、且符合所有规则的 Cypher 查询语句。

# 上下文
- 用户问题: '{query_str}'
- 图谱 Schema:
---
{schema}
---

# 核心规则
1.  强制状态过滤 (最重要!): 查询路径中的 每一个节点 都必须在 `WHERE` 子句中包含 `status = 'active'` 的过滤条件。例如 `WHERE n1.status = 'active' AND n2.status = 'active'`。
2.  严格遵循 Schema: 只能使用`图谱 Schema`中明确定义的节点标签、关系类型和属性。禁止猜测或使用不存在的元素。
3.  优先使用属性: 在 `MATCH` 子句中，优先使用属性（如 `{name: "实体名"}`）进行匹配，而不是仅仅依赖标签。
4.  关系方向: 对于不确定的关系查询（如“A和B有什么关系”），使用无方向匹配 `-[r]-`。对于明确的动作（如“A击败了B”），使用有方向匹配 `->`。
5.  字符串安全: 如果从用户问题中提取的实体名称包含双引号(`"`)，必须用反斜杠(`\\`)进行转义。

# 输出要求
1.  单行输出: 最终的 Cypher 查询必须是单行文本，不含任何换行符。
2.  无额外内容: 仅输出 Cypher 查询语句本身或 "INVALID_QUERY"。禁止添加任何解释、注释或代码块标记。
3.  无效查询: 如果问题无法基于给定的`图谱 Schema`回答，或者问题含糊不清，固定返回字符串 "INVALID_QUERY"。

# 查询策略与示例

## 1. 基础查询 (1-hop)
- 用户问题: '实体"龙傲天"和"赵日天"是什么关系?'
- Cypher 查询: `MATCH (a:__Entity__ {name: "龙傲天"})-[r]-(b:__Entity__ {name: "赵日天"}) WHERE a.status = 'active' AND b.status = 'active' RETURN type(r)`

## 2. 多跳查询 (Multi-hop)
- 用户问题: '龙傲天的宿敌的门派是什么？'
- Cypher 查询: `MATCH (a:__Entity__ {name: "龙傲天"})-[:宿敌是]-(enemy:__Entity__)-[:属于]->(faction:__Entity__) WHERE a.status = 'active' AND enemy.status = 'active' AND faction.status = 'active' RETURN faction.name`

## 3. 聚合查询 (Aggregation)
- 用户问题: '青云宗有多少个弟子?'
- Cypher 查询: `MATCH (p:__Entity__)-[:属于]->(s:__Entity__ {name: "青云宗"}) WHERE p.status = 'active' AND s.status = 'active' RETURN count(p)`

## 4. 排序与限制 (Sorting & Limiting)
- 用户问题: '列出与龙傲天关系最多的前3个实体。'
- Cypher 查询: `MATCH (a:__Entity__ {name:"龙傲天"})-[r]-(b:__Entity__) WHERE a.status = 'active' AND b.status = 'active' RETURN b.name, count(r) AS relationship_count ORDER BY relationship_count DESC LIMIT 3`

## 5. 上下文查询 (Contextual)
- 场景: `图谱 Schema` 中包含 `Event` 节点和 `date` 属性。
- 用户问题: '2023年在苍梧山发生了什么事件?'
- Cypher 查询: `MATCH (e:Event)-[:位于]->(l:__Entity__ {name: "苍梧山"}) WHERE e.status = 'active' AND l.status = 'active' AND e.date STARTS WITH '2023' RETURN e.name`

## 6. 属性查询 (Property)
- 用户问题: '介绍一下实体"赤霄剑"'
- Cypher 查询: `MATCH (n:__Entity__ {name: "赤霄剑"}) WHERE n.status = 'active' RETURN properties(n)`

# 指令
现在，请严格遵循以上所有规则和策略，为`用户问题`生成单行 Cypher 查询语句。
"""


###############################################################################


def get_kg_store(db_path: str) -> KuzuGraphStore:
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    db = kuzu.Database(db_path)
    kg_store = KuzuGraphStore(db)
    return kg_store


###############################################################################

def _check_content_unchanged(kg_store: KuzuGraphStore, doc_id: str, content: str) -> Tuple[bool, str]:
    """
    通过比较内容的 SHA256 哈希值，检查文档内容是否未发生变化。

    Args:
        kg_store (KuzuGraphStore): 知识图谱存储实例。
        doc_id (str): 文档的唯一标识符。
        content (str): 当前的文档内容。

    Returns:
        Tuple[bool, str]: 一个元组，第一个元素为布尔值，表示内容是否未变 (True) 或已变 (False)；
                          第二个元素为新内容的哈希值。
    """
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    query = "MATCH (d:__Document__ {doc_id: $doc_id}) RETURN d.content_hash AS old_hash"
    query_result = kg_store.query(query, param_map={"doc_id": doc_id})
    if query_result and query_result[0].get('old_hash') == new_content_hash:
        logger.info(f"内容 (doc_id: {doc_id}) 未发生变化，跳过更新。")
        return True, new_content_hash
    return False, new_content_hash


def _parse_and_update_vector_store(
    vector_store: VectorStore,
    doc_id: str,
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"]
) -> List[BaseNode]:
    """
    解析不同格式的内容为节点，并将其存入向量库。

    此函数会先删除向量库中与 `doc_id` 相关的旧节点，然后再插入新节点，
    从而实现内容的更新。

    Args:
        vector_store (VectorStore): 目标向量库实例。
        doc_id (str): 文档的唯一标识符。
        content (str): 待解析的文档内容。
        metadata (Dict[str, Any]): 与文档关联的元数据。
        content_format (str): 内容格式，支持 "md", "txt", "json"。

    Returns:
        List[BaseNode]: 解析成功时返回节点列表。如果无法解析出节点，则会引发异常。
    """
    doc = Document(id_=doc_id, text=content, metadata=metadata)
    if content_format == "json":
        nodes = [doc]
    elif content_format == "txt":
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
        nodes = node_parser.get_nodes_from_documents([doc])
    else:
        node_parser = MarkdownElementNodeParser(
            llm=Settings.llm,
            chunk_size=256,
            chunk_overlap=50,
            include_metadata=True,
            show_progress=False,
        )
        nodes = node_parser.get_nodes_from_documents([doc])

    if not nodes:
        logger.warning(f"内容 (doc_id: {doc_id}) 未解析出任何节点，跳过添加。")
        return []

    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    logger.info(f"正在从知识图谱的向量库中删除 doc_id '{doc_id}' 的旧节点...")
    vector_index.delete_ref_doc(doc_id, delete_from_docstore=True)
    logger.info(f"已删除 doc_id '{doc_id}' 的旧节点。")
    
    # time.sleep(0.5) # Give ChromaDB a moment. Removed for performance optimization.

    vector_index.insert_nodes(nodes)
    logger.info(f"已将 {len(nodes)} 个文本节点存入向量库。")
    return nodes


def _extract_and_normalize_triplets(
    nodes: List[BaseNode],
    max_triplets_per_chunk: int,
    kg_extraction_prompt: str
) -> List[tuple[str, str, str]]:
    """
    从文本节点中提取、规范化并去重知识三元组。

    使用临时的 LlamaIndex KnowledgeGraphIndex 进行三元组提取，然后对提取出的
    (主语, 关系, 宾语) 进行清洗（去除多余空格）和去重。

    Args:
        nodes (List[BaseNode]): 从文档解析出的节点列表。
        max_triplets_per_chunk (int): 每个节点块最多提取的三元组数量。
        kg_extraction_prompt (str): 用于指导 LLM 提取三元元组的提示。

    Returns:
        List[tuple[str, str, str]]: 规范化且去重后的三元组列表。
    """
    logger.info(f"正在从 {len(nodes)} 个节点中提取知识三元组, max_triplets_per_chunk={max_triplets_per_chunk}")
    
    temp_storage_context = StorageContext.from_defaults(graph_store=SimpleGraphStore())
    llm_extract_params = get_llm_params(llm_group="fast", temperature=llm_temperatures["summarization"])
    llm = LiteLLM(**llm_extract_params)

    temp_index = KnowledgeGraphIndex(
        nodes=nodes,
        storage_context=temp_storage_context,
        llm=llm,
        kg_extraction_prompt=PromptTemplate(kg_extraction_prompt),
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=False,
        show_progress=False,
    )

    graph_store = temp_index.graph_store
    raw_triplets = []
    for subj in list(graph_store.get_rel_map(depth=1).keys()):
        for rel, obj in graph_store.get(subj):
            raw_triplets.append((subj, rel, obj))

    normalized_triplets = set()
    for subj, rel, obj in raw_triplets:
        norm_subj = " ".join(subj.strip().split())
        norm_rel = " ".join(rel.strip().split())
        norm_obj = " ".join(obj.strip().split())
        if norm_subj and norm_rel and norm_obj:
            normalized_triplets.add((norm_subj, norm_rel, norm_obj))
    
    triplets = list(normalized_triplets)
    logger.info(f"提取了 {len(raw_triplets)} 个原始三元组，规范化和去重后剩余 {len(triplets)} 个。")
    return triplets


def _cleanup_old_graph_data(kg_store: KuzuGraphStore, doc_id: str):
    """
    清理（软删除）与指定 doc_id 相关的旧图谱数据。

    此函数执行一个 Cypher 查询，将目标 `doc_id` 从所有引用它的 `__Entity__` 节点的
    `doc_ids` 列表移除。如果一个节点的 `doc_ids` 列表变为空，则将其 `status`
    属性设置为 'inactive'。这是一种非破坏性的软删除。

    Args:
        kg_store (KuzuGraphStore): 知识图谱存储实例。
        doc_id (str): 需要清理其关联数据的文档ID。
    """
    logger.info(f"开始为 doc_id '{doc_id}' 清理旧的图谱数据。")
    # 优化后的查询：使用 WITH 预计算新列表，并用 CASE 表达式在一次 SET 操作中完成更新。
    cleanup_query = """
    MATCH (n:__Entity__) WHERE $doc_id IN n.doc_ids
    WITH n, [id IN n.doc_ids WHERE id <> $doc_id] AS new_doc_ids
    SET n.doc_ids = new_doc_ids,
        n.status = CASE WHEN size(new_doc_ids) = 0 THEN 'inactive' ELSE n.status END
    """
    kg_store.query(cleanup_query, param_map={"doc_id": doc_id})
    logger.info(f"已为 doc_id '{doc_id}' 执行了精细化清理。")


# 修复_write_new_graph_data函数中的语法错误
def _write_new_graph_data(kg_store: KuzuGraphStore, triplets: List[tuple[str, str, str]], doc_id: str):
    """
    将新的三元组数据批量写入图数据库。此函数已重构以支持向关系中添加属性。

    此函数首先使用 UNWIND 和 MERGE 语句批量创建或更新所有涉及的实体节点，并更新它们的
    `doc_ids` 列表和 `status` 属性。然后，按关系类型对三元组进行分组，并为每种
    关系类型批量创建或更新关系，同时向关系中添加 `doc_id`。此函数使用参数化查询以防止
    注入并正确处理数据类型。

    Args:
        kg_store (KuzuGraphStore): 知识图谱存储实例。
        triplets (List[tuple[str, str, str]]): 要写入的三元组列表。
        doc_id (str): 这些三元组来源的文档ID。
    """
    all_entities = set()
    for subj, _, obj in triplets:
        all_entities.add(subj)
        all_entities.add(obj)

    # 确保 __Entity__ 表存在
    if "__Entity__" not in kg_store.get_schema():
        kg_store.query("CREATE NODE TABLE __Entity__(name STRING, doc_ids STRING[], status STRING, PRIMARY KEY (name))")

    if all_entities:
        entities = [{"name": entity} for entity in all_entities]
        # 使用 list_append 优化列表追加，更符合 Kuzu 语法
        node_query = """
        UNWIND $entities AS entity
        MERGE (n:__Entity__ {{name: entity.name}})
        ON CREATE SET n.doc_ids = [$doc_id], n.status = 'active'
        ON MATCH SET n.doc_ids = CASE WHEN $doc_id IN n.doc_ids THEN n.doc_ids ELSE list_append(n.doc_ids, $doc_id) END,
                     n.status = 'active'
        """
        kg_store.query(node_query, param_map={"entities": entities, "doc_id": doc_id})
        logger.debug(f"已批量创建/更新 {len(all_entities)} 个实体节点。")

    triplets_by_rel = defaultdict(list)
    for subj, rel, obj in triplets:
        safe_rel = rel.replace('`', '')
        if safe_rel:
            triplets_by_rel[safe_rel].append({"subj": subj, "obj": obj})

    # 获取现有关系表以避免重复创建
    schema_str = kg_store.get_schema()
    for rel, pairs in triplets_by_rel.items():
        # 如果关系表不存在，则创建它
        if f"({rel})" not in schema_str:
            kg_store.query(f"CREATE REL TABLE `{rel}`(FROM __Entity__ TO __Entity__)")

        query = f"""
        UNWIND $pairs AS pair
        MATCH (s:__Entity__ {{name: pair.subj}})
        MATCH (o:__Entity__ {{name: pair.obj}})
        MERGE (s)-[:`{rel}`]->(o)
        """
        # Kuzu's MERGE for relationships doesn't support ON CREATE/MATCH.
        # We can simulate this by setting properties afterwards.
        # For now, we just merge the relationship structure.
        # A more advanced implementation could add properties to relationships:
        # MERGE (s)-[r:`{rel}`]->(o)
        # SET r.doc_ids = CASE WHEN r.doc_ids IS NULL THEN [$doc_id] ... END
        kg_store.query(query, param_map={"pairs": pairs, "doc_id": doc_id})
        logger.debug(f"已为关系 '{rel}' 批量创建/更新 {len(pairs)} 条记录。")

def _update_document_hash(kg_store: KuzuGraphStore, doc_id: str, content_hash: str):
    # 确保 __Document__ 表存在
    if "__Document__" not in kg_store.get_schema():
        kg_store.query("CREATE NODE TABLE __Document__(doc_id STRING, content_hash STRING, PRIMARY KEY (doc_id))")
    hash_update_query = """
    MERGE (d:__Document__ {doc_id: $doc_id})
    SET d.content_hash = $content_hash
    """
    kg_store.query(hash_update_query, param_map={"doc_id": doc_id, "content_hash": content_hash})
    logger.debug(f"已为 doc_id '{doc_id}' 更新内容哈希。")


def kg_add(
    kg_store: KuzuGraphStore,
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    content_format: Literal["md", "txt", "json"] = "md",
    max_triplets_per_chunk: int = 15,
    kg_extraction_prompt: str = kg_extraction_prompt
) -> None:
    """
    将内容原子化地添加到知识图谱和向量库，处理新增与更新。

    这是一个核心的流程编排函数，它按顺序执行以下操作：
    1. 检查内容哈希，如果内容未变则跳过。
    2. 解析内容并更新向量库中的文本节点。
    3. 从文本节点中提取知识三元组。
    4. 软删除与该文档ID相关的旧图谱实体引用。
    5. 写入新的图谱实体和关系。
    6. 更新文档的内容哈希记录。
    函数内部对关键步骤进行了异常处理，以确保流程的健壮性。

    Args:
        kg_store (KuzuGraphStore): 目标知识图谱存储实例。
        vector_store (VectorStore): 目标向量库实例，用于存储文本节点。
        content (str): 要添加的文档内容。
        metadata (Dict[str, Any]): 与文档关联的元数据。
        doc_id (str): 文档的唯一标识符，用于更新和删除。
        content_format (Literal["md", "txt", "json"], optional): 内容格式。默认为 "md"。
        max_triplets_per_chunk (int, optional): 每个文本块最多提取的三元组数量。默认为 15。
        kg_extraction_prompt (str, optional): 用于指导三元组提取的提示。默认为预设提示。
    """
    start_time = time.time()
    logger.info(f"开始处理 doc_id: {doc_id}...")

    # 1. 检查内容是否已是最新 (非关键，失败则继续)
    step_start_time = time.time()
    try:
        is_unchanged, content_hash = _check_content_unchanged(kg_store, doc_id, content)
        if is_unchanged:
            return
    except Exception as e:
        logger.warning("检查内容哈希时出错 (doc_id: {}, 可能是首次运行): {}", doc_id, e)
        # 即使检查失败，也计算哈希值以备后用
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    logger.info(f"步骤 1: 内容哈希检查完成, 耗时: {time.time() - step_start_time:.2f}s")
    
    # 2. 解析内容并更新向量库 (关键，失败则终止)
    step_start_time = time.time()
    try:
        nodes = _parse_and_update_vector_store(
            vector_store, doc_id, content, metadata, content_format
        )
        if not nodes:
            return
    except Exception as e:
        logger.error("为 doc_id '{}' 解析内容并更新向量库时失败: {}", doc_id, e, exc_info=True)
        return
    logger.info(f"步骤 2: 解析并更新向量库完成, 耗时: {time.time() - step_start_time:.2f}s")

    # 3. 从节点中提取三元组 (非关键，失败则继续，但无图谱更新)
    step_start_time = time.time()
    triplets = []
    try:
        triplets = _extract_and_normalize_triplets(
            nodes, max_triplets_per_chunk, kg_extraction_prompt
        )
    except Exception as e:
        logger.error("为 doc_id '{}' 从节点中提取三元组时失败: {}", doc_id, e, exc_info=True)
        # 允许继续执行，但图谱部分将为空
    logger.info(f"步骤 3: 提取三元组完成, 耗时: {time.time() - step_start_time:.2f}s")

    # 4. 清理旧图谱数据 (关键，失败则终止以防数据不一致)
    step_start_time = time.time()
    try:
        _cleanup_old_graph_data(kg_store, doc_id)
    except Exception as e:
        logger.error("为 doc_id '{}' 清理旧图谱数据时失败: {}", doc_id, e, exc_info=True)
        return
    logger.info(f"步骤 4: 清理旧图谱数据完成, 耗时: {time.time() - step_start_time:.2f}s")

    # 5. 写入新图谱数据 (关键，失败则终止)
    step_start_time = time.time()
    if triplets:
        try:
            _write_new_graph_data(kg_store, triplets, doc_id)
        except Exception as e:
            logger.error("为 doc_id '{}' 写入新图谱数据时失败: {}", doc_id, e, exc_info=True)
            return
    else:
        logger.warning(f"未能从内容 (doc_id: {doc_id}) 中提取任何三元组。")
    logger.info(f"步骤 5: 写入新图谱数据完成, 耗时: {time.time() - step_start_time:.2f}s")

    # 6. 更新文档哈希记录 (非关键，失败则记录警告)
    step_start_time = time.time()
    try:
        _update_document_hash(kg_store, doc_id, content_hash)
    except Exception as e:
        logger.warning("为 doc_id '{}' 更新文档哈希时失败: {}", doc_id, e, exc_info=True)
    logger.info(f"步骤 6: 更新文档哈希完成, 耗时: {time.time() - step_start_time:.2f}s")

    logger.success(f"成功处理内容 (doc_id: {doc_id}) 到知识图谱和向量库。总耗时: {time.time() - start_time:.2f}s")


###############################################################################


def get_kg_query_engine(
    kg_store: KuzuGraphStore,
    kg_vector_store: VectorStore,
    kg_similarity_top_k: int = 600,
    kg_rerank_top_n: int = 100,
    kg_nl2graphquery_prompt: str = kg_gen_cypher_prompt,
) -> BaseQueryEngine:
    logger.debug(f"参数: kg_similarity_top_k={kg_similarity_top_k}, kg_rerank_top_n={kg_rerank_top_n}")

    reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    kg_storage_context = StorageContext.from_defaults(
        graph_store=kg_store, 
        vector_store=kg_vector_store
    )
    kg_index = KnowledgeGraphIndex.from_documents(
        [], 
        storage_context=kg_storage_context, 
        llm=reasoning_llm,
        include_embeddings=True
    )

    reranker = SiliconFlowRerank(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        top_n=kg_rerank_top_n,
    )

    query_engine = kg_index.as_query_engine(
        llm=reasoning_llm,
        retriever_mode="hybrid", 
        similarity_top_k=kg_similarity_top_k,
        with_nl2graphquery=True, 
        graph_traversal_depth=2,
        nl2graphquery_prompt=PromptTemplate(kg_nl2graphquery_prompt), 
        response_synthesizer=synthesizer,
        node_postprocessors=[reranker],
        synonym_degree=2,
    )

    logger.success("知识图谱混合查询引擎创建成功。")
    return query_engine
