import os
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
import hashlib
import time
from collections import defaultdict
import kuzu
from loguru import logger

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms.litellm import LiteLLM
from llama_index.core.node_parser import SentenceSplitter, NodeParser, MarkdownElementNodeParser, SimpleNodeParser
from llama_index.extractors.entity import SimpleLLMPathExtractor
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

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

5.  处理修正与删除 (新规则):
    - 当文本包含明确的修正或删除指令时（如“删除...”、“修正为...”、“不再是...”），将其转换为一个表示最终状态或变化的三元组。
    - 例如，"删除'龙傲天是青云宗弟子'的设定" 或 "龙傲天不再是青云宗弟子"，可以提取为 `("龙傲天", "已离开", "青云宗")` 或 `("龙傲天", "关系结束", "青云宗")`。
    - 例如，"将龙傲天的门派从青云宗改为北冥魔殿"，应提取为 `("龙傲天", "已离开", "青云宗")` 和 `("龙傲天", "加入", "北冥魔殿")`。
    - 这种方式通过添加新事实来记录状态的变更，而不是真正删除信息。

# 提取流程
1.  识别实体: 通读文本，识别出所有符合实体定义的关键实体。
2.  解析关系: 分析实体之间的关系，包括动作、属性、从属等，并确定最精确的动词短语作为关系。特别注意表示状态变更的词语。
3.  指代消解: 在构建三元组前， mentally 解析所有代词（如“他”、“她”、“它”、“他们”），并用其指代的具体实体名称替换。
4.  构建三元组: 按照 `(主语, 关系, 宾语)` 的格式构建三元组列表。

# 示例分析 (请根据文本类型自动调整提取策略)

## 示例1: 小说/叙事文本
- 文本: "黄昏时分，在[地点A]的'[地点B]'，[角色A]展开了一张指向[地点C]的[物品A]。这张[物品A]是他在[组织A]的师父[角色B]所赠。"
- 最终三元组:
  - ("[角色A]", "位于", "[地点B]")
  - ("[地点B]", "位于", "[地点A]")
  - ("[角色A]", "展开", "[物品A]")
  - ("[物品A]", "指向", "[地点C]")
  - ("[角色B]", "赠送给", "[角色A]")
  - ("[角色B]", "是师父", "[角色A]")
  - ("[角色A]", "属于", "[组织A]")
  - ("[角色B]", "属于", "[组织A]")

## 示例2: 报告/分析文本
- 文本: "根据[机构A]的[年份A]报告，[国家A]的[行业A]市场的规模达到了[数值A]，并预测将在[年份B]增长至[数值B]。"
- 最终三元组:
  - ("[国家A]的[行业A]市场", "规模是", "[数值A]")
  - ("[数值A]", "统计年份", "[年份A]")
  - ("[国家A]的[行业A]市场", "预测规模", "[数值B]")
  - ("[数值B]", "预测年份", "[年份B]")
  - ("[国家A]的[行业A]市场", "数据来源", "[机构A]报告")

## 示例3: 技术手册/工具书
- 文本: "在[技术栈A]中，`[函数A]` Hook是一个函数，它允许你在函数组件中添加和管理状态(state)。它返回一个状态值和一个更新该值的函数。"
- 最终三元组:
  - ("[函数A] Hook", "属于", "[技术栈A]")
  - ("[函数A] Hook", "类型是", "函数")
  - ("[函数A] Hook", "允许", "在函数组件中管理状态")
  - ("[函数A] Hook", "返回", "状态值")
  - ("[函数A] Hook", "返回", "更新函数")

## 示例4: 表格 (Table) 数据
- 文本:
-  "| 姓名 | 组织   | 职位 |\n|---|---|---|\n| [角色A] | [组织A]   | [职位A] |\n| [角色B] | [组织B]   | [职位B] |"
- 最终三元组:
  - ("[角色A]", "属于", "[组织A]")
  - ("[角色A]", "职位是", "[职位A]")
  - ("[角色B]", "属于", "[组织B]")
  - ("[角色B]", "职位是", "[职位B]")

## 示例5: JSON 数据
- 文本:
-  '{ "character": "[角色A]", "alias": "[别名A]", "occupation": "[职业A]", "affiliation": { "name": "[组织A]", "role": "创始人" } }'
- 最终三元组:
  - ("[角色A]", "别名是", "[别名A]")
  - ("[角色A]", "职业是", "[职业A]")
  - ("[角色A]", "属于", "[组织A]")
  - ("[组织A]", "创始人是", "[角色A]")

## 示例6: 关系图 (Mermaid) 数据
- 文本:
""" + "```" + """mermaid
graph TD
    [角色A] -- 宿敌是 --> [角色B];
    [角色A] -- 属于 --> [组织A];
""" + "```" + """
- 最终三元组:
  - ("[角色A]", "宿敌是", "[角色B]")
  - ("[角色A]", "属于", "[组织A]")

## 劣质示例
- 文本: "他很强大。"
- 劣质三元组:
  - ("他", "是", "强大")  # 错误: 使用了代词"他"；"强大"是形容词而非实体；"是"关系模糊。
- 文本: "[角色A]去了[组织A]。"
- 劣质三元组:
  - ("[角色A]", "去", "地方") # 错误: "地方"不是命名实体，信息密度太低。应为 `("[角色A]", "前往", "[组织A]")`。

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
1.  严格遵循 Schema: 只能使用`图谱 Schema`中明确定义的节点标签、关系类型和属性。禁止猜测或使用不存在的元素。
2.  优先使用属性: 在 `MATCH` 子句中，优先使用属性（如 `{name: "实体名"}`）进行匹配，而不是仅仅依赖标签。
3.  关系方向: 对于不确定的关系查询（如“A和B有什么关系”），使用无方向匹配 `-[r]-`。对于明确的动作（如“A击败了B”），使用有方向匹配 `->`。
4.  字符串安全: 如果从用户问题中提取的实体名称包含双引号(`"`)，必须用反斜杠(`\\`)进行转义。

# 输出要求
1.  单行输出: 最终的 Cypher 查询必须是单行文本，不含任何换行符。
2.  无额外内容: 仅输出 Cypher 查询语句本身或 "INVALID_QUERY"。禁止添加任何解释、注释或代码块标记。
3.  无效查询: 如果问题无法基于给定的`图谱 Schema`回答，或者问题含糊不清，固定返回字符串 "INVALID_QUERY"。

# 查询策略与示例

## 1. 基础查询 (1-hop)
- 用户问题: '实体"[实体A]"和"[实体B]"是什么关系?'
- Cypher 查询: `MATCH (a:__Entity__ {name: "[实体A]"})-[r]-(b:__Entity__ {name: "[实体B]"}) RETURN type(r)`

## 2. 多跳查询 (Multi-hop)
- 用户问题: '[实体A]的宿敌的组织是什么？'
- Cypher 查询: `MATCH (a:__Entity__ {name: "[实体A]"})-[:宿敌是]-(enemy:__Entity__)-[:属于]->(faction:__Entity__) RETURN faction.name`

## 3. 聚合查询 (Aggregation)
- 用户问题: '[组织A]有多少个成员?'
- Cypher 查询: `MATCH (p:__Entity__)-[:属于]->(s:__Entity__ {name: "[组织A]"}) RETURN count(p)`

## 4. 排序与限制 (Sorting & Limiting)
- 用户问题: '列出与[实体A]关系最多的前3个实体。'
- Cypher 查询: `MATCH (a:__Entity__ {name:"[实体A]"})-[r]-(b:__Entity__) RETURN b.name, count(r) AS relationship_count ORDER BY relationship_count DESC LIMIT 3`

## 5. 上下文查询 (Contextual)
- 场景: `图谱 Schema` 中包含 `Event` 节点和 `date` 属性。
- 用户问题: '[年份A]在[地点A]发生了什么事件?'
- Cypher 查询: `MATCH (e:Event)-[:位于]->(l:__Entity__ {name: "[地点A]"}) WHERE e.date STARTS WITH '[年份A]' RETURN e.name`

## 6. 属性查询 (Property)
- 用户问题: '介绍一下实体"[实体A]"'
- Cypher 查询: `MATCH (n:__Entity__ {name: "[实体A]"}) RETURN properties(n)`

# 指令
现在，请严格遵循以上所有规则和策略，为`用户问题`生成单行 Cypher 查询语句。
"""


###############################################################################


def get_kg_store(db_path: str) -> KuzuGraphStore:
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE IF NOT EXISTS __Entity__(name STRING, doc_ids STRING[], PRIMARY KEY (name))")
    conn.execute("CREATE NODE TABLE IF NOT EXISTS __Document__(doc_id STRING, content_hash STRING, PRIMARY KEY (doc_id))")
    conn.execute("CREATE REL TABLE IF NOT EXISTS relationship(FROM __Entity__ TO __Entity__, label STRING)")
    kg_store = KuzuGraphStore(db)
    return kg_store


###############################################################################


def _is_content_unchanged(kg_store: KuzuGraphStore, doc_id: str, new_content_hash: str) -> bool:
    """检查内容哈希，如果内容未变则返回True。"""
    hash_check_query = "MATCH (d:__Document__ {doc_id: $doc_id}) RETURN d.content_hash AS old_hash"
    query_result = kg_store.query(hash_check_query, param_map={"doc_id": doc_id})
    return bool(query_result and query_result[0].get('old_hash') == new_content_hash)


def _get_kg_node_parser(content_format: Literal["md", "txt", "json"], content_length: int) -> NodeParser:
    """根据内容格式和长度获取合适的节点解析器。"""
    if content_format == "json":
        parser = SentenceSplitter(chunk_size=content_length * 2 if content_length > 0 else 1, chunk_overlap=0)
        logger.info(f"使用 JSON 整体解析策略，chunk_size={getattr(parser, 'chunk_size', 'N/A')}")
    elif content_format == "md":
        parser = MarkdownElementNodeParser(
            llm=None, 
            chunk_size=2048,
            chunk_overlap=400,
            include_metadata=True,
        )
        logger.info(f"使用 Markdown 元素解析策略 (无LLM摘要)，内部 chunk_size=2048")
    else:  # txt
        parser = SentenceSplitter(chunk_size=2048, chunk_overlap=400)
        logger.info(f"使用大文本块分割策略，chunk_size={parser.chunk_size}")
    return parser


def _update_document_hash(kg_store: KuzuGraphStore, doc_id: str, content_hash: str):
    """在知识图谱中更新文档的内容哈希记录。"""
    hash_update_query = """
    MERGE (d:__Document__ {doc_id: $doc_id})
    SET d.content_hash = $content_hash
    """
    kg_store.query(hash_update_query, param_map={"doc_id": doc_id, "content_hash": content_hash})
    logger.info(f"已更新 doc_id '{doc_id}' 的内容哈希。")


def kg_add(
    kg_store: KuzuGraphStore,
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: str,
    content_format: Literal["md", "txt", "json"] = "md",
    chars_per_triplet: int = 120,
    kg_extraction_prompt: str = kg_extraction_prompt
) -> None:
    start_time = time.time()
    logger.info(f"开始处理 doc_id: {doc_id} (格式: {content_format})...")

    # 检查内容哈希，如果内容未变则跳过
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    if _is_content_unchanged(kg_store, doc_id, new_content_hash):
        logger.info(f"内容 (doc_id: {doc_id}) 未发生变化，跳过更新。")
        return

    doc = Document(id_=doc_id, text=content, metadata=metadata)

    kg_node_parser = _get_kg_node_parser(content_format, len(content))

    # 动态计算每个块允许提取的最大三元组数量
    chunk_size = getattr(kg_node_parser, 'chunk_size', 2048)
    max_triplets_per_chunk = max(1, round(chunk_size / chars_per_triplet))
    logger.info(f"根据 chars_per_triplet={chars_per_triplet} 和 chunk_size={chunk_size}，动态设置 max_triplets_per_chunk={max_triplets_per_chunk}")

    storage_context = StorageContext.from_defaults(graph_store=kg_store, vector_store=vector_store)
    llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["classification"])
    llm_for_extraction = LiteLLM(**llm_params)

    path_extractor = SimpleLLMPathExtractor(
        llm=llm_for_extraction,
        max_triplets_per_chunk=max_triplets_per_chunk,
        kg_extraction_prompt=PromptTemplate(kg_extraction_prompt),
    )

    PropertyGraphIndex.from_documents(
        [doc],
        storage_context=storage_context,
        transformations=[kg_node_parser],
        kg_extractors=[path_extractor],
        include_embeddings=True,
        embed_model=Settings.embed_model,
        show_progress=False,
    )
    logger.info(f"已通过 PropertyGraphIndex 处理新内容，自动写入向量库和知识图谱。")

    # 更新文档哈希记录
    _update_document_hash(kg_store, doc_id, new_content_hash)

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

    # reasoning_llm_params = get_llm_params(llm_group="reasoning", temperature=llm_temperatures["reasoning"])
    reasoning_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["reasoning"])
    reasoning_llm = LiteLLM(**reasoning_llm_params)

    kg_storage_context = StorageContext.from_defaults(
        graph_store=kg_store, 
        vector_store=kg_vector_store
    )
    
    kg_index = PropertyGraphIndex.from_documents(
        [], 
        storage_context=kg_storage_context, 
        llm=reasoning_llm,
        include_embeddings=True,
        embed_model=Settings.embed_model,
    )

    reranker = SiliconFlowRerank(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        top_n=kg_rerank_top_n,
    )

    query_engine = kg_index.as_query_engine(
        llm=reasoning_llm,
        include_text=True,
        embedding_mode="hybrid", 
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
