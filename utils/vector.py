import os
import sys
import re
from llama_index.core.schema import TextNode
import threading
import asyncio
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from pydantic import Field
from litellm import arerank, rerank
import chromadb
from loguru import logger
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import CompactAndRefine, ResponseMode
from llama_index.core.schema import NodeWithScore, QueryBundle, RelatedNodeInfo
from llama_index.core.vector_stores import MetadataFilters, VectorStoreInfo, MetadataInfo
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeRelationship
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm import llm_temperatures, get_embedding_params, get_llm_params, get_rerank_params


def init_llama_settings():
    default_llm_params = get_llm_params(llm_group="fast", temperature=llm_temperatures["summarization"])
    Settings.llm = LiteLLM(**default_llm_params)
    default_llm_params['exceptions_to_fallback_on'] = get_llm_params()['exceptions_to_fallback_on']
    Settings.prompt_helper.context_window = default_llm_params.get('context_window', 4096)
    Settings.prompt_helper.num_output = default_llm_params.get('max_tokens', 512)

    embedding_params = get_embedding_params()
    embed_model_name = embedding_params.pop('model')
    Settings.embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)

init_llama_settings()


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


def get_nodes_from_document(doc: Document) -> List[Any]:
    parser = MarkdownElementNodeParser()
    nodes = parser.get_nodes_from_documents([doc])
    return nodes


def _convert_to_simple_text_nodes(nodes: List[Any], source_doc_id: str) -> List[TextNode]:
    new_nodes = []
    for n in nodes:
        text = n.get_content()
        
        # 过滤掉仅包含空白或Markdown分隔线的节点
        lines = [line.strip() for line in text.strip().split('\n')]
        meaningful_lines = [line for line in lines if line and line != '---']
        if not meaningful_lines:
            continue

        new_node = TextNode(
            id_=n.id_,
            text=text,
            metadata=n.metadata.copy(),
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=source_doc_id)}
        )
        new_nodes.append(new_node)
    return new_nodes


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
        
        nodes_for_doc = []
        if file_path.suffix == ".json":
            # 对于JSON，将其视为单个TextNode
            nodes_for_doc = [TextNode(
                id_=doc.id_,
                text=doc.text,
                metadata=doc.metadata,
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc.id_)}
            )]
        else:
            parsed_nodes = get_nodes_from_document(doc)
            nodes_for_doc = _convert_to_simple_text_nodes(parsed_nodes, doc.id_)

        logger.info(f"  - 文件 '{file_path.name}' 被解析成 {len(nodes_for_doc)} 个节点。")
        logger.trace(f"  - 为 '{file_path.name}' 创建的节点内容: {[n.get_content(metadata_mode='all') for n in nodes_for_doc]}")
        all_nodes.extend(nodes_for_doc)

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
    pipeline.run(nodes=unique_nodes, show_progress=True)

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
            logger.info(f"向量库内容变更, 使缓存的 VectorStoreIndex 失效 (key: {cache_key})。")
            del _vector_indices[cache_key]

    if doc_id:
        logger.info(f"正在从向量库中删除 doc_id '{doc_id}' 的旧节点...")
        vector_store.delete(ref_doc_id=doc_id)
        logger.info(f"已删除 doc_id '{doc_id}' 的旧节点。")

    final_metadata = metadata.copy()
    if "date" not in final_metadata:
        final_metadata["date"] = datetime.now().strftime("%Y-%m-%d")

    doc = Document(text=content, metadata=final_metadata, id_=doc_id)
    nodes_to_insert = []
    if content_format == "json":
        nodes_to_insert = [TextNode(
            id_=doc.id_,
            text=doc.text,
            metadata=doc.metadata,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc.id_)}
        )]
    else:
        parsed_nodes = get_nodes_from_document(doc)
        nodes_to_insert = _convert_to_simple_text_nodes(parsed_nodes, doc.id_)
    if not nodes_to_insert:
        logger.warning(f"内容 (doc_id: {doc_id}) 未解析出任何节点，跳过添加。")
        return False
    logger.debug(f"为 doc_id '{doc_id}' 创建的节点内容: {[n.get_content(metadata_mode='all') for n in nodes_to_insert]}")

    pipeline = IngestionPipeline(vector_store=vector_store)
    pipeline.run(nodes=nodes_to_insert)

    logger.success(f"成功将内容 (doc_id: {doc_id}, {len(nodes_to_insert)}个节点) 添加到向量库。")
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

        logger.debug(f"Reranker (同步) 收到 {len(nodes)} 个待重排节点。")
        for i, node in enumerate(nodes):
            logger.trace(f"  - 原始节点 {i+1} (score: {node.score:.4f}): {node.get_content()[:100]}...")

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
        if response and response.results:
            for result in response.results:
                original_node = nodes[result.index]
                original_node.score = result.relevance_score
                new_nodes_with_scores.append(original_node)
            logger.debug(f"重排后 (同步) 返回 {len(new_nodes_with_scores)} 个节点。")
        else:
            logger.warning(f"同步 rerank 调用返回了空或无效的结果。Response: {response}。将返回原始节点。")
            return nodes

        return new_nodes_with_scores

    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("必须提供查询信息 (QueryBundle) 才能进行重排。")
        if not nodes:
            return []

        logger.debug(f"Reranker (异步) 收到 {len(nodes)} 个待重排节点。")
        for i, node in enumerate(nodes):
            logger.trace(f"  - 原始节点 {i+1} (score: {node.score:.4f}): {node.get_content()[:100]}...")

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
        if response and response.results:
            for result in response.results:
                original_node = nodes[result.index]
                original_node.score = result.relevance_score
                new_nodes_with_scores.append(original_node)
        else:
            logger.warning(f"异步 arerank 调用返回了空或无效的结果。Response: {response}。将返回原始节点。")
            return nodes
        
        logger.debug(f"重排后 (异步) 返回 {len(new_nodes_with_scores)} 个节点。")
        return new_nodes_with_scores


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
            index = VectorStoreIndex.from_vector_store(vector_store)
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
    synthesis_llm_params['exceptions_to_fallback_on'] = get_llm_params()['exceptions_to_fallback_on']
    response_synthesizer = CompactAndRefine(
        llm=synthesis_llm,
        prompt_helper=PromptHelper(
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
        logger.info("使用标准 as_query_engine 模式创建查询引擎。")
        query_engine = index.as_query_engine(
            llm=synthesis_llm,
            response_synthesizer=response_synthesizer,
            filters=filters,
            similarity_top_k=similarity_top_k,
            node_postprocessors=postprocessors,
        )
        logger.success("标准查询引擎创建成功。")
        return query_engine


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
        if result and hasattr(result, "response") and result.response:
            answer = str(result.response).strip()
            final_answers.append(answer)
            logger.debug(f"问题 '{question}' 的回答: {answer}")
        else:
            logger.warning(f"查询 '{question}' 未生成有效回答。")
            final_answers.append("")

    logger.success(f"批量查询完成，共返回 {len(final_answers)} 个回答。")
    return final_answers


###############################################################################


if __name__ == '__main__':
    import asyncio
    import tempfile
    import shutil
    from pathlib import Path
    import json
    from utils.log import init_logger
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    import nest_asyncio

    init_logger("vector_test")

    nest_asyncio.apply()

    # 1. 初始化临时目录
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "chroma_db")
    input_dir = os.path.join(test_dir, "input_data")
    os.makedirs(input_dir, exist_ok=True)
    logger.info(f"测试目录已创建: {test_dir}")

    async def main():
        # 2. 准备多样化的测试文件
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

        # 3. 测试 get_vector_store
        logger.info("--- 3. 测试 get_vector_store ---")
        vector_store = get_vector_store(db_path=db_path, collection_name="test_collection")
        logger.info(f"成功获取 VectorStore: {vector_store}")

        # 4. 测试从目录添加
        logger.info("--- 4. 测试 vector_add_from_dir (常规) ---")
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

        # 7. 测试显式删除
        logger.info("--- 7. 测试显式删除 ---")
        vector_add(
            vector_store,
            "这是一个即将被删除的节点。",
            {"type": "disposable", "source": "delete_test"},
            doc_id="to_be_deleted"
        )
        query_engine_simple = get_vector_query_engine(vector_store, similarity_top_k=1, rerank_top_n=0)
        res_before_delete = await index_query(query_engine_simple, ["查找即将被删除的节点"])
        assert len(res_before_delete) > 0
        logger.info("删除前节点存在，验证通过。")

        vector_store.delete(ref_doc_id="to_be_deleted")
        logger.info("已调用删除方法。")

        # 直接操作 vector_store 后需要手动使索引缓存失效
        with _vector_index_lock:
            cache_key = id(vector_store)
            if cache_key in _vector_indices:
                del _vector_indices[cache_key]
        logger.info("已使向量索引缓存失效。")

        query_engine_after_delete = get_vector_query_engine(vector_store, similarity_top_k=1, rerank_top_n=0)
        res_after_delete = await index_query(query_engine_after_delete, ["查找即将被删除的节点"])
        assert len(res_after_delete) == 0
        logger.info("删除后节点不存在，验证通过。")

        # 8. 测试 get_vector_query_engine (标准模式)
        logger.info("--- 8. 测试 get_vector_query_engine (标准模式) ---")
        query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=2)
        logger.info(f"成功创建标准查询引擎: {type(query_engine)}")

        questions1 = [
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
        results1 = await index_query(query_engine, questions1)
        logger.info(f"标准查询结果:\n{results1}")
        assert any("龙傲天" in r for r in results1)
        assert any("虚空之石" in r for r in results1)
        assert any("萧炎" in r and "炎盟" in r for r in results1)
        assert any("药尘" in r for r in results1)
        assert any("萧炎" in r and "魂天帝" in r for r in results1)
        assert any("建木" in r for r in results1)
        assert any("宿敌" in r for r in results1)
        assert any("戮仙剑" in r and "绝仙剑" in r for r in results1)
        assert any("黑风寨" in r and "北冥魔殿" in r for r in results1)
        assert any("苍龙七宿" in r and "星宿之力" in r for r in results1)
        assert any("初入江湖" in r and "实力提升" in r for r in results1) # 验证Mermaid图内容
        assert any("北冥魔殿分舵" in r for r in results1) # 验证新增列表内容
        assert any("fibonacci" in r and "def" in r for r in results1) # 验证代码块内容
        # 验证被跳过或删除的内容不存在
        assert not any("错误信息" in r for r in results1)
        assert not any("即将被删除" in r for r in results1)

        # 9. 测试 get_vector_query_engine (带固定过滤器)
        logger.info("--- 9. 测试 get_vector_query_engine (带固定过滤器) ---")
        filters = MetadataFilters(filters=[ExactMatchFilter(key="type", value="item")])
        query_engine_filtered = get_vector_query_engine(vector_store, filters=filters)
        questions2 = ["介绍一下那个石头。"]
        results2 = await index_query(query_engine_filtered, questions2)
        logger.info(f"带过滤器的查询结果:\n{results2}")
        assert len(results2) > 0 and "虚空之石" in results2[0]

        questions3 = ["龙傲天是谁？"]  # 这个查询应该被过滤器挡住
        results3 = await index_query(query_engine_filtered, questions3)
        logger.info(f"被过滤器阻挡的查询结果:\n{results3}")
        assert len(results3) == 0

        # 10. 测试无重排器和同步查询
        logger.info("--- 10. 测试无重排器和同步查询 ---")
        query_engine_no_rerank = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=0)
        sync_question = "林动的功法是什么？"
        # 使用 .query() 来测试同步路径
        sync_response = query_engine_no_rerank.query(sync_question)
        logger.info(f"同步查询 (无重排器) 结果:\n{sync_response}")
        assert "大荒芜经" in str(sync_response)

        # 11. 测试 get_vector_query_engine (自动检索模式)
        logger.info("--- 11. 测试 get_vector_query_engine (自动检索模式) ---")
        query_engine_auto = get_vector_query_engine(vector_store, use_auto_retriever=True, similarity_top_k=5, rerank_top_n=2)
        logger.info(f"成功创建自动检索查询引擎: {type(query_engine_auto)}")

        # 这个查询应该能被 AutoRetriever 解析为针对 metadata 'type'='item' 的过滤
        auto_question = "请根据类型为 'item' 的文档，介绍一下那个物品。"
        auto_results = await index_query(query_engine_auto, [auto_question])
        logger.info(f"自动检索查询结果:\n{auto_results}")
        assert len(auto_results) > 0 and "虚空之石" in auto_results[0]

        # 12. 测试空查询
        logger.info("--- 12. 测试空查询 ---")
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
