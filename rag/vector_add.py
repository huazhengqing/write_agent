from typing import Any, Dict, List, Literal, Optional
from loguru import logger
from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStore



from rag.vector import init_llama_settings
init_llama_settings()



def filter_invalid_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    valid_nodes = []
    initial_count = len(nodes)
    for node in nodes:
        import re
        if node.text.strip() and re.search(r'\w', node.text):
            valid_nodes.append(node)
    removed_count = initial_count - len(valid_nodes)
    if removed_count > 0:
        logger.debug(f"过滤掉 {removed_count} 个无效或空节点。")
    return valid_nodes



def _parse_content_to_nodes(
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"],
    doc_id: Optional[str] = None,
) -> List[BaseNode]:
    logger.info(f"开始为 doc_id '{doc_id}' 解析内容为节点 (格式: {content_format})...")
    from datetime import datetime
    if "time" not in metadata:
        metadata["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = Document(text=content, metadata=metadata, id_=doc_id)
    from rag.splitter import get_vector_node_parser
    node_parser = get_vector_node_parser(content_format, content_length=len(content))
    nodes = filter_invalid_nodes(node_parser.get_nodes_from_documents([doc], show_progress=False))
    logger.info(f"为 doc_id '{doc_id}' 解析出 {len(nodes)} 个节点。")
    return nodes



def vector_add(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    content_format: Literal["md", "txt", "json"] = "md",
    doc_id: Optional[str] = None,
) -> bool:
    logger.info(f"开始向向量库添加内容, doc_id='{doc_id}', format='{content_format}'...")
    if not content or not content.strip() or "生成报告时出错" in content:
        logger.warning(f"🤷 内容为空或包含错误, 跳过存入向量库。元数据: {metadata}")
        return False

    import hashlib
    new_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    doc_cache = getattr(vector_store, "cache", None)
    if doc_cache and doc_cache.get(new_content_hash):
        return True

    effective_doc_id = doc_id or new_content_hash

    nodes_to_insert = _parse_content_to_nodes(content, metadata, content_format, effective_doc_id)
    if not nodes_to_insert:
        logger.warning(f"内容 (id: {effective_doc_id}) 未解析出任何有效节点, 跳过添加。")
        return False

    if doc_id:
        logger.info(f"为 doc_id '{doc_id}' 执行更新操作, 将首先删除旧节点。")
        vector_store.delete(doc_id)

    from llama_index.core.ingestion import IngestionPipeline
    pipeline = IngestionPipeline(vector_store=vector_store, transformations=[Settings.embed_model])
    pipeline.run(nodes=nodes_to_insert)

    if doc_cache:
        doc_cache.set(new_content_hash, True)

    logger.success(f"成功将内容 (id: {effective_doc_id}, {len(nodes_to_insert)}个节点) 添加到向量库。")
    return True



def vector_delete(
    vector_store: VectorStore,
    doc_id: str,
) -> None:
    """从向量库中删除指定 doc_id 的所有相关节点。"""
    logger.info(f"开始从向量库删除内容, doc_id='{doc_id}'...")
    if not doc_id:
        logger.warning("doc_id 为空, 跳过删除。")
        return
    
    vector_store.delete(doc_id)
    logger.success(f"成功从向量库中删除 doc_id='{doc_id}' 的相关节点。")
