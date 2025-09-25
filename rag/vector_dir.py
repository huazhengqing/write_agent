from functools import lru_cache
from pathlib import Path
from loguru import logger
from typing import Callable, Dict, List
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.schema import BaseNode



from rag.vector import init_llama_settings
init_llama_settings()



@lru_cache(maxsize=30)
def file_metadata_default(file_path_str: str) -> dict:
    file_path = Path(file_path_str)
    from datetime import datetime
    stat = file_path.stat()
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_path.name,
        "file_path": file_path_str,
        "creation_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modification_date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }



@lru_cache(maxsize=30)
def _load_and_filter_documents(
    input_dir: str,
    metadata_func: Callable[[str], dict]
) -> List[Document]:
    logger.info(f"开始从目录 '{input_dir}' 加载和过滤文档...")
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        required_exts=[".md", ".txt", ".json"],
        file_metadata=file_metadata_default,
        recursive=True,
        exclude_hidden=False
    )
    documents = reader.load_data()
    if not documents:
        logger.warning(f"🤷 在 '{input_dir}' 目录中未找到任何符合要求的文件。")
        return []
    
    logger.debug(f"从 '{input_dir}' 初始加载了 {len(documents)} 个文档。")
    valid_docs = []
    for doc in documents:
        file_path_str = doc.metadata.get("file_path") 
        if not file_path_str:
            logger.warning(f"文档缺少有效的文件路径ID, 跳过。元数据: {doc.metadata}")
            continue
        
        file_path = Path(file_path_str)
        custom_meta = metadata_func(file_path_str)
        doc.metadata.update(custom_meta)
        if not doc.text or not doc.text.strip():
            logger.warning(f"⚠️ 文件 '{file_path.name}' 内容为空, 已跳过。")
            continue
        
        valid_docs.append(doc)
    
    logger.success(f"完成文档加载和过滤, 共获得 {len(valid_docs)} 个有效文档。")
    return valid_docs



def _parse_docs_to_nodes_by_format(documents: List[Document]) -> List[BaseNode]:
    from rag.splitter import get_vector_node_parser
    logger.info("开始按文件格式解析文档为节点...")
    docs_by_format: Dict[str, List[Document]] = {
        "md": [], 
        "txt": [], 
        "json": []
    }
    for doc in documents:
        file_name = doc.metadata.get("file_name", "")
        file_extension = Path(file_name).suffix.lstrip('.')
        if file_extension in docs_by_format:
            docs_by_format[file_extension].append(doc)
        else:
            logger.warning(f"检测到未支持的文件扩展名 '{file_extension}', 将忽略。")

    all_nodes = []
    from rag.vector_add import filter_invalid_nodes
    for content_format, format_docs in docs_by_format.items():
        if not format_docs:
            continue
        
        logger.info(f"正在处理 {len(format_docs)} 个 '{content_format}' 文件...")
        nodes_for_format = []
        for doc in format_docs:
            node_parser = get_vector_node_parser(content_format, content_length=len(doc.text))
            parsed_nodes = node_parser.get_nodes_from_documents([doc], show_progress=False)
            nodes_for_format.extend(filter_invalid_nodes(parsed_nodes))
        logger.info(f"  - 从 '{content_format}' 文件中成功解析出 {len(nodes_for_format)} 个节点。")
        all_nodes.extend(nodes_for_format)
    
    logger.success(f"文档解析完成, 总共生成 {len(all_nodes)} 个节点。")
    return all_nodes



def vector_add_from_dir(
    vector_store: VectorStore,
    input_dir: str,
    metadata_func: Callable[[str], dict] = file_metadata_default,
) -> bool:
    logger.info(f"开始从目录 '{input_dir}' 添加内容到向量库...")
    documents = _load_and_filter_documents(input_dir, metadata_func)
    if not documents:
        return False

    all_nodes = _parse_docs_to_nodes_by_format(documents)
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

    if not unique_nodes:
        logger.warning("🤷‍♀️ 过滤后没有唯一的节点可供索引。")
        return False

    from llama_index.core.ingestion import IngestionPipeline
    pipeline = IngestionPipeline(vector_store=vector_store, transformations=[Settings.embed_model])
    pipeline.run(nodes=unique_nodes)

    logger.success(f"成功从目录 '{input_dir}' 添加 {len(unique_nodes)} 个节点到向量库。")
    return True
