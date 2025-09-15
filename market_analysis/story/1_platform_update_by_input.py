import os
from pathlib import Path
from loguru import logger
from datetime import datetime
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from utils.log import init_logger
from market_analysis.story.common import input_platform_dir, index

init_logger(os.path.splitext(os.path.basename(__file__))[0])


def get_file_metadata(file_path_str: str) -> dict:
    file_path = Path(file_path_str)
    return {
        "platform": file_path.stem,
        "type": "platform_profile",
        "source": str(file_path.resolve()),
        "date": datetime.now().strftime("%Y-%m-%d")
    }

if __name__ == "__main__":
    logger.info(f"正在从 '{input_platform_dir}' 目录加载文件...")
    reader = SimpleDirectoryReader(
        input_dir=input_platform_dir,
        required_exts=[".md", ".txt"],
        file_metadata=get_file_metadata
    )
    documents = reader.load_data()
    if not documents:
        logger.warning(f"未在 '{input_platform_dir.resolve()}' 目录中找到任何 .md 或 .txt 文件。")
        exit(1)

    logger.info(f"找到 {len(documents)} 个文件，开始构建索引...")
    md_parser = MarkdownNodeParser(include_metadata=True)
    txt_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    all_nodes = []
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", ""))
        if file_path.suffix == ".md":
            nodes = md_parser.get_nodes_from_documents([doc])
        else:
            nodes = txt_parser.get_nodes_from_documents([doc])
        all_nodes.extend(nodes)

    index.insert_nodes(all_nodes, show_progress=True)
    
    logger.success(f"成功处理 {len(documents)} 个平台文件并存入向量数据库。")


