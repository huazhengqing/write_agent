import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])
from market_analysis.story.common import get_market_vector_store, input_platform_dir
from utils.vector import store_from_directory


def get_file_metadata(file_path_str: str) -> dict:
    file_path = Path(file_path_str)
    return {
        "platform": file_path.stem,
        "type": "platform_profile",
        "source": str(file_path.resolve()),
        "date": datetime.now().strftime("%Y-%m-%d")
    }

def update_platform_from_input(platform_dir: str):
    logger.info(f"开始从目录 '{platform_dir}' 更新平台档案...")
    success = store_from_directory(
        vector_store=get_market_vector_store(),
        input_dir=str(platform_dir),
        file_metadata_func=get_file_metadata,
        recursive=True,
        required_exts=[".md", ".txt"]
    )
    if success:
        logger.success(f"成功从 '{platform_dir}' 更新平台档案到向量数据库。")
    else:
        logger.warning(f"从 '{platform_dir}' 更新平台档案失败或未找到文件。")


if __name__ == "__main__":
    update_platform_from_input(input_platform_dir)
    
