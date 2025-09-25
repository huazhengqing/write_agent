import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from utils.log import init_logger
init_logger(os.path.splitext(os.path.basename(__file__))[0])



def get_file_metadata(file_path_str: str) -> dict:
    from pathlib import Path
    file_path = Path(file_path_str)
    return {
        "type": "platform_profile",
        "platform": file_path.stem,
    }



def update_platform_from_input(platform_dir: str):
    from market_analysis.story.base import get_market_vector_store
    from rag.vector_dir import vector_add_from_dir
    return vector_add_from_dir(
        vector_store=get_market_vector_store(),
        input_dir=str(platform_dir),
        metadata_func=get_file_metadata,
    )



if __name__ == "__main__":
    from utils.file import data_platform_dir
    update_platform_from_input(data_platform_dir)
    
