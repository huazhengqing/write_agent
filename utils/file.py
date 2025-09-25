from functools import lru_cache
import os
from utils.models import Task



from dotenv import load_dotenv
load_dotenv()



from pathlib import Path
project_root = Path(__file__).resolve().parent.parent


data_dir = project_root / ".data"
data_dir.mkdir(parents=True, exist_ok=True)


data_market_dir = data_dir / "market"
data_market_dir.mkdir(parents=True, exist_ok=True)


prefect_dir = project_root / ".prefect"
os.environ["PREFECT_HOME"] = str(prefect_dir)
prefect_dir.mkdir(parents=True, exist_ok=True)
prefect_storage_path = prefect_dir / "storage"
prefect_storage_path.mkdir(parents=True, exist_ok=True)


cache_dir = project_root / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)


output_dir = project_root / "output"
output_dir.mkdir(parents=True, exist_ok=True)


log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)



@lru_cache(maxsize=30)
def sanitize_filename(name: str) -> str:
    import re
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s[:100]



@lru_cache(maxsize=30)
def get_text_file_path(task: Task) -> str:
    return os.path.join(output_dir, f"{task.run_id}.txt")



def text_file_append(file_path: str, content: str):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{content}")
        f.flush()
        os.fsync(f.fileno())



def text_file_read(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()




