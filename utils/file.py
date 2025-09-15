import os
from pathlib import Path
import re
from typing import get_args
from utils.models import Task, CategoryType
from dotenv import load_dotenv
load_dotenv()


project_root = Path(__file__).resolve().parent.parent

log_dir = project_root / ".logs"
log_dir.mkdir(parents=True, exist_ok=True)

prefect_dir = project_root / ".prefect"
os.environ["PREFECT_HOME"] = str(prefect_dir)
prefect_dir.mkdir(parents=True, exist_ok=True)
prefect_storage_path = prefect_dir / "storage"
prefect_storage_path.mkdir(parents=True, exist_ok=True)

sqlite_dir = project_root / ".sqlite"
sqlite_dir.mkdir(parents=True, exist_ok=True)
for category in get_args(CategoryType):
    (sqlite_dir / category).mkdir(exist_ok=True)

cache_dir = project_root / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
for category in get_args(CategoryType):
    (cache_dir / category).mkdir(exist_ok=True)

chroma_dir = project_root / ".chroma_db"
chroma_dir.mkdir(parents=True, exist_ok=True)
for category in get_args(CategoryType):
    (chroma_dir / category).mkdir(exist_ok=True)

kuzu_dir = project_root / ".kuzu_db"
kuzu_dir.mkdir(parents=True, exist_ok=True)

input_dir = project_root / ".input"
input_dir.mkdir(parents=True, exist_ok=True)
for category in get_args(CategoryType):
    (input_dir / category).mkdir(exist_ok=True)

output_dir = project_root / ".output"
output_dir.mkdir(parents=True, exist_ok=True)
for category in get_args(CategoryType):
    (output_dir / category).mkdir(exist_ok=True)


def sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s[:100]


def get_text_file_path(task: Task) -> str:
    return os.path.join(output_dir, task.category, f"{task.run_id}.txt")


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




