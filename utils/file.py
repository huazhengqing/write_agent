import os
import re
from utils.models import Task


def sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s[:100]

def get_text_file_path(task: Task) -> str:
    return os.path.join("output", task.category, f"{task.run_id}.txt")

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


