from datetime import datetime
from utils.loader import load_prompts
from rag.kg import kg_add
from rag.vector_add import vector_add
from story.base import get_story_kg_store, get_story_vector_store
from utils.models import Task



def design(task: Task, content: str) -> None:
    header_parts = [
        task.id,
        task.hierarchical_position,
        task.goal,
    ]
    header = " ".join(filter(None, header_parts))
    content = f"# 任务\n{header}\n\n{content}"
    vector_store = get_story_vector_store(task.run_id, "design")
    kg_store = get_story_kg_store(task.run_id, "design")
    doc_metadata = {
        "task_id": task.id,
        "hierarchical_position": task.hierarchical_position,
        "status": "active",                         # 状态, 用于标记/取消文档
        "created_at": datetime.now().isoformat()
    }
    vector_add(
        vector_store=vector_store,
        content=content,
        metadata=doc_metadata,
        content_format="md",
        doc_id=task.id
    )
    kg_add(
        kg_store=kg_store,
        content=content,
        metadata=doc_metadata,
        doc_id=task.id,
        content_format="md",
        kg_extraction_prompt=load_prompts(f"story.prompts.kg.design", "kg_extraction_prompt")[0]
    )



def search(task: Task, content: str) -> None:
    header_parts = [task.id, task.hierarchical_position, task.goal]
    header = " ".join(filter(None, header_parts))
    full_content = f"# 任务\n{header}\n\n{content}"
    vector_store = get_story_vector_store(task.run_id, "search")
    doc_metadata = {
        "task_id": task.id,
        "created_at": datetime.now().isoformat()
    }
    vector_add(
        vector_store=vector_store,
        content=full_content,
        metadata=doc_metadata,
        content_format="md",
        doc_id=task.id
    )



def write(task: Task, content: str) -> None:
    kg_store = get_story_kg_store(task.run_id, "write")
    doc_metadata = {
        "task_id": task.id,
        "hierarchical_position": task.hierarchical_position,
        "created_at": datetime.now().isoformat()
    }
    kg_add(
        kg_store=kg_store,
        content=content,
        metadata=doc_metadata,
        doc_id=task.id,
        content_format="txt",
        kg_extraction_prompt=load_prompts(f"story.prompts.kg.write", "kg_extraction_prompt")[0]
    )



def summary(task: Task, content: str) -> None:
    header_parts = [
        task.id,
        task.hierarchical_position,
        task.goal,
        task.length
    ]
    header = " ".join(filter(None, header_parts))
    full_content = f"# 任务\n{header}\n\n{content}"
    vector_store = get_story_vector_store(task.run_id, "summary")
    doc_metadata = {
        "task_id": task.id,
        "hierarchical_position": task.hierarchical_position,
        "created_at": datetime.now().isoformat()
    }
    vector_add(
        vector_store=vector_store,
        content=full_content,
        metadata=doc_metadata,
        content_format="md",
        doc_id=task.id
    )


