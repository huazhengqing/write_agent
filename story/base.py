from functools import lru_cache
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.kuzu import KuzuGraphStore
from utils.file import data_dir


@lru_cache(maxsize=None)
def get_story_vector_store(run_id: str, content_type: str) -> ChromaVectorStore:
    chroma_path = data_dir / run_id / content_type
    collection_name = f"{run_id}_{content_type}"
    from utils.vector import get_vector_store
    vector_store = get_vector_store(db_path=str(chroma_path), collection_name=collection_name)
    return vector_store


@lru_cache(maxsize=None)
def get_story_kg_store(run_id: str, content_type: str) -> KuzuGraphStore:
    kuzu_db_path = data_dir / run_id / content_type
    from utils.kg import get_kg_store
    graph_store = get_kg_store(db_path=str(kuzu_db_path))
    return graph_store
