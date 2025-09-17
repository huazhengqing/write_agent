import os
import sys
import threading
from typing import Dict
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.kg import get_kg_store
from utils.vector import get_vector_store
from utils.file import data_dir


vector_stores: Dict[str, ChromaVectorStore] = {}
graph_stores: Dict[str, KuzuGraphStore] = {}
_storage_lock = threading.Lock()


def get_story_vector_store(run_id: str, content_type: str) -> ChromaVectorStore:
    context_key = f"{run_id}_{content_type}"
    if context_key in vector_stores:
        return vector_stores[context_key]

    with _storage_lock:
        if context_key in vector_stores:
            return vector_stores[context_key]

        chroma_path = os.path.join(data_dir, run_id, content_type)
        collection_name = f"{run_id}_{content_type}"
        vector_store = get_vector_store(db_path=chroma_path, collection_name=collection_name)
        
        vector_stores[context_key] = vector_store
        return vector_store


def get_story_kg_store(run_id: str, content_type: str) -> KuzuGraphStore:
    context_key = f"{run_id}_{content_type}"
    if context_key in graph_stores:
        return graph_stores[context_key]

    with _storage_lock:
        if context_key in graph_stores:
            return graph_stores[context_key]

        kuzu_db_path = os.path.join(data_dir, run_id, content_type)
        graph_store = get_kg_store(db_path=kuzu_db_path)
        
        graph_stores[context_key] = graph_store
        return graph_store


