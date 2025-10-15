from functools import lru_cache
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.kuzu.kuzu_property_graph import KuzuPropertyGraphStore
from utils.file import data_dir



@lru_cache(maxsize=None)
def get_vector(run_id: str, content_type: str) -> ChromaVectorStore:
    chroma_path = data_dir / run_id / content_type
    
    # ChromaDB 的 collection 名称有严格限制, 不能包含中文等特殊字符。
    # 我们在这里专门为 collection 名称进行一次清理, 而不影响包含中文的 run_id。
    import re
    sanitized_run_id = re.sub(r'[^a-zA-Z0-9._-]', '_', run_id).strip('._')
    collection_name = f"{sanitized_run_id}_{content_type}"

    from rag.vector import get_vector_store
    vector_store = get_vector_store(db_path=str(chroma_path), collection_name=collection_name)
    return vector_store



@lru_cache(maxsize=None)
def get_kg(run_id: str, content_type: str) -> KuzuPropertyGraphStore:
    kuzu_db_path = data_dir / run_id / content_type / "kuzu_db"
    from rag.kg import get_kg_store
    graph_store = get_kg_store(db_path=str(kuzu_db_path))
    return graph_store


