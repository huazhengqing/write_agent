
import os
from pathlib import Path
from functools import lru_cache
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.response_synthesizers import TreeSummarize
from rag.vector_prompts import tree_summary_prompt



from llama_index.vector_stores.chroma import ChromaVectorStore
ChromaVectorStore.model_config['extra'] = 'allow'
if hasattr(ChromaVectorStore, 'model_rebuild'):
    ChromaVectorStore.model_rebuild(force=True)



@lru_cache(maxsize=None)
def init_llama_settings():
    Settings.llm = LiteLLM(
        model="openai/summary",
        temperature=0.2,
        max_tokens=None,
            max_retries=10,
        api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        api_base=os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
    )

    Settings.prompt_helper = PromptHelper(
        # context_window=8192, # 默认值
        # num_output=2048, # 默认值
        chunk_overlap_ratio=0.2,
    )
    Settings.embed_model = LiteLLMEmbedding(
        model_name="openai/embedding",
        api_base=os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
        api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
    )



init_llama_settings()



@lru_cache(maxsize=None)
def get_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    db_path_obj = Path(db_path)
    db_path_obj.mkdir(parents=True, exist_ok=True)

    import chromadb
    db = chromadb.PersistentClient(path=str(db_path_obj))
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    store_cache_path = db_path_obj / f"{collection_name}.cache.db"
    from diskcache import Cache  
    vector_store.cache = Cache(str(store_cache_path), size_limit=int(32 * (1024**2)))
    
    return vector_store



@lru_cache(maxsize=None)
def get_synthesizer():
    synthesizer = TreeSummarize(
        llm=LiteLLM(
            model="openai/summary",
            temperature=0.4,
            max_tokens=None,
                    max_retries=10,
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
                api_base=os.getenv("LITELLM_PROXY_URL", "http://0.0.0.0:4000"),
        ),
        summary_template=PromptTemplate(tree_summary_prompt),
        prompt_helper = PromptHelper(
            context_window=8192,
            num_output=2048,
            chunk_overlap_ratio=0.2,
        ),
        use_async=True,
    )
    return synthesizer
