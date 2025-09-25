
from pathlib import Path
from functools import lru_cache
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.response_synthesizers import TreeSummarize
from utils.llm_api import llm_temperatures, get_llm_params, get_embedding_params
from utils.llm_api import llm_temperatures, get_llm_params
from rag.vector_prompts import tree_summary_prompt



from llama_index.vector_stores.chroma import ChromaVectorStore
ChromaVectorStore.model_config['extra'] = 'allow'
if hasattr(ChromaVectorStore, 'model_rebuild'):
    ChromaVectorStore.model_rebuild(force=True)



@lru_cache(maxsize=None)
def init_llama_settings():
    llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["summarization"])
    Settings.llm = LiteLLM(**llm_params)

    Settings.prompt_helper = PromptHelper(
        context_window=llm_params.get('context_window', 8192),
        num_output=llm_params.get('max_tokens', 2048),
        chunk_overlap_ratio=0.2,
    )
    embedding_params = get_embedding_params()
    embed_model_name = embedding_params.pop('model')
    Settings.embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)



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
    synthesis_llm_params = get_llm_params(llm_group="summary", temperature=llm_temperatures["synthesis"])
    synthesizer = TreeSummarize(
        llm=LiteLLM(**synthesis_llm_params),
        summary_template=PromptTemplate(tree_summary_prompt),
        prompt_helper = PromptHelper(
            context_window=synthesis_llm_params.get('context_window', 8192),
            num_output=synthesis_llm_params.get('max_tokens', 2048),
            chunk_overlap_ratio=0.2,
        ),
        use_async=True,
    )
    return synthesizer
