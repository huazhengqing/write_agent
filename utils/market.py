import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from utils.llm import get_embedding_params


story_platforms_cn = ["番茄小说", "起点中文网", "飞卢小说网", "晋江文学城", "七猫免费小说", "纵横中文网", "17K小说网", "刺猬猫", "掌阅"]
story_platforms_en = []

story_input_dir = Path(".input")
story_input_dir.mkdir(parents=True, exist_ok=True)

story_output_dir = Path(".story/")
story_output_dir.mkdir(parents=True, exist_ok=True)

story_market_chroma_dir = ".chroma_db/story"
story_market_chroma_dir.mkdir(parents=True, exist_ok=True)

story_market_chroma_collection_name = "market"

embedding_params = get_embedding_params(embedding='bge-m3')
embed_model_name = embedding_params.pop('model')
embed_model = LiteLLMEmbedding(model_name=embed_model_name, **embedding_params)

db = chromadb.PersistentClient(path=str(story_market_chroma_dir))
chroma_collection = db.get_or_create_collection(story_market_chroma_collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

service_context = ServiceContext.from_defaults(embed_model=embed_model)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


