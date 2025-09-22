import nest_asyncio
nest_asyncio.apply()


import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"


from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()


from pathlib import Path
import kuzu
DB_NAME = "ex.kuzu"
Path(DB_NAME).unlink(missing_ok=True)
db = kuzu.Database(DB_NAME)


from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
extract_llm = OpenAI(model="gpt-4.1-mini", temperature=0.0)
generate_llm = OpenAI(model="gpt-4.1-mini", temperature=0.3)


from typing import Literal
entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]
# Define the relationship schema that we will pass to our graph store
# This must be a list of valid triples in the form (head_entity, relation, tail_entity)
validation_schema = [
    ("ORGANIZATION", "HAS", "PERSON"),
    ("PERSON", "WORKED_AT", "ORGANIZATION"),
    ("PERSON", "WORKED_WITH", "PERSON"),
    ("PERSON", "WORKED_ON", "ORGANIZATION"),
    ("PERSON", "PART_OF", "ORGANIZATION"),
    ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
    ("PERSON", "WORKED_AT", "PLACE"),
]


from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
graph_store = KuzuPropertyGraphStore(
    db,
    has_structured_schema=True,
    relationship_schema=validation_schema,
    use_vector_index=True,  # Enable vector index for similarity search
    embed_model=embed_model,  # Auto-detects embedding dimension from model
)

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=extract_llm,
            possible_entities=entities,
            possible_relations=relations,
            kg_validation_schema=validation_schema,
            strict=True,  # if false, will allow triples outside of the schema
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

# Switch to the generate LLM during retrieval
Settings.llm = generate_llm

query_text = "Tell me more about Interleaf and Viaweb?"
query_engine = index.as_query_engine(include_text=False)

response = query_engine.query(query_text)
print(str(response))

retriever = index.as_retriever(include_text=False)
nodes = retriever.retrieve(query_text)
nodes[0].text


from llama_index.core.vector_stores.types import VectorStoreQuery

query_text = "How much funding did Idelle Weber provide to Viaweb?"
query_embedding = embed_model.get_text_embedding(query_text)
# Perform direct vector search on the graph store
vector_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5
)

nodes, similarities = graph_store.vector_query(vector_query)

for i, (node, similarity) in enumerate(zip(nodes, similarities)):
    print(f"  {i + 1}. Similarity: {similarity:.3f}")
    print(f"     Text: {node.text}...")
    print()


from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.indices.property_graph import LLMSynonymRetriever
from typing import List


class GraphVectorRetriever(BaseRetriever):
    """
    A retriever that performs vector search on a property graph store.
    """

    def __init__(self, graph_store, embed_model, similarity_top_k: int = 5):
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get query embedding
        query_embedding = self.embed_model.get_text_embedding(
            query_bundle.query_str
        )

        # Perform vector search
        vector_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
        )
        nodes, similarities = self.graph_store.vector_query(vector_query)

        # Convert ChunkNodes to TextNodes
        nodes_with_scores = []
        for node, similarity in zip(nodes, similarities):
            # Convert ChunkNode to TextNode
            if hasattr(node, "text"):
                text_node = TextNode(
                    text=node.text,
                    id_=node.id,
                    metadata=getattr(node, "properties", {}),
                )
                nodes_with_scores.append(
                    NodeWithScore(node=text_node, score=similarity)
                )

        return nodes_with_scores


class CombinedGraphRetriever(BaseRetriever):
    """
    A retriever that performs that combines graph and vector search on a property graph store.
    """

    def __init__(
        self, graph_store, embed_model, llm, similarity_top_k: int = 5
    ):
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 1. Vector retrieval
        query_embedding = self.embed_model.get_text_embedding(
            query_bundle.query_str
        )
        vector_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
        )
        vector_nodes, similarities = self.graph_store.vector_query(
            vector_query
        )

        # Convert ChunkNodes to TextNodes for vector results
        vector_results = []
        for node, similarity in zip(vector_nodes, similarities):
            if hasattr(node, "text"):
                text_node = TextNode(
                    text=node.text,
                    id_=node.id,
                    metadata=getattr(node, "properties", {}),
                )
                vector_results.append(
                    NodeWithScore(node=text_node, score=similarity)
                )

        # 2. Graph traversal retrieval
        graph_retriever = LLMSynonymRetriever(
            self.graph_store, llm=self.llm, include_text=True
        )
        graph_results = graph_retriever.retrieve(query_bundle)

        # 3. Combine and deduplicate
        all_results = vector_results + graph_results
        seen_nodes = set()
        combined_results = []

        for node_with_score in all_results:
            node_id = node_with_score.node.node_id
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                combined_results.append(node_with_score)

        return combined_results


# Use the combined retriever
combined_retriever = CombinedGraphRetriever(
    graph_store=graph_store,
    llm=generate_llm,
    embed_model=embed_model,
    similarity_top_k=5,
)

# Test the combined retriever
query_text = "What was the role of Idelle Weber in Viaweb?"
query_bundle = QueryBundle(query_str=query_text)
results = combined_retriever.retrieve(query_bundle)
for i, node_with_score in enumerate(results):
    print(f"{i + 1}. Score: {node_with_score.score:.3f}")
    print(
        f"   Text: {node_with_score.node.text[:100]}..."
    )  # Print first 100 chars
    print(f"   Node ID: {node_with_score.node.node_id}")
    print()


from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# Create query engine with your combined retriever
query_engine = RetrieverQueryEngine.from_args(
    retriever=combined_retriever,
    llm=generate_llm,
)

# Create response synthesizer
response_synthesizer = get_response_synthesizer(
    llm=generate_llm, use_async=False
)

# Create query engine
query_engine = RetrieverQueryEngine(
    retriever=combined_retriever, response_synthesizer=response_synthesizer
)

# Query and get answer
query_text = "What was the role of Idelle Weber in Viaweb?"
response = query_engine.query(query_text)
print(response)



