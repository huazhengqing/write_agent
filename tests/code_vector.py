



from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    BasePydanticVectorStore,
)

class VectorStoreQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    TEXT_SEARCH = "text_search"
    SEMANTIC_HYBRID = "semantic_hybrid"

    # fit learners
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"

    # maximum marginal relevance
    MMR = "mmr"



















