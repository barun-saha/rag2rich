import os

from llama_index import ServiceContext, get_response_synthesizer, VectorStoreIndex
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores import MilvusVectorStore

from helper.vertex_ai import CustomVertexAIEmbeddings, get_llm


def get_service_context(chunk_size: int, chunk_overlap: int) -> ServiceContext:
    """
    Create a LlamaIndex service context with ChatVertexAI as the LLM.

    :param chunk_size: The chunk size
    :param chunk_overlap: The chunk overlap
    :return: The service context
    """

    return ServiceContext.from_defaults(
        chunk_size=chunk_size,  # Based on optimal richness
        chunk_overlap=chunk_overlap,
        llm=get_llm(),
        embed_model=CustomVertexAIEmbeddings()
    )


def get_vector_store(overwrite: bool) -> MilvusVectorStore:
    """
    Return an instance of MilvusVectorStore on Zilliz cloud.

    :param overwrite: Whether or not to overwrite the existing collection
    :return: The vector store
    """

    return MilvusVectorStore(
        uri=os.environ['ZILLIZ_URI'],
        token=os.environ['ZILLIZ_TOKEN'],
        collection_name=os.environ['ZILLIZ_COLLECTION_NAME'],
        similarity_metric="L2",
        dim=int(os.environ['ZILLIZ_DIMENSION']),
        overwrite=overwrite
    )


def get_top_k_query_engine(k: int, chunk_size: int, chunk_overlap: int) -> RetrieverQueryEngine:
    vector_store = get_vector_store(overwrite=False)
    service_context = get_service_context(chunk_size, chunk_overlap)
    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    # Configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=k,
        service_context=service_context
    )

    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer(service_context=service_context)

    # Assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    return query_engine

