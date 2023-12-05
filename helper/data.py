import os

from llama_index import ServiceContext
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
