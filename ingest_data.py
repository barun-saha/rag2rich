import os
import requests
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from pathlib import Path

import helper.data as dhelp


load_dotenv()

LOCAL_FILE_NAME = 'data/TR-61850.pdf'


def build_index() -> VectorStoreIndex:
    """
    Load data into Milvus vector store on Zilliz and build the index.

    :return: The vector store index
    """

    print('Building index from scratch...')

    if not os.path.exists(LOCAL_FILE_NAME) or not os.path.isfile(LOCAL_FILE_NAME):
        print('Downloading file...')
        filename = Path(LOCAL_FILE_NAME)
        url = 'https://www.fit.vut.cz/research/publication-file/11832/TR-61850.pdf'
        response = requests.get(url)
        filename.write_bytes(response.content)

    documents = SimpleDirectoryReader(
        input_files=[LOCAL_FILE_NAME]
    ).load_data()

    print('Indexing data...')
    # Milvus index automatically persists:
    # https://github.com/run-llama/llama_index/issues/6754#issuecomment-1624037933
    vector_store = dhelp.get_vector_store(overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = dhelp.get_service_context(chunk_size=512, chunk_overlap=100)
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True
    )

    return index


if __name__ == '__main__':
    build_index()
