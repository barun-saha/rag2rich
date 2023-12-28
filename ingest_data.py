import os
import requests

from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from pathlib import Path

import helper.data as dhelp


load_dotenv()

LOCAL_FILE_NAME = 'data/TR-61850.pdf'
INDEX_STORAGE_DIR = './storage'


def build_index() -> VectorStoreIndex:
    """
    Load data into a vector store and build the index.

    :return: The vector store index
    """

    # Using the previously identified optimal settings
    service_context = dhelp.get_service_context(chunk_size=512, chunk_overlap=75)

    try:
        print('Loading index from storage...')
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
        # load index
        index = load_index_from_storage(
            storage_context,
            service_context=service_context
        )
    except Exception as ex:
        print(f'Building index from scratch because of exception: {ex}')

        if not os.path.exists(LOCAL_FILE_NAME) or os.path.isfile(LOCAL_FILE_NAME):
            filename = Path(LOCAL_FILE_NAME)
            url = 'https://www.fit.vut.cz/research/publication-file/11832/TR-61850.pdf'
            response = requests.get(url)
            filename.write_bytes(response.content)

        documents = SimpleDirectoryReader(
            input_files=[LOCAL_FILE_NAME]
        ).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            show_progress=True
        )
        index.storage_context.persist(INDEX_STORAGE_DIR)

    return index


if __name__ == '__main__':
    build_index()
