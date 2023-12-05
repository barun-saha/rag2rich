import os.path
import chainlit as cl
import time
import pymilvus as mil
import requests

from pathlib import Path
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index import (
    StorageContext,
)
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.chat_models import ChatVertexAI
from llama_index.vector_stores import MilvusVectorStore
from dotenv import load_dotenv

import helper.data as dhelp
import helper.vertex_ai as vai


def load_index() -> VectorStoreIndex:
    """
    Load an existing index from Zilliz cloud datastore.

    :return: The vector store index
    """

    print('Loading Milvus index from Zilliz cloud...')
    vector_store = dhelp.get_vector_store(overwrite=False)
    service_context = dhelp.get_service_context(chunk_size=512, chunk_overlap=100)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    return index


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content='Hello there!',
    ).send()

    index = load_index()
    query_engine = index.as_query_engine(
        streaming=True,
        service_context=dhelp.get_service_context(chunk_size=512, chunk_overlap=100)
    )
    cl.user_session.set('query_engine', query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get('query_engine')  # type: RetrieverQueryEngine
    start_time = time.perf_counter()
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    end_time = time.perf_counter()
    print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()


@cl.on_chat_end
def end():
    print('Goodbye!', cl.user_session.get("id"))


