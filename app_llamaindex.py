import os.path
import chainlit as cl
import time
import requests

from pathlib import Path
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index import (
    StorageContext,
    load_index_from_storage,
)
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.chat_models import ChatVertexAI

from vai_util import VERTEX_AI_LLM_PARAMS, CustomVertexAIEmbeddings


LOCAL_FILE_NAME = 'data/TR-61850.pdf'
INDEX_STORAGE_DIR = './storage'

llm = ChatVertexAI(
    model='chat-bison-32k',
    temperature=0,
    additional_kwargs=VERTEX_AI_LLM_PARAMS
)
service_context = ServiceContext.from_defaults(
    chunk_size=512,  # Based on optimal richness
    chunk_overlap=100,
    llm=llm,
    embed_model=CustomVertexAIEmbeddings()
)

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


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content='Hello there!',
    ).send()

    query_engine = index.as_query_engine(streaming=True, service_context=service_context)
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
    print('Goodbye', cl.user_session.get("id"))


