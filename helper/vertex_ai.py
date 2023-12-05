import time
from typing import List

from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.pydantic_v1 import BaseModel


VERTEX_AI_LLM_PARAMS = {
    'temperature': 0.0,
    'max_output_tokens': 8192,
    'verbose': True,
    'model_name': 'chat-bison-32k',
    "top_p": 0.8,
    "top_k": 40
}

EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5


def rate_limit(max_per_minute):
    """
    Utility functions for Embeddings API with rate limiting.

    :param max_per_minute: The number of requests to issue per minute.
    """

    period = 60 / max_per_minute
    print('Waiting to clear the rate limit...', end='')
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print('.', end='')
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int = EMBEDDING_QPM
    num_instances_per_batch: int = EMBEDDING_NUM_BATCH

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]


def get_llm() -> ChatVertexAI:
    """
    Return an instance of the chat-bison model.

    :return: The LLM
    """

    return ChatVertexAI(
        model='chat-bison-32k',
        temperature=0,
        max_output_tokens=2048,
        additional_kwargs=VERTEX_AI_LLM_PARAMS
    )

