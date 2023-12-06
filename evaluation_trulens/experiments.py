import pathlib
import time
import numpy as np
import litellm

from langchain.llms import VertexAI
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.litellm import LiteLLM
from typing import List, Tuple, Iterable

import helper.data
from helper import vertex_ai as vai_util


litellm.set_verbose = False


def get_lite_llm_provider() -> LiteLLM:
    """
    Get an instance of ChatVertexAI via LiteLLM.

    :return: The LLM provider
    """

    return LiteLLM(model_engine='chat-bison-32k', max_output_tokens=2048)


def load_questions(file_name: str) -> List[str]:
    """
    Read the questions from a file. Each line contains one questions.

    :param file_name: The input file name
    :return: The list of questions
    """

    with open(file_name, 'r') as in_file:
        return in_file.readlines()


def experiment_with_chunks(chunk_size_overlap=Iterable[List[Tuple[int]]]) -> None:
    """
    Run experiments with different chunk sizes and overlaps.

    :param chunk_size_overlap: A list of (chunk size, overlap) values
    """

    questions = load_questions('questions.txt')

    tru_llm = LiteLLM(model_engine='chat-bison-32k')
    grounded = Groundedness(groundedness_provider=tru_llm)

    # Define a groundedness feedback function
    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(tru_llm.relevance).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(tru_llm.qs_relevance).on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    llm = VertexAI(
        model='text-bison',
        temperature=0,
        additional_kwargs=vai_util.VERTEX_AI_LLM_PARAMS
    )
    embeddings = vai_util.CustomVertexAIEmbeddings()

    n_configs = len(chunk_size_overlap)
    # https://cloud.google.com/vertex-ai/docs/quotas#generative-ai
    rate_limiter = vai_util.rate_limit(60)

    for idx, a_config in enumerate(chunk_size_overlap):
        chunk_size, chunk_overlap = a_config
        print(f'Config {idx + 1} of {n_configs}: {chunk_size=}, {chunk_overlap=}')

        service_context = ServiceContext.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            llm=llm,
            embed_model=embeddings
        )

        documents = SimpleDirectoryReader(
            input_files=['../data/TR-61850.pdf']
        ).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )

        app_id = f'RAG2Rich_LlamaIndex_App_{idx + 1}'
        query_engine = index.as_query_engine()
        tru_query_engine_recorder = TruLlama(
            query_engine,
            app_id=app_id,
            feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
        )

        with tru_query_engine_recorder as _:
            print('Running queries...')
            for a_question in questions:
                start_time = time.perf_counter()
                response = query_engine.query(a_question)
                end_time = time.perf_counter()
                print('Response:', response)
                print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')
                next(rate_limiter)
                time.sleep(3)


def experiment_with_summary_index(chunk_size_overlap=Iterable[tuple[int]]) -> None:
    """
    Run experiments with summary index using the optimal chunk size and overlap.

    :param chunk_size_overlap: The optimal (chunk size, overlap) values
    """

    questions = load_questions('questions.txt')

    tru_llm = LiteLLM(model_engine='chat-bison-32k')
    grounded = Groundedness(groundedness_provider=tru_llm)

    # Define a groundedness feedback function
    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(tru_llm.relevance).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(tru_llm.qs_relevance).on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    llm = VertexAI(
        model='text-bison',
        temperature=0,
        additional_kwargs=vai_util.VERTEX_AI_LLM_PARAMS
    )
    embeddings = vai_util.CustomVertexAIEmbeddings()

    # https://cloud.google.com/vertex-ai/docs/quotas#generative-ai
    rate_limiter = vai_util.rate_limit(60)

    chunk_size, chunk_overlap = chunk_size_overlap
    service_context = ServiceContext.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        llm=llm,
        embed_model=embeddings
    )

    documents = SimpleDirectoryReader(
        input_files=['../data/TR-61850.pdf']
    ).load_data()

    index = SummaryIndex.from_documents(
        documents,
        service_context=service_context
    )

    app_id = f'RAG2Rich_LlamaIndex_App_SummaryIndex'
    query_engine = index.as_query_engine()
    tru_query_engine_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
    )

    with tru_query_engine_recorder as _:
        print('Running queries...')
        for a_question in questions:
            start_time = time.perf_counter()
            response = query_engine.query(a_question)
            end_time = time.perf_counter()
            print('Response:', response)
            print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')
            next(rate_limiter)
            time.sleep(3)


def experiment_with_top_k(top_k=Iterable[List[int]], similarity_cutoff=Iterable[List[float]]) -> None:
    """
    Run experiments with different top-k values and similarity cutoff for vector search.

    :param top_k: A list of top-k values for vector search
    :param similarity_cutoff: Threshold for ignoring nodes
    """

    questions = load_questions('questions.txt')

    tru_llm = get_lite_llm_provider()
    grounded = Groundedness(groundedness_provider=tru_llm)

    # Define a groundedness feedback function
    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(tru_llm.relevance).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(tru_llm.qs_relevance).on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    n_configs = len(top_k) * len(similarity_cutoff)
    idx = 1
    rate_limiter = vai_util.rate_limit(60)

    service_context = helper.data.get_service_context(chunk_size=500, chunk_overlap=100)
    documents = SimpleDirectoryReader(
        input_files=['../data/TR-61850.pdf']
    ).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    the_runs = []

    for k in top_k:
        for c in similarity_cutoff:
            print(f'Config {idx} of {n_configs}: {k=}, {c=}')
            the_runs.append((k, c))

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
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=c)],
            )

            app_id = f'RAG2Rich_Exp_top_k={k}_cutoff={c}'
            tru_query_engine_recorder = TruLlama(
                query_engine,
                app_id=app_id,
                feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
            )

            with tru_query_engine_recorder as _:
                print('Running queries...')
                for a_question in questions:
                    start_time = time.perf_counter()
                    response = query_engine.query(a_question)
                    end_time = time.perf_counter()
                    print('Response:', response)
                    print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')
                    next(rate_limiter)
                    time.sleep(4)

            time.sleep(4)

    print('The runs are:')
    for idx, run in enumerate(the_runs):
        print(f'Index: {idx}:: {run}')


if __name__ == '__main__':
    tru = Tru()
    tru.start_dashboard(
        # force=True,  # Not supported on Windows
        _dev=pathlib.Path().cwd().parent.parent.resolve()
    )

    # If needed, you can reset the trulens_eval dashboard database
    # tru.reset_database()

    config_values = [
        (512, 50),
        (512, 100),  # Found to be optimal
        (768, 50),
        (768, 100),
        (1024, 50),
        (1024, 100)
    ]

    # Uncomment to run the experiment
    # experiment_with_chunks(chunk_size_overlap=config_values)

    # Does not perform well
    # experiment_with_summary_index((512, 100))

    # The optimal top-k value is 6 without similarity cutoff
    # A positive value of similarity cutoff degraded the scores
    # experiment_with_top_k(top_k=[2, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])
    # experiment_with_top_k(top_k=[4, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])
    # experiment_with_top_k(top_k=[6, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])
    # experiment_with_top_k(top_k=[8, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])

