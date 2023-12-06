import chainlit as cl
import time
import llama_index

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

import helper.data as dhelp


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content='Hello there!',
    ).send()

    # index = load_index()
    # query_engine = index.as_query_engine(
    #     streaming=True,
    #     service_context=dhelp.get_service_context(chunk_size=512, chunk_overlap=100)
    # )
    query_engine = dhelp.get_top_k_query_engine(k=6, chunk_size=512, chunk_overlap=100)
    cl.user_session.set('query_engine', query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get('query_engine')  # type: RetrieverQueryEngine
    start_time = time.perf_counter()
    response = await cl.make_async(query_engine.query)(message.content)  # type: llama_index.response.schema.Response
    end_time = time.perf_counter()
    print(f'Computation time: {end_time - start_time} s')

    answer = response.response
    source_documents = response.source_nodes  # type: list[llama_index.schema.NodeWithScore]
    text_elements = []  # type: list[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f'(p. {int(source_doc.metadata["page_label"])})'
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.text, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    ui_msg = cl.Message(
        content=answer,
        elements=text_elements,
    )

    await ui_msg.send()


@cl.on_chat_end
def end():
    print('Goodbye!', cl.user_session.get('id'))


