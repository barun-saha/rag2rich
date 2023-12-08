import chainlit as cl
import time
import llama_index

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

import helper.data as dhelp


@cl.on_chat_start
async def on_chat_start():
    welcome_message = (
        'Hello there!'
        ' I am here to answer your questions based on the'
        ' [Description of IEC 61850 Communication](https://www.fit.vut.cz/research/publication-file/11832/TR-61850.pdf).'
        ' Try out one of these sample questions or ask something different:'
        '\n\n'
        '- Summarize GOOSE'
        '\n\n'
        '- Write an email to a prospective customer highlighting the importance and features of 61850 communication.'
        '\n\n'
        '- Show the differences between GOOSE and MMS.'
    )
    await cl.Message(
        content=welcome_message,
        disable_human_feedback=True
    ).send()

    try:
        query_engine = dhelp.get_top_k_query_engine(k=6, chunk_size=512, chunk_overlap=100)
        cl.user_session.set('query_engine', query_engine)
    except Exception as ex:
        message = f'*** An error occurred while trying to load the application: {ex}'
        print(message)
        await cl.ErrorMessage(
            content=message,
            author='Error'
        ).send()


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


