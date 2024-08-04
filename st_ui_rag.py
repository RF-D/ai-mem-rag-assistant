import os
import tempfile
import streamlit as st
from typing import List, Tuple, Dict
from functools import lru_cache
from operator import itemgetter
from dotenv import load_dotenv
from utils.llm_manager import LLMManager

# Langchain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import format_document

# Vector store and embeddings
from langchain_pinecone import PineconeVectorStore
from tools.voyage_embeddings import vo_embed

# Custom tools and loaders
from tools.doc_loader import load_documents
from tools.retriever_tools import retriever_tool_meta

from tools.text_splitter import split_md, split_text
from scrape_sitemap import scrape_sitemap


# Components
from components.rag_sidebar import (
    setup_sidebar,
    file_uploader,
    index_name_for_sitemap_scraper,
    crawl_parameters,
    youtube_chat_submit,
)
from components.streamlit_app_initializer import initialize_streamlit_app

load_dotenv()


LLMManager.initialize_ollama_models()
MAX_HISTORY_TOKENS = LLMManager.MAX_HISTORY_TOKENS


# Initialize the Streamlit app and get the sidebar configuration
sidebar_config = initialize_streamlit_app()


@st.cache_resource(ttl=3600)
def load_vectorstore(index_name):
    embeddings = vo_embed()
    return PineconeVectorStore.from_existing_index(
        embedding=embeddings, index_name=index_name
    )


@lru_cache(maxsize=1)
def load_retriever():
    return retriever_tool_meta(vectorstore)


@lru_cache(maxsize=1)
def load_condense_question_prompt():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    return PromptTemplate.from_template(_template)


@lru_cache(maxsize=1)
def load_answer_prompt():
    template = """Here is the context you have access to:
<context>
{context}
</context>

If the context provided is sufficient to answer the question, use it to formulate your response. If the context is insufficient or empty, you should rely on your own knowledge to answer the question. If you cannot answer the question based on the context or your own knowledge, indicate what additional information would be needed to provide a complete answer.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )


# Load LLMs
chain_llm = LLMManager.load_llm(
    sidebar_config.chain_provider, sidebar_config.chain_model
)
search_query_llm = LLMManager.load_llm(
    sidebar_config.search_query_provider, sidebar_config.search_query_model
)


# Setup VectorDB
vectorstore = load_vectorstore(sidebar_config.pinecone_index_name)


# Setup retriever
retriever = load_retriever()


# RAG setup
CONDENSE_QUESTION_PROMPT = load_condense_question_prompt()


ANSWER_PROMPT = load_answer_prompt()

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [
        f"{format_document(doc, document_prompt)}\nSource: {doc.metadata.get('source', 'Unknown')}"
        for doc in docs
    ]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for message in chat_history:
        if message["role"] == "user":
            buffer.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            buffer.append(AIMessage(content=message["content"]))
    return buffer


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(
                x["chat_history"][:-1]
                if x["chat_history"]
                and x["chat_history"][-1]["content"] == x["question"]
                else x["chat_history"]
            ),
            question=lambda x: x["question"],
        )
        | CONDENSE_QUESTION_PROMPT
        | search_query_llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(
            x["chat_history"][:-1]
            if x["chat_history"] and x["chat_history"][-1]["content"] == x["question"]
            else x["chat_history"]
        ),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = _inputs | ANSWER_PROMPT | chain_llm | StrOutputParser(verbose=True)


# Load and cache the embedding model for vector embeddings.
@st.cache_resource
def load_embedding_model():
    return vo_embed()


embeddings = load_embedding_model()


# Create a sidebar
sidebar = st.sidebar


split_result = None


# Show crawl parameters only when "Crawl" is selected
if sidebar_config.selected_function == "Crawl":
    crawl_parameters()

# Show the input field for index name only if the selected function is "Sitemap Scraper"
index_name_for_sitemap_scraper(sidebar_config.selected_function)


# SUBMIT URL per chosen function
if st.sidebar.button("URL Submit", key="url_submit"):
    if sidebar_config.url:
        if sidebar_config.selected_function == "Sitemap Scraper":
            try:
                # Display a progress bar in the sidebar while the function is running
                progress_bar = st.sidebar.progress(0)

                def progress_callback(current, total):
                    progress_bar.progress(current / total)

                scrape_sitemap(
                    sidebar_config.url, st.session_state.index_name, progress_callback
                )
                st.sidebar.success("Sitemap scraped and results embedded successfully!")
            except Exception as e:
                st.sidebar.error(f"Sitemap scraping and embedding failed: {str(e)}")
                st.sidebar.error("Please check the error message and try again.")
        elif sidebar_config.selected_function == "YouTube Chat":
            youtube_chat_submit()
            # Store the split_result in session state
            st.session_state.split_result = split_result
        elif sidebar_config.selected_function == "Crawl":
            fn_result = sidebar_config.functions[sidebar_config.selected_function](
                sidebar_config.url, params=st.session_state.crawl_params
            )
            if fn_result is not None:
                split_result = split_md(fn_result)
                st.session_state.split_result = split_result
            else:
                st.error(
                    "An error occurred during crawling. Please check the logs for more information."
                )
        else:
            # Call the selected function with the provided URL and split the result
            fn_result = sidebar_config.functions[sidebar_config.selected_function](
                sidebar_config.url
            )
            split_result = split_md(fn_result)
            if split_result:
                st.session_state.split_result = split_result
                st.success(
                    f"{sidebar_config.selected_function} completed successfully!"
                )
            else:
                st.warning(
                    f"{sidebar_config.selected_function} completed, but no results were found."
                )
    else:
        st.warning("Please enter a valid URL.")

# Check if split_result exists in the session state and the selected function is not "Sitemap Scraper"
if (
    "split_result" in st.session_state
    and sidebar_config.selected_function != "Sitemap Scraper"
):
    split_result = st.session_state.split_result

    # Show the input field for index name only if split_result is not empty
    if split_result:
        # Display the index name input field and update the session state
        st.session_state.index_name = st.sidebar.text_input(
            "Index Name", value=st.session_state.index_name
        )

        if st.sidebar.button("Add to Memory", key="add_to_memory"):
            try:
                embeddings = vo_embed()
                PineconeVectorStore.from_documents(
                    documents=split_result,
                    embedding=embeddings,
                    index_name=st.session_state.index_name,
                )
                st.sidebar.success("Embedding completed successfully!")
            except Exception as e:
                st.sidebar.error(f"Embedding failed: {str(e)}")
                st.sidebar.error("Please check the error message and try again.")
    else:
        st.warning("No results to embed.")

# Upload File or Text documents
st.cache_data()
file_uploader()

# Reset chat history button
if st.sidebar.button("Reset Chat History"):
    st.session_state.messages = []


@st.cache_resource(ttl=3600)
def trim_chat_history(
    messages: List[Dict[str, str]], max_tokens: int = 8000
) -> Tuple[List[Dict[str, str]], int]:
    trimmed_messages = []
    total_prompt_tokens = 0
    user_message_found = False

    for message in reversed(messages):
        message_tokens = LLMManager.count_tokens(
            message["content"],
            sidebar_config.chain_provider,
            sidebar_config.chain_model,
        )

        if total_prompt_tokens + message_tokens > max_tokens and user_message_found:
            break

        total_prompt_tokens += message_tokens
        trimmed_messages.insert(0, message)

        if message["role"] == "user":
            user_message_found = True

    # Ensure at least one user message is included
    if not user_message_found and messages:
        for message in messages:
            if message["role"] == "user":
                if trimmed_messages:
                    trimmed_messages.insert(0, message)
                else:
                    trimmed_messages.append(message)
                total_prompt_tokens += LLMManager.count_tokens(
                    message["content"],
                    sidebar_config.chain_provider,
                    sidebar_config.chain_model,
                )
                break

    # If still no user message, add a dummy user message
    if not trimmed_messages or trimmed_messages[0]["role"] != "user":
        dummy_message = {"role": "user", "content": "Start of conversation"}
        trimmed_messages.insert(0, dummy_message)
        total_prompt_tokens += LLMManager.count_tokens(
            dummy_message["content"],
            sidebar_config.chain_provider,
            sidebar_config.chain_model,
        )

    return trimmed_messages, total_prompt_tokens


calculate_total_tokens = LLMManager.calculate_total_tokens

# Initialize chat history
chat_history = []


# Initialize token counters
update_token_count = LLMManager.update_token_count

st.title("Persistent Memory Conversational Agent")


for message in st.session_state.messages:
    avatar = (
        "utils/images/user_avatar.png"
        if message["role"] == "user"
        else "utils/images/queryqueen.png"
    )
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


user_input = st.chat_input("Write something here...", key="input")

# Display the scrape result below the user input
if split_result:
    with st.expander("Scrape Result", expanded=False):
        st.write(split_result)

# Chat Container
if user_input:
    # Display user input in chat message container
    with st.chat_message("user", avatar="utils/images/user_avatar.png"):
        st.text(user_input)

    # Append to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Trim chat history before sending to the model
    trimmed_history, history_tokens = trim_chat_history(
        st.session_state.messages, MAX_HISTORY_TOKENS
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="utils/images/queryqueen.png"):
        loading_message = st.empty()
        # Display "Thinking..." message
        loading_message.markdown("Thinking...")

        result = ""
        for token in chain.invoke(
            input={"question": user_input, "chat_history": trimmed_history}
        ):
            result += token
            loading_message.markdown(result)

    # Calculate prompt tokens
    prompt_tokens = LLMManager.count_tokens(
        str({"question": user_input, "chat_history": trimmed_history}),
        sidebar_config.chain_provider,
        sidebar_config.chain_model,
    )

    # Calculate completion tokens
    completion_tokens = LLMManager.count_tokens(
        result, sidebar_config.chain_provider, sidebar_config.chain_model
    )

    # Update token counts
    LLMManager.update_token_count(st.session_state, prompt_tokens, completion_tokens)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})

    # Trim chat history if needed, based on total tokens
    if st.session_state.total_tokens > LLMManager.MAX_HISTORY_TOKENS:
        st.session_state.messages, new_prompt_tokens = trim_chat_history(
            st.session_state.messages, LLMManager.MAX_HISTORY_TOKENS
        )
        LLMManager.reset_token_count(st.session_state, new_prompt_tokens)

    # Display updated token counts
    LLMManager.display_token_counts(st.sidebar, st.session_state)
