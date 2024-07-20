
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
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import format_document

# Vector store and embeddings
from langchain_pinecone import PineconeVectorStore
from tools.voyage_embeddings import vo_embed

# Custom tools and loaders
from tools.doc_loader import load_documents
from tools.retriever_tools import retriever_tool_meta
from tools.firecrawl_scrape_loader import scrape
from tools.text_splitter import split_md, split_text
from tools.firecrawl_crawl_loader import crawl
from scrape_sitemap import scrape_sitemap
from tools.youtube_chat import youtube_chat

load_dotenv()


LLMManager.initialize_ollama_models()
MAX_HISTORY_TOKENS = LLMManager.MAX_HISTORY_TOKENS


# Streamlit app

# Set page configuration
st.set_page_config(page_title="AI MEM", page_icon=":guardsman:", layout="wide")


def setup_sidebar():
    st.sidebar.title("AI MEM Configuration") 
    
    # LLM selection for Response Generation
    chain_provider = st.sidebar.selectbox("Select LLM Provider for Response Generation", 
                                          list(LLMManager.get_provider_models().keys()))
    chain_model = st.sidebar.selectbox("Select Model for Response Generation", 
                                       LLMManager.get_models_for_provider(chain_provider))

    st.sidebar.write("---")

    # LLM selection for Querying Retriever
    search_query_provider = st.sidebar.selectbox("Select LLM Provider for Querying Retriever", 
                                                 list(LLMManager.get_provider_models().keys()))
    search_query_model = st.sidebar.selectbox("Select Model for Querying Retriever", 
                                              LLMManager.get_models_for_provider(search_query_provider))

    return chain_provider, chain_model, search_query_provider, search_query_model

# In your main app logic
chain_provider, chain_model, search_query_provider, search_query_model = setup_sidebar()

# Load LLMs
chain_llm = LLMManager.load_llm(chain_provider, chain_model)
search_query_llm = LLMManager.load_llm(search_query_provider, search_query_model)



# Setup VectorDB

@lru_cache(maxsize=1)
def load_vectorstore(index_name):
    embeddings = vo_embed()
    return PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)

pinecone_index_name = st.sidebar.text_input("IndexName for Retreiving", value="langchain")
vectorstore = load_vectorstore(pinecone_index_name)

# Setup retriever
@lru_cache(maxsize=1)
def load_retriever():
    return retriever_tool_meta(vectorstore)

retriever = load_retriever()

# RAG setup
@lru_cache(maxsize=1)
def load_condense_question_prompt():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    return PromptTemplate.from_template(_template)

CONDENSE_QUESTION_PROMPT = load_condense_question_prompt()

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

ANSWER_PROMPT = load_answer_prompt()

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content}")


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
    chat_history: List[Tuple[str, str]] = Field(..., extra={
                                                "widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  
        # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"][:-1] if x["chat_history"] and x["chat_history"][-1]["content"] == x["question"] else x["chat_history"]),
            question=lambda x: x["question"]
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
        "chat_history": lambda x: _format_chat_history(x["chat_history"][:-1] if x["chat_history"] and x["chat_history"][-1]["content"] == x["question"] else x["chat_history"]),
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

# Add sidebar title and description
sidebar.title("Rag Chat Tools")


# Create a dropdown menu to select the function to call
functions = {"Scrape": scrape, "Crawl": crawl, "Sitemap Scraper": scrape_sitemap, "Youtube Chat": youtube_chat}

selected_function = st.sidebar.selectbox("Select a function", list(functions.keys()))


url = st.sidebar.text_input("Enter a URL")

split_result = None

# Show the input field for index name only if the selected function is "Sitemap Scraper"
if selected_function == "Sitemap Scraper":
    # Check if index_name exists in the session state
    if "index_name" not in st.session_state:
        st.session_state.index_name = ""

    # Display the index name input field and update the session state
    st.session_state.index_name = st.sidebar.text_input("Index Name", value=st.session_state.index_name)


if st.sidebar.button("Submit URL Function", key="url_submit"):
    if url:
        if selected_function == "Sitemap Scraper":
            try:
                # Display a progress bar in the sidebar while the function is running
                progress_bar = st.sidebar.progress(0)

                def progress_callback(current, total):
                    progress_bar.progress(current / total)

                scrape_sitemap(url, st.session_state.index_name, progress_callback)

                st.sidebar.success("Sitemap scraped and results embedded successfully!")
            except Exception as e:
                st.sidebar.error(f"Sitemap scraping and embedding failed: {str(e)}")
                st.sidebar.error("Please check the error message and try again.")
        elif selected_function == "YouTube Chat":
            # Call the scrape_sitemap function without splitting the result
            fn_result = youtube_chat(url)
            split_result = split_text(fn_result)
            # Store the split_result in session state
            st.session_state.split_result = split_result
        else:
            # Call the selected function with the provided URL and split the result
            fn_result = functions[selected_function](url)
            split_result = split_md(fn_result)
            # Store the split_result in session state
            st.session_state.split_result = split_result
    else:
        st.warning("Please enter a valid URL.")

# Check if split_result exists in the session state and the selected function is not "Sitemap Scraper"
if "split_result" in st.session_state and selected_function != "Sitemap Scraper":
    split_result = st.session_state.split_result

    # Show the input field for index name only if split_result is not empty
    if split_result:
        # Check if index_name exists in the session state
        if "index_name" not in st.session_state:
            st.session_state.index_name = ""

        # Display the index name input field and update the session state
        st.session_state.index_name = st.sidebar.text_input("Index Name", value=st.session_state.index_name)

        if st.sidebar.button("Add to Memory", key="add_to_memory"):
            try:
                embeddings = vo_embed()
                PineconeVectorStore.from_documents(
                    documents=split_result, embedding=embeddings, index_name=st.session_state.pinecone_index_name)
                st.sidebar.success("Embedding completed successfully!")
            except Exception as e:
                st.sidebar.error(f"Embedding failed: {str(e)}")
                st.sidebar.error("Please check the error message and try again.")
    else:
        st.warning("No results to embed.")
# File Uploader
with st.sidebar.expander("Upload and Embed Documents"):
    upload_method = st.radio("Upload Method", ["File", "Text"])

    loaded_docs = None
    if upload_method == "File":
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
        if uploaded_file:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # Load documents from the temporary file
            loaded_docs = load_documents(file_path=temp_file_path)

            # Clean up the temporary file
            os.remove(temp_file_path)
    else:
        text_input = st.text_area("Paste your text here")
        if text_input:
            loaded_docs = load_documents(text=text_input)

    if st.button("Embed Documents", key="embed_documents"):
        if loaded_docs:
            try:
                embeddings = vo_embed()
                PineconeVectorStore.from_documents(
                    documents=loaded_docs, embedding=embeddings, index_name="qa")
                st.success("Embedding completed successfully!")
            except Exception as e:
                st.error(f"Embedding failed: {str(e)}")
                st.error("Please check the error message and try again.")
        else:
            st.warning("No documents to embed.")

# Reset chat history button
if st.sidebar.button("Reset Chat History"):
    st.session_state.messages = []

def trim_chat_history(messages: List[Dict[str, str]], max_tokens: int = 8000) -> Tuple[List[Dict[str, str]], int]:
    trimmed_messages = []
    total_prompt_tokens = 0
    user_message_found = False

    for message in reversed(messages):
        message_tokens = LLMManager.count_tokens(message["content"], chain_provider, chain_model)

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
                total_prompt_tokens += LLMManager.count_tokens(message["content"], chain_provider, chain_model)
                break

    # If still no user message, add a dummy user message
    if not trimmed_messages or trimmed_messages[0]["role"] != "user":
        dummy_message = {"role": "user", "content": "Start of conversation"}
        trimmed_messages.insert(0, dummy_message)
        total_prompt_tokens += LLMManager.count_tokens(dummy_message["content"], chain_provider, chain_model)

    return trimmed_messages, total_prompt_tokens



calculate_total_tokens = LLMManager.calculate_total_tokens

# Initialize chat history
chat_history = []


# Initialize token counters
update_token_count = LLMManager.update_token_count

st.title("Persistent Memory Conversational Agent")

    
#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize token counters in session state
if "total_prompt_tokens" not in st.session_state:
    st.session_state.total_prompt_tokens = 0
if "total_completion_tokens" not in st.session_state:
    st.session_state.total_completion_tokens = 0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    


for message in st.session_state.messages:
    avatar = "utils/images/user_avatar.png" if message["role"] == "user" else "utils/images/queryqueen.png"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Write something here...", key="input")

# Display the scrape result below the user input 
if split_result:
    with st.expander("Scrape Result", expanded=False):
        st.write(split_result)

#Chat Container
if user_input:
    # Display user input in chat message container
    with st.chat_message("user", avatar="utils/images/user_avatar.png"):
        st.text(user_input)

    # Append to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Trim chat history before sending to the model
    trimmed_history, history_tokens = trim_chat_history(st.session_state.messages, MAX_HISTORY_TOKENS)
    
   
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="utils/images/queryqueen.png"):
        loading_message = st.empty()
        # Display "Thinking..." message
        loading_message.markdown("Thinking...")

        result = ""
        for token in chain.invoke(input={"question": user_input, "chat_history": trimmed_history}):
            result += token
            loading_message.markdown(result)
           
   # Calculate prompt tokens
    prompt_tokens = LLMManager.count_tokens(str({"question": user_input, "chat_history": trimmed_history}), chain_provider, chain_model)

    # Calculate completion tokens
    completion_tokens = LLMManager.count_tokens(result, chain_provider, chain_model)
    
    # Update token counts
    LLMManager.update_token_count(st.session_state, prompt_tokens, completion_tokens)
    
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})
    
    
    # Trim chat history if needed, based on total tokens
    if st.session_state.total_tokens > LLMManager.MAX_HISTORY_TOKENS:
        st.session_state.messages, new_prompt_tokens = trim_chat_history(st.session_state.messages, LLMManager.MAX_HISTORY_TOKENS)
        LLMManager.reset_token_count(st.session_state, new_prompt_tokens)

    # Display updated token counts
    LLMManager.display_token_counts(st.sidebar, st.session_state)

     
