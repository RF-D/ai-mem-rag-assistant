import os
import streamlit as st
from operator import itemgetter
from typing import List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_pinecone import PineconeVectorStore
from tools.voyage_embeddings import vo_embed
from tools.retriever_tools import retriever_tool, retriever_tool_meta
from tools.firecrawl_scrape_loader import scrape
from tools.text_splitter import split_md,split_text
from tools.firecrawl_crawl_loader import crawl
from tools.firecrawl_scrape_loader import scrape
from scrape_sitemap import scrape_sitemap
from tools.youtube_chat import youtube_chat
from dotenv import load_dotenv


load_dotenv()

# Create the LLM
llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.8)


# Setup VectorDB
embeddings = vo_embed()

index_name = "langchain"

vectorstore = PineconeVectorStore.from_existing_index(
    embedding=embeddings, index_name=index_name)

retriever = retriever_tool_meta(vectorstore)


# RAG setup
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Provide a detailed and comprehensive answer to the question, using the context provided. If the context is insufficient, indicate what additional information would be needed to answer the question.
<context>
{context}
</context>

Question: {question}"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        doc_string = format_document(doc, document_prompt)
        doc_string += f"\nSource: {source}"
        doc_strings.append(doc_string)
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
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = _inputs | ANSWER_PROMPT | llm | StrOutputParser()



#Streamlit app

# Set page configuration
st.set_page_config(page_title="Rag Chat", page_icon=":guardsman:", layout="wide")



# Create a sidebar
sidebar = st.sidebar

# Add sidebar title and description
sidebar.title("Rag Chat Tools")
sidebar.write("Ingest Knowledge here with your preferred method")


# Create a dropdown menu to select the function to call
functions = {"Scrape": scrape, "Crawl": crawl, "Sitemap Scraper": scrape_sitemap, "Youtube Chat": youtube_chat}
selected_function = st.sidebar.selectbox("Select a function", list(functions.keys()))

url = st.sidebar.text_input("Enter a URL")

split_result = None

if st.sidebar.button("Process"):
    if url:
        if selected_function == "Sitemap Scraper":
            # Call the scrape_sitemap function without splitting the result
            fn_result = scrape_sitemap(url)
        elif selected_function == "Youtube Chat":
            # Call the scrape_sitemap function without splitting the result
            fn_result = youtube_chat(url)
            split_result = split_text(fn_result)
        else:
            # Call the selected function with the provided URL and split the result
            fn_result = functions[selected_function](url)
            split_result = split_md(fn_result)
    else:
        st.warning("Please enter a URL.")

    
# Initialize chat history
chat_history = []

st.title("Rag Chat")

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    

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


if user_input:
    # Display user input in chat message container
    with st.chat_message("user", avatar="utils/images/user_avatar.png"):
        st.markdown(user_input)

    # Append to chat history
    chat_history.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="utils/images/queryqueen.png"):
        loading_message = st.empty()
        loading_message.markdown("Thinking...")

        result = ""
        for word in chain.invoke(input={"question": user_input, "chat_history": st.session_state.messages}):
            result += word
            loading_message.markdown(result)

        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": result})
        st.session_state.messages.append({"role": "assistant", "content": result})

        

     
