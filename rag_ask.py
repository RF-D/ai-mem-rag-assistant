
from tools.youtube_chat import youtube_chat
from tools.voyage_embeddings import text_voyageai
from tools.webloader_tool import load_web_url
from tools.retriever_tool import retriever_tool
from tools.voyage_embeddings import setup_voyageai
from tools.text_splitter import split_text
from tools.firecrawl_crawl_loader import crawl
from tools.firecrawl_scrape_loader import scrape
from utils.env_loader import load_env_vars
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import FireCrawlLoader
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
5from langchain_voyageai import VoyageAIEmbeddings


# Load environment variables
anthropic_api_key, v_api_key, firecrawl_api_key, pinecone_api_key, openai_key = load_env_vars()

# VoyageAI Setup
embeddings = setup_voyageai("voyage-large-2-instruct")


def scrape_flow():
    url = input("Provide URL to Scrape: ")
    # Use Firecrawl to scrape or crawl URL
    data = scrape(url)

    # Split text into documents
    docs = split_text(data)

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="claude01")


def crawl_flow():
    url = input("Provide URL to Scrape: ")
    # Use Firecrawl to scrape or crawl URL
    data = crawl(url)

    # Split text into documents
    docs = split_text(data)

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="claude01")


while True:
    user_input = input("Enter a command (scrape/crawl/load/paste/ask): ")

    if user_input.lower() == "scrape":
        scrape_flow()
    elif user_input.lower() == "crawl":
        crawl_flow()
    elif user_input.lower() == "load":
        url = input("Provide URL to Load: ")

        data = load_web_url(url)

    # Split text into documents
        docs = split_text(data)

        vectorstore = PineconeVectorStore.from_documents(
            documents=docs, embedding=embeddings, index_name="claude01")

        retriever = retriever_tool(vectorstore)
    elif user_input.lower() == "yt":
        yt_url = input("Provide YT URL to Load: ")
        data = scrape(yt_url)
        data += youtube_chat(yt_url)
        # Ensure data is a string before splitting
    # Split text into documents
        docs = split_text(data)

        vectorstore = PineconeVectorStore.from_documents(
            documents=docs, embedding=embeddings, index_name="claude01")

    elif user_input.lower() == "ask":
        query = input("Ask Memory: ")
        # Invoke current index
        vectorstore = PineconeVectorStore.from_existing_index(
            embedding=embeddings, index_name="claude01")

        retriever = retriever_tool(vectorstore)

        # Chat setup
        llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.8)

        template = """Use the following pieces of context to answer the question at the end.



        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
    elif user_input.lower() == "quit":
        # Exit the loop and end the program if user says "quit"
        print("Exiting the program...")
        break
    else:
        print("Invalid command. Please enter 'scrape', 'ask', or 'quit'.")
