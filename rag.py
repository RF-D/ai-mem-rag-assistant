
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
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


from utils.env_loader import load_env_vars
from tools.firecrawl_scrape_loader import scrape
from tools.firecrawl_crawl_loader import crawl
from tools.text_splitter import split_text
from tools.voyage_embeddings import setup_voyageai
from tools.retriever_tool import retriever_tool


# Load environment variables
anthropic_api_key, v_api_key, firecrawl_api_key, pinecone_api_key = load_env_vars()

# VoyageAI Setup
embeddings = setup_voyageai("voyage-large-2-instruct")


def scrape_flow():
    url = input("Provide URL to Scrape: ")
    # Use Firecrawl to scrape or crawl URL
    data = scrape(url)

    # Split text into documents
    docs = split_text(data)

    vectorstore = PineconeVectorStore.from_documents(
        docs, embeddings, index_name="claude01")


while True:
    user_input = input("Enter a command (scrape/ask): ")

    if user_input.lower() == "scrape":
        scrape_flow()

    elif user_input.lower() == "ask":
        query = input("Ask Memory: ")
        # Invoke current index
        vectorstore = PineconeVectorStore.from_existing_index(
            embedding=embeddings, index_name="claude01")

        retriever = retriever_tool(vectorstore)

        # Chat setup
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.8)

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
