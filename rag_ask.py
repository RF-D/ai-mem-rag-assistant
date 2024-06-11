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
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()



# VoyageAI Setup
embeddings = setup_voyageai("voyage-large-2-instruct")


def scrape_flow():
    """
    This function handles the scraping flow.
    It prompts the user to enter a URL to scrape,
    uses Firecrawl to scrape the URL,
    splits the scraped text into documents,
    and stores them in a PineconeVectorStore.
    """
    url = input("Provide URL to Scrape: ")
    # Use Firecrawl to scrape or crawl URL
    data = scrape(url)

    # Split text into documents
    docs = split_text(data)

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="claude01")


def crawl_flow():
    """
    This function handles the crawling flow.
    It prompts the user to enter a URL to crawl,
    uses Firecrawl to crawl the URL,
    splits the crawled text into documents,
    and stores them in a PineconeVectorStore.
    """
    url = input("Provide URL to Scrape: ")
    # Use Firecrawl to scrape or crawl URL
    data = crawl(url)

    # Split text into documents
    docs = split_text(data)

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="claude01")


while True:
    user_input = input("Enter a command (scrape/crawl/load/paste/ask/yt): ")
    
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
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.8)

        template = """Use the following pieces of context to answer the question at the end.
        {context}
        Question: {question}
        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            """
            This function formats the documents into a string.
            """
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )
        print()
        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
        print()
    elif user_input.lower() == "quit":
        # Exit the loop and end the program if user says "quit"
        print("Exiting the program...")
        break
    else:
        print("Invalid command. Please enter 'crape', 'ask', or 'quit'.")