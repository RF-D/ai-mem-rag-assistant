# modules/retriever_tool.py
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIRerank
from functools import lru_cache

def retriever_tool(vectorstore, search_type="similarity", search_kwargs={"k": 25}):
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever


@lru_cache(maxsize=1000)  # Adjust the cache size based on your requirements
def retriever_tool_meta(vectorstore):
    def retrieve_documents(query):
        # Retrieve a smaller number of documents initially
        docs = vectorstore.similarity_search(query, k=15)  # Adjust the value of k as needed

        # Add source information to the documents' metadata
        for doc in docs:
            doc.metadata["source"] = doc.metadata.get("source", "Unknown")

        # Create a VoyageAIRerank instance with optimized configuration
        reranker = VoyageAIRerank(
            model="rerank-lite-1",
            voyageai_api_key=os.getenv("VOYAGE_API_KEY"),
            top_k=6  # Adjust the value of top_k as needed
        )

        # Create a ContextualCompressionRetriever with the reranker
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 15})  # Adjust the value of k as needed
        )

        # Rerank and compress the retrieved documents using invoke
        compressed_docs = compression_retriever.invoke(query)

        return compressed_docs

    return retrieve_documents


def retrieve_qa(llm, vectorstore):
    retriever = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25}))
    
    return retriever

def retrieve_qa_with_sources(llm, vectorstore):
    retriever = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25}))
    
    return retriever