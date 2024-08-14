# modules/retriever_tool.py
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIRerank
from functools import lru_cache
from langchain.schema import Document


def retriever_tool(vectorstore, search_type="similarity", search_kwargs={"k": 25}):
    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def retriever_tool_meta(vectorstore):
    def retrieve_documents(query):
        # Retrieve documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

        # Use VoyageAIRerank for reranking
        reranker = VoyageAIRerank(
            model="rerank-1",
            voyageai_api_key=os.getenv("VOYAGE_API_KEY"),
            top_k=10,
        )

        # Create a ContextualCompressionRetriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=retriever
        )

        # Retrieve and rerank documents
        compressed_docs = compression_retriever.get_relevant_documents(query)

        # Enhance metadata usage
        for doc in compressed_docs:
            # Ensure basic metadata fields are present
            basic_fields = ["source", "title", "author", "publish_date"]
            for field in basic_fields:
                doc.metadata[field] = doc.metadata.get(field, "Unknown")

            # Handle YouTube-specific metadata if present
            youtube_fields = ["title", "view_count", "description", "thumbnail_url"]
            for field in youtube_fields:
                if field in doc.metadata:
                    doc.metadata[field] = doc.metadata.get(field, "Unknown")

            # Convert view_count to integer if it exists and is not "Unknown"
            if "view_count" in doc.metadata and doc.metadata["view_count"] != "Unknown":
                try:
                    doc.metadata["view_count"] = int(doc.metadata["view_count"])
                except ValueError:
                    pass  # Keep as string if conversion fails

        return compressed_docs

    return retrieve_documents


def retrieve_qa(llm, vectorstore):
    retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 25}
        ),
    )

    return retriever


def retrieve_qa_with_sources(llm, vectorstore):
    retriever = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 25}
        ),
    )

    return retriever
