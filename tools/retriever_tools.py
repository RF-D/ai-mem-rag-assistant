# modules/retriever_tool.py
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

def retriever_tool(vectorstore, search_type="similarity", search_kwargs={"k": 25}):
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever


def retriever_tool_meta(vectorstore):
    def retrieve_documents(query):
        docs = vectorstore.similarity_search(query, k=8)
        # Add source information to the documents' metadata
        for doc in docs:
            doc.metadata["source"] 
        return docs
    return retrieve_documents


def retrieve_qa(llm, vectorstore):
    retriever = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25}))
    
    return retriever

def retrieve_qa_with_sources(llm, vectorstore):
    retriever = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25}))
    
    return retriever