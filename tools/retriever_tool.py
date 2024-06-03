# modules/retriever_tool.py


def retriever_tool(vectorstore, search_type="similarity", search_kwargs={"k": 6}):
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
