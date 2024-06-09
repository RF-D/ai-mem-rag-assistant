from langchain_community.retrievers import TavilySearchAPIRetriever
from utils.env_loader import load_env_vars



tavily_api_key = load_env_vars()[5]
def tavily_search(query: str, num_results: int = 10) -> str:
    """
    This function performs a web search using the Tavily API and returns the search results.

    Parameters:
    query (str): The search query.
    num_results (int): The number of search results to retrieve (default: 5).

    Returns:
    str: The formatted search results.
    """
    retriever = TavilySearchAPIRetriever(search = True, max_tokens = 6000, k=num_results, search_depth= "advanced", format = "markdown",use_search_contex = True)
    results = retriever.invoke(query, includ_anwer = True)
    
    formatted_results = ""
    for i, doc in enumerate(results):
        formatted_results += f"Result {i+1}:\n"
        formatted_results += f"URL: {doc.metadata.get('source')}\n"  # Use get() with default value 'N/A'
        formatted_results += f"Title: {doc.metadata.get('title', 'N/A')}\n"  # Use get() with default value 'N/A'
        
        if 'published_at' in doc.metadata:
            formatted_results += f"Published At: {doc.metadata['published_at']}\n"
        
        formatted_results += f"Summary: {doc.page_content}\n\n"  # Return full summary without truncation
        
  
   

    return formatted_results