from langchain_community.tools.tavily_search import TavilySearchResults
from utils.env_loader import load_env_vars

tavily_api_key = load_env_vars()[5]

def tavily_search_r(query: str) -> str:
    tool = TavilySearchResults()
    results = tool.invoke({"query": query})

    formatted_results = ""
    for i, doc in enumerate(results, start=1):
        formatted_results += f"Result {i}:\n"
        formatted_results += f"URL: {doc.get('url')}\n"  # Access URL directly from doc
        formatted_results += f"Title: {doc.get('title', 'N/A')}\n"  # Access title directly from doc
        
        if 'published_at' in doc:
            formatted_results += f"Published At: {doc['published_at']}\n"
        
        formatted_results += f"Summary: {doc.get('content', 'N/A')}\n\n"  # Access summary directly from doc
    
    return formatted_results