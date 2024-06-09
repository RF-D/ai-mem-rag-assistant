from langchain_community.retrievers import TavilySearchAPIRetriever
from tools.firecrawl_scrape_loader import scrape
from utils.env_loader import load_env_vars
from tools.text_splitter import split_text



tavily_api_key = load_env_vars()[5]
firecrawl_api_key = load_env_vars()[4]

def scrape_search(query: str) -> str:
    """
    This function performs a web search using the Tavily API and returns the search results.
    It also scrapes the URLs provided by the search and adds the scraped content to the results.

    Parameters:
    query (str): The search query.
    num_results (int): The number of search results to retrieve (default: 5).

    Returns:
    str: The formatted search results with scraped content.
    """
    retriever = TavilySearchAPIRetriever(search = True, max_tokens = 6000, k=1, search_depth= "advanced", format = "markdown",use_search_contex = True)
    results = retriever.invoke(query, includ_anwer = True)
    
    formatted_results = ""
    for i, doc in enumerate(results):
        formatted_results += f"Result {i+1}:\n"
        formatted_results += f"URL: {doc.metadata.get('source')}\n"
        formatted_results += f"Title: {doc.metadata.get('title', 'N/A')}\n"
        formatted_results += f"Published At: {doc.metadata.get('published_at', 'N/A')}\n"
        formatted_results += f"Summary: {doc.page_content}\n"
        
        formatted_results += f"Scraped Content: {scrape(doc.metadata.get('source'))}\n\n"
        
        split_text(formatted_results)
    return formatted_results