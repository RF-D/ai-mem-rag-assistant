from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from utils.env_loader import load_env_vars

serper_api_key = load_env_vars()[6]

def serper_search(query) -> str:
    search = GoogleSerperAPIWrapper()
    search_results = search.run(query)
    
    return search_results