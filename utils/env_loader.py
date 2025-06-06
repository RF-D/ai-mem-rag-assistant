# utils/env_loader.py
import os
from dotenv import load_dotenv


def load_env():
    load_dotenv(dotenv_path='config/.env')
    return {
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "v_api_key": os.getenv("VOYAGE_API_KEY"),
        "firecrawl_api_key": os.getenv("FIRECRAWL_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "openai_key": os.getenv("OPENAI_API_KEY"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "serper_api_key": os.getenv("SERPER_API_KEY"),
        
    }


def load_env_vars():
    env_vars = load_env()
    anthropic_api_key = env_vars["anthropic_api_key"]
    v_api_key = env_vars["v_api_key"]
    firecrawl_api_key = env_vars["firecrawl_api_key"]
    pinecone_api_key = env_vars["pinecone_api_key"]
    openai_key = env_vars["openai_key"]
    tavily_api_key = env_vars["tavily_api_key"]
    serper_api_key = env_vars["serper_api_key"]
    
    return anthropic_api_key, v_api_key, firecrawl_api_key, pinecone_api_key, openai_key, tavily_api_key,serper_api_key
