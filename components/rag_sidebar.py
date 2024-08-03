import streamlit as st
import os
from dataclasses import dataclass
from utils.llm_manager import LLMManager
from scrape_sitemap import scrape_sitemap
from tools.youtube_chat import youtube_chat
from tools.firecrawl_scrape_loader import scrape
from tools.firecrawl_crawl_loader import crawl
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


SITEMAP_SCRAPER = "Sitemap Scraper"
YOUTUBE_CHAT = "Youtube Chat"
SCRAPE = "Scrape"
CRAWL = "Crawl"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_pinecone_indexes():
    return [index.name for index in pc.list_indexes()]


@dataclass
class SidebarConfig:
    chain_provider: str
    chain_model: str
    search_query_provider: str
    search_query_model: str
    pinecone_index_name: str
    url: str
    selected_function: str
    functions: dict


# Update callback function
def update_selected_function():
    st.session_state.selected_function = st.session_state.function_selector


def setup_sidebar() -> SidebarConfig:
    st.sidebar.title("AI MEM Configuration")

    chain_provider = st.sidebar.selectbox(
        "Choose AI assistant response generation",
        list(LLMManager.get_provider_models().keys()),
    )
    chain_model = st.sidebar.selectbox(
        "Select specific model for responses",
        LLMManager.get_models_for_provider(chain_provider),
    )

    st.sidebar.write("---")

    search_query_provider = st.sidebar.selectbox(
        "Choose AI assistant for relevant information retrieval",
        list(LLMManager.get_provider_models().keys()),
    )
    search_query_model = st.sidebar.selectbox(
        "Select specific model for retrieval",
        LLMManager.get_models_for_provider(search_query_provider),
    )
    # Use the cached function to get Pinecone indexes
    PINECONE_INDEXES = get_pinecone_indexes()

    pinecone_index_name = st.sidebar.selectbox(
        "Choose where the AI should look for information:",
        options=PINECONE_INDEXES,
        index=1 if PINECONE_INDEXES else 0 else None ,
    )
    st.sidebar.title("Rag Chat Tools")

    # Create a dropdown menu to select the function to call
    functions = {
        SCRAPE: scrape,
        CRAWL: crawl,
        SITEMAP_SCRAPER: scrape_sitemap,
        YOUTUBE_CHAT: youtube_chat,
    }

    # Function selection
    selected_function = st.sidebar.selectbox(
        "Select a function",
        list(functions.keys()),
        key="function_selector",
        index=list(functions.keys()).index(st.session_state.selected_function),
        on_change=update_selected_function,
    )
    url = st.sidebar.text_input("Enter a URL")
    return SidebarConfig(
        chain_provider=chain_provider,
        chain_model=chain_model,
        search_query_provider=search_query_provider,
        search_query_model=search_query_model,
        pinecone_index_name=pinecone_index_name,
        url=url,
        selected_function=selected_function,
        functions=functions,
    )
