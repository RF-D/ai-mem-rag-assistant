import streamlit as st
from dataclasses import dataclass
from utils.llm_manager import LLMManager


@dataclass
class SidebarConfig:
    chain_provider: str
    chain_model: str
    search_query_provider: str
    search_query_model: str
    pinecone_index_name: str


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
    pinecone_index_name = st.sidebar.text_input(
        "Choose where the AI should look for information: (IndexName)",
        value="langchain",  # TODO: make this value empty or pull directly from pinecone API
    )

    return SidebarConfig(
        chain_provider=chain_provider,
        chain_model=chain_model,
        search_query_provider=search_query_provider,
        search_query_model=search_query_model,
        pinecone_index_name=pinecone_index_name,
    )
