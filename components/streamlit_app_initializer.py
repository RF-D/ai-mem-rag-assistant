# components/app_initializer.py

import streamlit as st
from tools.firecrawl_crawl_loader import get_default_crawl_params
from components.rag_sidebar import setup_sidebar


def initialize_streamlit_app():
    # Set page configuration
    st.set_page_config(page_title="AI MEM", page_icon=":guardsman:", layout="wide")

    # Initialize session state variables
    if "selected_function" not in st.session_state:
        st.session_state.selected_function = "Scrape"

    if "crawl_params" not in st.session_state:
        st.session_state.crawl_params = get_default_crawl_params()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "total_prompt_tokens" not in st.session_state:
        st.session_state.total_prompt_tokens = 0

    if "total_completion_tokens" not in st.session_state:
        st.session_state.total_completion_tokens = 0

    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    # Initialize other session state variables
    if "index_name" not in st.session_state:
        st.session_state.index_name = ""

    if "upload_index_name" not in st.session_state:
        st.session_state.upload_index_name = ""

    if "split_result" not in st.session_state:
        st.session_state.split_result = None
    if "sidebar_config" not in st.session_state:
        st.session_state.sidebar_config = "None"
    # Set up sidebar
    sidebar_config = setup_sidebar()

    return sidebar_config
