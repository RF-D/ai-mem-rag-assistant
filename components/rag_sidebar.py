import streamlit as st
import os
from dataclasses import dataclass
from utils.llm_manager import LLMManager
from scrape_sitemap import scrape_sitemap
from tools.youtube_chat import youtube_chat
from tools.firecrawl_scrape_loader import scrape
from tools.firecrawl_crawl_loader import crawl
from pinecone import Pinecone
from tools.doc_loader import load_documents
import tempfile
from langchain_pinecone import PineconeVectorStore
from tools.voyage_embeddings import vo_embed
from tools.text_splitter import split_md, split_text


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
        index=1 if len(PINECONE_INDEXES) > 1 else 0,
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


def index_name_for_sitemap_scraper(selected_function):
    if selected_function == "Sitemap Scraper":
        st.session_state.index_name = st.sidebar.text_input(
            "Index Name", value=st.session_state.index_name
        )


def file_uploader():
    # File Uploader
    with st.sidebar.expander("Upload and Embed Documents"):
        upload_method = st.radio("Upload Method", ["File", "Text"])

        loaded_docs = None
        if upload_method == "File":
            uploaded_file = st.file_uploader(
                "Choose a file", type=["txt", "pdf", "docx"]
            )
            if uploaded_file:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}'
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                # Load documents from the temporary file
                loaded_docs = load_documents(file_path=temp_file_path)

                # Clean up the temporary file
                os.remove(temp_file_path)
        else:
            text_input = st.text_area("Paste your text here")
            if text_input:
                loaded_docs = load_documents(text=text_input)

        st.session_state.upload_index_name = st.text_input(
            "Index Name for File/Text Uploader",
            value=st.session_state.upload_index_name,
        )

        if st.button("Embed Documents", key="embed_documents"):
            if loaded_docs and st.session_state.upload_index_name:
                try:
                    embeddings = vo_embed()
                    PineconeVectorStore.from_documents(
                        documents=loaded_docs,
                        embedding=embeddings,
                        index_name=st.session_state.upload_index_name,
                    )
                    st.success("Embedding completed successfully!")
                except Exception as e:
                    st.error(f"Embedding failed: {str(e)}")
                    st.error("Please check the error message and try again.")
            elif not loaded_docs:
                st.warning("No documents to embed.")
            else:
                st.warning("Please enter an index name for embedding.")


def crawl_parameters():
    with st.sidebar.form("crawl_params_form"):
        st.header("Crawl Parameters")

        max_depth = st.number_input(
            "Max Depth",
            min_value=1,
            max_value=10,
            value=st.session_state.crawl_params["crawlerOptions"]["maxDepth"],
        )

        limit = st.number_input(
            "Limit",
            min_value=1,
            max_value=1000,
            value=st.session_state.crawl_params["crawlerOptions"]["limit"],
        )

        crawl_delay = st.number_input(
            "Crawl Delay",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.crawl_params["crawlerOptions"]["crawldelay"],
            step=0.1,
        )

        only_main_content = st.checkbox(
            "Only Main Content",
            value=st.session_state.crawl_params["pageOptions"]["onlyMainContent"],
        )

        if st.form_submit_button("Apply Crawl Parameters"):
            st.session_state.crawl_params["crawlerOptions"]["maxDepth"] = max_depth
            st.session_state.crawl_params["crawlerOptions"]["limit"] = limit
            st.session_state.crawl_params["crawlerOptions"]["crawldelay"] = crawl_delay
            st.session_state.crawl_params["pageOptions"][
                "onlyMainContent"
            ] = only_main_content


def youtube_chat_submit(url):
    fn_result = youtube_chat(url)
    split_result = split_text(fn_result)
    return split_result
