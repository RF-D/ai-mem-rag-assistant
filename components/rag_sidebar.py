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
import traceback
from langchain.schema import Document

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
    api_keys: dict
    selected_providers: list


# Update callback function
def update_selected_function():
    st.session_state.selected_function = st.session_state.function_selector


def setup_sidebar() -> SidebarConfig:
    st.sidebar.title("AI MEM Configuration")

    # Provider selection with an expander
    with st.sidebar.expander(
        "Hide/Show Providers",
        expanded=not st.session_state.get("provider_selection_confirmed", False),
    ):
        available_providers = list(LLMManager.get_provider_models().keys())
        selected_providers = st.multiselect(
            "Select AI providers to use",
            options=available_providers,
            default=st.session_state.get("selected_providers", available_providers),
            key="provider_multiselect",
        )

        # Button to confirm provider selection
        if st.button("Confirm Provider Selection"):
            st.session_state.selected_providers = selected_providers
            st.session_state.provider_selection_confirmed = True

    if st.session_state.get("provider_selection_confirmed", False):
        st.session_state.provider_selection_confirmed = False

    # Use the confirmed selection or default to all providers
    selected_providers = st.session_state.get("selected_providers", available_providers)

    # Ensure at least one provider is selected
    if not selected_providers:
        st.sidebar.warning("Please select at least one AI provider.")
        selected_providers = ["Ollama"]  # Default to Ollama if no provider is selected

    # API Key inputs (only shown if selected and not set, excluding Ollama)
    api_keys = {}
    for provider in selected_providers:
        if provider != "Ollama":
            env_var_name = f"{provider.upper()}_API_KEY"
            key_state = f"{provider.lower()}_api_key_set"

            # Initialize session state for this provider if not already done
            if key_state not in st.session_state:
                st.session_state[key_state] = False

            # Check if the key is in the environment or has been set in this session
            if not os.getenv(env_var_name) and not st.session_state[key_state]:
                with st.sidebar.expander(f"{provider} API Key", expanded=False):
                    api_key = st.text_input(
                        f"Enter your {provider} API key",
                        type="password",
                        key=f"{provider.lower()}_api_key_input",
                    )
                    if api_key:
                        if LLMManager.validate_api_key(provider, api_key):
                            api_keys[provider] = api_key
                            os.environ[env_var_name] = api_key
                            st.session_state[key_state] = True
                            st.success(f"{provider} API key validated successfully!")
                            st.rerun()  # Rerun to hide the input field
                        else:
                            st.error(f"Invalid {provider} API key. Please try again.")
                    else:
                        st.warning(f"{provider} API key not found or invalid.")

    if api_keys:
        st.sidebar.markdown("---")

    # Provider and model selection
    chain_provider = st.sidebar.selectbox(
        "Choose AI assistant response generation", options=selected_providers, index=0
    )
    chain_model = st.sidebar.selectbox(
        "Select specific model for responses",
        LLMManager.get_models_for_provider(chain_provider),
    )

    st.sidebar.write("---")

    search_query_provider = st.sidebar.selectbox(
        "Choose AI assistant for relevant information retrieval",
        options=selected_providers,
        index=0,
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
        api_keys=api_keys,
        selected_providers=selected_providers,
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
    try:
        fn_result = youtube_chat(url)
        if fn_result is not None:
            split_result = split_md(fn_result)
            st.session_state.split_result = split_result
            st.session_state.results_to_display = True
            st.success("YouTube chat completed successfully!")
            return split_text(fn_result)
        else:
            st.error(
                "The YouTube chat function returned None. Please check the input and try again."
            )
            st.session_state.split_result = None
            st.session_state.results_to_display = False
            return None
    except Exception as e:
        error_message = f"An error occurred during the YouTube chat: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        st.error(error_message)
        st.session_state.split_result = None
        st.session_state.results_to_display = False
        return None


def crawl_submit(url):
    try:
        fn_result = crawl(url, params=st.session_state.crawl_params)
        if fn_result is not None:
            if isinstance(fn_result, str):
                # Convert the string to a Document object
                doc = Document(page_content=fn_result, metadata={"source": url})
                split_result = split_md([doc])
            else:
                split_result = fn_result  # Assume it's already split if not a string
            st.session_state.split_result = split_result
            st.session_state.results_to_display = True
            st.success("Crawl completed successfully!")
            return split_result
        else:
            st.error(
                "An error occurred during crawling. Please check the logs for more information."
            )
            st.session_state.split_result = None
            st.session_state.results_to_display = False
            return None
    except Exception as e:
        error_message = f"An error occurred during crawling: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        st.error(error_message)
        st.session_state.split_result = None
        st.session_state.results_to_display = False
        return None


def sitemap_scraper_submit(url):
    try:
        progress_bar = st.sidebar.progress(0)
        progress_bar.progress_callback
        result = scrape_sitemap(url, st.session_state.index_name, progress_callback)
        st.sidebar.success("Sitemap scraped and results embedded successfully!")
        st.session_state.results_to_display = True
        return result
    except Exception as e:
        st.sidebar.error(f"Sitemap scraping and embedding failed: {str(e)}")
        st.sidebar.error("Please check the error message and try again.")
        st.session_state.results_to_display = False
        return None


def add_to_memory_button(split_result):
    if st.sidebar.button("Add to Memory", key="add_to_memory"):
        try:
            embeddings = vo_embed()
            PineconeVectorStore.from_documents(
                documents=split_result,
                embedding=embeddings,
                index_name=st.session_state.index_name,
            )
            st.sidebar.success("Embedding completed successfully!")
        except Exception as e:
            st.sidebar.error(f"Embedding failed: {str(e)}")
            st.sidebar.error("Please check the error message and try again.")


def display_results(split_result, selected_function):
    if split_result and selected_function != "Sitemap Scraper":
        with st.expander(f"{selected_function} Result", expanded=False):
            st.write(split_result)
    elif selected_function != "Sitemap Scraper":
        st.info("No results to display. Try submitting a URL first.")


def handle_split_result(selected_function):
    if (
        "split_result" in st.session_state
        and st.session_state.split_result
        and selected_function != "Sitemap Scraper"
    ):
        split_result = st.session_state.split_result
        st.session_state.index_name = st.sidebar.text_input(
            "Index Name", value=st.session_state.get("index_name", "")
        )
        add_to_memory_button(split_result)
        return split_result
    return None
