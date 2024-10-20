import time
import json
import logging
import requests
from langchain_community.document_loaders import FireCrawlLoader
from langchain.schema import Document
from firecrawl import FirecrawlApp
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated default FireCrawl Setup
default_crawl_params = {
    "max_depth": 3,
    "max_pages": 50,
    "crawl_delay": 0.5,
    "only_main_content": True,
    "allow_backward_links": True,
}


def get_default_crawl_params():
    return default_crawl_params.copy()


def is_url_accessible(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def crawl(url, params=None, wait_until_done=True, timeout=300, check_interval=5):
    try:
        logger.info(f"Starting crawl for URL: {url}")

        if not is_url_accessible(url):
            logger.error(f"URL {url} is not accessible")
            return None

        app = FirecrawlApp()
        crawl_params = params if params is not None else get_default_crawl_params()

        # Prepare the parameters in the format expected by FirecrawlApp
        firecrawl_params = {
            "limit": crawl_params.get("max_pages", default_crawl_params["max_pages"]),
            "maxDepth": crawl_params.get(
                "max_depth", default_crawl_params["max_depth"]
            ),
            "allowBackwardLinks": crawl_params.get(
                "allow_backward_links", default_crawl_params["allow_backward_links"]
            ),
            "scrapeOptions": {
                "onlyMainContent": crawl_params.get(
                    "only_main_content", default_crawl_params["only_main_content"]
                ),
                "formats": ["markdown", "html"],
            },
        }

        # Store crawl_delay separately as it's not recognized by the API, need to check if we can still use it
        crawl_delay = crawl_params.get(
            "crawl_delay", default_crawl_params["crawl_delay"]
        )

        logger.info(f"Crawl parameters: {firecrawl_params}")
        logger.info(f"Crawl delay: {crawl_delay}")

        try:
            crawl_status = app.crawl_url(
                url, params=firecrawl_params, poll_interval=check_interval
            )
            logger.info("Crawl started successfully")
        except Exception as e:
            logger.error(f"Exception during crawl_url call: {str(e)}", exc_info=True)
            return None

        # If the crawl started successfully, implement the delay here
        time.sleep(crawl_delay)

        if not wait_until_done:
            return {"status": "started", "crawl_status": crawl_status}

        start_time = time.time()
        status_text = st.empty()

        while True:
            if crawl_status.get("status") == "completed":
                logger.info("Crawl completed successfully")
                break
            elif crawl_status.get("status") == "failed":
                logger.error(f"Crawl failed: {crawl_status.get('error')}")
                st.error(f"Crawl failed: {crawl_status.get('error')}")
                return None

            status_text.text(f"Crawled: {crawl_status.get('pagesCrawled', 0)} pages")

            if time.time() - start_time > timeout:
                logger.warning("Crawl job timed out")
                status_text.text("Crawl job timed out")
                return None

            time.sleep(check_interval)
            crawl_status = app.check_crawl_status(crawl_status.get("id"))

        # Process and return the crawl results
        documents = process_crawl_result(crawl_status.get("data", []))
        return documents

    except Exception as e:
        logger.error(f"Error during crawling: {str(e)}", exc_info=True)
        st.error(f"Error during crawling: {str(e)}")
        return None


def process_crawl_result(result):
    if isinstance(result, list) and len(result) > 0:
        documents = []
        for item in result:
            content = f"Title: {item.get('title', '')}\n\n"
            content += f"Description: {item.get('description', '')}\n\n"
            content += f"Content:\n{item.get('markdown', '')}\n\n"
            content += f"Source: {item.get('sourceURL', '')}"
            documents.append(content)
        return "\n\n".join(documents)
    else:
        logger.error(f"Unexpected crawl result format: {result}")
        return None
