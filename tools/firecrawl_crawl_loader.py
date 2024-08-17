import time
import json
import logging
import requests
from langchain_community.document_loaders import FireCrawlLoader
from langchain.schema import Document  # Add this import
from firecrawl import FirecrawlApp
import streamlit as st

from dotenv import load_dotenv

load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default FireCrawl Setup
default_crawl_params = {
    "crawlerOptions": {
        "excludes": [],
        "includes": [],
        "maxDepth": 3,
        "limit": 50,
        "crawldelay": 0.5,
    },
    "pageOptions": {"onlyMainContent": True},
}


def get_default_crawl_params():
    return default_crawl_params


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
        crawl_params_to_use = params if params is not None else default_crawl_params
        logger.info(f"Crawl parameters: {crawl_params_to_use}")

        try:
            crawl_result = app.crawl_url(
                url, params=crawl_params_to_use, wait_until_done=wait_until_done
            )
            logger.info(f"Raw crawl result: {crawl_result}")
        except Exception as e:
            logger.error(f"Exception during crawl_url call: {str(e)}", exc_info=True)
            return None

        if not crawl_result:
            logger.error("Crawl result is empty. The API call might have failed.")
            st.error(
                "Crawl result is empty. The API call might have failed. Please check your API key and parameters."
            )
            return None

        # Check if the result is already in the expected format
        if (
            isinstance(crawl_result, list)
            and len(crawl_result) > 0
            and "markdown" in crawl_result[0]
        ):
            logger.info("Crawl completed successfully")
            return "\n\n".join([item["markdown"] for item in crawl_result])

        # If we don't have a jobId, assume the crawl is already complete
        if "jobId" not in crawl_result:
            logger.warning(
                "No jobId in crawl result. Assuming crawl is already complete."
            )
            return process_crawl_result(crawl_result)

        job_id = crawl_result["jobId"]
        logger.info(f"Job ID: {job_id}")

        if not wait_until_done:
            return crawl_result

        start_time = time.time()
        status_text = st.empty()

        while True:
            status = app.check_crawl_status(job_id)
            logger.info(f"Current status: {status['status']}")
            status_text.text(f"Crawling: {status['status']}")
            if status["status"] in ["completed", "failed"]:
                break

            if time.time() - start_time > timeout:
                logger.warning("Crawl job timed out")
                status_text.text("Crawl job timed out")
                return None

            time.sleep(check_interval)

        # If the crawl is completed, process the result
        if status["status"] == "completed":
            logger.info("Crawl completed successfully")

            # Save the raw result to a file
            with open(f"crawl_result_{job_id}.json", "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4)

            # Process the data for Langchain
            documents = []
            for item in status.get("data", []):
                content = f"Title: {item.get('title', '')}\n\nDescription: {item.get('description', '')}\n\nContent:\n{item.get('markdown', '')}\n\nSource: {item.get('sourceURL', '')}"
                documents.append(content)

            return "\n\n".join(documents)  # Return as a single string
        else:
            logger.error(f"Crawl failed with status: {status['status']}")
            st.error(f"Crawl failed with status: {status['status']}")
            return None

    except Exception as e:
        logger.error(f"Error during crawling: {str(e)}", exc_info=True)
        st.error(f"Error during crawling: {str(e)}")
        return None


def process_crawl_result(result):
    if isinstance(result, list) and len(result) > 0:
        documents = []
        for item in result:
            content = f"Title: {item.get('metadata', {}).get('title', '')}\n\n"
            content += (
                f"Description: {item.get('metadata', {}).get('description', '')}\n\n"
            )
            content += f"Content:\n{item.get('markdown', '')}\n\n"
            content += f"Source: {item.get('metadata', {}).get('sourceURL', '')}"
            documents.append(content)
        return "\n\n".join(documents)
    else:
        logger.error(f"Unexpected crawl result format: {result}")
        return None
