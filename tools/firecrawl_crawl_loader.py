import time
import json
import logging
from langchain_community.document_loaders import FireCrawlLoader
from langchain.schema import Document  # Add this import
from utils.env_loader import load_env_vars
from firecrawl import FirecrawlApp
import streamlit as st

firecrawl_api_key = load_env_vars()[2]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FireCrawl Setup
crawl_params = {
    'crawlerOptions': {
        'excludes': [],
        'includes': [],  # leave empty for all pages
        'pagelimit': 5,
        'crawldelay': 0.5, 
    },
    'pageOptions': {
        'onlyMainContent': True
    }
}

def crawl(url, params=crawl_params, wait_until_done=True, timeout=300, check_interval=5):
    try:
        app = FirecrawlApp(api_key=firecrawl_api_key)

        # Start the crawl job
        crawl_result = app.crawl_url(url, params=params, wait_until_done=False)
        logger.info(f"Crawl job started: {crawl_result}")

        if not wait_until_done:
            return crawl_result

        # Check the status of the crawl job continually until done
        job_id = crawl_result['jobId']
        start_time = time.time()
        status_text = st.empty()

        while True:
            status = app.check_crawl_status(job_id)
            logger.info(f"Current status: {status['status']}")
            status_text.text(f"Crawling: {status['status']}")
            if status['status'] in ['completed', 'failed']:
                break

            if time.time() - start_time > timeout:
                logger.warning("Crawl job timed out")
                status_text.text("Crawl job timed out")
                return None

            time.sleep(check_interval)

        # If the crawl is completed, process the result
        if status['status'] == 'completed':
            logger.info("Crawl completed successfully")
            

            # Save the raw result to a file
            # with open(f"crawl_result_{job_id}.json", "w", encoding="utf-8") as f:
            #     json.dump(status, f, indent=4)

            # Process the data for Langchain
            documents = []
            for item in status.get('data', []):
                doc = Document(
                    page_content=item.get('markdown', ''),
                    metadata={
                        'url': item.get('sourceURL', ''),
                        'title': item.get('title', ''),
                        'description': item.get('description', '')
                    }
                )
                documents.append(doc)

            return documents
        else:
            logger.error(f"Crawl failed with status: {status['status']}")
            return None

    except Exception as e:
        logger.error(f"Error during crawling: {str(e)}", exc_info=True)
        return None