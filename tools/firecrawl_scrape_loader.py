# modules/firecrawl_setup.py
from langchain_community.document_loaders import FireCrawlLoader
from utils.env_loader import load_env_vars

firecrawl_api_key = load_env_vars()[2]


def scrape(url):
    loader = FireCrawlLoader(api_key=firecrawl_api_key, url=url, mode="scrape")
    return loader.load()
