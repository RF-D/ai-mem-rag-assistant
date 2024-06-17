from langchain_community.document_loaders import FireCrawlLoader
from utils.env_loader import load_env_vars

firecrawl_api_key = load_env_vars()[2]


# FireCrawl Setup
crawl_params = {
    'crawlerOptions': {
        'excludes': ['blog/*'],
        'includes': [],  # leave empty for all pages
        'pagelimit': 1000,
    }

}


def crawl(url, params=crawl_params):
    loader = FireCrawlLoader(api_key=firecrawl_api_key,
                             url=url, mode="crawl", params=params)
    return loader.load()
