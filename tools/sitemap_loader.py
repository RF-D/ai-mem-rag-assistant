# fixes a bug with asyncio and jupyter

from langchain_community.document_loaders.sitemap import SitemapLoader



def get_xml(url):

    sitemap_loader = SitemapLoader(web_path=url)
    sitemap_loader.requests_per_second = 3
    
    # Optional: avoid `[SSL: CERTIFICATE_VERIFY_FAILED]` issue
    sitemap_loader.requests_kwargs = {"verify": False}
    docs = sitemap_loader.load()
    return docs