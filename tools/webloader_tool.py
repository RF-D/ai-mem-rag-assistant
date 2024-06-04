from langchain_community.document_loaders import WebBaseLoader


def load_web_url(url):
    loader = WebBaseLoader(url)

    # Load documents from the specified URL
    data = loader.load()
    return data
