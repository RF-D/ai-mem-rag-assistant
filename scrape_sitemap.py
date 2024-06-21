from tqdm.auto import tqdm
from tools.sitemap_loader import get_xml
from tools.text_splitter import split_text
from tools.retriever_tools import retriever_tool
from tools.voyage_embeddings import vo_embed
from langchain_pinecone import PineconeVectorStore


def scrape_sitemap(url, index_name, progress_callback):
    # Provide sitemap url here
    site = get_xml(url)

    docs = split_text(site)

    embeddings = vo_embed()

    total_docs = len(docs)
    for i, doc in enumerate(docs):
        PineconeVectorStore.from_documents(
            documents=[doc], embedding=embeddings, index_name=index_name)

        # Update the progress bar
        progress_callback(i + 1, total_docs)


