from tqdm.auto import tqdm
from tools.sitemap_loader import get_xml
from tools.text_splitter import split_text
from tools.retriever_tools import retriever_tool
from tools.voyage_embeddings import vo_embed
from langchain_pinecone import PineconeVectorStore

#Provide sitemap url here
site = get_xml("https://docs.streamlit.io/sitemap.xml")

docs = split_text(site)

embeddings = vo_embed()

# Wrap the documents with tqdm
docs_with_progress = tqdm(docs, desc="Loading documents")

PineconeVectorStore.from_documents(
            documents=docs_with_progress, embedding=embeddings, index_name="langchain")


