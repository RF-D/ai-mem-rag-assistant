import sys
import concurrent.futures
from tools.sitemap_loader import get_xml
from tools.text_splitter import split_text
from tools.voyage_embeddings import vo_embed
from langchain_pinecone import PineconeVectorStore
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def upsert_batch(batch, embeddings, index_name):
    PineconeVectorStore.from_documents(
        documents=batch, embedding=embeddings, index_name=index_name)
    
def scrape_sitemap(url, index_name, progress_callback):
    site = get_xml(url)
    docs = split_text(site)
    embeddings = vo_embed()

    def estimate_batch_size(batch):
        return sys.getsizeof(str(batch))  # Rough estimate

    def process_batch(batch):
        upsert_batch(batch, embeddings, index_name)
        return len(batch)

    max_payload_size = 2 * 1024 * 1024  # 2MB in bytes

    current_batch = []
    current_size = 0
    total_docs = len(docs)
    processed_docs = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_batch = {}
        for doc in docs:
            doc_size = estimate_batch_size(doc)
            if current_size + doc_size > max_payload_size:
                future = executor.submit(process_batch, current_batch)
                future_to_batch[future] = len(current_batch)
                current_batch = [doc]
                current_size = doc_size
            else:
                current_batch.append(doc)
                current_size += doc_size

        if current_batch:
            future = executor.submit(process_batch, current_batch)
            future_to_batch[future] = len(current_batch)

        for future in concurrent.futures.as_completed(future_to_batch):
            batch_size = future_to_batch[future]
            processed_docs += batch_size
            progress_callback(processed_docs, total_docs)