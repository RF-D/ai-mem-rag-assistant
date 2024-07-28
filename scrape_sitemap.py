import sys
import concurrent.futures
from tools.sitemap_loader import get_xml
from tools.text_splitter import split_text
from tools.voyage_embeddings import vo_embed
from langchain_pinecone import PineconeVectorStore
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import time

logging.basicConfig(level=logging.INFO)

# Define rate limit: 250 calls per 60 seconds (leaving some buffer)
CALLS = 250
RATE_LIMIT = 60

def rate_limited(max_per_minute):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    def decorate(func):
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return rate_limited_function
    return decorate

@rate_limited(CALLS)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def upsert_batch(batch, embeddings, index_name):
    try:
        PineconeVectorStore.from_documents(
            documents=batch, embedding=embeddings, index_name=index_name)
        logging.info(f"Successfully upserted batch of {len(batch)} documents")
    except Exception as e:
        logging.error(f"Error upserting batch: {str(e)}")
        raise

def scrape_sitemap(url, index_name, progress_callback):
    try:
        site = get_xml(url)
        docs = split_text(site)
        embeddings = vo_embed()

        def estimate_batch_size(batch):
            return sys.getsizeof(str(batch))  # Rough estimate

        def process_batch(batch):
            upsert_batch(batch, embeddings, index_name)
            return len(batch)

        max_payload_size = 4 * 1024 * 1024  # Increased to 4MB

        current_batch = []
        current_size = 0
        total_docs = len(docs)
        processed_docs = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
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
                try:
                    batch_size = future.result()
                    processed_docs += batch_size
                    progress_callback(processed_docs, total_docs)
                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")

        logging.info(f"Completed processing {processed_docs} out of {total_docs} documents")
    except Exception as e:
        logging.error(f"Error in scrape_sitemap: {str(e)}")
        raise

    return processed_docs