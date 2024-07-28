from langchain_community.embeddings import OllamaEmbeddings

def ollama_embed():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings