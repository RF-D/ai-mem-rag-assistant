

# modules/voyageai_setup.py
from langchain_voyageai import VoyageAIEmbeddings
from utils.env_loader import load_env_vars


api_key = load_env_vars()[1]


def setup_voyageai(model):
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=api_key, model=model)
    return embeddings
