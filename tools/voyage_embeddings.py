

# modules/voyageai_setup.py
from langchain_voyageai import VoyageAIEmbeddings
from utils.env_loader import load_env_vars
import voyageai


api_key = load_env_vars()[1]
vo = voyageai.Client(api_key=api_key)


def setup_voyageai(model):
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=api_key, model=model)
    return embeddings


def text_voyageai(text, model, input_type):
    text_embeddings = vo.embed(text, model=model, input_type=input_type)
    return text_embeddings
