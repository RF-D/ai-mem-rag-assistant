from functools import lru_cache
from typing import Dict
from functools import lru_cache

# Langchain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Streamlit is needed for type hinting and accessing session state
import streamlit as st

class LLMManager:
    provider_models = {
        "Anthropic": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
        "OpenAI": ["gpt-3.5-turbo", "gpt-4o"],
        "Groq": ["llama3-70b-8192"],
    }
    MAX_HISTORY_TOKENS = 40000

    @staticmethod
    @lru_cache(maxsize=None)
    def load_llm(provider: str, model: str):
        providers = {
            "Anthropic": lambda: ChatAnthropic(model=model, temperature=0.7, streaming=True),
            "OpenAI": lambda: ChatOpenAI(model=model, temperature=0.7),
            "Groq": lambda: ChatGroq(model_name="llama3-70b-8192", temperature=0.7)
        }
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")
        if model not in LLMManager.provider_models[provider]:
            raise ValueError(f"Unsupported model {model} for provider {provider}")
        return providers[provider]()

    @staticmethod
    def count_tokens(text: str, provider: str, model: str) -> int:
        llm = LLMManager.load_llm(provider, model)
        return llm.get_num_tokens(text)

    @staticmethod
    def get_provider_models():
        return LLMManager.provider_models

    @staticmethod
    def get_models_for_provider(provider: str):
        return LLMManager.provider_models.get(provider, [])

    @staticmethod
    def calculate_total_tokens(prompt_tokens: int, completion_tokens: int) -> int:
        return prompt_tokens + completion_tokens
     
    @staticmethod
    def update_token_count(st_session_state, prompt_tokens: int, completion_tokens: int):
        st_session_state.total_prompt_tokens += prompt_tokens
        st_session_state.total_completion_tokens += completion_tokens
        st_session_state.total_tokens = LLMManager.calculate_total_tokens(
            st_session_state.total_prompt_tokens, 
            st_session_state.total_completion_tokens
        )

    @staticmethod
    def reset_token_count(st_session_state, new_prompt_tokens: int):
        st_session_state.total_prompt_tokens = new_prompt_tokens
        st_session_state.total_completion_tokens = 0
        st_session_state.total_tokens = new_prompt_tokens

    @staticmethod
    def get_token_usage_percentage(st_session_state):
        return (st_session_state.total_tokens / LLMManager.MAX_HISTORY_TOKENS) * 100    
    @staticmethod
    def display_token_counts(st_sidebar, st_session_state):
        st_sidebar.write(f"Prompt tokens: {st_session_state.total_prompt_tokens}")
        st_sidebar.write(f"Completion tokens: {st_session_state.total_completion_tokens}")
        st_sidebar.write(f"Total tokens: {st_session_state.total_tokens}")
        st_sidebar.write(f"Max history tokens: {LLMManager.MAX_HISTORY_TOKENS}")
        usage_percentage = LLMManager.get_token_usage_percentage(st_session_state)
        st_sidebar.progress(usage_percentage / 100)
        st_sidebar.write(f"Token usage: {usage_percentage:.2f}%")