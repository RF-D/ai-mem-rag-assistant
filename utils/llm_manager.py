from functools import lru_cache
from typing import Dict
from sys import stdout
import re

# Langchain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_mistralai.chat_models import ChatMistralAI


# Local imports for Ollama
import os
import subprocess
import time
import atexit
import psutil


# Streamlit is needed for type hinting and accessing session state
import streamlit as st


class LLMManager:
    provider_models = {
        "Anthropic": [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
        ],
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "Groq": [
            "llama-3.1-70b-versatile",
            "llama-3.1-405b-reasoning",
            "mixtral-8x7b-32768",
            "llama3-groq-70b-8192-tool-use-preview",
        ],
        "Mistral": [
            "mistral-large-latest",
            "open-mixtral-8x22b",
            "codestral-latest",
            "open-codestral-mamba",
            "open-mistral-nemo",
        ],
        "Ollama": [],
    }
    # TODO: Make this dynamic based on the model
    MAX_HISTORY_TOKENS = 200000
    ollama_process = None

    @staticmethod
    def check_api_key(provider: str) -> bool:
        env_var_names = {
            "Anthropic": "ANTHROPIC_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Groq": "GROQ_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
        }
        return bool(os.getenv(env_var_names.get(provider, "")))

    @staticmethod
    def get_api_key_input(provider: str) -> str:
        return st.sidebar.text_input(f"Enter your {provider} API key", type="password")

    @staticmethod
    def start_ollama_server():
        # Check if Ollama is already running
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == "ollama":
                print("Ollama is already running.")
                return

        print("Starting Ollama server...")
        try:
            # Start Ollama server
            process = subprocess.Popen(
                ["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for the server to start (adjust the sleep time if needed)
            time.sleep(5)

            # Check if the process is running
            if process.poll() is None:
                print("Ollama server started successfully.")
            else:
                print("Failed to start Ollama server.")
                stdout, stderr = process.communicate()
                print(f"Error: {stderr.decode()}")
        except FileNotFoundError:
            print(
                "Ollama executable not found. Make sure Ollama is installed and in your PATH."
            )
        except Exception as e:
            print(f"An error occurred while starting Ollama: {str(e)}")

    @staticmethod
    def stop_ollama_server():
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == "ollama":
                proc.terminate()
                proc.wait()
                print("Ollama server stopped.")
                return
        print("Ollama server was not running.")

    @staticmethod
    def initialize_ollama_models():
        LLMManager.start_ollama_server()
        LLMManager.provider_models["Ollama"] = LLMManager.get_installed_ollama_models()

    @staticmethod
    def get_installed_ollama_models():
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                models = []
                for line in result.stdout.split("\n")[1:]:  # Skip the header line
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                print(f"Error running 'ollama list': {result.stderr}")
                return []
        except FileNotFoundError:
            print(
                "Ollama command not found. Make sure Ollama is installed and in your PATH."
            )
            return []

    @staticmethod
    def load_llm(provider: str, model: str):
        if not provider:
            return None  # Return None if no provider is selected

        if provider == "Ollama":
            LLMManager.initialize_ollama_models()
        else:
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                return None

        providers = {
            "Anthropic": lambda api_key: ChatAnthropic(
                api_key=api_key, model=model, temperature=0.6
            ),
            "OpenAI": lambda api_key: ChatOpenAI(
                api_key=api_key, model=model, temperature=0.7
            ),
            "Groq": lambda api_key: ChatGroq(
                api_key=api_key, model_name=model, temperature=0.9, max_tokens=2048
            ),
            "Ollama": lambda: ChatOllama(
                model=model,
                temperature=0.8,
                presence_penalty=0.2,
                n_sentence_context=2,
                streaming=True,
            ),
            "Mistral": lambda api_key: ChatMistralAI(
                api_key=api_key, model=model, temperature=1, streaming=True
            ),
        }
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")
        if model not in LLMManager.provider_models[provider]:
            raise ValueError(f"Unsupported model {model} for provider {provider}")

        return (
            providers[provider](api_key)
            if provider != "Ollama"
            else providers[provider]()
        )

    @staticmethod
    def handle_api_key_input(provider: str) -> str:
        env_var_name = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_var_name)
        if not api_key:
            api_key = st.sidebar.text_input(
                f"Enter your {provider} API key",
                type="password",
                key=f"{provider.lower()}_api_key_input",  # Use a unique key for each provider
            )
            if api_key:
                os.environ[env_var_name] = api_key
        return api_key

    @staticmethod
    def simple_token_count(text: str) -> int:
        # This is a very rough estimation and won't be as accurate as model-specific tokenizers
        return len(re.findall(r"\w+|[^\w\s]", text))

    @staticmethod
    def count_tokens(text: str, provider: str, model: str) -> int:
        try:
            llm = LLMManager.load_llm(provider, model)
            return llm.get_num_tokens(text)
        except Exception as e:
            print(f"Error using LLM token counter: {str(e)}. Using simple token count.")
            return LLMManager.simple_token_count(text)

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
    def update_token_count(
        st_session_state, prompt_tokens: int, completion_tokens: int
    ):
        st_session_state.total_prompt_tokens += prompt_tokens
        st_session_state.total_completion_tokens += completion_tokens
        st_session_state.total_tokens = LLMManager.calculate_total_tokens(
            st_session_state.total_prompt_tokens,
            st_session_state.total_completion_tokens,
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
        st_sidebar.write(
            f"Completion tokens: {st_session_state.total_completion_tokens}"
        )
        st_sidebar.write(f"Total tokens: {st_session_state.total_tokens}")
        st_sidebar.write(f"Max history tokens: {LLMManager.MAX_HISTORY_TOKENS}")
        usage_percentage = LLMManager.get_token_usage_percentage(st_session_state)
        st_sidebar.progress(usage_percentage / 100)
        st_sidebar.write(f"Token usage: {usage_percentage:.2f}%")

    @staticmethod
    def validate_api_key(provider: str, api_key: str) -> bool:
        try:
            if provider == "Anthropic":
                ChatAnthropic(
                    api_key=api_key, model_name="claude-3-sonnet-20240229"
                ).invoke("Test")
            elif provider == "OpenAI":
                ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo").invoke("Test")
            elif provider == "Groq":
                ChatGroq(api_key=api_key, model_name="mixtral-8x7b-32768").invoke(
                    "Test"
                )
            elif provider == "Mistral":
                ChatMistralAI(api_key=api_key, model="mistral-small-latest").invoke(
                    "Test"
                )
            else:
                return False  # Unsupported provider
            return True
        except Exception as e:
            print(f"API key validation failed for {provider}: {str(e)}")
            return False
