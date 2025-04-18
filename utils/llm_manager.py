from functools import lru_cache
from typing import Dict, ClassVar, List, Optional, Iterator, Any
from sys import stdout
import re

# Langchain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult, BaseMessage, ChatGeneration, AIMessage
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning
import warnings
from requests.exceptions import SSLError


# Local imports for Ollama
import os
import subprocess
import time
import atexit
import psutil


# Streamlit is needed for type hinting and accessing session state
import streamlit as st

from pydantic import Field

from openai import OpenAI


class ChatGrok(BaseChatModel):
    """Custom wrapper for xAI's Grok API"""

    api_key: str = Field(...)  # ... means required
    model: str = Field(default="grok-beta")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    streaming: bool = Field(default=True)
    base_url: str = "https://api.x.ai/v1"
    client: Any = Field(default=None)

    # Add role mapping as ClassVar to indicate it's not a model field
    ROLE_MAP: ClassVar[Dict[str, str]] = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "grok-beta",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @property
    def _llm_type(self) -> str:
        """Return identifier for the LLM type"""
        return "grok"

    def _map_role(self, role: str) -> str:
        """Map LangChain roles to OpenAI roles"""
        return self.ROLE_MAP.get(role, "user")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted_messages = [
            {"role": self._map_role(msg.type), "content": msg.content}
            for msg in messages
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                stream=False,
                **kwargs,
            )

            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=response.choices[0].message.content),
                        generation_info={
                            "usage": response.usage.dict() if response.usage else {}
                        },
                    )
                ]
            )
        except Exception as e:
            print(f"Error in ChatGrok _generate: {str(e)}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatResult]:
        formatted_messages = [
            {"role": self._map_role(msg.type), "content": msg.content}
            for msg in messages
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                stream=True,
                **kwargs,
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(
                                    content=chunk.choices[0].delta.content
                                )
                            )
                        ]
                    )
        except Exception as e:
            print(f"Error in ChatGrok _stream: {str(e)}")
            raise


class LLMManager:
    provider_models = {
        "Anthropic": [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            
        ],
        "OpenAI": [
            "gpt-4.1-2025-04-14",
            "o4-mini-2025-04-16",
            "o3-2025-04-16",
            "gpt-4.5-preview-2025-02-27",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
            "gpt-3.5-turbo",
            "o3-mini-2025-01-31",
            "o1-2024-12-17",
            "o1-mini-2024-09-12",
        ],
        "Groq": [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-qwen-32b"
            
        ],
        "Mistral": [
            "mistral-large-latest",
            "open-mixtral-8x22b",
            "codestral-latest",
            "open-codestral-mamba",
            "open-mistral-nemo",
        ],
        "Ollama": [],
        "OpenRouter": [
            "openai/o1-mini-2024-09-12",
            "openai/o1-preview-2024-09-12",
            "cohere/command-r-08-2024",
            "google/gemini-pro-1.5",
            "qwen/qwen-2.5-72b-instruct",
        ],
        "xAI": ["grok-beta"],
    }
    # Mapping of model names to their context window sizes (max tokens)
    MODEL_CONTEXT_WINDOWS = {
        # OpenAI
        "gpt-4.1-2025-04-14": 1_047_576,
        "gpt-4.5-preview-2025-02-27": 128_000,
        "gpt-4o-mini-2024-07-18": 128_000,
        "gpt-4o-2024-08-06": 128_000,
        "gpt-3.5-turbo": 16_385,
        "o3-mini-2025-01-31": 200_000,
        "o3-2025-04-16": 200_000,
        "o4-mini-2025-04-16": 200_000,
        "o1-2024-12-17": 200_000,
        "o1-mini-2024-09-12": 128_000,
        # Anthropic
        "claude-3-5-haiku-latest": 200_000,
        "claude-3-5-sonnet-latest": 200_000,
        "claude-3-opus-latest": 200_000,
        "claude-3-7-sonnet-latest": 200_000,
    }
    DEFAULT_MAX_HISTORY_TOKENS = 128_000

    @staticmethod
    def get_max_history_tokens(provider: str, model: str) -> int:
        # Only OpenAI and Anthropic have special cases, others use default
        key = model
        if provider in ("OpenAI", "Anthropic"):
            return LLMManager.MODEL_CONTEXT_WINDOWS.get(key, LLMManager.DEFAULT_MAX_HISTORY_TOKENS)
        return LLMManager.DEFAULT_MAX_HISTORY_TOKENS

    ollama_process = None

    @staticmethod
    def check_api_key(provider: str) -> bool:
        env_var_names = {
            "Anthropic": "ANTHROPIC_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Groq": "GROQ_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
            "OpenRouter": "OPENROUTER_API_KEY",
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
            return None

        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        if not api_key:
            return None

        # Models that do NOT support temperature
        no_temp_models = {"o3-2025-04-16", "o4-mini-2025-04-16", "o3-mini-2025-01-31", "o1-2024-12-17", "o1-mini-2024-09-12"}

        if provider == "OpenAI":
            if model in no_temp_models:
                return ChatOpenAI(api_key=api_key, model=model)
            else:
                return ChatOpenAI(api_key=api_key, model=model, temperature=0.7)
        elif provider == "Ollama":
            LLMManager.initialize_ollama_models()
        else:
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                return None

        providers = {
            "Anthropic": lambda api_key: ChatAnthropic(
                api_key=api_key, model=model, temperature=0.6
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
            "OpenRouter": lambda api_key: ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7,
            ),
            "xAI": lambda api_key: ChatGrok(
                api_key=api_key, model=model, temperature=0.7, streaming=True
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
    def get_token_usage_percentage(st_session_state, provider: str, model: str):
        max_tokens = LLMManager.get_max_history_tokens(provider, model)
        return (st_session_state.total_tokens / max_tokens) * 100

    @staticmethod
    def display_token_counts(st_sidebar, st_session_state, provider: str, model: str):
        st_sidebar.write(f"Prompt tokens: {st_session_state.total_prompt_tokens}")
        st_sidebar.write(
            f"Completion tokens: {st_session_state.total_completion_tokens}"
        )
        st_sidebar.write(f"Total tokens: {st_session_state.total_tokens}")
        max_tokens = LLMManager.get_max_history_tokens(provider, model)
        st_sidebar.write(f"Max history tokens: {max_tokens}")
        usage_percentage = LLMManager.get_token_usage_percentage(st_session_state, provider, model)
        st_sidebar.progress(usage_percentage / 100)
        st_sidebar.write(f"Token usage: {usage_percentage:.2f}%")

    @staticmethod
    def validate_api_key(provider: str, api_key: str) -> bool:
        try:
            if provider == "Anthropic":
                ChatAnthropic(
                    api_key=api_key, model_name="claude-3-5-haiku-latest"
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
            elif provider == "OpenRouter":
                ChatOpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="qwen/qwen-2-vl-7b-instruct:free",
                ).invoke("Test")
            elif provider == "xAI":
                ChatGrok(api_key=api_key, model="grok-beta").invoke("Test")
            else:
                return False  # Unsupported provider
            return True
        except Exception as e:
            print(f"API key validation failed for {provider}: {str(e)}")
            return False

    def _create_session(self):
        """Create a requests session with retry strategy and SSL verification handling"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)

        # Disable SSL verification warnings if needed
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)

        return session

    def get_embeddings(self, texts):
        """Get embeddings with proper error handling and SSL verification"""
        try:
            # First try with SSL verification
            response = self.client.embeddings.create(
                model=self.embedding_model, input=texts
            )
            return response
        except SSLError:
            # If SSL verification fails, try with verification disabled
            # Note: This is a fallback solution and should be used cautiously
            import os

            os.environ["REQUESTS_CA_BUNDLE"] = (
                ""  # Temporarily disable SSL cert verification
            )
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model, input=texts
                )
                return response
            finally:
                # Reset the environment variable
                os.environ.pop("REQUESTS_CA_BUNDLE", None)
        except Exception as e:
            # Handle other potential errors
            print(f"Error getting embeddings: {str(e)}")
            raise
