# ai-mem-rag-assistant

## App Capabilities

AI Memory is an advanced conversational RAG (Retrieval-Augmented Generation) assistant with persistent memory capabilities. The application leverages multiple large language models (LLMs) to provide accurate, contextually relevant responses by retrieving information from various sources.

### Key Features:

1. **Multi-provider LLM Support**: Seamlessly integrate with various AI providers including Anthropic, OpenAI, Groq, and Ollama for both response generation and information retrieval tasks.

2. **Persistent Memory Storage**: Efficiently store and retrieve information in vector databases (Pinecone) to maintain context across conversations.

3. **Multiple Content Ingestion Methods**:
   - **Web Crawling**: Intelligently crawl websites with configurable parameters
   - **Sitemap Scraping**: Extract and process content from website sitemaps
   - **YouTube Analysis**: Process video content and transcripts from YouTube URLs
   - **Document Upload**: Support for various document formats (TXT, PDF, DOCX)
   - **Manual Text Input**: Directly paste text to be processed and stored

4. **Adaptive Token Management**: Automatically manages conversation history to stay within token limits while preserving context.

5. **Robust Error Handling**: Implements retry mechanisms and graceful degradation for API errors.

6. **User-friendly Interface**: Built with Streamlit for intuitive interaction and configuration.

7. **Vector Database Integration**: Uses Pinecone for efficient semantic search capabilities.

8. **Contextualized Embeddings**: Utilizes Voyage AI's instruction-tuned embeddings (voyage-large-2-instruct) for superior semantic search performance. These state-of-the-art contextualized embeddings significantly improve RAG accuracy by:
   - Capturing nuanced semantic relationships between concepts
   - Better understanding of context and intent in queries
   - Reducing retrieval of irrelevant information
   - Enhancing the precision of knowledge retrieval for complex topics
   - Maintaining contextual coherence across different domains

This application is ideal for researchers, content creators, developers, and anyone needing to have extended conversations with AI that can recall specific information from diverse sources.

## Setup

### Requirements

#### Python

You should first set up your Python environment before installing dependencies. You have two options:

#### Option 1: Using venv

If you used Homebrew to install python, you'll get an error about your environment being externally managed when you try to install with `pip`.

> error: externally-managed-environment

To resolve, use Python's virtual environments feature to install the packages.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Verify you're using the version of python from the virtual environment (rather than the system)
which python #=> should list this project's virtual env

# Use the virtual environment's version of Python to install your pip dependencies
python -m pip install -r requirements.txt # note the use of python instead of python3 (system)

```

#### Option 2: Using Conda

If you don't have Conda installed, download it from the official website: https://docs.conda.io/en/latest/miniconda.html

Alternatively, you can use Conda to manage your Python environment:

```bash
# Create a new conda environment
conda create --name ai-mem-rag

# Activate the conda environment
conda activate ai-mem-rag

# If you installed Miniconda, you may need to install pip first:
conda install pip

# Install the requirements
pip install -r requirements.txt

```

### Ollama (Optional: For Local Models)

Ollama is only required if you want to use local models. If you plan to use only cloud-based models (OpenAI, Anthropic, Groq, etc.), you can skip this section.

Use Homebrew to setup Ollama and then install your first model.

```bash
brew install ollama

ollama run llama3:latest
```

This app will start up Ollama as part of its run environment but you can also have Ollama running in stand-alone.

```bash
ollama serve
```

### API Keys

Setup the service API keys by duplicating the example dotenv file.

```bash
cp config/.env.example config/.env
```

#### Required Keys

##### Voyage AI

Visit https://www.voyageai.com/ to setup an account and generate an API key.

Add billing information to avoid rate limit error messages.

##### Pinecone

Visit https://www.pinecone.io/ to setup an account.

Feel free to use Pinecone's curl script to create your first index (called
"quickstart").

Use the same options as the quickstart curl script to create new indexes.

##### Langsmith (Optional: For Tracing)

Langsmith is only required if you want to enable tracing of your LLM calls and workflows. If you do not need tracing, you can skip this step.

Visit https://smith.langchain.com/ to create an API key

#### Optional Keys

You can start right away with a model from Ollama but if you'd like to query
another model, you'll need to setup keys and billing.

#### OpenAI

Visit the OpenAI platform to create an API key
https://platform.openai.com/

#### Anthropic

Visit the Anthropic console to create an API key
https://console.anthropic.com/dashboard

#### Groq

Visit the Groq console to create an API key
https://console.groq.com

#### Mistral

Visit the Mistral AI website to create an API key
https://www.mistral.ai/

#### xAI

Visit the xAI developer portal to create an API key
https://x.ai/api

#### Firecrawl (Optional: For Web Scraping)

Firecrawl is used for web scraping capabilities. If you want to enable web crawling and scraping features, set up a Firecrawl API key. Otherwise, you can skip this step.

Visit https://www.firecrawl.dev/app to create an API key

## Usage

Start the Streamlit application.

```bash
# Activate your venv if you're using a virtual environment.
# source venv/bin/activate or conda activate 

streamlit run st_ui_rag.py

# Remember to deactive your virtual environment when done using the application.
# deactivate  or conda deactivate
```

You can now view the Streamlit app in your browser.

