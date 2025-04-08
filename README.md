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

Use `pip` to install the libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Ollama

Use Homebrew to setup Ollama and then install your first model.

```bash
brew install ollama

ollama run llama3:latest
```

This app will start up Ollama as part of its run environment but you can also
have Ollama running in stand-alone.

```bash
ollama serve
```

### Virtual Environment

You have two options for managing your Python environment:

#### Option 1: Using venv

If you used Homebrew to install python, you'll get an error about your
environment being externally managed when you try to install with `pip`.

> error: externally-managed-environment

To resolve, use Python's virtual environments feature to install the packages.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Verify your using the version of python from the virtual environment (rather than the system)
which python #=> should list this project's virtual env

# Use the virtual environment's version of Python to install your pip dependencies
python -m pip install -r requirements.txt # note the use of python instead of python3 (system)

# When you're done, deactivate the virtual environment
deactivate
```

#### Option 2: Using Conda

Alternatively, you can use Conda to manage your Python environment:

```bash
# Create a new conda environment
conda create --name ai-mem-rag

# Activate the conda environment
conda activate ai-mem-rag

#Make sure pip is installed in your conda env 
conda install pip

# Install the requirements
pip install -r requirements.txt

# When you're done, deactivate the conda environment
conda deactivate
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

##### Langsmith

Visit https://smith.langchain.com/ to create an API key

#### Optional Keys

You can start right away with a model from Ollama but if you'd like to query
another model, you'll need to setup keys and billing.

#### Anthropic

Visit the Anthropic console to create an API key
https://console.anthropic.com/dashboard

#### Groq

Visit the Groq console to create an API key
https://console.groq.com

#### OpenAI

Visit the OpenAI platform to create an API key
https://platform.openai.com/

#### Firecrawl

Visit https://www.firecrawl.dev/app to create an API key

## Usage

Start the Streamlit application.

```bash
# Activate your venv if you're using a virtual environment.
# source venv/bin/activate

streamlit run st_ui_rag.py

# Remember to deactive your virtual environment when done.
# deactivate
```

You can now view the Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.68.61:8501
