# agent_claude

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

