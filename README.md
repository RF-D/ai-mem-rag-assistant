# agent_claude

## Setup

### Requirements

#### Python

Use `pip` to install the libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

#### Ollama

```bash
brew install ollama
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

#### Voyage AI

Visit https://www.voyageai.com/ to setup an account and generate an API key.

#### Pinecone

Visit https://www.pinecone.io/ to setup an account. 

Feel free to use Pinecone's curl script to create your first index (called
"quickstart"). However, for this project, you'll need to create a new index
called "langchain".

Use the same options as the quickstart curl script to create the "langchanin"
index.

## Usage

Start the Streamlit application

```bash
# Activate your venv if you're using a virtual environment
# source venv/bin/activate

streamlit run st_ui_rag.py

# Remember to deactive your virtual environment
# deactivate
```

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.68.61:8501

