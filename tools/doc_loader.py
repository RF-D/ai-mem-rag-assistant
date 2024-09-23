from langchain.document_loaders import (
    TextLoader,
    UnstructuredFileLoader,
    PyPDFLoader,
)
from langchain.schema import Document
import os


def load_documents(file_path=None, text=None):
    if file_path:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Get the file extension
        _, file_extension = os.path.splitext(file_path)

        # Choose the appropriate loader based on the file extension
        if file_extension.lower() == ".txt":
            loader = TextLoader(file_path)
        elif file_extension.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            # Use UnstructuredFileLoader for other file types
            loader = UnstructuredFileLoader(file_path)

        try:
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            return None

    elif text:
        # If text is provided directly, create a Document object
        try:
            return [Document(page_content=text, metadata={"source": "user_input"})]
        except Exception as e:
            print(f"Error creating document from text: {str(e)}")
            return None
    else:
        return None
