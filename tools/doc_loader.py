from langchain.document_loaders import TextLoader, UnstructuredFileLoader, PyPDFLoader
import os

def load_documents(file_path=None, text=None):
    if file_path:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Get the file extension
        _, file_extension = os.path.splitext(file_path)

        # Choose the appropriate loader based on the file extension
        if file_extension.lower() == '.txt':
            loader = TextLoader(file_path)
        elif file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        else:
            # Use UnstructuredFileLoader for other file types
            loader = UnstructuredFileLoader(file_path)
    elif text:
        # If text is provided directly, use TextLoader with a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(text)
            temp_file_path = temp_file.name
        loader = TextLoader(temp_file_path)
    else:
        return None

    try:
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return None
    finally:
        # Clean up the temporary file if it was created
        if 'temp_file_path' in locals():
            os.remove(temp_file_path)