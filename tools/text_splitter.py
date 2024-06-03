from langchain_text_splitters import CharacterTextSplitter


def split_text(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)
