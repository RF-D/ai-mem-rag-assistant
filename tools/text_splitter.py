from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    return text_splitter.split_documents(data)


def split_paste(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    return text_splitter.split_text(data)
