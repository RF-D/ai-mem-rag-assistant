from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)



def split_text(data, chunk_size=1297, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    return text_splitter.split_documents(data)


def split_paste(data, chunk_size=1297, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    return text_splitter.split_text(data)

def split_md(data, chunk_size=1297, chunk_overlap=0):
    md_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    md_docs = md_splitter.create_documents(data)
    return md_docs
