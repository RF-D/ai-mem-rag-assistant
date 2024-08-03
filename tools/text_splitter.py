from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
import re


def split_text(data, chunk_size=1297, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return text_splitter.split_documents(data)


def split_paste(data, chunk_size=1297, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return text_splitter.split_text(data)


def split_md(data, chunk_size=2000, chunk_overlap=300):
    def markdown_text_splitter(text, max_length=chunk_size):
        chunks = []
        current_chunk = ""

        # Split the document into sections based on Markdown headings
        sections = re.split(r"(#+\s+.+)", text)
        for section in sections:
            if section.strip() == "":
                continue

            # Check if the section is a heading
            if section.startswith("#"):
                if current_chunk.strip() != "":
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                current_chunk += section + "\n"
            else:
                # Split the section into paragraphs
                paragraphs = section.split("\n\n")
                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) > max_length:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    current_chunk += paragraph + "\n\n"

        if current_chunk.strip() != "":
            chunks.append(current_chunk.strip())

        return chunks

    md_docs = []
    start_index = 0
    for doc in data:
        text = doc.page_content
        chunks = markdown_text_splitter(text)

        # Apply RecursiveCharacterTextSplitter on each chunk
        for chunk in chunks:
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            recursive_chunks = recursive_splitter.split_text(chunk)

            for recursive_chunk in recursive_chunks:
                metadata = doc.metadata.copy()
                metadata["start_index"] = start_index
                md_docs.append(
                    Document(page_content=recursive_chunk, metadata=metadata)
                )
                start_index += len(recursive_chunk)

    return md_docs
