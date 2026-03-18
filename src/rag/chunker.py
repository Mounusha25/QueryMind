"""
RAG Layer – Text Chunking
Splits documents into overlapping chunks using RecursiveCharacterTextSplitter.
Token-aware sizing via tiktoken.
"""
import os
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """Split documents into fixed-size, overlapping chunks.

    Args:
        documents:     Raw Document objects from the loader.
        chunk_size:    Target chunk size in characters (~500 chars ≈ 125 tokens).
                       Falls back to CHUNK_SIZE env var, then 2000 chars.
        chunk_overlap: Overlap between chunks. Falls back to CHUNK_OVERLAP env var, then 200.

    Returns:
        List of smaller Document chunks, each inheriting source metadata.
    """
    # ~4 chars per token → 500 tokens ≈ 2000 chars
    _size = chunk_size or int(os.getenv("CHUNK_SIZE", 2000))
    _overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_size,
        chunk_overlap=_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"[Chunker] {len(documents)} docs → {len(chunks)} chunks "
          f"(size={_size}, overlap={_overlap})")
    return chunks
