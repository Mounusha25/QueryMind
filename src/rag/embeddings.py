"""
RAG Layer – Embeddings & Vector Store
Uses OpenAI text-embedding-3-small and FAISS for local vector search.
Supports persistence: saves/loads from disk so you don't re-embed on every run.
"""
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def _get_embeddings() -> OpenAIEmbeddings:
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def build_or_load_vectorstore(
    chunks: List[Document] | None = None,
    index_path: str | None = None,
    force_rebuild: bool = False,
) -> FAISS:
    """Build a FAISS vector store from chunks, or load from disk if it exists.

    Args:
        chunks:        List of document chunks (required when building fresh).
        index_path:    Directory path to save/load the FAISS index.
                       Defaults to FAISS_INDEX_PATH env var → 'data/faiss_index'.
        force_rebuild: If True, always rebuild even when saved index exists.

    Returns:
        A FAISS vectorstore ready for similarity search.
    """
    _path = index_path or os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    embeddings = _get_embeddings()

    if not force_rebuild and Path(_path).exists():
        print(f"[Embeddings] Loading existing FAISS index from '{_path}'")
        vectorstore = FAISS.load_local(
            _path, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore

    if not chunks:
        raise ValueError("chunks must be provided when building a new FAISS index")

    print(f"[Embeddings] Building FAISS index for {len(chunks)} chunks …")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    Path(_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(_path)
    print(f"[Embeddings] FAISS index saved to '{_path}'")
    return vectorstore
