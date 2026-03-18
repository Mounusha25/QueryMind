"""
RAG Layer – Document Loading
Supports PDF files and whole directories.
"""
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_core.documents import Document


def load_documents(source: str) -> List[Document]:
    """Load documents from a single PDF path or a directory of PDFs/TXTs.

    Args:
        source: Path to a PDF file or a directory.

    Returns:
        List of LangChain Document objects with page_content and metadata.
    """
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    elif path.is_dir():
        # Load PDFs
        pdf_loader = DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        txt_loader = DirectoryLoader(
            str(path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs = pdf_loader.load() + txt_loader.load()
    else:
        raise ValueError(f"Source must be a file or directory: {source}")

    # Normalise metadata: make sure 'source' key is always a string
    for doc in docs:
        doc.metadata.setdefault("source", str(path))

    print(f"[Loader] Loaded {len(docs)} document chunks from '{source}'")
    return docs
