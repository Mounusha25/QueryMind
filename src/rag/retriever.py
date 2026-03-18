"""
RAG Layer – Retriever
Returns top-k chunks with similarity scores.
The score is later used by the Reviewer agent as a confidence signal.
"""
import os
from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    document: Document
    score: float  # cosine similarity, higher = more relevant

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def page(self) -> int | None:
        return self.document.metadata.get("page")


def build_retriever(vectorstore: FAISS):
    """Return a retrieval function bound to the given vectorstore.

    The returned function signature:
        retrieve(query: str, k: int | None = None) -> List[RetrievedChunk]
    """
    def retrieve(query: str, k: int | None = None) -> List[RetrievedChunk]:
        top_k = k or int(os.getenv("RETRIEVER_TOP_K", 5))
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        chunks = [RetrievedChunk(document=doc, score=float(score))
                  for doc, score in results]
        # Sort descending by score (FAISS returns L2 distance; lower = better)
        # Convert L2 distance to a 0-1 similarity for intuitive use downstream
        if chunks:
            max_score = max(c.score for c in chunks)
            for c in chunks:
                # Normalise: similarity = 1 - (dist / max_dist)
                c.score = round(1.0 - (c.score / (max_score + 1e-9)), 4)
        chunks.sort(key=lambda c: c.score, reverse=True)
        print(f"[Retriever] Query: '{query[:60]}…' → {len(chunks)} chunks retrieved")
        return chunks

    return retrieve
