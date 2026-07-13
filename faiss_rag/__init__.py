"""
faiss_rag — FIELD-MIND Offline FAISS Vector Database (RAG)
===========================================================
Provides retrieval-augmented generation (RAG) context for the FIELD-MIND
Scientific Reasoning Core (Layer 3). Indexes scientific literature, mining
safety regulations, incident protocols, and operational manuals into a
FAISS vector store for offline semantic search.

Components
----------
DocumentChunker     — load .md/.txt docs, split into overlapping windows
SentenceEmbedder    — embed text chunks via all-MiniLM-L6-v2 (22 MB, CPU)
FAISSIndexBuilder   — build & persist FAISS IndexFlatIP to disk
RAGRetriever        — query the index, return ranked chunks + LLM context

Usage
-----
# One-time build:
from faiss_rag import RAGRetriever
retriever = RAGRetriever.build_from_directory(
    docs_dir="faiss_rag/knowledge_base",
    save_dir="faiss_rag/data"
)

# Query:
results = retriever.retrieve("methane safe concentration limit", top_k=5)
context = retriever.format_context(results)
"""

from .chunker       import DocumentChunker
from .embedder      import SentenceEmbedder
from .index_builder import FAISSIndexBuilder
from .retriever     import RAGRetriever

__all__ = [
    "DocumentChunker",
    "SentenceEmbedder",
    "FAISSIndexBuilder",
    "RAGRetriever",
]
