"""
retriever.py — RAG Retriever (Main Public Interface)
=====================================================
High-level query interface over the FAISS vector index.
This is what the Scientific Reasoning Core (Layer 3 / LangGraph agent)
calls to retrieve relevant knowledge grounding before generating advice.

Workflow
--------
1. Embed the query string with SentenceEmbedder
2. Search the FAISS index for top-K nearest neighbours (cosine similarity)
3. Return ranked chunks with metadata and similarity scores
4. Format results as a compact LLM-ready context block

Integration Point
-----------------
The RAGRetriever is designed to be called from:
  - MineOrchestratorAgent (on EMERGENCY state → fetch safety protocols)
  - LangGraph reasoning loop (RAG-RETRIEVE step)
  - CLI / demo queries
"""

import os
import time
from typing import List, Dict, Any, Optional

import numpy as np

from .chunker       import DocumentChunker
from .embedder      import SentenceEmbedder
from .index_builder import FAISSIndexBuilder


# ---------------------------------------------------------------------------
# Default file paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_DOCS_DIR      = "faiss_rag/knowledge_base"
DEFAULT_INDEX_PATH    = "faiss_rag/data/faiss_index.bin"
DEFAULT_METADATA_PATH = "faiss_rag/data/chunks_metadata.json"


class RAGRetriever:
    """
    Offline semantic retrieval over a FAISS vector index.

    Parameters
    ----------
    index_path    : str             Path to the .bin FAISS index file
    metadata_path : str             Path to the .json chunks metadata file
    embedder      : SentenceEmbedder  Shared embedder instance
    """

    def __init__(
        self,
        index_path    : str,
        metadata_path : str,
        embedder      : Optional[SentenceEmbedder] = None,
    ):
        self._embedder = embedder or SentenceEmbedder()
        self._builder  = FAISSIndexBuilder.load(index_path, metadata_path)
        self._query_count  = 0
        self._total_query_ms = 0.0

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query : str,
        top_k : int = 5,
        min_score : float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Parameters
        ----------
        query     : str    Natural language query
        top_k     : int    Number of results to return (default 5)
        min_score : float  Minimum cosine similarity threshold (default 0.0)

        Returns
        -------
        list of result dicts:
            {rank, score, text, source, chunk_id, start_char, end_char}
        """
        t0 = time.time()

        # Embed query
        query_vec = self._embedder.embed(query)   # shape (1, 384)
        query_vec = query_vec.astype(np.float32)

        # FAISS search
        scores, indices = self._builder.index.search(query_vec, top_k)

        # Build results — deduplicate overlapping chunks from the same source
        seen: List[Dict[str, Any]] = []   # track (source, start_char) of included chunks
        results: List[Dict[str, Any]] = []
        rank_out = 1

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                    # FAISS returns -1 for empty slots
                continue
            if float(score) < min_score:
                continue

            meta = self._builder.metadata[idx]
            # Skip if a chunk from the same source with very close start_char already included
            is_duplicate = any(
                s["source"] == meta["source"]
                and abs(s["start_char"] - meta["start_char"]) < 300
                for s in seen
            )
            if is_duplicate:
                continue

            seen.append({"source": meta["source"], "start_char": meta["start_char"]})
            results.append({
                "rank"      : rank_out,
                "score"     : float(score),
                "text"      : meta["text"],
                "source"    : meta["source"],
                "chunk_id"  : meta["chunk_id"],
                "start_char": meta["start_char"],
                "end_char"  : meta["end_char"],
            })
            rank_out += 1
            if rank_out > top_k:
                break

        elapsed_ms = (time.time() - t0) * 1000
        self._query_count    += 1
        self._total_query_ms += elapsed_ms

        return results

    # ------------------------------------------------------------------
    # Formatting for LLM context
    # ------------------------------------------------------------------

    def format_context(
        self,
        results       : List[Dict[str, Any]],
        max_chars     : int = 3000,
        include_scores: bool = True,
    ) -> str:
        """
        Format retrieved chunks into a compact LLM-ready context block.

        Parameters
        ----------
        results       : list    Output of retrieve()
        max_chars     : int     Max total characters in context (default 3000)
        include_scores: bool    Whether to include similarity scores

        Returns
        -------
        str   Formatted context string ready to inject into an LLM prompt
        """
        if not results:
            return "[RAG] No relevant documents found."

        lines = ["=== RETRIEVED KNOWLEDGE (RAG) ==="]
        total_chars = 0

        for r in results:
            score_tag = f" [score={r['score']:.3f}]" if include_scores else ""
            header    = f"\n--- [{r['rank']}] Source: {r['source']}{score_tag} ---"
            body      = r["text"].strip()

            chunk_str = header + "\n" + body
            if total_chars + len(chunk_str) > max_chars:
                # Truncate last chunk to fit
                remaining = max_chars - total_chars - len(header) - 10
                if remaining > 100:
                    lines.append(header)
                    lines.append(body[:remaining] + " ...")
                break

            lines.append(chunk_str)
            total_chars += len(chunk_str)

        lines.append("\n=== END RETRIEVED KNOWLEDGE ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience method: build + save + return retriever
    # ------------------------------------------------------------------

    @classmethod
    def build_from_directory(
        cls,
        docs_dir      : str,
        save_dir      : str,
        embedder      : Optional[SentenceEmbedder] = None,
        chunk_size    : int = 1200,
        chunk_overlap : int = 200,
    ) -> "RAGRetriever":
        """
        One-shot utility: chunk docs → embed → build index → save → return retriever.

        Parameters
        ----------
        docs_dir      : str   Directory of .md / .txt knowledge base files
        save_dir      : str   Directory to save the index and metadata
        embedder      : SentenceEmbedder  (optional, created if None)
        chunk_size    : int
        chunk_overlap : int

        Returns
        -------
        RAGRetriever instance ready to query
        """
        embedder = embedder or SentenceEmbedder()

        # 1. Load & chunk documents
        print(f"\n[RAG Build] Loading documents from: {docs_dir}")
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs    = chunker.load_directory(docs_dir)
        if not docs:
            raise ValueError(f"No .md or .txt files found in: {docs_dir}")
        chunks  = chunker.chunk_all(docs)

        # 2. Embed
        print(f"\n[RAG Build] Generating embeddings ...")
        embeddings = embedder.embed_chunks(chunks)

        # 3. Build FAISS index
        print(f"\n[RAG Build] Building FAISS index ...")
        builder = FAISSIndexBuilder(embed_dim=embeddings.shape[1])
        builder.build(chunks, embeddings)

        # 4. Save to disk
        os.makedirs(save_dir, exist_ok=True)
        index_path    = os.path.join(save_dir, "faiss_index.bin")
        metadata_path = os.path.join(save_dir, "chunks_metadata.json")
        print(f"\n[RAG Build] Saving index to: {save_dir}")
        builder.save(index_path, metadata_path)

        # 5. Return loaded retriever
        print(f"\n[RAG Build] Index ready. Loading retriever ...")
        return cls(index_path=index_path, metadata_path=metadata_path, embedder=embedder)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        return self._builder.total_vectors

    def stats(self) -> Dict[str, Any]:
        avg_ms = (
            self._total_query_ms / self._query_count
            if self._query_count > 0 else 0.0
        )
        return {
            "total_chunks"   : self.total_chunks,
            "embed_dim"      : self._builder.embed_dim,
            "queries_made"   : self._query_count,
            "avg_query_ms"   : round(avg_ms, 2),
        }
