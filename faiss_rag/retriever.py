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
from collections import OrderedDict
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
        self._cache_hits = 0
        # Query text is repeated frequently by the streaming agents.  A small
        # in-process cache avoids paying the embedding cost again for identical
        # questions while keeping memory bounded for Jetson-class devices.
        self._query_cache: "OrderedDict[tuple, List[Dict[str, Any]]]" = OrderedDict()
        self._max_cache_entries = 128

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def warmup(self) -> float:
        """Load the embedding model before the first operator query.

        Returns the warm-up time in milliseconds. This keeps the first chat
        response from appearing artificially slow because model initialization
        is mixed into query latency.
        """
        t0 = time.perf_counter()
        self._embedder.embed("underground mine safety warmup")
        return round((time.perf_counter() - t0) * 1000, 2)

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
        self._validate_query_args(query, top_k, min_score)
        cache_key = (query.strip(), top_k, float(min_score))
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            self._query_cache.move_to_end(cache_key)
            return [dict(item) for item in cached]

        t0 = time.time()
        query_vec = self._embedder.embed(query).astype(np.float32)
        results = self._search_vectors(query_vec, top_k=top_k, min_score=min_score)[0]
        self._record_query(time.time() - t0)
        self._cache_put(cache_key, results)
        return [dict(item) for item in results]

    def retrieve_many(
        self,
        queries: List[str],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve several queries in one embedding and FAISS search batch.

        The reasoning core commonly asks for gas, vibration, environment, and
        navigation guidance in the same turn.  Encoding those queries in one
        batch materially reduces Python/model overhead compared with calling
        :meth:`retrieve` once per domain.  Results are returned under the
        original query strings, preserving a simple public API.
        """
        if not isinstance(queries, list):
            raise TypeError("queries must be a list of strings")
        self._validate_query_args("batch", top_k, min_score)

        unique_queries = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))
        if not unique_queries:
            return {}

        output: Dict[str, List[Dict[str, Any]]] = {}
        missing: List[str] = []
        for query in unique_queries:
            cache_key = (query, top_k, float(min_score))
            cached = self._query_cache.get(cache_key)
            if cached is None:
                missing.append(query)
            else:
                self._cache_hits += 1
                self._query_cache.move_to_end(cache_key)
                output[query] = [dict(item) for item in cached]

        if missing:
            t0 = time.time()
            vectors = self._embedder.embed(missing).astype(np.float32)
            batch_results = self._search_vectors(vectors, top_k=top_k, min_score=min_score)
            self._record_query(time.time() - t0, count=len(missing))
            for query, results in zip(missing, batch_results):
                cache_key = (query, top_k, float(min_score))
                self._cache_put(cache_key, results)
                output[query] = [dict(item) for item in results]

        return {query: output[query] for query in unique_queries}

    @staticmethod
    def _validate_query_args(query: str, top_k: int, min_score: float) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(min_score, (int, float)):
            raise TypeError("min_score must be numeric")

    def _search_vectors(
        self,
        vectors: np.ndarray,
        top_k: int,
        min_score: float,
    ) -> List[List[Dict[str, Any]]]:
        if self._builder.index is None or self._builder.total_vectors == 0:
            return [[] for _ in range(len(vectors))]

        # Search more candidates than the requested K because adjacent chunks
        # are intentionally deduplicated after the FAISS search.
        candidate_k = min(self._builder.total_vectors, max(top_k, top_k * 4))
        scores, indices = self._builder.index.search(vectors, candidate_k)
        all_results: List[List[Dict[str, Any]]] = []

        for row_scores, row_indices in zip(scores, indices):
            seen: List[Dict[str, Any]] = []
            results: List[Dict[str, Any]] = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0 or float(score) < min_score:
                    continue
                meta = self._builder.metadata[int(idx)]
                is_duplicate = any(
                    item["source"] == meta["source"]
                    and abs(item["start_char"] - meta["start_char"]) < 300
                    for item in seen
                )
                if is_duplicate:
                    continue
                seen.append({"source": meta["source"], "start_char": meta["start_char"]})
                results.append({
                    "rank": len(results) + 1,
                    "score": float(score),
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                    "start_char": meta["start_char"],
                    "end_char": meta["end_char"],
                })
                if len(results) >= top_k:
                    break
            all_results.append(results)
        return all_results

    def _cache_put(self, key: tuple, results: List[Dict[str, Any]]) -> None:
        self._query_cache[key] = [dict(item) for item in results]
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._max_cache_entries:
            self._query_cache.popitem(last=False)

    def _record_query(self, elapsed_seconds: float, count: int = 1) -> None:
        self._query_count += count
        self._total_query_ms += elapsed_seconds * 1000

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
            "cache_hits"     : self._cache_hits,
            "avg_query_ms"   : round(avg_ms, 2),
        }
