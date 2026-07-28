"""
index_builder.py — FAISS Index Construction & Persistence
==========================================================
Builds a FAISS IndexFlatIP (Inner Product) index from pre-computed
L2-normalized embeddings, then saves both the index and chunk metadata
to disk for offline retrieval.

Index Type: IndexFlatIP
  - Exact nearest-neighbour search (no approximation)
  - Inner product on L2-normalized vectors = cosine similarity
  - No training step required
  - Easily swappable to IndexIVFFlat for larger corpora (>100k chunks)

Persistence
  - FAISS index  → faiss_index.bin   (binary, FAISS native format)
  - Chunk metadata → chunks_metadata.json  (text + source for each chunk)
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

import faiss


class FAISSIndexBuilder:
    """
    Builds, saves, and loads a FAISS vector index for text chunk retrieval.

    Parameters
    ----------
    embed_dim : int   Dimensionality of embedding vectors (default 384)
    """

    def __init__(self, embed_dim: int = 384):
        self.embed_dim = embed_dim
        self._index    : Optional[faiss.Index] = None
        self._metadata : List[Dict[str, Any]]  = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunks     : List[Dict[str, Any]],
        embeddings : np.ndarray,
    ) -> "FAISSIndexBuilder":
        """
        Create a FAISS index from chunks + pre-computed embeddings.

        Parameters
        ----------
        chunks     : list[dict]   Chunk dicts from DocumentChunker
        embeddings : np.ndarray   Shape (N, embed_dim), L2-normalized float32

        Returns
        -------
        self (for method chaining)
        """
        assert len(chunks) == len(embeddings), (
            f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
        )
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        n, d = embeddings.shape
        assert d == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {d}"

        print(f"  [IndexBuilder] Building IndexFlatIP ({n} vectors, dim={d}) ...")
        self._index = faiss.IndexFlatIP(d)
        self._index.add(embeddings)

        # Store only serialisable metadata (no numpy arrays)
        self._metadata = [
            {
                "chunk_id"  : c.get("chunk_id",   i),
                "source"    : c.get("source",      "unknown"),
                "path"      : c.get("path",        ""),
                "text"      : c.get("text",        ""),
                "start_char": c.get("start_char",  0),
                "end_char"  : c.get("end_char",    0),
                "length"    : c.get("length",      0),
            }
            for i, c in enumerate(chunks)
        ]

        print(f"  [IndexBuilder] Index built: {self._index.ntotal} vectors indexed.")
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and chunk metadata to disk.

        Parameters
        ----------
        index_path    : str   Path for the .bin FAISS index file
        metadata_path : str   Path for the .json metadata file
        """
        if self._index is None:
            raise RuntimeError("No index built yet. Call build() first.")

        os.makedirs(os.path.dirname(index_path),    exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        faiss.write_index(self._index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

        index_size_mb = os.path.getsize(index_path) / (1024 ** 2)
        print(f"  [IndexBuilder] Saved FAISS index  → {index_path} ({index_size_mb:.2f} MB)")
        print(f"  [IndexBuilder] Saved metadata      → {metadata_path} ({len(self._metadata)} chunks)")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndexBuilder":
        """
        Load a FAISS index + metadata from disk.

        Parameters
        ----------
        index_path    : str   Path to faiss_index.bin
        metadata_path : str   Path to chunks_metadata.json

        Returns
        -------
        FAISSIndexBuilder instance ready for searching
        """
        assert os.path.exists(index_path), f"Index file not found: {index_path}"
        assert os.path.exists(metadata_path), f"Metadata file not found: {metadata_path}"

        print(f"  [IndexBuilder] Loading index from {index_path} ...")
        index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"  [IndexBuilder] Loaded index: {index.ntotal} vectors (dim={index.d})")
        builder = cls(embed_dim=index.d)
        builder._index    = index
        builder._metadata = metadata
        builder._last_index_path = index_path
        builder._last_metadata_path = metadata_path
        return builder

    def add_single_experience(
        self,
        chunk: Dict[str, Any],
        embedding: np.ndarray,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ) -> None:
        """
        Dynamically appends a new self-learned experience rule chunk and its vector embedding
        to the active FAISS index and persists to disk.
        """
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.embed_dim)

        emb_32 = np.ascontiguousarray(embedding, dtype=np.float32)
        if len(emb_32.shape) == 1:
            emb_32 = emb_32.reshape(1, -1)

        self._index.add(emb_32)

        meta = {
            "chunk_id"  : len(self._metadata),
            "source"    : chunk.get("source",   "self_learned_experience"),
            "path"      : chunk.get("path",     "dynamic_reflection"),
            "text"      : chunk.get("text",     ""),
            "start_char": 0,
            "end_char"  : len(chunk.get("text", "")),
            "length"    : len(chunk.get("text", "")),
        }
        self._metadata.append(meta)

        idx_p = index_path or getattr(self, "_last_index_path", None)
        meta_p = metadata_path or getattr(self, "_last_metadata_path", None)

        if idx_p and meta_p:
            self.save(idx_p, meta_p)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index(self) -> Optional[faiss.Index]:
        return self._index

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self._metadata

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
