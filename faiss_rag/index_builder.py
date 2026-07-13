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
        Load a previously saved FAISS index and metadata from disk.

        Parameters
        ----------
        index_path    : str   Path to the .bin FAISS index file
        metadata_path : str   Path to the .json metadata file

        Returns
        -------
        FAISSIndexBuilder instance with index and metadata loaded
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        builder = cls()
        builder._index = faiss.read_index(index_path)
        builder.embed_dim = builder._index.d

        with open(metadata_path, "r", encoding="utf-8") as f:
            builder._metadata = json.load(f)

        print(
            f"  [IndexBuilder] Loaded index: {builder._index.ntotal} vectors "
            f"(dim={builder.embed_dim}) from {index_path}"
        )
        return builder

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
