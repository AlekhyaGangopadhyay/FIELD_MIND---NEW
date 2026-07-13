"""
embedder.py — Sentence Embedding via all-MiniLM-L6-v2
======================================================
Wraps the sentence-transformers library to generate dense 384-dimensional
embeddings for text chunks. Uses all-MiniLM-L6-v2 — a tiny (22 MB) model
that runs in ~5 ms/chunk on CPU, making it suitable for Jetson Nano 4 GB.

The model is downloaded once to the local HuggingFace cache and works
100% offline after the first run.

Key Properties
--------------
- Output dimension : 384
- Normalization    : L2-normalized (unit vectors → cosine similarity = dot product)
- Hardware         : CPU-only, no GPU required
- Model size       : ~22 MB quantized (all-MiniLM-L6-v2)
"""

import os
import numpy as np
from typing import List, Union


# ---------------------------------------------------------------------------
# Model identifier — swap here to try other sentence-transformers models
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM     = 384


class SentenceEmbedder:
    """
    Generates L2-normalized sentence embeddings using a local sentence-transformer.

    Parameters
    ----------
    model_name : str   HuggingFace model name (default: all-MiniLM-L6-v2)
    batch_size : int   Inference batch size (default: 64)
    device     : str   'cpu' or 'cuda' (default: 'cpu')
    """

    def __init__(
        self,
        model_name : str = DEFAULT_MODEL,
        batch_size : int = 64,
        device     : str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device     = device
        self._model     = None          # lazy loading
        self.embed_dim  = EMBED_DIM

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load model on first use (lazy init)."""
        if self._model is None:
            print(f"  [Embedder] Loading model: {self.model_name} ...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Confirm embedding dimension
            test_emb = self._model.encode(["test"], normalize_embeddings=True)
            self.embed_dim = test_emb.shape[1]
            print(f"  [Embedder] Model loaded. Embedding dim: {self.embed_dim}")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one or more text strings.

        Parameters
        ----------
        texts : str or list[str]

        Returns
        -------
        np.ndarray of shape (N, embed_dim), L2-normalized float32
        """
        self._ensure_model()
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size        = self.batch_size,
            normalize_embeddings = True,   # L2-normalize → cosine sim = dot product
            show_progress_bar = len(texts) > 100,
            convert_to_numpy  = True,
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: List[dict]) -> np.ndarray:
        """
        Embed a list of chunk dicts (as returned by DocumentChunker).

        Parameters
        ----------
        chunks : list[dict]   Each dict must have a 'text' key

        Returns
        -------
        np.ndarray of shape (len(chunks), embed_dim)
        """
        texts = [c["text"] for c in chunks]
        print(f"  [Embedder] Embedding {len(texts)} chunks ...")
        embeddings = self.embed(texts)
        print(f"  [Embedder] Done. Matrix shape: {embeddings.shape}")
        return embeddings
