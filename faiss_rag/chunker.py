"""
chunker.py — Document Loading & Text Chunking
=============================================
Loads Markdown and plain-text documents from a directory and splits
them into overlapping fixed-size chunks for embedding.

Chunk Strategy
--------------
- Fixed character-window chunking (default: 1200 chars)
- Overlap of 200 chars to preserve context at boundaries
- Each chunk records: text, source_file, chunk_id, start_char, end_char

Supported file types: .md, .txt
"""

import os
import re
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Default chunking parameters
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE    = 1200   # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200    # overlap between consecutive chunks
MIN_CHUNK_LENGTH      = 80     # discard chunks shorter than this


class DocumentChunker:
    """
    Loads documents from a directory and splits them into overlapping
    text chunks with source metadata.

    Parameters
    ----------
    chunk_size    : int   Max characters per chunk (default 1200)
    chunk_overlap : int   Overlap between consecutive chunks (default 200)
    """

    def __init__(
        self,
        chunk_size    : int = DEFAULT_CHUNK_SIZE,
        chunk_overlap : int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Directory loading
    # ------------------------------------------------------------------

    def load_directory(
        self,
        directory: str,
        extensions: tuple = (".md", ".txt"),
        recursive: bool   = True,
    ) -> List[Dict[str, str]]:
        """
        Scan a directory for documents and load their content.

        Parameters
        ----------
        directory  : str    Root directory to scan
        extensions : tuple  File extensions to include
        recursive  : bool   Whether to recurse into subdirectories

        Returns
        -------
        list of dicts: [{"path": str, "source": str, "content": str}]
        """
        docs: List[Dict[str, str]] = []

        for root, dirs, files in os.walk(directory):
            for fname in sorted(files):
                if any(fname.lower().endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        docs.append({
                            "path"    : full_path,
                            "source"  : os.path.relpath(full_path, directory),
                            "content" : content,
                        })
                        print(f"  [Chunker] Loaded: {os.path.relpath(full_path, directory)} ({len(content)} chars)")
                    except Exception as e:
                        print(f"  [Chunker] Failed to load {full_path}: {e}")

            if not recursive:
                break

        return docs

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        content : str,
        source  : str,
        path    : str = "",
    ) -> List[Dict[str, Any]]:
        """
        Split a single document into overlapping text chunks.

        Parameters
        ----------
        content : str   Full document text
        source  : str   Relative file path (used as metadata)
        path    : str   Absolute file path

        Returns
        -------
        list of chunk dicts:
            {chunk_id, source, path, text, start_char, end_char, length}
        """
        # Normalise whitespace (collapse multiple blank lines)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        chunks: List[Dict[str, Any]] = []
        start    = 0
        chunk_id = 0
        total    = len(content)

        while start < total:
            end  = min(start + self.chunk_size, total)
            text = content[start:end].strip()

            # Try to break at a sentence or paragraph boundary
            if end < total:
                # Walk back to find a good break point
                for sep in ("\n\n", "\n", ". ", "? ", "! "):
                    last_sep = text.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        text = text[: last_sep + len(sep)].strip()
                        end  = start + last_sep + len(sep)
                        break

            if len(text) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    "chunk_id"  : chunk_id,
                    "source"    : source,
                    "path"      : path,
                    "text"      : text,
                    "start_char": start,
                    "end_char"  : start + len(text),
                    "length"    : len(text),
                })
                chunk_id += 1

            # Advance with overlap
            step  = max(1, len(text) - self.chunk_overlap)
            start += step

        return chunks

    def chunk_all(
        self,
        docs: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Chunk all loaded documents and return a flat list of all chunks.

        Parameters
        ----------
        docs : list   Output of load_directory()

        Returns
        -------
        list of all chunk dicts across all documents
        """
        all_chunks: List[Dict[str, Any]] = []
        for doc in docs:
            doc_chunks = self.chunk_document(
                content = doc["content"],
                source  = doc["source"],
                path    = doc["path"],
            )
            all_chunks.extend(doc_chunks)
            print(
                f"  [Chunker] {doc['source']:<40} → {len(doc_chunks)} chunks"
            )
        print(f"  [Chunker] Total chunks: {len(all_chunks)}")
        return all_chunks
