"""
demo_rag.py — FAISS RAG System End-to-End Demo
===============================================
Demonstrates the full FAISS Vector Database pipeline:

  1. Load all knowledge base documents from faiss_rag/knowledge_base/
  2. Chunk documents into overlapping text windows
  3. Embed chunks with all-MiniLM-L6-v2 (384-dim, CPU)
  4. Build a FAISS IndexFlatIP and save to faiss_rag/data/
  5. Run sample mining safety queries and display ranked results

Run from project root:
    py -X utf8 faiss_rag/demo_rag.py

Options:
    --rebuild     Force rebuild of index even if it exists
    --top-k N     Number of results to return per query (default: 3)
"""

import os
import sys
import argparse
import time

# ── Project root setup ────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from faiss_rag.chunker       import DocumentChunker
from faiss_rag.embedder      import SentenceEmbedder
from faiss_rag.index_builder import FAISSIndexBuilder
from faiss_rag.retriever     import RAGRetriever

# ── Paths ─────────────────────────────────────────────────────────────────
DOCS_DIR      = os.path.join(SCRIPT_DIR, "knowledge_base")
SAVE_DIR      = os.path.join(SCRIPT_DIR, "data")
INDEX_PATH    = os.path.join(SAVE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(SAVE_DIR, "chunks_metadata.json")


# ═══════════════════════════════════════════════════════════════════════════
# Sample queries to demonstrate retrieval quality
# ═══════════════════════════════════════════════════════════════════════════
DEMO_QUERIES = [
    # Gas Safety
    "What is the safe methane concentration limit in underground mines?",
    "Carbon monoxide exposure thresholds and evacuation levels",
    "LPG lower explosive limit and OSHA permissible exposure limit",

    # Vibration
    "PPV peak particle velocity safe limits for residential structures",
    "IS 6922 blast vibration standard India mining operations",
    "scaled distance USBM formula blast prediction",

    # Environmental
    "safe temperature and humidity range for underground mine workers",
    "humidex calculation formula heat stress mining",
    "CO2 concentration occupancy detection threshold",

    # Navigation / Ultrasonic
    "robot collision avoidance proximity alert underground mine",
    "ultrasonic sensor sharp turn collision risk protocol",
    "ATEX certification robot explosion proof underground",

    # System
    "FIELD-MIND anomaly triggered reasoning IDLE ACTIVE EMERGENCY states",
    "experience replay buffer self learning sensor agent FAISS",
]


# ═══════════════════════════════════════════════════════════════════════════
# Build or Load Index
# ═══════════════════════════════════════════════════════════════════════════

def build_or_load_index(rebuild: bool = False) -> RAGRetriever:
    """Build a new FAISS index or load the existing one from disk."""

    if not rebuild and os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        print(f"\n[Demo] Found existing index at {INDEX_PATH}")
        print("[Demo] Loading index from disk (use --rebuild to force rebuild)...\n")
        embedder  = SentenceEmbedder()
        retriever = RAGRetriever(
            index_path    = INDEX_PATH,
            metadata_path = METADATA_PATH,
            embedder      = embedder,
        )
        return retriever

    print("\n[Demo] Building FAISS index from knowledge base documents...\n")
    t0 = time.time()
    retriever = RAGRetriever.build_from_directory(
        docs_dir      = DOCS_DIR,
        save_dir      = SAVE_DIR,
        chunk_size    = 1200,
        chunk_overlap = 200,
    )
    elapsed = time.time() - t0
    print(f"\n[Demo] Index built in {elapsed:.1f}s — {retriever.total_chunks} chunks indexed.")
    return retriever


# ═══════════════════════════════════════════════════════════════════════════
# Query Display
# ═══════════════════════════════════════════════════════════════════════════

def run_query(retriever: RAGRetriever, query: str, top_k: int = 3) -> None:
    """Run a single query and pretty-print results."""
    print(f"\n{'═' * 70}")
    print(f"  QUERY: {query}")
    print(f"{'─' * 70}")

    t0      = time.time()
    results = retriever.retrieve(query, top_k=top_k)
    elapsed = (time.time() - t0) * 1000

    if not results:
        print("  No results found.")
        return

    for r in results:
        score_bar = "█" * int(r["score"] * 20)
        print(f"\n  [{r['rank']}] Source: {r['source']}  |  Score: {r['score']:.4f}  {score_bar}")
        # Print first 400 chars of chunk text
        preview = r["text"].strip().replace("\n", " ")[:400]
        if len(r["text"]) > 400:
            preview += " ..."
        print(f"      {preview}")

    print(f"\n  ⏱  Query time: {elapsed:.1f} ms")


# ═══════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════

def run_demo(rebuild: bool = False, top_k: int = 3) -> None:
    print("\n" + "█" * 70)
    print("  FIELD-MIND — FAISS RAG Vector Database Demo")
    print("  Offline semantic search over mining safety knowledge base")
    print("█" * 70)

    # ── Build or load index ────────────────────────────────────────────
    retriever = build_or_load_index(rebuild=rebuild)

    # ── Print index stats ──────────────────────────────────────────────
    stats = retriever.stats()
    print(f"\n{'─' * 70}")
    print(f"  Index Statistics")
    print(f"{'─' * 70}")
    print(f"  Total indexed chunks : {stats['total_chunks']}")
    print(f"  Embedding dimension  : {stats['embed_dim']}")
    print(f"  Top-K per query      : {top_k}")
    print(f"{'─' * 70}")

    # ── Run demo queries ───────────────────────────────────────────────
    print(f"\n  Running {len(DEMO_QUERIES)} sample queries...\n")

    for query in DEMO_QUERIES:
        run_query(retriever, query, top_k=top_k)

    # ── Final stats ────────────────────────────────────────────────────
    final_stats = retriever.stats()
    print(f"\n\n{'█' * 70}")
    print("  DEMO COMPLETE — Final Stats")
    print(f"{'─' * 70}")
    print(f"  Total queries run    : {final_stats['queries_made']}")
    print(f"  Avg query time       : {final_stats['avg_query_ms']:.1f} ms")
    print(f"  Index path           : {INDEX_PATH}")
    print(f"  Metadata path        : {METADATA_PATH}")
    print(f"{'█' * 70}\n")

    # ── Show a full formatted LLM context block for the last query ─────
    print("  === SAMPLE LLM CONTEXT BLOCK (last query) ===\n")
    last_results = retriever.retrieve(DEMO_QUERIES[-1], top_k=top_k)
    print(retriever.format_context(last_results, max_chars=2000))


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIELD-MIND FAISS RAG Demo")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of FAISS index even if it already exists"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks per query (default: 3)"
    )
    args = parser.parse_args()

    run_demo(rebuild=args.rebuild, top_k=args.top_k)
