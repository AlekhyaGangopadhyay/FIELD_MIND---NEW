# FAISS Vector Database (RAG) — FIELD-MIND Layer 2B

The **FAISS RAG module** is FIELD-MIND's offline semantic search layer. It indexes mining safety literature, operational protocols, and system documentation into a FAISS vector store and retrieves the most relevant knowledge chunks to ground the downstream Scientific Reasoning Core (Layer 3 / LangGraph + Llama-3.2-3B).

All processing runs **100% offline** — no API calls, no internet required. Designed to operate within the 4 GB RAM envelope of an NVIDIA Jetson Nano.

---

## Architecture

```
   [ .md / .txt Knowledge Base Files ]
                    |
                    v
     +----------------------------------+
     |   DocumentChunker (chunker.py)   |
     |  Overlapping 1200-char windows   |
     |  200-char overlap, min 80 chars  |
     +----------------------------------+
                    |
                    v
     +----------------------------------+
     |  SentenceEmbedder (embedder.py)  |
     |  all-MiniLM-L6-v2  (22 MB)      |
     |  384-dim · L2-normalized · CPU   |
     +----------------------------------+
                    |
                    v
     +----------------------------------+
     | FAISSIndexBuilder (index_builder)|
     |  IndexFlatIP · Cosine Similarity |
     |  faiss_index.bin + metadata.json |
     +----------------------------------+
                    |
                    v
     +----------------------------------+
     |   RAGRetriever (retriever.py)    |
     |  embed query → top-K search      |
     |  deduplicate → format context    |
     +----------------------------------+
                    |
                    v
     [ Layer 3: Scientific Reasoning Core ]
     [ LangGraph RAG-RETRIEVE step        ]
```

---

## Module Structure

```
faiss_rag/
├── __init__.py               # Package exports (chunker, embedder, builder, retriever)
├── chunker.py                # Load .md/.txt files, split into overlapping chunks
├── embedder.py               # Sentence embedding via all-MiniLM-L6-v2 (CPU, lazy-loaded)
├── index_builder.py          # Build & persist FAISS IndexFlatIP to disk
├── retriever.py              # Query index → top-K chunks → LLM-ready context string
├── demo_rag.py               # End-to-end build + 14 sample mining safety queries
├── knowledge_base/           # Source documents indexed into FAISS
│   ├── gas_safety.md         # Gas thresholds: CH4, CO, H2S, NOx, LPG (OSHA/NIOSH/IDLH)
│   ├── vibration_limits.md   # PPV standards: IS 6922, ISEE, USBM scaled distance
│   ├── env_safety.md         # Temp/humidity/CO2 occupational limits, humidex
│   ├── navigation_safety.md  # Robot collision protocols, ATEX, ISO 10218
│   └── field_mind_overview.md# Full FIELD-MIND system architecture reference
└── data/                     # Auto-created at build time
    ├── faiss_index.bin        # Serialized FAISS index (binary)
    └── chunks_metadata.json   # Chunk text + source metadata (JSON)
```

---

## Knowledge Base Contents

| Document | Domain | Key Topics Covered |
|----------|--------|--------------------|
| `gas_safety.md` | Gas Sensors | CH4 LEL/UEL, CO OSHA PEL, H2S IDLH, NOx TLV, LPG explosive limits, PM2.5 |
| `vibration_limits.md` | Blast Vibration | IS 6922 PPV table, ISEE frequency limits, USBM/Langefors scaled distance, post-blast re-entry |
| `env_safety.md` | Environmental | MHSA wet bulb temp, OSHA humidity, humidex formula, CO2 occupancy thresholds |
| `navigation_safety.md` | Robot Navigation | Collision avoidance levels, ATEX Zone 1/2, AS 4024, ISO 10218, NavigationEvent EKG logging |
| `field_mind_overview.md` | System Reference | All 6 layers, all sensor agents, EKG schema, RAG integration, operational modes |

---

## Index Specifications

| Parameter | Value |
|-----------|-------|
| Index type | `faiss.IndexFlatIP` (exact search) |
| Similarity metric | Cosine (inner product on L2-normalized vectors) |
| Embedding model | `all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Model size | ~22 MB |
| Hardware | CPU-only (no GPU required) |
| Chunk size | 1,200 characters |
| Chunk overlap | 200 characters |
| Total chunks indexed | 636 |
| Avg query latency | **~7 ms** (after model load) |
| Cold-start (model load) | ~10 s (once, then cached) |

---

## Running the Demo

```bash
# Build index from scratch and run 14 sample queries
py -X utf8 faiss_rag/demo_rag.py --rebuild

# Load existing index and run queries (skips rebuild)
py -X utf8 faiss_rag/demo_rag.py

# Use top-5 results per query instead of default 3
py -X utf8 faiss_rag/demo_rag.py --top-k 5
```

**Demo output** (verified run):

| Query | Top Source | Score |
|-------|-----------|-------|
| methane safe concentration limit | `gas_safety.md` | 0.761 |
| IS 6922 blast vibration standard India | `vibration_limits.md` | 0.673 |
| temperature humidity safe range miners | `env_safety.md` | 0.646 |
| robot collision avoidance underground | `navigation_safety.md` | 0.637 |
| experience replay buffer sensor agent | `gas_safety.md` | 0.680 |

---

## Python API

### One-time index build

```python
from faiss_rag import RAGRetriever

# Build from knowledge_base/ directory and save index to data/
retriever = RAGRetriever.build_from_directory(
    docs_dir="faiss_rag/knowledge_base",
    save_dir="faiss_rag/data",
)
```

### Load existing index

```python
from faiss_rag import RAGRetriever, SentenceEmbedder

embedder  = SentenceEmbedder()    # all-MiniLM-L6-v2, lazy-loaded
retriever = RAGRetriever(
    index_path    = "faiss_rag/data/faiss_index.bin",
    metadata_path = "faiss_rag/data/chunks_metadata.json",
    embedder      = embedder,
)
```

### Query the index

```python
# Retrieve top-5 most relevant chunks
results = retriever.retrieve(
    query   = "maximum safe methane concentration in underground mine",
    top_k   = 5,
    min_score = 0.3,   # optional similarity threshold
)

# Format as LLM-ready context block
context = retriever.format_context(results, max_chars=3000)
print(context)
```

### Compare model predictions with protocols

The evaluator is the bridge between Tier-1 model output and the chatbot. It
keeps safety decisions deterministic while FAISS supplies the supporting
OSHA/NIOSH/IS/ISEE/AS-4024 passages. It also flags possible model misses and
false alarms for later calibration work:

```python
from faiss_rag import SafetyProtocolEvaluator

evaluator = SafetyProtocolEvaluator(retriever)
assessment = evaluator.assess(
    readings={
        "MQ4_CH4_ppm": 12500,
        "MQ7_CO_ppm": 35,
        "predicted_ppv": 2.4,
        "min_distance": 0.8,
    },
    predictions={
        "methane_hazard": 1,
        "co_nox_hazard": 0,
        "vibration_hazard": 1,
        "sharp_turn_required": 0,
    },
)

print(assessment.overall_status)  # CRITICAL, WARNING, SAFE, or REVIEW_MODEL_DISAGREEMENT
print(assessment.format_report())
```

For a streaming turn, `RAGRetriever.retrieve_many()` embeds all active domain
queries in one batch. Repeated queries are kept in a small bounded in-process
cache, reducing embedding overhead without changing the existing `retrieve()`
API.

### Streaming simulation usage

The multi-node streaming simulation (`unified_demo/streaming_safety_simulation.py`)
instantiates a `SafetyProtocolEvaluator` and calls `.assess()` for every
node × timestamp sample (30 calls in the default 3-node × 10-timestamp run).
Each call produces a status (`SAFE`, `WARNING`, `CRITICAL`,
`REVIEW_MODEL_DISAGREEMENT`) that is recorded in the history JSON and fed
into the final trend summary:

```python
from faiss_rag import SafetyProtocolEvaluator

evaluator = SafetyProtocolEvaluator()          # retriever is optional
for tick in range(10):
    for node in range(3):
        readings = generate_readings(rng, node, tick)
        result   = evaluate_sample(monitor, readings)
        check    = evaluator.assess(result["readings"], result["predictions"])
        # check.overall_status → "SAFE" | "WARNING" | "CRITICAL" | "REVIEW_MODEL_DISAGREEMENT"
```

### Sample output

```
=== RETRIEVED KNOWLEDGE (RAG) ===

--- [1] Source: gas_safety.md [score=0.761] ---
## Monitored Gases and Safety Thresholds

### Methane (CH4) — MQ-4 Sensor
...
| Warning     | > 0.5% (5,000 ppm) | Increase ventilation, alert personnel |
| Evacuation  | > 1.0% (10,000 ppm) | Immediate evacuation of section |
| Explosive   | 5% – 15% (LEL–UEL)  | Full site shutdown, no ignition sources |
...

=== END RETRIEVED KNOWLEDGE ===
```

### Individual components

```python
from faiss_rag import DocumentChunker, SentenceEmbedder, FAISSIndexBuilder

# Chunk custom documents
chunker = DocumentChunker(chunk_size=1200, chunk_overlap=200)
docs    = chunker.load_directory("my_docs/")
chunks  = chunker.chunk_all(docs)

# Embed chunks
embedder   = SentenceEmbedder(model_name="all-MiniLM-L6-v2")
embeddings = embedder.embed_chunks(chunks)   # np.ndarray (N, 384)

# Build and save index
builder = FAISSIndexBuilder(embed_dim=384)
builder.build(chunks, embeddings)
builder.save("data/faiss_index.bin", "data/chunks_metadata.json")
```

---

## Integration with FIELD-MIND

### MineOrchestratorAgent → RAG (on EMERGENCY)

When the `MineOrchestratorAgent` raises an EMERGENCY state (global_score ≥ 0.60), the reasoning core retrieves relevant safety grounding before generating advice:

```python
# Conceptual integration (Layer 3 hook)
if orchestrator.device_state == "EMERGENCY":
    active_sources = orchestrator.get_event_log()[-1]["active_sources"]

    # Build a natural language query from active hazard context
    query = f"emergency response protocol for {', '.join(active_sources)}"

    # Retrieve and inject as LLM context
    results = retriever.retrieve(query, top_k=5)
    context = retriever.format_context(results)
    # → Pass `context` as system prompt grounding to Llama-3.2-3B
```

### LangGraph RAG-RETRIEVE Step

The RAG retriever serves the `RAG-RETRIEVE` node in the planned LangGraph reasoning loop:

```
OBSERVE → EKG-RETRIEVE → RAG-RETRIEVE → HYPOTHESIZE → SUGGEST → UPDATE_EKG
```

---

## Extending the Knowledge Base

To add new documents:

1. Drop `.md` or `.txt` files into `faiss_rag/knowledge_base/`
2. Rebuild the index:
   ```bash
   py -X utf8 faiss_rag/demo_rag.py --rebuild
   ```
3. The new documents are automatically chunked, embedded, and merged into the existing index.

Suggested additions:
- Mining incident reports (PDF → convert to text first)
- Equipment manufacturer manuals
- Site-specific blasting regulations
- Emergency response plans (ERPs)

---

## Design Decisions

- **`IndexFlatIP` over `IndexIVFFlat`**: Exact search is used for correctness on a corpus of 636 chunks. `IndexIVFFlat` would require a training step but offers sub-linear search at > 100k vectors — migrate when the knowledge base grows significantly.
- **Character-based chunking over token-based**: Avoids a tokenizer dependency. At ~4 chars/token, 1200 chars ≈ 300 tokens — well within the 512-token limit of `all-MiniLM-L6-v2`.
- **Deduplication**: Chunks from the same source within 300 chars of an already-returned chunk are filtered to prevent redundant overlapping results.
- **Lazy model loading**: The embedding model is only loaded on the first `embed()` call, keeping import time near-zero for the orchestrator's startup path.
