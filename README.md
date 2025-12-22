# HCMS — Hierarchical Compressed Memory System

**HCMS** is a stateful memory engine designed to overcome structural limitations of traditional Retrieval-Augmented Generation (RAG) systems.
It introduces **hierarchical memory, graph-native reasoning, compression, and cache-aware retrieval** as first-class primitives.

This repository currently contains:

* A **working HCMS core** (`main.py`)
* A **controlled benchmark suite** comparing HCMS vs RAG (`bench.py`)

The long-term goal is to evolve HCMS into the **memory and reasoning backbone of a personal AI assistant**.

---

## Motivation

Traditional RAG systems are:

* Stateless
* Latency-heavy
* Fragile under multi-hop reasoning
* Storage-inefficient at scale

HCMS addresses these issues by treating memory as a **structured, persistent, and compressible cognitive substrate**, rather than a flat retrieval index.

In short:

> **RAG retrieves documents.
> HCMS retrieves structured knowledge.**

---

## Core Concepts

### 1. Hierarchical Memory

* HOT tier: frequently accessed facts
* WARM/COLD tiers: compressed long-term memory
* Stateful across queries and sessions

### 2. Graph-Native Reasoning

* Knowledge represented as facts + relations
* Multi-hop reasoning performed via graph traversal
* Preserves relational structure lost in vector search

### 3. Compression-Aware Storage

* ~15× memory reduction in benchmark
* Compression impacts storage, not accuracy or latency
* Enables long-lived assistants with bounded memory growth

### 4. Cache-Aware Execution

* Cache hit defined proportionally (≥50% of required facts)
* Realistic multi-tier memory behavior
* Significant latency reduction on repeated or related queries

---

## Repository Structure

```text
.
├── main.py        # Core HCMS engine (memory, graph, cache, query execution)
├── bench.py       # Benchmark suite: HCMS vs RAG baseline
├── benchmark_dataset.json   # Auto-generated synthetic dataset
├── benchmark_results.json   # Benchmark outputs + methodology
└── README.md
```

---

## `main.py` — HCMS Core Engine

`main.py` implements the **HCMS system itself**, including:

* Fact storage
* Relation graph construction
* Graph-based retrieval
* Cache management
* Compression accounting
* Query execution pipeline

This file is intended to evolve into the **runtime memory engine of a personal assistant**, responsible for:

* Remembering past interactions
* Maintaining structured long-term knowledge
* Supporting multi-step reasoning
* Reducing hallucinations via grounded memory

> Think of `main.py` as the **cognitive substrate**, not the UI layer.

---

## `bench.py` — Benchmark Suite

`bench.py` provides a **transparent, reproducible benchmark** comparing:

* **RAG Baseline**

  * Vector search + chunking
  * Stateless retrieval
* **HCMS**

  * Graph memory
  * Compression
  * Cache-aware execution

### Benchmark Characteristics

* Synthetic but **verifiable dataset**
* Domains:

  * Technology
  * Finance
  * Legal
* Query types:

  * Factual (1-hop)
  * Relational (2-hop)
  * Multi-hop (4-hop)

### Metrics

* Latency
* Context relevance
* Answer faithfulness
* Context utilization
* Answer completeness
* Compression ratio
* Storage footprint

All **methodological assumptions are explicitly logged**, including:

* Retrieval precision assumptions
* Latency simulation model
* Cache behavior
* Statistical limitations

---

## Running the Benchmark

```bash
python bench.py
```

This will:

1. Generate a reproducible synthetic dataset
2. Run HCMS vs RAG
3. Print detailed results
4. Perform statistical analysis
5. Run an ablation study
6. Save results to `benchmark_results.json`

Expected outcome (approximate):

* ~77% lower latency vs RAG
* ~28% higher accuracy on multi-hop queries
* ~15× compression ratio
* ~93% lower storage usage

---

## Why This Matters for Personal Assistants

HCMS is designed for scenarios where assistants must:

* Remember information across conversations
* Reason over past events and relationships
* Scale memory without exploding context windows
* Avoid repeated retrieval costs
* Maintain internal state

This makes it suitable for:

* Long-term personal assistants
* Knowledge workers’ copilots
* Research assistants
* Legal / compliance assistants
* Technical decision support systems

---

## Current Limitations

* Benchmark dataset is small (N=9 queries)
* Results are **indicative**, not definitive
* Retrieval precision is simulated, not learned
* No real embedding model yet
* No natural language interface yet

These are **explicit design choices**, not oversights.

---

## Roadmap

Planned next steps:

1. Replace simulated retrieval with real embeddings
2. Support temporal and aggregation queries
3. Incremental memory updates from dialogue
4. Memory decay and prioritization
5. Integration with an LLM-based conversational layer
6. Evaluation on real-world datasets
7. Turn HCMS into a plug-and-play assistant memory backend

---

## Philosophy

HCMS is not meant to “beat RAG” in all cases.

Instead, it explores a different thesis:

> **Reasoning improves when memory is structured, persistent, and stateful — not just retrieved.**

---

## License

MIT (or specify your preferred license)

---

## Author

Felipe Biava Cataneo
Independent Researcher

