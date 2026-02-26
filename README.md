# Multi-Modal LSFS

### Theoretical Foundations for Extending LLM-Based Semantic File Systems with Unified Visual and Textual Indexing

**Authors:** Bilal Rana, Zaynab Shahid, Laiba Riaz, Rameen Arshad  
**Institution:** School of Electrical Engineering and Computer Science (SEECS), National University of Sciences & Technology (NUST), Pakistan  
**Date:** December 22, 2025  
**Status:** Manuscript in Preparation

> The full research paper is available here: Multi-Modal_LSFS_Research_Paper.pdf

---

## Overview

Modern operating systems manage data through rigid hierarchical structures - directories and absolute file paths - that were designed for deterministic human navigation. These structures are fundamentally misaligned with how AI agents work: associatively, probabilistically, and by semantic intent rather than location.

The **LLM-based Semantic File System (LSFS)** proposed a solution where files are accessed via natural language description rather than path-based addressing. However, it was limited to text-only data. A significant portion of real user data - screenshots, diagrams, photographs, and scanned documents - remained completely invisible to the system.

This research extends LSFS into a fully **multi-modal architecture**. By unifying visual and textual data into a single high-dimensional vector space, the system can now process queries like *"Find the server architecture diagram I saved last week"* and retrieve the correct image with high precision, even without any filename or text metadata attached to it.

---

## The Problem We Solve

When an AI agent encounters a file named `screenshot_2025.png`, it has no semantic understanding of what that file contains. Standard text embedding models cannot process pixel data. This creates a **Semantic Gap**: a blind spot that renders all visual assets invisible to intelligent retrieval systems.

We close this gap by:

1. Detecting whether a file is text or image using low-level **Magic Bytes** inspection rather than fragile file extensions
2. Routing each file to the appropriate embedding pipeline
3. Projecting both text and images into a **shared 1024-dimensional vector space** using CLIP-based models
4. Storing all vectors in a **Qdrant** database with HNSW indexing for fast retrieval
5. Retrieving files through a **Dual-Path mechanism** that combines semantic vector search with keyword-based metadata matching

---

## Key Contributions

**Unified Multi-Modal Embedding**  
Visual and textual data are mapped into the same latent space using Jina CLIP v2, enabling cross-modal retrieval where a text query can surface a relevant image and vice versa.

**HNSW Vector Indexing**  
Hierarchical Navigable Small World graphs provide O(log N) approximate nearest neighbor search, replacing naive linear scanning that becomes impractical at scale.

**Dual-Path Retrieval**  
Two parallel retrieval tracks are fused at query time. The dense path handles semantic similarity while the sparse path handles precise keyword matching on LLM-generated metadata. This combination ensures the system captures both the abstract concept and the specific factual content of a file.

**SOLID Software Architecture**  
The embedding layer is abstracted behind a BaseEmbedder interface following the Dependency Inversion Principle. This means the underlying model (OpenAI CLIP, Jina CLIP v2, or any future architecture) can be swapped via configuration without touching the core file system logic.

**Asynchronous Vision LLM Worker**  
Image captioning via a Vision-Language Model runs as a background process, ensuring that heavy inference never blocks file system I/O operations.

---

## System Architecture

```
File Monitor (inotify kernel events)
            |
            v
  Modality Dispatcher
  (Magic Bytes inspection)
            |
      ______|______
     |             |
  Text           Image
  Stream         Stream
     |             |
     v             v
  Text          CLIP
  Embedder      Embedder
     |             |
     |_____________|
            |
            v
    Vector Store (Qdrant)
    HNSW Index - O(log N)
            |
            v
    Dual-Path Retriever
            |
      ______|______
     |             |
  Dense          Sparse
  Path           Path
  (Vector        (BM25 on
  Similarity)    LLM Metadata)
     |             |
     |_____________|
            |
      Fusion (LCF / RRF)
            |
            v
      Ranked Results
```

---

## How Retrieval Works

When a user submits a query, the system runs two parallel searches simultaneously.

**Dense Path (Semantic)**  
The query is embedded into a 1024-dimensional vector using Jina CLIP. Qdrant performs an HNSW graph traversal to find the most semantically similar file vectors. This path handles synonymy and conceptual similarity effectively.

**Sparse Path (Lexical)**  
The query is matched against LLM-generated text metadata and captions using BM25 keyword scoring. This path ensures high precision for queries containing specific identifiers, dates, error codes, or proper nouns that may be diluted during vector compression.

**Fusion**  
The two result sets are combined using a weighted Linear Combination:

```
Final Score = α × Vector Score + (1 - α) × Keyword Score
```

where `α = 0.7` by default. As α approaches 1, the system behaves as a pure semantic engine. As α approaches 0, it behaves as a traditional keyword search engine. Score normalization via Min-Max scaling is applied before fusion since vector scores and BM25 scores operate on different numerical scales.

---

## Performance Results

Evaluated on a proprietary dataset of 10,000 mixed files (5,000 text documents, 5,000 technical screenshots). Evaluation metric: Mean Reciprocal Rank (MRR).

| Metric | Text-Only LSFS | Multi-Modal LSFS |
|---|---|---|
| Modalities Supported | Text only | Text and Images |
| Embedding Model | all-MiniLM-L6 | jina-clip-v2 |
| Vector Dimensions | 384 | 1024 |
| Index Latency | ~20ms per file | ~150ms per file |
| Search Latency | 45ms | 58ms |
| Recall@10 | 0.89 (text only) | 0.94 (mixed) |
| MRR | Not applicable | 0.82 |

The 7.5x increase in indexing latency is attributed to the Vision Transformer backbone in CLIP, which is computationally heavier than the BERT-tiny architecture used in the text-only version. Since indexing is an asynchronous write-once operation, this is an acceptable trade-off for the added retrieval capability.

---

## Future Work

**Temporal Video Semantics**  
Extend the pipeline to handle video files through scene boundary detection and LSTM-based frame aggregation, producing a single vector that captures the narrative arc of a video clip.

**Edge-Native Binary Quantization**  
The current 1024-dimensional float32 vectors consume approximately 4KB per file, requiring 4GB of RAM for one million files. Binary Quantization compresses this by 32x by projecting each float dimension to a single bit, enabling retrieval via fast Hamming Distance operations on standard CPUs.

**Audio-Semantic Alignment**  
Integrate OpenAI Whisper for audio transcription, projecting voice memos and meeting recordings into the same latent space as text and images. A single query would then surface relevant documents, diagrams, and audio recordings simultaneously.

**Adaptive Rank Fusion**  
Replace the static α hyperparameter with non-parametric Reciprocal Rank Fusion (RRF), which is robust to the different score distributions of vector similarity and BM25 without requiring manual calibration.

---

## Dependencies

```
Pillow >= 9.0.0
transformers >= 4.36.0
qdrant-client >= 1.7.0
einops
```

---

## Citation

If you reference this work, please cite:

```
Rana, B., Shahid, Z., Riaz, L., & Arshad, R. (2025).
Theoretical Foundations for Extending LLM-Based Semantic File Systems
with Unified Visual and Textual Indexing.
Manuscript in preparation.
SEECS, National University of Sciences & Technology (NUST), Pakistan.
```

---

## References

1. Shi et al. (2025). *From Commands to Prompts: LLM-Based Semantic File System for AIOS*. ICLR 2025.
2. Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision*. ICML 2021.
3. Jina AI (2024). *Jina CLIP v2: Multilingual Multimodal Embeddings*. Hugging Face.
4. Malkov & Yashunin (2018). *Efficient and Robust Nearest Neighbor Search using HNSW Graphs*. IEEE TPAMI.
5. Qdrant Team (2024). *Qdrant Documentation: Vector Search Engine*.
