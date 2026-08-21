---
name: rag-search
version: 1.0.0
description: Grounded knowledge retrieval, multi-source synthesis, and citation verification
category: research
author: QorVix
tags: ["rag", "search", "retrieval", "embeddings"]
required_tools: ["rag_search"]
---

# Grounded RAG Search Playbook

Execute knowledge-grounded retrieval workflows adhering to strict factual verification:

## 1. Query Formulation
- Deconstruct user questions into targeted semantic queries and lexical keywords.
- Formulate alternative queries if initial results lack sufficient context.

## 2. Hybrid Retrieval
- Invoke `rag_search` with appropriate `top_k` (default: 5).
- Scrutinize retrieval scores to assess chunk relevance.

## 3. Strict Grounding
- Never hallucinate facts outside the retrieved context.
- Explicitly state when source documents lack the requested information.

## 4. Citation Formatting
- Attribute every claim with explicit bracketed citation markers: `[Source: <filename>, Chunk: #<index>]`.
