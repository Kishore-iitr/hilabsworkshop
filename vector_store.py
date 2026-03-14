"""
vector_store.py — Pinecone integration layer.

Responsibilities:
  • Create / connect to index (idempotent)
  • Embed chunks using Anthropic voyage-3 (or OpenAI as fallback)
  • Upsert chunks in batches
  • Query by text → top-k chunks with metadata
  • Delete namespace (per-file cleanup)

Usage (standalone test):
    python vector_store.py path/to/chart.md
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Any

from config import Config
from chunker import chunk_markdown, Chunk


# ── Lazy imports (keeps startup fast) ─────────────────────────────────────────

def _pinecone():
    try:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore
        return Pinecone, ServerlessSpec
    except ImportError:
        raise ImportError("Run: pip install pinecone")


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Return embeddings via OpenRouter (default) or direct OpenAI.
    Controlled by EMBEDDING_PROVIDER in .env.
    """
    from openai import OpenAI

    provider = Config.EMBEDDING_PROVIDER.lower()

    if provider == "openrouter":
        client = OpenAI(
            api_key  = Config.OPENROUTER_API_KEY,
            base_url = Config.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": Config.OPENROUTER_SITE_URL,
                "X-Title":      Config.OPENROUTER_SITE_NAME,
            },
        )
    elif provider == "openai":
        import os
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {provider!r}. Use 'openrouter' or 'openai'."
        )

    response = client.embeddings.create(
        model = Config.EMBEDDING_MODEL,
        input = texts,
    )
    return [item.embedding for item in response.data]


# ── Index management ───────────────────────────────────────────────────────────

def get_or_create_index():
    """Connect to Pinecone and return the index object (creates if missing)."""
    Pinecone, ServerlessSpec = _pinecone()
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if Config.PINECONE_INDEX_NAME not in existing:
        print(f"[vector_store] Creating Pinecone index '{Config.PINECONE_INDEX_NAME}' …")
        pc.create_index(
            name      = Config.PINECONE_INDEX_NAME,
            dimension = Config.EMBEDDING_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(
                cloud  = "aws",
                region = Config.PINECONE_ENVIRONMENT,
            ),
        )
        # Wait until ready
        for _ in range(20):
            desc = pc.describe_index(Config.PINECONE_INDEX_NAME)
            if desc.status.get("ready", False):
                break
            time.sleep(3)
        print(f"[vector_store] Index ready.")

    return pc.Index(Config.PINECONE_INDEX_NAME)


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: list[Chunk],
    namespace: str,
    batch_size: int = 50,
) -> int:
    """
    Embed and upsert chunks into Pinecone under `namespace`.
    Returns total vectors upserted.
    """
    index = get_or_create_index()
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        embeddings = _get_embeddings(texts)

        vectors = [
            {
                "id":       c.chunk_id,
                "values":   emb,
                "metadata": c.to_pinecone_metadata(),
            }
            for c, emb in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors, namespace=namespace)
        total += len(vectors)
        print(f"  [upsert] {total}/{len(chunks)} chunks → namespace='{namespace}'")

    return total


# ── Query ─────────────────────────────────────────────────────────────────────

def query_chunks(
    query_text: str,
    namespace: str,
    top_k: int = 5,
    filter_meta: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over a namespace.
    Returns list of { chunk_id, score, metadata, text } dicts.
    """
    index   = get_or_create_index()
    emb     = _get_embeddings([query_text])[0]

    kwargs: dict = dict(
        vector    = emb,
        top_k     = top_k,
        namespace = namespace,
        include_metadata = True,
    )
    if filter_meta:
        kwargs["filter"] = filter_meta

    results = index.query(**kwargs)
    hits = []
    for match in results.matches:
        hits.append({
            "chunk_id": match.id,
            "score":    match.score,
            "metadata": match.metadata,
        })
    return hits


# ── Delete namespace ──────────────────────────────────────────────────────────

def delete_namespace(namespace: str) -> None:
    """Remove all vectors for a given file namespace."""
    index = get_or_create_index()
    index.delete(delete_all=True, namespace=namespace)
    print(f"[vector_store] Deleted namespace '{namespace}'")


# ── Ingest a whole md file ────────────────────────────────────────────────────

def ingest_md_file(md_path: str | Path, force: bool = False) -> str:
    """
    Full pipeline: chunk → embed → upsert.
    Namespace = file stem (e.g. "chart_01").
    If `force=False`, skips re-ingestion if namespace already has vectors.
    Returns the namespace used.
    """
    md_path   = Path(md_path)
    namespace = md_path.stem   # e.g. "chart_01"
    index     = get_or_create_index()

    if not force:
        # Check if namespace already populated
        stats = index.describe_index_stats()
        ns_stats = stats.namespaces.get(namespace, {})
        if ns_stats.get("vector_count", 0) > 0:
            print(f"[ingest] '{namespace}' already in Pinecone ({ns_stats['vector_count']} vecs). Skipping.")
            return namespace

    print(f"[ingest] Chunking '{md_path.name}' …")
    chunks = chunk_markdown(md_path)
    print(f"[ingest] {len(chunks)} chunks created.")

    upsert_chunks(chunks, namespace=namespace)
    return namespace


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <path_to_md_file> [query_text]")
        sys.exit(1)

    md   = Path(sys.argv[1])
    ns   = ingest_md_file(md, force=True)

    query = sys.argv[2] if len(sys.argv) > 2 else "COPD exacerbation diagnosis"
    print(f"\n🔍  Query: {query!r}")
    hits = query_chunks(query, namespace=ns, top_k=3)
    for h in hits:
        print(f"  score={h['score']:.3f}  section={h['metadata'].get('section', '')[:40]}")
