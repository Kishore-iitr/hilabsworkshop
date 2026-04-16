"""
test.py — Main entry point for the clinical AI evaluation pipeline.

Usage:
    python test.py <input.json> <output.json> [--md path/to/chart.md] [--force-ingest]

The script:
  1. Loads the system output JSON (entities to evaluate).
  2. Finds the companion .md file (auto-discovered or --md flag).
  3. Chunks and ingests the MD into Pinecone (skips if already done).
  4. Runs LLM evaluation on every entity (RAG-augmented).
  5. Aggregates verdicts into the required output schema.
  6. Writes the output JSON.

Example:
    python test.py test_data/chart_01/chart_01.json output/chart_01.json
"""

import streamlit as st

st.title("Test App")
st.write("heeelllo")





from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config import Config
from chunker import chunk_markdown
from vector_store import ingest_md_file, get_or_create_index
from evaluator import evaluate_batch
from metrics import aggregate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_md(json_path: Path) -> Path | None:
    """
    Auto-discover the companion .md file.
    Looks in: same directory, parent directory.
    """
    stem = json_path.stem  # e.g. "chart_01"
    candidates = [
        json_path.parent / f"{stem}.md",
        json_path.parent.parent / f"{stem}.md",
        json_path.parent / f"{stem.split('__')[0]}.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _progress(i: int, total: int, entity_name: str) -> None:
    bar_len = 30
    filled  = int(bar_len * (i + 1) / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = (i + 1) / total * 100
    print(f"\r  [{bar}] {pct:5.1f}%  [{i+1}/{total}] {entity_name[:40]:<40}", end="", flush=True)
    if i + 1 == total:
        print()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    json_path: Path,
    output_path: Path,
    md_path: Path | None = None,
    force_ingest: bool = False,
    max_entities: int | None = None,
    model: str | None = None,
) -> dict:
    t0 = time.time()

    # ── 0. Resolve MD path ────────────────────────────────────────────────────
    if md_path is None:
        md_path = _discover_md(json_path)
    if md_path is None or not md_path.exists():
        raise FileNotFoundError(
            f"Cannot find companion .md for '{json_path}'. "
            f"Pass --md <path> explicitly."
        )

    print(f"\n{'='*60}")
    print(f"📄  Input JSON : {json_path}")
    print(f"📋  MD chart   : {md_path}")
    print(f"📤  Output     : {output_path}")
    print(f"{'='*60}\n")

    # ── 1. Ingest MD into Pinecone ────────────────────────────────────────────
    print("🔵  Stage 1 — Chunking & ingesting MD into Pinecone …")
    namespace = ingest_md_file(md_path, force=force_ingest)
    print(f"    Namespace: '{namespace}'  ✓\n")

    # ── 2. Load system entities ───────────────────────────────────────────────
    print("🔵  Stage 2 — Loading system output entities …")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    entities: list[dict] = raw if isinstance(raw, list) else raw.get("entities", [])
    if max_entities:
        entities = entities[:max_entities]
    print(f"    {len(entities)} entities loaded.\n")

    # ── 3. LLM Evaluation (RAG-augmented) ─────────────────────────────────────
    print("🔵  Stage 3 — LLM Evaluation (RAG over Pinecone) …")

    def _prog(i, total):
        _progress(i, total, entities[i].get("entity", ""))

    verdicts = evaluate_batch(entities, namespace=namespace, progress_callback=_prog, model=model)
    print(f"    {len(verdicts)} verdicts collected.\n")

    # ── 4. Aggregate into output schema ───────────────────────────────────────
    print("🔵  Stage 4 — Aggregating metrics …")
    result = aggregate(verdicts, file_name=output_path.name)

    # ── 5. Write output ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    elapsed = time.time() - t0
    print(f"\n✅  Done in {elapsed:.1f}s  →  {output_path}\n")
    print(json.dumps(result, indent=2))

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clinical AI evaluation pipeline (OpenRouter backend)."
    )
    parser.add_argument("input_json",  type=Path, help="System output JSON file")
    parser.add_argument("output_json", type=Path, help="Output evaluation JSON file")
    parser.add_argument("--md",           type=Path, default=None, help="Companion .md chart file")
    parser.add_argument("--force-ingest", action="store_true",     help="Re-ingest MD even if cached")
    parser.add_argument("--max-entities", type=int, default=None,  help="Limit entities (dev mode)")
    parser.add_argument("--model",        type=str, default=None,
                        help="Override LLM model slug, e.g. openai/gpt-4o-mini")
    parser.add_argument("--list-models",  action="store_true",
                        help="Print candidate model list and exit")
    args = parser.parse_args()

    if args.list_models:
        Config.print_model_info()
        return

    Config.print_model_info()

    run_pipeline(
        json_path    = args.input_json,
        output_path  = args.output_json,
        md_path      = args.md,
        force_ingest = args.force_ingest,
        max_entities = args.max_entities,
        model        = args.model,
    )


if __name__ == "__main__":
    main()
