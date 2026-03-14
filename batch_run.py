"""
batch_run.py — Runs the full pipeline over all folders in test_data/.

Usage:
    python batch_run.py --data-dir test_data/ --out-dir output/ [--workers 3] [--max-entities 100]

This script:
  • Discovers all (folder.json, folder.md) pairs under --data-dir
  • Runs test.py's pipeline for each pair
  • Collects all output JSONs in --out-dir
  • Prints a summary error-rate table at the end
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from test import run_pipeline


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """
    Scan data_dir for directories that contain a matching .json + .md pair.
    Returns list of (json_path, md_path).
    """
    pairs = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        jsons = list(folder.glob("*.json"))
        mds   = list(folder.glob("*.md"))
        if jsons and mds:
            # Match by stem if multiple files exist
            for jf in jsons:
                matching_mds = [m for m in mds if m.stem == jf.stem]
                md = matching_mds[0] if matching_mds else mds[0]
                pairs.append((jf, md))
    return pairs


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    from config import Config

    print("\n" + "=" * 80)
    print("📊  BATCH EVALUATION SUMMARY")
    print("=" * 80)

    # Entity type errors
    print("\n🔹  Entity Type Error Rates (avg across files):")
    for et in Config.ENTITY_TYPES:
        vals = [r["entity_type_error_rate"].get(et, 0.0) for r in results]
        avg  = sum(vals) / len(vals) if vals else 0.0
        bar  = "█" * int(avg * 20)
        print(f"    {et:<20} {avg:.3f}  {bar}")

    print("\n🔹  Assertion Error Rates:")
    for a in Config.ASSERTION_TYPES:
        vals = [r["assertion_error_rate"].get(a, 0.0) for r in results]
        avg  = sum(vals) / len(vals) if vals else 0.0
        print(f"    {a:<20} {avg:.3f}")

    print("\n🔹  Temporality Error Rates:")
    for t in Config.TEMPORALITY_TYPES:
        vals = [r["temporality_error_rate"].get(t, 0.0) for r in results]
        avg  = sum(vals) / len(vals) if vals else 0.0
        print(f"    {t:<20} {avg:.3f}")

    print("\n🔹  Subject Error Rates:")
    for s in Config.SUBJECT_TYPES:
        vals = [r["subject_error_rate"].get(s, 0.0) for r in results]
        avg  = sum(vals) / len(vals) if vals else 0.0
        print(f"    {s:<20} {avg:.3f}")

    avg_date = sum(r["event_date_accuracy"] for r in results) / len(results) if results else 0.0
    avg_attr = sum(r["attribute_completeness"] for r in results) / len(results) if results else 0.0

    print(f"\n🔹  Avg Event Date Accuracy  : {avg_date:.3f}")
    print(f"🔹  Avg Attribute Completeness: {avg_attr:.3f}")
    print("=" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation runner.")
    parser.add_argument("--data-dir",     type=Path, required=True, help="Root folder with test_data subfolders")
    parser.add_argument("--out-dir",      type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--workers",      type=int,  default=1, help="Parallel workers (be careful with API rate limits)")
    parser.add_argument("--max-entities", type=int,  default=None, help="Limit entities per file (for testing)")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model for all files (any OpenRouter slug)")
    parser.add_argument("--list-models", action="store_true",
                        help="Print candidate model list and exit")
    args = parser.parse_args()

    if args.list_models:
        from config import Config
        Config.print_model_info()
        return

    pairs = discover_pairs(args.data_dir)
    print(f"\n🗂️   Found {len(pairs)} file pair(s) in '{args.data_dir}'\n")

    if not pairs:
        print("No .json/.md pairs found. Check your --data-dir.")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results      = []
    failed       = []

    def _process(pair: tuple[Path, Path]) -> dict | None:
        jf, md = pair
        out    = args.out_dir / jf.name
        try:
            return run_pipeline(
                json_path    = jf,
                output_path  = out,
                md_path      = md,
                force_ingest = args.force_ingest,
                max_entities = args.max_entities,
                model        = args.model,
            )
        except Exception as e:
            print(f"\n❌  FAILED: {jf.name}  →  {e}")
            traceback.print_exc()
            failed.append(jf.name)
            return None

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, p): p for p in pairs}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)
    else:
        for pair in pairs:
            res = _process(pair)
            if res:
                results.append(res)

    print(f"\n✅  Completed: {len(results)}/{len(pairs)}")
    if failed:
        print(f"❌  Failed   : {failed}")

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
