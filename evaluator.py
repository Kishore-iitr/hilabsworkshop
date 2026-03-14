"""
evaluator.py — Stage-1 LLM evaluation engine (OpenRouter backend).

LLM returns confidence scores (0.0–1.0) per dimension instead of booleans,
so the final error rates are nuanced floats, not just 0 or 1.

Usage (standalone test):
    python evaluator.py chart_01.json chart_01.md 10
    python evaluator.py chart_01.json chart_01.md 10 --model openai/gpt-oss-120b:free
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import Config
from vector_store import query_chunks, ingest_md_file


# ── OpenRouter client (singleton) ────────────────────────────────────────────

_client: OpenAI | None = None

def _llm() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key  = Config.OPENROUTER_API_KEY,
            base_url = Config.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": Config.OPENROUTER_SITE_URL,
                "X-Title":      Config.OPENROUTER_SITE_NAME,
            },
        )
    return _client


def _call_llm(system: str, user: str, model: str | None = None) -> str:
    """
    Call OpenRouter with automatic fallback through CANDIDATE_MODELS.
    Returns raw text from the model.
    """
    active = model or Config.LLM_MODEL
    to_try = [active] + [m for m in Config.CANDIDATE_MODELS if m != active]
    last_err = None

    for m in to_try:
        try:
            resp = _llm().chat.completions.create(
                model      = m,
                max_tokens = Config.MAX_TOKENS,
                messages   = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            if m != active:
                print(f"\n  ⚠️  Fell back to: {m}")
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            print(f"\n  ⚠️  '{m}' failed: {exc}")
            continue

    raise RuntimeError(f"All models failed. Last: {last_err}")


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior clinical NLP auditor evaluating an AI entity-extraction pipeline on medical charts.

For each entity, score EVERY dimension on a continuous scale from 0.0 to 1.0:
  - 1.0  = completely correct / fully present
  - 0.0  = completely wrong / entirely missing
  - 0.3–0.7 = partially correct, ambiguous, or uncertain

This is critical: DO NOT return only 0 or 1. Use the full range to reflect nuance.

ALWAYS respond ONLY with a valid JSON object. No prose. No markdown fences. No extra keys.

Exact required schema:
{
  "entity_type_score":   <float 0.0–1.0>,
  "assertion_score":     <float 0.0–1.0>,
  "temporality_score":   <float 0.0–1.0>,
  "subject_score":       <float 0.0–1.0>,
  "event_date_score":    <float 0.0–1.0 | null>,
  "attribute_score":     <float 0.0–1.0>,
  "reasoning":           "<one concise sentence explaining your scores>"
}

Scoring guide per dimension:
- entity_type_score  : How correct is the assigned type (PROBLEM, MEDICINE, PROCEDURE, etc.)?
                       1.0=perfect, 0.5=plausible but debatable, 0.0=clearly wrong category.
- assertion_score    : How well does POSITIVE/NEGATIVE/UNCERTAIN match what the text implies?
                       1.0=clear match, 0.5=text is ambiguous, 0.0=clear mismatch.
- temporality_score  : How well does CURRENT/CLINICAL_HISTORY/UPCOMING/UNCERTAIN match context?
                       1.0=clear match, 0.5=debatable, 0.0=clearly wrong.
- subject_score      : How well does PATIENT/FAMILY_MEMBER match who the text refers to?
                       1.0=clear match, 0.5=ambiguous, 0.0=wrong.
- event_date_score   : If a date/time is present in context: 1.0=date captured correctly,
                       0.5=date present but imprecise, 0.0=date missing or wrong.
                       Set to null if no date is relevant for this entity.
- attribute_score    : How complete is metadata_from_qa (values, units, relations)?
                       1.0=all values/units captured, 0.5=partially captured, 0.0=empty when data exists.
"""


def _build_user_prompt(entity: dict, hits: list[dict]) -> str:
    parts = []
    for h in hits:
        meta = h.get("metadata", {})
        parts.append(
            f"[Section: {meta.get('section','?')}  Page: {meta.get('page_no','?')}]\n"
            f"{meta.get('text', entity.get('text',''))[:500]}"
        )
    context = "\n\n---\n\n".join(parts) or entity.get("text", "")[:600]

    meta_qa   = entity.get("metadata_from_qa") or {}
    relations = meta_qa.get("relations", [])
    rel_str   = json.dumps(relations[:8], indent=2) if relations else "none"

    return f"""## CHART SOURCE CONTEXT (retrieved from MD)

{context}

## SYSTEM EXTRACTED ENTITY

Entity Text  : {entity.get('entity', '')}
Entity Type  : {entity.get('entity_type', '')}
Assertion    : {entity.get('assertion', '')}
Temporality  : {entity.get('temporality', '')}
Subject      : {entity.get('subject', '')}
Heading      : {entity.get('heading', '')[:120]}
Context Snip : {entity.get('text', '')[:300]}
Relations    : {rel_str}

## YOUR VERDICT (JSON only, continuous scores 0.0–1.0, no fences):"""


# ── Parsing & validation ──────────────────────────────────────────────────────

def _clamp(val: Any, default: float = 0.5) -> float:
    """Ensure value is a float in [0.0, 1.0]. Falls back to default."""
    try:
        f = float(val)
        return max(0.0, min(1.0, round(f, 4)))
    except (TypeError, ValueError):
        return default


def _parse_verdict(raw: str, entity: dict) -> dict[str, Any]:
    """Parse LLM output into a clean verdict with clamped float scores."""
    # Strip markdown fences some models add
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean).strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        # Full fallback — use 0.5 (uncertain) so we don't bias error rates
        return {
            "entity_type_score":  0.5,
            "assertion_score":    0.5,
            "temporality_score":  0.5,
            "subject_score":      0.5,
            "event_date_score":   None,
            "attribute_score":    0.5,
            "reasoning": f"LLM output unparseable — defaulted to 0.5: {raw[:120]}",
            "_entity": entity,
        }

    # Normalise: accept both old boolean schema and new float schema
    def _resolve(score_key: str, bool_key: str, invert: bool = False) -> float:
        if score_key in parsed:
            return _clamp(parsed[score_key])
        # boolean fallback (true=correct → score 1.0, false=wrong → 0.0)
        bval = parsed.get(bool_key)
        if isinstance(bval, bool):
            return 1.0 if (bval and not invert) else 0.0
        return 0.5  # unknown

    ev_raw = parsed.get("event_date_score", parsed.get("event_date_accurate"))
    if ev_raw is None and parsed.get("event_date_present") is False:
        event_date_score = None
    elif ev_raw is None:
        event_date_score = None
    elif isinstance(ev_raw, bool):
        event_date_score = 1.0 if ev_raw else 0.0
    else:
        event_date_score = _clamp(ev_raw)

    return {
        "entity_type_score":  _resolve("entity_type_score",  "entity_type_correct"),
        "assertion_score":    _resolve("assertion_score",     "assertion_correct"),
        "temporality_score":  _resolve("temporality_score",   "temporality_correct"),
        "subject_score":      _resolve("subject_score",       "subject_correct"),
        "event_date_score":   event_date_score,
        "attribute_score":    _resolve("attribute_score",     "attribute_complete"),
        "reasoning":          parsed.get("reasoning", ""),
        "_entity":            entity,
    }


# ── Core API ──────────────────────────────────────────────────────────────────

def evaluate_entity(
    entity: dict,
    namespace: str,
    top_k: int = 4,
    model: str | None = None,
) -> dict[str, Any]:
    query       = f"{entity.get('entity','')} {entity.get('heading','')} {entity.get('entity_type','')}"
    hits        = query_chunks(query, namespace=namespace, top_k=top_k)
    user_prompt = _build_user_prompt(entity, hits)
    raw         = _call_llm(SYSTEM_PROMPT, user_prompt, model=model)
    return _parse_verdict(raw, entity)


def evaluate_batch(
    entities: list[dict],
    namespace: str,
    progress_callback=None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    results = []
    for i, ent in enumerate(entities):
        if progress_callback:
            progress_callback(i, len(entities))
        results.append(evaluate_entity(ent, namespace=namespace, model=model))
    return results


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test evaluator on N entities.")
    parser.add_argument("system_json",  type=Path)
    parser.add_argument("chart_md",     type=Path)
    parser.add_argument("max_entities", type=int, nargs="?", default=10)
    parser.add_argument("--model",      type=str, default=None,
                        help="OpenRouter model slug, e.g. openai/gpt-oss-120b:free")
    args = parser.parse_args()

    Config.print_model_info()

    active_model = args.model or Config.LLM_MODEL
    print(f"\n  Using : {active_model}\n")

    print(f"📥  Ingesting '{args.chart_md.name}' …")
    namespace = ingest_md_file(args.chart_md)

    raw_json = json.loads(args.system_json.read_text(encoding="utf-8"))
    entities = raw_json if isinstance(raw_json, list) else raw_json.get("entities", [])
    entities = entities[:args.max_entities]

    print(f"🔬  Evaluating {len(entities)} entities …\n")

    def _prog(i, total):
        ent_name = entities[i].get("entity", "")[:55]
        print(f"  [{i+1:>3}/{total}] {ent_name}")

    verdicts = evaluate_batch(entities, namespace=namespace,
                               progress_callback=_prog, model=active_model)

    print("\n📊  Average Scores (1.0 = perfect, 0.0 = all wrong):")
    score_keys = ["entity_type_score","assertion_score","temporality_score",
                  "subject_score","attribute_score"]
    n = len(verdicts)
    for k in score_keys:
        vals = [v[k] for v in verdicts if v.get(k) is not None]
        avg  = sum(vals) / len(vals) if vals else 0.0
        bar  = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {k:<22} avg={avg:.3f}  [{bar}]")

    out = args.system_json.with_suffix(".verdicts.json")
    out.write_text(json.dumps(verdicts, indent=2, default=str))
    print(f"\n💾  Verdicts → {out}")