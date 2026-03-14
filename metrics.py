"""
metrics.py — Aggregates per-entity LLM confidence scores into the output schema.

Each verdict now carries float scores (0.0–1.0) per dimension.
Error rate = 1.0 - avg(score) per sub-category, giving smooth values like 0.23.

Output schema:
{
  "file_name": "...",
  "entity_type_error_rate":  { "MEDICINE": 0.0–1.0, ... },
  "assertion_error_rate":    { "POSITIVE": 0.0–1.0, ... },
  "temporality_error_rate":  { "CURRENT": 0.0–1.0, ... },
  "subject_error_rate":      { "PATIENT": 0.0–1.0, ... },
  "event_date_accuracy":     0.0–1.0,
  "attribute_completeness":  0.0–1.0,
}

Usage:
    python metrics.py chart_01.verdicts.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import Config


def aggregate(verdicts: list[dict[str, Any]], file_name: str) -> dict[str, Any]:
    """
    Convert per-entity score dicts → output JSON.

    For each sub-category (e.g. entity_type=MEDICINE):
        error_rate  = 1.0 - mean(entity_type_score)   for all entities of that type
        accuracy    = mean(score)                       (for event_date, attribute)

    This gives smooth values like 0.27 instead of only 0.0 or 1.0.
    """
    output = Config.empty_output(file_name)

    # accumulators: sum of scores and count per sub-category
    et_sum:  dict[str, float] = defaultdict(float)
    et_cnt:  dict[str, int]   = defaultdict(int)

    as_sum:  dict[str, float] = defaultdict(float)
    as_cnt:  dict[str, int]   = defaultdict(int)

    tm_sum:  dict[str, float] = defaultdict(float)
    tm_cnt:  dict[str, int]   = defaultdict(int)

    su_sum:  dict[str, float] = defaultdict(float)
    su_cnt:  dict[str, int]   = defaultdict(int)

    date_scores: list[float] = []
    attr_scores: list[float] = []

    for v in verdicts:
        ent = v.get("_entity", {})

        # ── Entity type ───────────────────────────────────────────────────
        et = (ent.get("entity_type") or "UNKNOWN").upper()
        if et in Config.ENTITY_TYPES:
            score = _safe_float(v.get("entity_type_score"), default=0.5)
            et_sum[et] += score
            et_cnt[et] += 1

        # ── Assertion ─────────────────────────────────────────────────────
        ass = (ent.get("assertion") or "UNKNOWN").upper()
        if ass in Config.ASSERTION_TYPES:
            score = _safe_float(v.get("assertion_score"), default=0.5)
            as_sum[ass] += score
            as_cnt[ass] += 1

        # ── Temporality ───────────────────────────────────────────────────
        tmp = (ent.get("temporality") or "UNKNOWN").upper()
        if tmp in Config.TEMPORALITY_TYPES:
            score = _safe_float(v.get("temporality_score"), default=0.5)
            tm_sum[tmp] += score
            tm_cnt[tmp] += 1

        # ── Subject ───────────────────────────────────────────────────────
        sub = (ent.get("subject") or "UNKNOWN").upper()
        if sub in Config.SUBJECT_TYPES:
            score = _safe_float(v.get("subject_score"), default=0.5)
            su_sum[sub] += score
            su_cnt[sub] += 1

        # ── Event date ────────────────────────────────────────────────────
        ev = v.get("event_date_score")
        if ev is not None:
            date_scores.append(_safe_float(ev, default=0.5))

        # ── Attribute completeness ────────────────────────────────────────
        attr_scores.append(_safe_float(v.get("attribute_score"), default=0.5))

    # ── Fill output: error_rate = 1 - avg_score ───────────────────────────────

    for et_key in Config.ENTITY_TYPES:
        cnt = et_cnt.get(et_key, 0)
        avg = (et_sum[et_key] / cnt) if cnt else None
        # error_rate = 1 - correctness; 0.0 means no errors, 1.0 means all wrong
        output["entity_type_error_rate"][et_key] = round(1.0 - avg, 4) if avg is not None else 0.0

    for as_key in Config.ASSERTION_TYPES:
        cnt = as_cnt.get(as_key, 0)
        avg = (as_sum[as_key] / cnt) if cnt else None
        output["assertion_error_rate"][as_key] = round(1.0 - avg, 4) if avg is not None else 0.0

    for tm_key in Config.TEMPORALITY_TYPES:
        cnt = tm_cnt.get(tm_key, 0)
        avg = (tm_sum[tm_key] / cnt) if cnt else None
        output["temporality_error_rate"][tm_key] = round(1.0 - avg, 4) if avg is not None else 0.0

    for su_key in Config.SUBJECT_TYPES:
        cnt = su_cnt.get(su_key, 0)
        avg = (su_sum[su_key] / cnt) if cnt else None
        output["subject_error_rate"][su_key] = round(1.0 - avg, 4) if avg is not None else 0.0

    # event_date_accuracy = mean of date scores (higher = more accurate)
    output["event_date_accuracy"] = (
        round(sum(date_scores) / len(date_scores), 4) if date_scores else 0.0
    )

    # attribute_completeness = mean of attr scores (higher = more complete)
    output["attribute_completeness"] = (
        round(sum(attr_scores) / len(attr_scores), 4) if attr_scores else 0.0
    )

    return output


def _safe_float(val: Any, default: float = 0.5) -> float:
    """Coerce to float in [0,1]. Handles booleans, strings, None."""
    if val is None:
        return default
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return default


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metrics.py <verdicts.json>")
        sys.exit(1)

    vpath    = Path(sys.argv[1])
    verdicts = json.loads(vpath.read_text(encoding="utf-8"))
    fname    = vpath.stem.replace(".verdicts", "") + ".json"
    result   = aggregate(verdicts, fname)

    print(json.dumps(result, indent=2))

    out = vpath.with_suffix("").with_suffix(".output.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\n💾  Output → {out}")