"""
chunker.py — Semantic markdown chunker.

Splits an .md clinical document into overlapping chunks that respect:
  • heading boundaries  (H1 / H2 / H3)
  • page boundaries     (<ocr_service_page_start>N<ocr_service_page_start>)
  • configurable token-window with overlap

Each chunk carries rich metadata so it can be stored in Pinecone and later
retrieved by entity/heading context.

Usage (standalone test):
    python chunker.py path/to/chart.md
"""

from __future__ import annotations

import re
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator

from config import Config


# ── Simple whitespace-based token counter (no external deps needed) ──────────
def _token_count(text: str) -> int:
    """Approximate token count ≈ word-split.  Replace with tiktoken if available."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str          # "<file_stem>__<seq>"
    file_name: str
    page_no: int | None
    section: str           # last seen H1/H2/H3 heading
    text: str
    token_count: int
    char_start: int        # character offset in original file
    char_end: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pinecone_metadata(self) -> dict:
        """Keep only Pinecone-safe scalar fields (no nested objects)."""
        return {
            "chunk_id":   self.chunk_id,
            "file_name":  self.file_name,
            "page_no":    self.page_no if self.page_no is not None else -1,
            "section":    self.section,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end":   self.char_end,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

_PAGE_RE    = re.compile(r"<ocr_service_page_start>(\d+)<ocr_service_page_start>")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)


def _detect_page(line: str) -> int | None:
    m = _PAGE_RE.search(line)
    return int(m.group(1)) if m else None


def _detect_heading(line: str) -> str | None:
    m = _HEADING_RE.match(line)
    return m.group(2).strip() if m else None


def _sliding_window(
    tokens: list[str],
    chunk_size: int,
    overlap: int,
) -> Generator[tuple[int, int], None, None]:
    """Yield (start, end) token-index pairs."""
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        yield start, end
        if end == len(tokens):
            break
        start += chunk_size - overlap


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_markdown(
    md_path: str | Path,
    chunk_size: int = Config.CHUNK_SIZE,
    overlap: int = Config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Read a clinical .md file and return a list of Chunk objects.

    Strategy
    --------
    1. Walk lines, tracking current page and section heading.
    2. Group lines into logical "sections" (heading → next heading / page break).
    3. Apply a sliding-window over each section's tokens to produce chunks.
    4. Each chunk inherits the section's page/heading metadata.
    """
    md_path  = Path(md_path)
    raw_text = md_path.read_text(encoding="utf-8", errors="replace")
    file_stem = md_path.stem

    lines   = raw_text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    seq     = 0

    # ── Pass 1: annotate each line with (page_no, section_heading) ──────────
    current_page    = None
    current_heading = "DOCUMENT_START"
    annotated: list[tuple[str, int | None, str]] = []   # (line, page, heading)

    for line in lines:
        detected_page = _detect_page(line)
        if detected_page is not None:
            current_page = detected_page

        detected_heading = _detect_heading(line)
        if detected_heading is not None:
            current_heading = detected_heading

        annotated.append((line, current_page, current_heading))

    # ── Pass 2: group consecutive lines that share same (page, heading) ─────
    # Then chunk each group with sliding window
    i = 0
    char_cursor = 0

    while i < len(annotated):
        _, pg, hd = annotated[i]

        # collect run of lines with this (page, heading)
        j = i
        while j < len(annotated) and annotated[j][1] == pg and annotated[j][2] == hd:
            j += 1

        segment_lines = [a[0] for a in annotated[i:j]]
        segment_text  = "".join(segment_lines)
        seg_char_start = char_cursor
        char_cursor   += len(segment_text)

        # sliding window over this segment
        words = segment_text.split()
        if not words:
            i = j
            continue

        for w_start, w_end in _sliding_window(words, chunk_size, overlap):
            chunk_text = " ".join(words[w_start:w_end])
            tc = _token_count(chunk_text)

            # approximate char offsets within segment
            pre_text  = " ".join(words[:w_start])
            c_start   = seg_char_start + len(pre_text) + (1 if pre_text else 0)
            c_end     = c_start + len(chunk_text)

            chunks.append(Chunk(
                chunk_id    = f"{file_stem}__{seq:04d}",
                file_name   = md_path.name,
                page_no     = pg,
                section     = hd,
                text        = chunk_text,
                token_count = tc,
                char_start  = c_start,
                char_end    = c_end,
            ))
            seq += 1

        i = j

    return chunks


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_md_file>")
        sys.exit(1)

    path   = Path(sys.argv[1])
    result = chunk_markdown(path)

    print(f"\n✅  Chunked '{path.name}'  →  {len(result)} chunks\n")
    for c in result[:5]:   # preview first 5
        print(f"  [{c.chunk_id}]  page={c.page_no}  section='{c.section[:40]}'")
        print(f"    tokens={c.token_count}  text[:80]={c.text[:80]!r}")
        print()

    # Dump full output as JSON for inspection
    out_path = path.with_suffix(".chunks.json")
    out_path.write_text(json.dumps([c.to_dict() for c in result], indent=2))
    print(f"📄  Full chunk list saved → {out_path}")
