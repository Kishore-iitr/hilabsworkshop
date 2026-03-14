"""
config.py — Centralised configuration loader.
All values come from environment / .env — zero hardcoding.

LLM backend : OpenRouter  (https://openrouter.ai)
Embeddings  : OpenRouter free embedding model  OR  Pinecone's built-in inference
Vector store: Pinecone
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Required env var '{key}' is not set. "
            f"Copy .env.example → .env and fill in your credentials."
        )
    return val


def _optional(key: str, default):
    val = os.getenv(key)
    return val if val is not None else default


def _list_from_env(key: str, default: list[str]) -> list[str]:
    """Parse a comma-separated env var into a list, stripping whitespace."""
    raw = os.getenv(key, "")
    if not raw.strip():
        return default
    return [m.strip() for m in raw.split(",") if m.strip()]


class Config:
    # ── OpenRouter ─────────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str  = _require("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = _optional(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )

    # ── Active LLM model (picked from CANDIDATE_MODELS list) ───────────────
    #    Set LLM_MODEL in .env to pin a specific model, or leave blank to use
    #    the first entry in CANDIDATE_MODELS automatically.
    _raw_active: str = _optional("LLM_MODEL", "")

    # ── ✏️  CANDIDATE MODEL LIST ───────────────────────────────────────────
    #    Add / remove OpenRouter model slugs here (or override via .env as a
    #    comma-separated string in CANDIDATE_MODELS).
    #    Free-tier models on OpenRouter as of 2026-03:
    CANDIDATE_MODELS: list[str] = _list_from_env(
        "CANDIDATE_MODELS",
        default=[
            # ── Primary: OpenAI OSS 120B (free via OpenRouter) ─────────────
            "openai/gpt-oss-120b:free",            # ← active default
            # ── Free OpenAI models via OpenRouter ──────────────────────────
            "openai/gpt-4o-mini",                  # fast fallback
            "openai/gpt-4.1-mini",                 # latest mini
            "openai/gpt-3.5-turbo",                # classic
            # ── Other free heavy-hitters ───────────────────────────────────
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
        ],
    )

    # Active model: env pin > first in list
    LLM_MODEL: str = _raw_active if _raw_active else (
        CANDIDATE_MODELS[0] if CANDIDATE_MODELS else "openai/gpt-4o-mini"
    )

    MAX_TOKENS: int = int(_optional("MAX_TOKENS", "2048"))

    # ── Optional site metadata sent to OpenRouter (for free-tier routing) ──
    OPENROUTER_SITE_URL: str  = _optional("OPENROUTER_SITE_URL", "http://localhost")
    OPENROUTER_SITE_NAME: str = _optional("OPENROUTER_SITE_NAME", "ClinicalEval")

    # ── Pinecone ───────────────────────────────────────────────────────────
    PINECONE_API_KEY: str     = _require("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str  = _optional("PINECONE_INDEX_NAME", "clinical-eval-chunks")
    PINECONE_ENVIRONMENT: str = _optional("PINECONE_ENVIRONMENT", "us-east-1-aws")

    # ── Embeddings ─────────────────────────────────────────────────────────
    #    "openrouter" → uses OpenAI-compatible /embeddings on OpenRouter
    #    "openai"     → direct OpenAI API (needs OPENAI_API_KEY)
    EMBEDDING_PROVIDER: str = _optional("EMBEDDING_PROVIDER", "openrouter")
    EMBEDDING_MODEL: str    = _optional(
        "EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    EMBEDDING_DIM: int      = int(_optional("EMBEDDING_DIM", "1536"))

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_SIZE: int    = int(_optional("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(_optional("CHUNK_OVERLAP", "80"))

    # ── Processing ─────────────────────────────────────────────────────────
    BATCH_SIZE: int = int(_optional("BATCH_SIZE", "5"))

    # ── Clinical entity schema ─────────────────────────────────────────────
    ENTITY_TYPES: list = [
        "MEDICINE", "PROBLEM", "PROCEDURE", "TEST",
        "VITAL_NAME", "IMMUNIZATION", "MEDICAL_DEVICE",
        "MENTAL_STATUS", "SDOH", "SOCIAL_HISTORY",
    ]
    ASSERTION_TYPES: list   = ["POSITIVE", "NEGATIVE", "UNCERTAIN"]
    TEMPORALITY_TYPES: list = ["CURRENT", "CLINICAL_HISTORY", "UPCOMING", "UNCERTAIN"]
    SUBJECT_TYPES: list     = ["PATIENT", "FAMILY_MEMBER"]

    # ── Output schema (zeroed baseline) ───────────────────────────────────
    @staticmethod
    def empty_output(file_name: str) -> dict:
        return {
            "file_name": file_name,
            "entity_type_error_rate":  {k: 0.0 for k in Config.ENTITY_TYPES},
            "assertion_error_rate":    {k: 0.0 for k in Config.ASSERTION_TYPES},
            "temporality_error_rate":  {k: 0.0 for k in Config.TEMPORALITY_TYPES},
            "subject_error_rate":      {k: 0.0 for k in Config.SUBJECT_TYPES},
            "event_date_accuracy":     0.0,
            "attribute_completeness":  0.0,
        }

    @classmethod
    def print_model_info(cls) -> None:
        """Print active model and full candidate list to stdout."""
        print(f"\n{'─'*55}")
        print(f"  🤖  Active LLM  : {cls.LLM_MODEL}")
        print(f"  📋  Candidates  ({len(cls.CANDIDATE_MODELS)}):")
        for i, m in enumerate(cls.CANDIDATE_MODELS):
            marker = "→" if m == cls.LLM_MODEL else " "
            print(f"       {marker} [{i}] {m}")
        print(f"  🔗  Via         : OpenRouter ({cls.OPENROUTER_BASE_URL})")
        print(f"{'─'*55}\n")