"""Langfuse tracing for the agent, with PII masking (adapted from course lab P11 `obs.py` / `samarkan`).

One trace per question:
    answer (agent) ─┬─ step_0 … (generation: model, tokens, latency)
                    ├─ tool:<name> (tool / retriever: arguments, result, evidence id)
                    ├─ finalize (generation)
                    └─ grounding (guardrail: ok, problems) [+ repair (generation)]
Trace scores: grounding_ok, repair_level, status.

Tracing is off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and the `langfuse` package is
installed). It never changes or blocks an answer: with tracing off every call below is a no-op.

Masking runs inside the SDK before anything leaves the process: e-mails, phone numbers, long numeric ids,
32-hex keys (hashed employee keys), and capitalised name-like sequences containing a word that is not part of
HRUM's own vocabulary (departments, jobs, metric catalogue, knowledge base, prompts). The warehouse has no names; the only way a name
reaches a trace is the user's question ("Berapa gaji Stephen Lee?"), and that is what the name rule targets.
It errs on the side of masking: an unknown capitalised phrase also becomes [NAME].
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any

from common.config import ROOT, db_path

log = logging.getLogger(__name__)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?![\w.])")
_HEX32 = re.compile(r"\b[0-9a-f]{32}\b")
_LONG_ID = re.compile(r"\b\d{12,}\b")
_CAP_SEQ = re.compile(r"\b[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)+\b")
_WORD = re.compile(r"[a-z][a-z'’-]+")

# Question words and other capitalised words that are not names (Indonesian + English)
_COMMON = {
    "berapa", "apa", "apakah", "kenapa", "mengapa", "bagaimana", "siapa", "mana", "kapan", "tolong",
    "sebutkan", "tampilkan", "bandingkan", "hitung", "jelaskan", "ini", "itu", "di", "ke", "dari", "untuk",
    "dan", "atau", "yang", "saya", "kamu", "anda", "hubungi", "tahun", "bulan", "data", "rata", "total",
    "jumlah", "gaji", "kompensasi", "karyawan", "pegawai", "departemen", "jabatan", "organisasi", "what",
    "how", "why", "which", "who", "when", "where", "please", "show", "list", "compare", "count", "give",
    "tell", "the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are", "january", "february",
    "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "januari", "februari", "maret", "mei", "juni", "juli", "agustus", "oktober", "desember",
}


@lru_cache
def known_vocabulary() -> frozenset[str]:
    """Lower-case words of HRUM's own texts: warehouse dimensions, metric catalogue, knowledge base, prompts."""
    words = set(_COMMON)
    texts: list[str] = []
    # Only sources that never contain people's names. Developer docs are excluded on purpose: they quote example
    # questions with names, which would put those names on the allowlist.
    for pattern in ("docs/kb/*.md", "warehouse/semantic/*.yaml", "agent/prompts/*.py"):
        texts += [p.read_text(errors="ignore") for p in ROOT.glob(pattern)]
    try:
        import duckdb

        with duckdb.connect(str(db_path()), read_only=True) as con:
            for sql in ("select distinct department || ' ' || organization_group from mart.dim_department",
                        "select distinct job || ' ' || job_family from mart.dim_job"):
                texts += [r[0] or "" for r in con.execute(sql).fetchall()]
    except Exception as e:  # noqa: BLE001 - masking still works with the docs vocabulary
        log.warning("tracing: warehouse vocabulary unavailable (%s); name masking is stricter", e)
    for t in texts:
        words.update(_WORD.findall(t.lower()))
    return frozenset(words)


def _mask_names(text: str) -> str:
    """In each capitalised sequence, replace the span from the first to the last unknown word with [NAME]."""
    vocab = known_vocabulary()

    def repl(m: re.Match[str]) -> str:
        tokens = m.group(0).split()
        unknown = [i for i, t in enumerate(tokens) if t.lower().strip("'’-") not in vocab]
        if not unknown:
            return m.group(0)
        return " ".join([*tokens[:unknown[0]], "[NAME]", *tokens[unknown[-1] + 1:]])

    return _CAP_SEQ.sub(repl, text)


def mask_text(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _HEX32.sub("[KEY]", text)
    text = _LONG_ID.sub("[ID]", text)
    text = _PHONE.sub("[PHONE]", text)
    return _mask_names(text)


def mask(data: Any = None, **_: Any) -> Any:
    """Langfuse mask hook: applied to every input, output and metadata value before export."""
    if isinstance(data, str):
        return mask_text(data)
    if isinstance(data, dict):
        return {k: mask(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(v) for v in data]
    return data


# ------------------------------------------------------------------ client

_override: Any = None   # tests inject a client wired to an in-memory exporter


def set_client(client: Any) -> None:
    global _override
    _override = client


@lru_cache
def _default_client() -> Any:
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() in ("0", "false", "no"):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("tracing: LANGFUSE_* is set but the langfuse package is missing; tracing off")
        return None
    from agent.prompts import PROMPT_VERSION

    return Langfuse(base_url=base_url(), mask=mask,
                    environment=os.environ.get("HRUM_ENV", "local"), release=f"prompt-{PROMPT_VERSION}")


def base_url() -> str | None:
    """LANGFUSE_BASE_URL (name used by the Langfuse UI) or LANGFUSE_HOST (older name); either works."""
    return os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or None


def client() -> Any:
    return _override if _override is not None else _default_client()


def enabled() -> bool:
    return client() is not None


class _Noop:
    def update(self, **_: Any) -> _Noop:
        return self

    def score_trace(self, **_: Any) -> None:
        return None


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Nested Langfuse observation (span / generation / tool / retriever / guardrail / agent) or a no-op."""
    c = client()
    if c is None:
        yield _Noop()
        return
    with c.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


def trace_attributes(**kwargs: Any) -> Any:
    """Trace-level name / user / session / tags / metadata for everything opened inside."""
    if client() is None:
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(**{k: v for k, v in kwargs.items() if v is not None})


def flush() -> None:
    c = client()
    if c is not None:
        c.flush()


def trace_url() -> str | None:
    c = client()
    try:
        return c.get_trace_url() if c is not None else None
    except Exception:  # noqa: BLE001
        return None
