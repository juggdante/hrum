"""Numeric grounding: every non-trivial number in an answer must be supported by cited evidence.

Adapted from course lab P4 (`angka_dari`, `cocok`, `validasi`): the model must cite evidence ids from
this run's ledger, and each number it writes must match (within display rounding) a number inside
the cited evidence. Pure functions, no LLM calls.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

_SCALE_WORDS = {
    "k": 1e3, "rb": 1e3, "ribu": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "mio": 1e6, "jt": 1e6, "juta": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9, "miliar": 1e9, "milyar": 1e9, "mılyar": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12, "triliun": 1e12,
}
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>-)?(?P<cur>\$|usd\s?|rp\.?\s?)?(?P<num>\d+(?:[.,]\d+)*)"
    r"(?:\s?(?P<scale>[a-zı]+)\b)?(?P<pct>\s?%)?",
    re.IGNORECASE,
)
YEAR_RANGE = (1990, 2035)
RELATIVE_TOL = 0.005


@dataclass
class Number:
    raw: str
    value: float
    step: float          # display precision in absolute units (0.5 * step = rounding tolerance)
    percent: bool
    trivial: bool
    alternatives: tuple[float, ...] = ()   # other readings of an ambiguous number ("42.930": 42.93 or 42930)


def _parse_digits(s: str) -> tuple[float, int]:
    """Return (value, decimals). Handles 7,523,952,094 / 7.523.952.094 / 7.52 / 7,52."""
    if "," in s and "." in s:
        dec_sep = "." if s.rfind(".") > s.rfind(",") else ","
    elif s.count(",") == 1 and len(s.split(",")[1]) != 3:
        dec_sep = ","
    elif s.count(".") == 1 and len(s.split(".")[1]) != 3:
        dec_sep = "."
    elif s.count(".") == 1 and s.count(",") == 0:
        dec_sep = "."  # "7.523" ambiguous: read as decimal (English style is requested in prompts)
    else:
        dec_sep = None
    if dec_sep:
        whole, _, frac = s.rpartition(dec_sep)
        whole = re.sub(r"[.,]", "", whole)
        return float(f"{whole or 0}.{frac}"), len(frac)
    return float(re.sub(r"[.,]", "", s)), 0


def _alternative(s: str) -> float | None:
    """'42.930' / '42,930' (one separator, exactly 3 digits after it) can be a decimal or a thousands group.

    English writes 42,930 and Indonesian writes 42.930 for the same integer, so both readings are kept.
    """
    if re.fullmatch(r"\d{1,3}[.,]\d{3}", s):
        whole, frac = re.split(r"[.,]", s)
        primary, _ = _parse_digits(s)
        as_int, as_dec = float(whole + frac), float(f"{whole}.{frac}")
        return as_dec if primary == as_int else as_int
    return None


def extract_numbers(text: str) -> list[Number]:
    out = []
    for m in _NUMBER.finditer(text or ""):
        value, decimals = _parse_digits(m["num"])
        alt = _alternative(m["num"])
        scale_word = (m["scale"] or "").lower()
        scale = _SCALE_WORDS.get(scale_word, 1.0)
        if m["sign"]:
            value = -value
        step = 10 ** (-decimals) * scale
        percent = bool(m["pct"])
        is_year = decimals == 0 and scale == 1 and not percent and not m["cur"] \
            and YEAR_RANGE[0] <= value <= YEAR_RANGE[1] and "," not in m["num"]
        trivial = is_year or (decimals == 0 and scale == 1 and not percent and not m["cur"] and abs(value) <= 10)
        alts = () if alt is None else ((-alt if m["sign"] else alt) * scale,)
        out.append(Number(m.group(0).strip(), value * scale, step, percent, trivial, alts))
    return out


def evidence_numbers(obj: Any) -> list[float]:
    """All numeric values inside an evidence payload (rows, scalars, nested dicts/lists)."""
    nums: list[float] = []
    if isinstance(obj, bool):
        return nums
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and math.isnan(obj)):
            nums.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "text" and isinstance(v, str):  # document passages: numbers written in prose
                nums.extend(n.value for n in extract_numbers(v))
            else:
                nums.extend(evidence_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            nums.extend(evidence_numbers(v))
    return nums


def supported(n: Number, evidence: list[float]) -> bool:
    tol = 0.5 * n.step * 1.001
    candidates = [(n.value, tol)] + [(a, 0.0) for a in n.alternatives]   # alternatives: relative tol only
    if n.percent:  # "14.6%" may be cited from a 0.146 ratio
        candidates += [(c / 100, t / 100) for c, t in list(candidates)]
    for c, tol_abs in candidates:
        for e in evidence:
            for ev in (e, abs(e)):
                if abs(c - ev) <= tol_abs or (ev and abs(c - ev) / abs(ev) <= RELATIVE_TOL):
                    return True
    return False


@dataclass
class Validation:
    ok: bool
    claims_ok: int
    claims_total: int
    problems: list[str] = field(default_factory=list)


def validate(answer: dict[str, Any], ledger: dict[str, Any]) -> Validation:
    """ledger: evidence_id -> evidence payload (anything JSON-like)."""
    problems: list[str] = []
    claims = answer.get("claims") or []
    ok = 0
    cited_numbers: list[float] = []
    for i, claim in enumerate(claims, 1):
        ids = [x for x in claim.get("evidence_ids", []) if x]
        fake = [x for x in ids if x not in ledger]
        if not ids:
            problems.append(f"claim {i} has no evidence_ids")
            continue
        if fake:
            problems.append(f"claim {i} cites evidence that was never retrieved: {fake[:3]}")
            continue
        ev = [v for x in ids for v in evidence_numbers(ledger[x])]
        cited_numbers.extend(ev)
        bad = [n.raw for n in extract_numbers(claim.get("text", "")) if not n.trivial and not supported(n, ev)]
        if bad:
            problems.append(f"claim {i}: number(s) {bad[:3]} not found in cited evidence")
            continue
        ok += 1
    loose = [n.raw for n in extract_numbers(answer.get("answer", ""))
             if not n.trivial and not supported(n, cited_numbers)]
    if loose:
        problems.append(f"answer text contains number(s) {loose[:3]} that no cited evidence supports")
    if answer.get("status") == "answered" and not claims:
        problems.append("status is 'answered' but there are no claims")
    return Validation(ok=not problems, claims_ok=ok, claims_total=len(claims), problems=problems)
