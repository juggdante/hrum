"""Rewrite unambiguous Indonesian-style numbers in an answer to English style (9,9% -> 9.9%).

The prompt asks for English number style, but the models still write a decimal comma in about a quarter of
Indonesian answers. Only rewrites that cannot change the value are applied:
  - "1.234.567" / "1.234,5"      (two or more dot groups, or dot groups plus a comma decimal) -> 1,234,567 / 1,234.5
  - "14,56" / "9,9"              (comma followed by 1-2 digits)                               -> 14.56 / 9.9
  - "0,1456"                     (leading zero, comma, any digits)                             -> 0.1456
"42.930" and "42,930" stay as written: they are ambiguous, and grounding accepts both readings.
"""
from __future__ import annotations

import re
from typing import Any

_ID_GROUPED = re.compile(r"(?<![\d.,])(\d{1,3})((?:\.\d{3})+)(?:,(\d+))?(?![\d.,]*\d)")
_DEC_COMMA = re.compile(r"(?<![\d.,])(\d+),(\d{1,2})(?![\d,])")
_ZERO_COMMA = re.compile(r"(?<![\d.,])0,(\d+)(?![\d,])")


def _grouped(m: re.Match[str]) -> str:
    head, groups, dec = m.group(1), m.group(2), m.group(3)
    if groups.count(".") < 2 and dec is None:
        return m.group(0)                      # "42.930": ambiguous, leave it
    out = head + groups.replace(".", ",")
    return f"{out}.{dec}" if dec is not None else out


def normalize_numbers(text: str) -> str:
    text = _ID_GROUPED.sub(_grouped, text)
    text = _ZERO_COMMA.sub(lambda m: f"0.{m.group(1)}", text)
    return _DEC_COMMA.sub(lambda m: f"{m.group(1)}.{m.group(2)}", text)


def normalize_answer(answer: dict[str, Any]) -> dict[str, Any]:
    if isinstance(answer.get("answer"), str):
        answer["answer"] = normalize_numbers(answer["answer"])
    for c in answer.get("claims", []):
        if isinstance(c.get("text"), str):
            c["text"] = normalize_numbers(c["text"])
    return answer
