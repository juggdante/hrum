"""Hybrid retrieval: BM25 + dense e5, fused with Reciprocal Rank Fusion, optional cross-encoder rerank.

Adapted from course lab P4 (`cari_leksikal`, `cari_vektor`, `rrf`, `rerank`), plus the P10 document filter
(`penyaring_dokumen`): lines that look like instructions to the assistant are removed from retrieved text.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

from agent.rag.index import device, embed, load, rag_config

_TOKEN = re.compile(r"[a-z0-9_]+")
_INJECTION = re.compile(
    r"(ignore|disregard|abaikan)\s+(all\s+|the\s+|semua\s+)?(previous|prior|above|sebelumnya)|"
    r"(system prompt|you are now|kamu sekarang|new instructions|instruksi baru)|"
    r"(reveal|show|tampilkan)\s+.*(salary|gaji)\s+of\s+",
    re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
            out[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return out


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda x: -x[1])


def sanitize(text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for line in text.splitlines():
        if _INJECTION.search(line):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(rag_config()["reranker"]["model"], device=device())


def _rerank_enabled() -> bool:
    return str(rag_config()["reranker"]["enabled"]).lower() in ("1", "true", "yes")


class Retriever:
    def __init__(self) -> None:
        self.chunks, self.vectors = load()
        self.bm25 = BM25(self.chunks["text"].tolist())

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        cfg = rag_config()["retrieval"]
        k = k or int(cfg["top_k"])
        n = int(cfg["candidates_per_retriever"])
        # BM25 only votes for chunks that share a distinctive term with the query. Without this, a query with
        # no real overlap (Indonesian question, English docs) ranks chunks by noise words like "data", and that
        # noise outvotes the dense retriever in RRF.
        bm = self.bm25.scores(query)
        min_score = float(cfg.get("min_lexical_score", 0))
        lexical = [int(i) for i in np.argsort(-bm)[:n] if bm[i] >= min_score and bm[i] > 0]
        dense_scores = self.vectors @ embed([query], "query")[0]
        dense = [int(i) for i in np.argsort(-dense_scores)[:n]]
        fused = rrf([lst for lst in (lexical, dense) if lst], int(cfg["rrf_k"]))
        candidates = [i for i, _ in fused[: max(n, k)]]
        if _rerank_enabled() and candidates:
            pairs = [(query, self.chunks.iloc[i]["text"]) for i in candidates]
            scores = _reranker().predict(pairs)
            candidates = [candidates[j] for j in np.argsort(-np.asarray(scores))]
        out = []
        for rank, i in enumerate(candidates[:k], 1):
            row = self.chunks.iloc[i]
            text, removed = sanitize(row["text"])
            out.append({"rank": rank, "chunk_id": row["chunk_id"], "source": row["source"],
                        "section": row["section"], "text": text, "lines_removed": removed})
        return out


@lru_cache
def retriever() -> Retriever:
    return Retriever()


def warm_up() -> bool:
    """Load the index and the embedding model now, so the first docs question does not pay ~10 s of loading.

    Returns False (and does nothing else) when the index has not been built yet.
    """
    try:
        retriever().search("warm-up", 1)
        return True
    except FileNotFoundError:
        return False
