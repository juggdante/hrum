"""Build the documentation index: markdown chunks + metric chunks -> parquet + embeddings.

Adapted from course labs P3/P4 (`buat_chunk`, `embed`): heading-aware chunks with the heading path kept in
the text, e5 passage embeddings (normalized). Rebuild with:  python -m agent.rag.index
"""
from __future__ import annotations

import glob
import hashlib
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common.config import ROOT, load_config
from common.semantic import SEMANTIC_PATH


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section: str
    text: str


def rag_config() -> dict:
    return load_config("rag")


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) + 2 > max_chars:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def chunk_markdown(path: Path, max_chars: int) -> list[Chunk]:
    rel = str(path.relative_to(ROOT))
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks, heads, buf = [], [], []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            section = " > ".join(heads) or path.stem
            for i, piece in enumerate(_split_long(body, max_chars)):
                cid = hashlib.md5(f"{rel}|{section}|{i}".encode()).hexdigest()[:10]
                chunks.append(Chunk(cid, rel, section, f"{section}\n{piece}"))
        buf.clear()

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            heads[:] = heads[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks


def metric_chunks() -> list[Chunk]:
    cat = yaml.safe_load(SEMANTIC_PATH.read_text(encoding="utf-8"))
    out = []
    for name, m in cat["metrics"].items():
        model = cat["models"][m["model"]]
        synthetic = m.get("synthetic") or model["is_synthetic"]
        text = (f"Metric definition > {name}\n{name}: {m['description']}\n"
                f"Data: {'SYNTHETIC' if synthetic else 'REAL'} ({model['table']}). "
                f"Can be grouped by: {', '.join(model['dimensions'])}.")
        out.append(Chunk(f"metric_{name}", str(SEMANTIC_PATH.relative_to(ROOT)), f"Metric {name}", text))
    return out


def collect_chunks() -> list[Chunk]:
    cfg = rag_config()
    chunks: list[Chunk] = []
    for pattern in cfg["sources"]:
        for p in sorted(glob.glob(str(ROOT / pattern))):
            path = Path(p)
            if path.suffix == ".md":
                chunks += chunk_markdown(path, int(cfg["chunk_max_chars"]))
            elif path.resolve() == SEMANTIC_PATH.resolve():
                chunks += metric_chunks()
    return chunks


def device() -> str:
    d = rag_config()["embedding"]["device"]
    if d != "auto":
        return d
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(rag_config()["embedding"]["model"], device=device())


def embed(texts: list[str], kind: str) -> np.ndarray:
    cfg = rag_config()["embedding"]
    prefix = cfg["query_prefix"] if kind == "query" else cfg["passage_prefix"]
    vecs = embedder().encode([prefix + t for t in texts], batch_size=32, normalize_embeddings=True,
                             show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def index_dir() -> Path:
    return ROOT / rag_config()["index_dir"]


def build() -> pd.DataFrame:
    chunks = collect_chunks()
    df = pd.DataFrame([asdict(c) for c in chunks])
    out = index_dir()
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "chunks.parquet", index=False)
    np.save(out / "embeddings.npy", embed(df["text"].tolist(), "passage"))
    (out / "model.txt").write_text(rag_config()["embedding"]["model"])
    return df


def load() -> tuple[pd.DataFrame, np.ndarray]:
    d = index_dir()
    if not (d / "chunks.parquet").exists():
        build()
    if (d / "model.txt").read_text().strip() != rag_config()["embedding"]["model"]:
        build()  # embedding model changed: vectors are not comparable
    return pd.read_parquet(d / "chunks.parquet"), np.load(d / "embeddings.npy")


if __name__ == "__main__":
    frame = build()
    print(f"indexed {len(frame)} chunks from {frame['source'].nunique()} sources into {index_dir()}")
