"""Per-question run state: evidence ledger, trace, tool cache and usage meter (course P4 `Konteks`/`Meter`)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import duckdb

from common.config import db_path


@dataclass
class Meter:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    tool_calls: int = 0


@dataclass
class Evidence:
    id: str
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class RunContext:
    role: str = "chro"
    con: duckdb.DuckDBPyConnection | None = None
    ledger: dict[str, Evidence] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    meter: Meter = field(default_factory=Meter)
    started: float = field(default_factory=time.time)
    on_event: Callable[[dict[str, Any]], None] | None = None   # observability hook (Langfuse)

    def connection(self) -> duckdb.DuckDBPyConnection:
        if self.con is None:
            self.con = duckdb.connect(str(db_path()), read_only=True)  # agent can never write
        return self.con

    def add_evidence(self, tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
        eid = f"E{len(self.ledger) + 1}"
        self.ledger[eid] = Evidence(eid, tool, arguments, result)
        return eid

    def log(self, kind: str, **data: Any) -> None:
        event = {"kind": kind, "t": round(time.time() - self.started, 3), **data}
        self.trace.append(event)
        if self.on_event:
            self.on_event(event)

    def evidence_payloads(self) -> dict[str, Any]:
        return {k: v.result for k, v in self.ledger.items()}

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None
