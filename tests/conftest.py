from __future__ import annotations

import json
from typing import Any

import pytest

from agent.llm.base import LLMClient, LLMResponse, ToolCall
from common.config import db_path


class ScriptedLLM(LLMClient):
    """Fake LLM that replays scripted responses; each item is a ToolCall list or a text/JSON string."""

    name = model = "scripted"

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, json_schema=None, max_tokens=800) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "json_schema": json_schema})
        item = self.script.pop(0)
        if isinstance(item, list):
            return LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a))
                                           for i, (n, a) in enumerate(item)])
        return LLMResponse(text=item if isinstance(item, str) else json.dumps(item))


@pytest.fixture
def scripted():
    return ScriptedLLM


requires_db = pytest.mark.skipif(not db_path().exists(), reason="run `make build-data` first")
