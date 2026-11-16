"""Single LLM interface. All providers (Qwen on vLLM, OpenAI, Anthropic) implement LLMClient.

Agent code imports ONLY from this module (CLAUDE.md rule 4): `get_client()` picks the provider
from configs/llm.yaml.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from common.config import load_config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model; validated by the tool runner


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


class LLMClient(ABC):
    name: str = "llm"
    model: str = ""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_schema: dict | None = None, max_tokens: int = 800) -> LLMResponse:
        """Send OpenAI-style messages, optionally with tool schemas or a JSON schema for the output."""
        raise NotImplementedError


def get_client(provider: str | None = None) -> LLMClient:
    """Return the client for `provider` (default: configs/llm.yaml `default`)."""
    cfg = load_config("llm")
    name = provider or cfg["default"]
    if name not in cfg["providers"]:
        raise ValueError(f"Unknown LLM provider '{name}'. Options: {sorted(cfg['providers'])}")
    p = cfg["providers"][name]
    if p["kind"] in ("openai_compatible", "openai"):
        from agent.llm.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(name=name, **{k: v for k, v in p.items() if k != "kind"})
    raise NotImplementedError(f"Provider kind '{p['kind']}' is not implemented yet")
