"""OpenAI-compatible client: vLLM (Qwen) locally, or OpenAI itself.

Adapted from the course labs (P4 `chat`, P10/P11 client setup): temperature 0, Qwen thinking off via
`chat_template_kwargs`, tools with tool_choice=auto, structured output via `response_format`.
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from agent.llm.base import LLMClient, LLMResponse, ToolCall, Usage


class OpenAICompatibleClient(LLMClient):
    def __init__(self, name: str, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, enable_thinking: bool | None = None, timeout: float = 180,
                 **_: Any) -> None:
        self.name, self.model, self.temperature = name, model, temperature
        self.enable_thinking = enable_thinking
        self._client = OpenAI(base_url=base_url or None,
                              api_key=api_key or os.environ.get("OPENAI_API_KEY") or "none",
                              timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_schema: dict | None = None, max_tokens: int = 800) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages,
                                  "temperature": self.temperature, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"], kwargs["tool_choice"] = tools, "auto"
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "answer", "schema": json_schema}}
        if self.enable_thinking is not None:  # vLLM / Qwen chat-template switch
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        r = self._client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        calls = [ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
                 for c in (msg.tool_calls or [])]
        u = r.usage
        return LLMResponse(text=msg.content or "", tool_calls=calls,
                           usage=Usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0), raw=r)
