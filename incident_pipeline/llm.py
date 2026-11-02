"""Talking to the local model through Ollama.

Every call asks for JSON that matches a JSON Schema (Ollama's ``format`` option)
and is checked again by the caller with Pydantic. The ``LLM`` protocol keeps
the rest of the pipeline independent of Ollama, so tests can use a fake model.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from .config import Config
from .utils import estimate_tokens, log

# Tokens kept free for the model's answer when checking the context budget.
ANSWER_RESERVE = 2048


class ContextOverflowError(RuntimeError):
    pass


class LLM(Protocol):
    num_ctx: int
    model: str

    def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        """Return the raw JSON string produced by the model."""
        ...


def prompt_budget(num_ctx: int) -> int:
    """Tokens available for the prompt once room for the answer is kept."""
    return max(num_ctx - ANSWER_RESERVE, num_ctx // 2)


def check_fits(messages: list[dict[str, str]], num_ctx: int) -> int:
    tokens = sum(estimate_tokens(m["content"]) for m in messages)
    if tokens > prompt_budget(num_ctx):
        raise ContextOverflowError(
            f"Prompt is about {tokens} tokens but num_ctx={num_ctx} leaves room for "
            f"{prompt_budget(num_ctx)}. Ollama would silently cut it. Raise num_ctx or use less text."
        )
    return tokens


class OllamaLLM:
    def __init__(self, cfg: Config):
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError("The ollama package is not installed: pip install ollama") from exc
        self.client = ollama.Client(host=cfg.ollama_host, timeout=cfg.request_timeout)
        self.model = cfg.model
        self.num_ctx = cfg.num_ctx
        self.think = cfg.think
        self.options = {
            "temperature": cfg.temperature,
            "seed": cfg.seed,
            "num_ctx": cfg.num_ctx,
            **cfg.ollama_options,
        }

    def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        estimated = check_fits(messages, self.num_ctx)
        kwargs: dict[str, Any] = {}
        if self.think is not None:
            kwargs["think"] = self.think
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema,
            options=self.options,
            **kwargs,
        )
        used = response.get("prompt_eval_count") or 0
        # Ollama reports how many prompt tokens it actually read. If that is at the
        # limit, the prompt was almost certainly truncated.
        if used >= self.num_ctx - 8:
            log.warning(
                "Prompt filled the whole context window (%d tokens, estimate %d). "
                "Input was probably truncated; raise num_ctx.",
                used,
                estimated,
            )
        return response["message"]["content"]

    def ping(self) -> None:
        """Fail early with a clear message if Ollama or the model is missing."""
        try:
            names = {m.get("model") for m in self.client.list().get("models", [])}
        except Exception as exc:
            raise RuntimeError(
                "Cannot reach Ollama. Is it installed and running ('ollama serve')?"
            ) from exc
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if names and wanted not in names:
            raise RuntimeError(
                f"Model {self.model!r} is not pulled. Run: ollama pull {self.model}"
            )


def parse_json(raw: str) -> Any:
    """Parse model output, tolerating stray text around the JSON object."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise
