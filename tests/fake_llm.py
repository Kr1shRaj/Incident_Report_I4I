"""A stand-in for the local model, used by the tests.

It answers from a queue of canned replies, or from a function of the prompt,
and records every request so tests can check what was sent.
"""

from __future__ import annotations

import json
from typing import Any, Callable


class FakeLLM:
    def __init__(self, replies: list[Any] | Callable[[list[dict], dict], Any] | None = None,
                 num_ctx: int = 16384, model: str = "fake-model"):
        self.replies = replies if replies is not None else []
        self.num_ctx = num_ctx
        self.model = model
        self.calls: list[dict] = []

    def chat_json(self, messages, schema):
        self.calls.append({"messages": messages, "schema": schema})
        if callable(self.replies):
            reply = self.replies(messages, schema)
        else:
            reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)
