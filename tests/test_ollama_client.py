"""Runs the real ``ollama`` client against a tiny stand-in server that speaks Ollama's HTTP API.

This checks what actually goes over the wire (format schema, num_ctx, temperature)
without needing Ollama or a model installed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("ollama")

from incident_pipeline.config import Config  # noqa: E402
from incident_pipeline.llm import ContextOverflowError, OllamaLLM, check_fits  # noqa: E402


class FakeOllama(BaseHTTPRequestHandler):
    requests: list = []
    reply = {"ok": True}
    prompt_eval_count = 100
    models = ["llama3.1:8b"]

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, body: dict):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"model": m, "name": m} for m in self.models]})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        self._send(
            {
                "model": body["model"],
                "created_at": "2025-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": json.dumps(FakeOllama.reply)},
                "done": True,
                "prompt_eval_count": FakeOllama.prompt_eval_count,
            }
        )


@pytest.fixture
def server():
    FakeOllama.requests = []
    FakeOllama.prompt_eval_count = 100
    FakeOllama.models = ["llama3.1:8b"]
    httpd = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def make(host, **kw):
    return OllamaLLM(Config(ollama_host=host, model="llama3.1:8b", num_ctx=8192, **kw))


def test_request_carries_schema_and_options(server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    llm = make(server)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    raw = llm.chat_json([{"role": "user", "content": "hi"}], schema)
    assert json.loads(raw) == {"ok": True}
    sent = FakeOllama.requests[0]
    assert sent["format"] == schema
    assert sent["options"]["num_ctx"] == 8192
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["seed"] == 42
    assert sent["stream"] is False
    assert "think" not in sent or sent["think"] is None


def test_think_flag_is_passed_when_set(server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    make(server, think=False).chat_json([{"role": "user", "content": "hi"}], {"type": "object"})
    assert FakeOllama.requests[0]["think"] is False


def test_warns_when_context_window_was_filled(server, monkeypatch, caplog):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    FakeOllama.prompt_eval_count = 8192
    make(server).chat_json([{"role": "user", "content": "hi"}], {"type": "object"})
    assert "probably truncated" in caplog.text


def test_ping_reports_missing_model(server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    make(server).ping()  # model present: no error
    FakeOllama.models = ["qwen2.5:7b"]
    with pytest.raises(RuntimeError, match="ollama pull llama3.1:8b"):
        make(server).ping()


def test_ping_reports_unreachable_server():
    llm = OllamaLLM(Config(ollama_host="http://127.0.0.1:9", request_timeout=2))
    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        llm.ping()


def test_prompt_too_big_is_refused_before_sending():
    with pytest.raises(ContextOverflowError, match="num_ctx"):
        check_fits([{"role": "user", "content": "x" * 40000}], num_ctx=4096)
    assert check_fits([{"role": "user", "content": "x" * 4000}], num_ctx=4096) > 0
