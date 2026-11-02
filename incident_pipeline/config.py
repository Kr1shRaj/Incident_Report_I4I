"""Pipeline settings.

Defaults live here. They can be overridden by a TOML file (``--config``) and
then by command-line flags, in that order.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # Folders and files
    reports_dir: Path = Path("reports")
    output_dir: Path = Path("output")
    schema_file: Path = Path("schema.json")
    gold_file: Path = Path("gold.csv")

    # Parsing (step 1)
    parser: str = "unstructured"  # "unstructured", "docling" or "plain"
    pdf_strategy: str = "hi_res"  # "fast" for text PDFs, "hi_res" keeps table structure

    # Model (steps 2 and 4)
    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    # Ollama's default context window is only a few thousand tokens and it
    # silently drops whatever does not fit. Always set it explicitly.
    num_ctx: int = 16384
    temperature: float = 0.0
    seed: int = 42
    think: bool | None = None  # set False for "thinking" models such as qwen3
    max_retries: int = 2
    request_timeout: float = 600.0

    # Schema discovery (step 2)
    topic: str = "incident reports"  # what the batch is about; used in prompts
    sample_size: int = 6
    sample_seed: int = 13
    min_support: float = 0.5  # keep fields found in at least this share of the sample

    # Extra Ollama options passed through unchanged
    ollama_options: dict[str, Any] = field(default_factory=dict)

    # ---- derived paths -------------------------------------------------
    @property
    def parsed_dir(self) -> Path:
        return self.output_dir / "parsed"

    @property
    def schema_runs_dir(self) -> Path:
        return self.output_dir / "schema_runs"

    @property
    def schema_draft_file(self) -> Path:
        return self.output_dir / "schema.draft.json"

    @property
    def schema_candidates_file(self) -> Path:
        return self.output_dir / "schema_candidates.csv"

    @property
    def extracted_dir(self) -> Path:
        return self.output_dir / "extracted"

    @property
    def extractions_file(self) -> Path:
        return self.output_dir / "extractions.jsonl"

    @property
    def template_schema_file(self) -> Path:
        return self.output_dir / "template.schema.json"

    @property
    def comparison_file(self) -> Path:
        return self.output_dir / "comparison.csv"

    @property
    def fill_rate_file(self) -> Path:
        return self.output_dir / "fill_rate.csv"

    @property
    def eval_file(self) -> Path:
        return self.output_dir / "eval.csv"

    @property
    def eval_details_file(self) -> Path:
        return self.output_dir / "eval_details.csv"

    # ---- loading ---------------------------------------------------------
    def update(self, values: dict[str, Any]) -> None:
        known = {f.name: f for f in fields(self)}
        for key, value in values.items():
            if value is None:
                continue
            if key not in known:
                raise ValueError(f"Unknown setting: {key!r}")
            if key.endswith("_dir") or key.endswith("_file"):
                value = Path(value)
            setattr(self, key, value)

    @classmethod
    def load(cls, path: Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        cfg = cls()
        if path is not None:
            with open(path, "rb") as fh:
                cfg.update(tomllib.load(fh))
        if overrides:
            cfg.update(overrides)
        return cfg
