"""Small helpers shared by the pipeline steps."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("incident_pipeline")

# Words dropped when deciding whether two field names mean the same thing,
# e.g. "severity_of_injury" and "injury_severity".
_NAME_STOPWORDS = {"of", "the", "a", "an", "and", "or", "for", "to", "in", "on", "by", "at", "with"}


def snake_case(name: str) -> str:
    """Turn any label into a clean snake_case identifier ("Incident ID" -> "incident_id")."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    if not name:
        return "field"
    if name[0].isdigit():
        name = "f_" + name
    return name


def canonical_key(name: str) -> str:
    """Key used to merge near-duplicate field names: sorted tokens without stopwords."""
    tokens = [t for t in snake_case(name).split("_") if t and t not in _NAME_STOPWORDS]
    # crude singularisation so "recommendation" and "recommendations" merge
    tokens = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens]
    return "_".join(sorted(tokens)) or snake_case(name)


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English)."""
    return len(text) // 4 + 1


def doc_id(path: Path, root: Path) -> str:
    """Stable, filesystem-safe id for a source document."""
    rel = path.relative_to(root).as_posix()
    return re.sub(r"[^0-9A-Za-z._-]+", "_", rel)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_into_chunks(text: str, max_tokens: int) -> list[str]:
    """Split text on paragraph boundaries into pieces of at most ``max_tokens`` (estimated)."""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    max_chars = max_tokens * 4
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for para in re.split(r"\n\s*\n", text):
        # a single paragraph larger than the budget is cut hard
        while len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        if size + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks
