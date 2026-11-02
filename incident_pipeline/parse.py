"""Step 1 - read and clean the documents.

Input:  every supported file in ``reports_dir`` (searched recursively).
Output: for each document
          output/parsed/<doc_id>.md    clean text, headings as '#', tables as Markdown
          output/parsed/<doc_id>.json  the labelled pieces (type, text, page, depth)
        plus output/parsed/index.json listing all parsed documents.

Backends
  unstructured  (default) the open-source ``unstructured`` library, run locally
  docling       IBM's Docling (optional; often better with PDF tables)
  plain         no dependency; only .txt and .md
"""

from __future__ import annotations

import os
from html.parser import HTMLParser
from pathlib import Path

from .config import Config
from .utils import doc_id, log, write_json

SUPPORTED = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm"}
PLAIN_TEXT = {".txt", ".md", ".markdown"}


def find_documents(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Reports folder not found: {root}")
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED]
    skipped = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() not in SUPPORTED]
    for p in skipped:
        if p.suffix.lower() == ".gdoc":
            log.warning("Skipping %s: export Google Docs as .docx or .md first", p.name)
        elif not p.name.startswith("."):
            log.warning("Skipping unsupported file %s", p.name)
    return files


# ---------------------------------------------------------------------------
# HTML table -> Markdown table (stdlib only)
# ---------------------------------------------------------------------------
class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(html: str) -> str | None:
    parser = _TableParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    esc = lambda c: c.replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _parse_plain(path: Path) -> tuple[str, list[dict]]:
    if path.suffix.lower() not in PLAIN_TEXT:
        raise RuntimeError(
            f"The 'plain' parser only reads .txt/.md files, not {path.name}. "
            "Install unstructured: pip install \"unstructured[pdf,docx,md]\""
        )
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text, [{"type": "Text", "text": text}]


def _parse_unstructured(path: Path, cfg: Config) -> tuple[str, list[dict]]:
    os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")
    try:
        from unstructured.partition.auto import partition
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            'unstructured is not installed: pip install "unstructured[pdf,docx,md]"'
        ) from exc

    kwargs: dict = {}
    if path.suffix.lower() == ".pdf":
        kwargs = {"strategy": cfg.pdf_strategy, "infer_table_structure": True}
    elements = partition(filename=str(path), **kwargs)

    pieces: list[dict] = []
    lines: list[str] = []
    for el in elements:
        text = (el.text or "").strip()
        md = el.metadata
        depth = md.category_depth
        html = getattr(md, "text_as_html", None)
        pieces.append(
            {
                "type": el.category,
                "text": text,
                "page": md.page_number,
                "depth": depth,
                "html": html,
            }
        )
        if not text and not html:
            continue
        if el.category == "Title":
            lines.append("#" * min((depth or 0) + 1, 6) + " " + text)
        elif el.category == "ListItem":
            lines.append("  " * max((depth or 0) - 1, 0) + "- " + text)
        elif el.category == "Table":
            table = html_table_to_markdown(html) if html else None
            if table is None:
                log.warning(
                    "%s: a table was flattened to plain text. For PDFs use pdf_strategy='hi_res'.",
                    path.name,
                )
                table = text
            lines.append(table)
        elif el.category in ("Header", "Footer", "PageNumber"):
            continue  # repeated page furniture adds noise
        else:
            lines.append(text)
    return "\n\n".join(lines).strip(), pieces


def _parse_docling(path: Path) -> tuple[str, list[dict]]:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError("docling is not installed: pip install docling") from exc
    result = DocumentConverter().convert(str(path))
    text = result.document.export_to_markdown().strip()
    return text, [{"type": "Markdown", "text": text}]


def parse_document(path: Path, cfg: Config) -> tuple[str, list[dict]]:
    # .txt is already clean text. Partitioning it joins short "Key: value" lines into one
    # run-on paragraph, which hides the structure, so it is read as-is by every backend.
    if cfg.parser == "plain" or path.suffix.lower() == ".txt":
        return _parse_plain(path)
    if cfg.parser == "docling":
        return _parse_docling(path)
    if cfg.parser == "unstructured":
        return _parse_unstructured(path, cfg)
    raise ValueError(f"Unknown parser {cfg.parser!r}")


def run(cfg: Config, force: bool = False) -> list[dict]:
    """Parse every report. Returns the index: [{doc_id, source_file, md_file, chars}]."""
    docs = find_documents(cfg.reports_dir)
    if not docs:
        raise RuntimeError(f"No supported documents in {cfg.reports_dir}")
    cfg.parsed_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for path in docs:
        did = doc_id(path, cfg.reports_dir)
        md_file = cfg.parsed_dir / f"{did}.md"
        json_file = cfg.parsed_dir / f"{did}.json"
        if force or not md_file.exists() or md_file.stat().st_mtime < path.stat().st_mtime:
            log.info("Parsing %s", path)
            try:
                text, pieces = parse_document(path, cfg)
            except Exception as exc:
                log.error("Could not parse %s: %s", path, exc)
                continue
            if not text:
                log.warning("%s produced no text (scanned PDF without OCR?)", path.name)
            md_file.write_text(text + "\n", encoding="utf-8")
            write_json(json_file, {"source_file": str(path), "pieces": pieces})
        else:
            log.info("Up to date: %s", path)
        text = md_file.read_text(encoding="utf-8")
        index.append(
            {
                "doc_id": did,
                "source_file": path.relative_to(cfg.reports_dir).as_posix(),
                "md_file": md_file.name,
                "chars": len(text),
            }
        )
    write_json(cfg.parsed_dir / "index.json", index)
    log.info("Parsed %d documents into %s", len(index), cfg.parsed_dir)
    return index


def load_index(cfg: Config) -> list[dict]:
    from .utils import read_json

    index_file = cfg.parsed_dir / "index.json"
    if not index_file.exists():
        raise RuntimeError("No parsed documents yet. Run the 'parse' step first.")
    return read_json(index_file)


def load_text(cfg: Config, entry: dict) -> str:
    return (cfg.parsed_dir / entry["md_file"]).read_text(encoding="utf-8")
