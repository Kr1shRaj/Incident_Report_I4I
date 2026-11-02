"""Step 4 - extract every report into the template.

Each report goes to the model with the field list (names, types, descriptions)
in the prompt and the template's JSON Schema as Ollama's ``format``. The answer
is then validated with Pydantic. Invalid answers are retried with the error
message; if they stay invalid, the valid fields are kept and the rest set to null.

Reports that do not fit in the context window are split into chunks; the chunk
answers are merged (first value wins for single fields, union for lists).

Outputs
    output/extracted/<doc_id>.json   one record per report (cached)
    output/extractions.jsonl         all records, the input for the table and graph
    output/template.schema.json      the JSON Schema sent to the model
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ValidationError

from .config import Config
from .llm import LLM, parse_json, prompt_budget
from .parse import load_index, load_text
from .template import build_model, describe_fields, merge_records, validate_schema
from .utils import estimate_tokens, log, read_json, split_into_chunks, write_json, write_jsonl

SYSTEM = (
    "You extract facts from {topic} into a fixed set of fields. "
    "You answer only with JSON that matches the requested format."
)

PROMPT = """Fill in the fields below using ONLY information stated in the report.

Rules:
- If the report does not state a value, use null. Never guess or invent a value.
- Copy names, numbers and identifiers exactly as written.
- Dates: use YYYY-MM-DD when the full date is given; otherwise copy what is written.
- For fields with a fixed list of options, pick the closest option, or "other" if none fits.
- For list fields, include every value mentioned, or null if there are none.
{part_note}
FIELDS:
{fields}

REPORT:
<<<
{text}
>>>"""


def load_schema(cfg: Config, accept_draft: bool = False) -> dict:
    path = cfg.schema_file
    if not path.exists():
        if accept_draft and cfg.schema_draft_file.exists():
            log.warning("No %s; using the unreviewed draft %s", path, cfg.schema_draft_file)
            path = cfg.schema_draft_file
        else:
            raise RuntimeError(
                f"{cfg.schema_file} not found. Run 'discover', review {cfg.schema_draft_file}, "
                f"then save it as {cfg.schema_file} (or pass --accept-draft)."
            )
    return validate_schema(read_json(path))


def schema_hash(schema: dict) -> str:
    payload = json.dumps(schema["fields"], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def inline_refs(schema: dict) -> dict:
    """Replace "$ref" pointers with their definitions so the grammar builder sees one flat schema."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def validate_answer(model: type[BaseModel], data: Any) -> tuple[dict, list[str]]:
    """Validate a whole answer; on failure keep the fields that are valid on their own."""
    if not isinstance(data, dict):
        return {name: None for name in model.model_fields}, [f"answer is not an object: {type(data).__name__}"]
    try:
        return model.model_validate(data).model_dump(), []
    except ValidationError:
        pass
    result: dict[str, Any] = {}
    errors: list[str] = []
    for name in model.model_fields:
        try:
            result[name] = model.model_validate({name: data.get(name)}).model_dump()[name]
        except ValidationError as exc:
            result[name] = None
            errors.append(f"{name}: {exc.errors()[0]['msg']}")
    return result, errors


def extract_text(
    llm: LLM, model: type[BaseModel], schema: dict, text: str, topic: str, max_retries: int = 2
) -> tuple[dict, dict]:
    """Extract one report. Returns (data, meta)."""
    json_schema = inline_refs(model.model_json_schema())
    fields_text = describe_fields(schema)
    system = SYSTEM.format(topic=topic)
    overhead = estimate_tokens(system + PROMPT + fields_text) + 100
    chunks = split_into_chunks(text, prompt_budget(llm.num_ctx) - overhead)

    results: list[dict] = []
    errors: list[str] = []
    attempts = 0
    for i, chunk in enumerate(chunks, start=1):
        part_note = (
            f"\nThis is part {i} of {len(chunks)} of the report. Fill in only what this part states.\n"
            if len(chunks) > 1
            else ""
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": PROMPT.format(part_note=part_note, fields=fields_text, text=chunk)},
        ]
        chunk_errors: list[str] = []
        data: dict = {}
        for attempt in range(max_retries + 1):
            attempts += 1
            raw = llm.chat_json(messages, json_schema)
            try:
                parsed = parse_json(raw)
            except ValueError as exc:
                chunk_errors = [f"not JSON: {exc}"]
                parsed = None
            if parsed is not None:
                data, chunk_errors = validate_answer(model, parsed)
                if not chunk_errors:
                    break
            log.warning("  invalid answer (attempt %d): %s", attempt + 1, "; ".join(chunk_errors)[:200])
            messages = messages[:2] + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "Your answer had these problems:\n- "
                    + "\n- ".join(chunk_errors)
                    + "\nAnswer again with corrected JSON.",
                },
            ]
        results.append(data or {name: None for name in model.model_fields})
        errors.extend(f"chunk {i}: {e}" if len(chunks) > 1 else e for e in chunk_errors)

    merged = results[0] if len(results) == 1 else merge_records(results, schema)
    meta = {"chunks": len(chunks), "attempts": attempts, "valid": not errors, "errors": errors}
    return merged, meta


def run(
    cfg: Config,
    llm: LLM,
    accept_draft: bool = False,
    only: list[str] | None = None,
    force: bool = False,
) -> list[dict]:
    schema = load_schema(cfg, accept_draft)
    model = build_model(schema)
    write_json(cfg.template_schema_file, inline_refs(model.model_json_schema()))
    shash = schema_hash(schema)
    index = load_index(cfg)
    if only:
        wanted = set(only)
        index = [e for e in index if e["source_file"] in wanted or e["doc_id"] in wanted]
        if not index:
            raise RuntimeError("None of the requested documents have been parsed.")

    records: list[dict] = []
    for n, entry in enumerate(index, start=1):
        out_file = cfg.extracted_dir / f"{entry['doc_id']}.json"
        if out_file.exists() and not force:
            cached = read_json(out_file)
            m = cached.get("meta", {})
            if m.get("schema_hash") == shash and m.get("model") == llm.model:
                log.info("[%d/%d] cached: %s", n, len(index), entry["source_file"])
                records.append(cached)
                continue
        log.info("[%d/%d] extracting %s", n, len(index), entry["source_file"])
        text = load_text(cfg, entry)
        if not text.strip():
            data, meta = {name: None for name in model.model_fields}, {
                "chunks": 0, "attempts": 0, "valid": False, "errors": ["document has no text"],
            }
        else:
            data, meta = extract_text(llm, model, schema, text, cfg.topic, cfg.max_retries)
        record = {
            "source_file": entry["source_file"],
            "data": data,
            "meta": {
                **meta,
                "model": llm.model,
                "num_ctx": llm.num_ctx,
                "schema_hash": shash,
                "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }
        write_json(out_file, record)
        records.append(record)

    # extractions.jsonl holds every report extracted with the current schema and model,
    # including ones done in earlier runs (e.g. with --only).
    current = []
    for entry in load_index(cfg):
        f = cfg.extracted_dir / f"{entry['doc_id']}.json"
        if f.exists():
            rec = read_json(f)
            if rec["meta"].get("schema_hash") == shash and rec["meta"].get("model") == llm.model:
                current.append(rec)
    write_jsonl(cfg.extractions_file, current)
    bad = sum(1 for r in records if not r["meta"]["valid"])
    log.info("Extracted %d reports (%d with validation problems) -> %s", len(records), bad, cfg.extractions_file)
    return records
