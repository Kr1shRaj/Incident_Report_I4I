"""Step 2 - find the schema.

2a  Ask the model, one sample report at a time, which fields the report holds.
    (One report per call fits in the context window; six at once usually do not.)
2b  Merge the proposals in code: clean the names, merge near-duplicates, count in
    how many reports each field appears and keep those above ``min_support``.

Outputs
    output/schema_runs/<doc_id>.json   raw proposals for each sample report
    output/schema_candidates.csv       every candidate field with its support
    output/schema.draft.json           the kept fields, ready for you to review

Nothing is used until you review the draft and save it as ``schema.json``.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Literal, Optional

from pydantic import BaseModel, ValidationError

from .config import Config
from .llm import LLM, parse_json, prompt_budget
from .parse import load_index, load_text
from .template import TYPES, normalize_type, safe_field_name, validate_schema
from .utils import canonical_key, estimate_tokens, log, read_json, split_into_chunks, write_json

FLAT_TYPES = tuple(t for t in TYPES if t != "object_list")
MAX_OPTIONS = 15
FUZZY_MERGE = 0.88


class ProposedField(BaseModel):
    name: str
    type: Literal[FLAT_TYPES]  # type: ignore[valid-type]
    description: str
    example: Optional[str] = None
    options: Optional[List[str]] = None


class Proposal(BaseModel):
    fields: List[ProposedField]


SYSTEM = (
    "You design data schemas for collections of documents. "
    "You answer only with JSON that matches the requested format."
)

PROMPT = """Below is one report from a collection of {topic}.

List the pieces of information this report contains that would most likely also be
recorded in OTHER reports of the same kind (for example: dates, places, people,
equipment, causes, outcomes, actions taken). Leave out details that only make sense
for this one report.

For each field give:
- name: a short snake_case name
- type: one of string, integer, number, boolean, date, list_of_strings, category
- description: one sentence saying what the field holds, written for any report
- example: the value in this report, as text
- options: only for category fields, the small set of values such a field can take

Use category for values that come from a small fixed set (such as severity levels).
Use list_of_strings when a report can hold several values (such as several witnesses).
Return between 5 and 25 fields.

REPORT:
<<<
{text}
>>>"""


def choose_sample(index: list[dict], cfg: Config, names: list[str] | None = None) -> list[dict]:
    if names:
        wanted = set(names)
        chosen = [e for e in index if e["source_file"] in wanted or e["doc_id"] in wanted]
        missing = wanted - {e["source_file"] for e in chosen} - {e["doc_id"] for e in chosen}
        if missing:
            raise RuntimeError(f"Sample documents not found: {', '.join(sorted(missing))}")
        return chosen
    usable = [e for e in index if e["chars"] > 0]
    if len(usable) <= cfg.sample_size:
        return usable
    return random.Random(cfg.sample_seed).sample(usable, cfg.sample_size)


def propose_fields(llm: LLM, text: str, topic: str, max_retries: int = 2) -> list[dict]:
    """Ask the model for the fields of one report (split into chunks if long)."""
    overhead = estimate_tokens(SYSTEM + PROMPT) + 50
    chunks = split_into_chunks(text, prompt_budget(llm.num_ctx) - overhead)
    if len(chunks) > 1:
        log.info("  report is long; proposing fields for %d chunks", len(chunks))
    fields: list[dict] = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(topic=topic, text=chunk)},
        ]
        for attempt in range(max_retries + 1):
            raw = llm.chat_json(messages, Proposal.model_json_schema())
            try:
                proposal = Proposal.model_validate(parse_json(raw))
                break
            except (ValidationError, ValueError) as exc:
                log.warning("  invalid proposal (attempt %d): %s", attempt + 1, str(exc)[:200])
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That answer was invalid: {str(exc)[:500]}\nTry again."},
                ]
        else:
            log.error("  giving up on this chunk")
            continue
        fields.extend(f.model_dump() for f in proposal.fields)
    return fields


def merge_proposals(per_doc: dict[str, list[dict]], min_support: float) -> tuple[list[dict], list[dict]]:
    """Merge field proposals from several reports.

    Returns (kept_fields, all_candidates). Support = share of reports that proposed the field.
    """
    n_docs = len(per_doc)
    groups: dict[str, dict] = {}  # canonical key -> group

    def find_group(key: str) -> str:
        if key in groups:
            return key
        for other in groups:
            if SequenceMatcher(None, key, other).ratio() >= FUZZY_MERGE:
                return other
        groups[key] = {"names": Counter(), "types": Counter(), "docs": set(), "descriptions": {}, "options": Counter()}
        return key

    for did, fields in per_doc.items():
        for f in fields:
            name = safe_field_name(f.get("name", ""))
            g = groups[find_group(canonical_key(name))]
            g["names"][name] += 1
            g["types"][normalize_type(f.get("type"))] += 1
            g["docs"].add(did)
            g["descriptions"].setdefault(name, f.get("description", ""))
            for opt in f.get("options") or []:
                opt = str(opt).strip().lower()
                if opt:
                    g["options"][opt] += 1

    candidates: list[dict] = []
    for key, g in groups.items():
        name = sorted(g["names"].items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
        ftype = g["types"].most_common(1)[0][0]
        options = [o for o, _ in g["options"].most_common(MAX_OPTIONS)]
        if ftype == "category" and not options:
            ftype = "string"
        support = len(g["docs"]) / n_docs if n_docs else 0.0
        cand = {
            "name": name,
            "type": ftype,
            "description": g["descriptions"].get(name, ""),
            "support": round(support, 3),
            "docs": len(g["docs"]),
            "aliases": sorted(set(g["names"]) - {name}),
            "type_votes": dict(g["types"]),
            "kept": support >= min_support,
        }
        if ftype == "category":
            cand["options"] = options
        candidates.append(cand)

    candidates.sort(key=lambda c: (-c["support"], c["name"]))
    # two groups can still end up with the same display name; keep the stronger one
    seen: set[str] = set()
    for c in candidates:
        if c["name"] in seen:
            c["kept"] = False
        seen.add(c["name"])

    kept = []
    for c in candidates:
        if c["kept"]:
            field = {k: c[k] for k in ("name", "type", "description")}
            if "options" in c:
                field["options"] = c["options"]
            kept.append(field)
    return kept, candidates


def write_candidates_csv(path, candidates: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["kept", "name", "type", "support", "docs", "aliases", "options", "type_votes", "description"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in candidates:
            w.writerow(
                {
                    **{k: c.get(k, "") for k in cols},
                    "aliases": "; ".join(c["aliases"]),
                    "options": "; ".join(c.get("options", [])),
                    "type_votes": "; ".join(f"{k}={v}" for k, v in c["type_votes"].items()),
                }
            )


def run(cfg: Config, llm: LLM, sample: list[str] | None = None, force: bool = False) -> dict:
    index = load_index(cfg)
    chosen = choose_sample(index, cfg, sample)
    if not chosen:
        raise RuntimeError("No documents with text to sample from.")
    log.info("Schema discovery on %d sample reports: %s", len(chosen), ", ".join(e["source_file"] for e in chosen))

    per_doc: dict[str, list[dict]] = {}
    for entry in chosen:
        run_file = cfg.schema_runs_dir / f"{entry['doc_id']}.json"
        if run_file.exists() and not force:
            data = read_json(run_file)
            if data.get("model") == llm.model:
                log.info("Reusing proposal for %s", entry["source_file"])
                per_doc[entry["doc_id"]] = data["fields"]
                continue
        log.info("Proposing fields for %s", entry["source_file"])
        fields = propose_fields(llm, load_text(cfg, entry), cfg.topic, cfg.max_retries)
        write_json(run_file, {"source_file": entry["source_file"], "model": llm.model, "fields": fields})
        per_doc[entry["doc_id"]] = fields

    kept, candidates = merge_proposals(per_doc, cfg.min_support)
    write_candidates_csv(cfg.schema_candidates_file, candidates)
    if not kept:
        raise RuntimeError(
            f"No field reached min_support={cfg.min_support}. See {cfg.schema_candidates_file}."
        )
    draft = {
        "topic": cfg.topic,
        "generated": {
            "model": llm.model,
            "sample": [e["source_file"] for e in chosen],
            "min_support": cfg.min_support,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "fields": kept,
    }
    validate_schema(draft)  # names were cleaned, so this should always pass
    write_json(cfg.schema_draft_file, draft)
    log.info("Kept %d of %d candidate fields -> %s", len(kept), len(candidates), cfg.schema_draft_file)
    return draft
