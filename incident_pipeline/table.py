"""Step 5 - build the comparison table.

Input:  output/extractions.jsonl and schema.json
Output: output/comparison.csv   one row per report, one column per field
        output/fill_rate.csv    how often each field has a value

A field that is almost always empty is either a bad schema field or a sign
that input was lost (for example, a context window that was too small).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import Config
from .extract import load_schema, schema_hash
from .template import is_empty
from .utils import log, read_jsonl

LIST_SEP = "; "


def cell(value: Any, ftype: str) -> Any:
    if is_empty(value):
        return ""
    if ftype == "list_of_strings":
        return LIST_SEP.join(str(v) for v in value)
    if ftype == "object_list":
        return json.dumps(value, ensure_ascii=False)
    return value


def build_table(records: list[dict], schema: dict) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"source_file": r["source_file"]}
        for f in schema["fields"]:
            row[f["name"]] = cell(r["data"].get(f["name"]), f["type"])
        row["_valid"] = r["meta"].get("valid", True)
        row["_errors"] = " | ".join(r["meta"].get("errors", []))
        rows.append(row)
    columns = ["source_file"] + [f["name"] for f in schema["fields"]] + ["_valid", "_errors"]
    return pd.DataFrame(rows, columns=columns)


def fill_rates(records: list[dict], schema: dict) -> pd.DataFrame:
    total = len(records)
    rows = []
    for f in schema["fields"]:
        filled = sum(1 for r in records if not is_empty(r["data"].get(f["name"])))
        rows.append(
            {
                "field": f["name"],
                "type": f["type"],
                "filled": filled,
                "total": total,
                "fill_rate": round(filled / total, 3) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("fill_rate", kind="stable")


def run(cfg: Config, accept_draft: bool = False) -> pd.DataFrame:
    schema = load_schema(cfg, accept_draft)
    if not cfg.extractions_file.exists():
        raise RuntimeError("No extractions yet. Run the 'extract' step first.")
    shash = schema_hash(schema)
    records = [r for r in read_jsonl(cfg.extractions_file)]
    stale = [r for r in records if r["meta"].get("schema_hash") != shash]
    if stale:
        log.warning("Skipping %d records extracted with an older schema; re-run 'extract'.", len(stale))
        records = [r for r in records if r not in stale]
    if not records:
        raise RuntimeError("No records match the current schema. Re-run 'extract'.")

    table = build_table(records, schema)
    cfg.comparison_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.comparison_file, index=False)
    rates = fill_rates(records, schema)
    rates.to_csv(cfg.fill_rate_file, index=False)

    log.info("Wrote %d rows x %d fields -> %s", len(table), len(schema["fields"]), cfg.comparison_file)
    low = rates[rates["fill_rate"] < 0.2]
    if len(low):
        log.warning("Fields filled in fewer than 20%% of reports: %s", ", ".join(low["field"]))
    return table
