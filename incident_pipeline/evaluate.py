"""Accuracy check against reports you filled in by hand.

gold.csv has the same layout as comparison.csv: a ``source_file`` column and one
column per field (you may leave out fields you do not want to score). Leave a
cell empty when the report does not state that value. Separate list items with
";".

Each cell is scored as one of:
    correct       same value after normalising case, spaces and punctuation
    close         nearly the same text (similarity >= 0.85), e.g. small wording changes
    wrong         both have a value and they differ
    missed        the report states it, the model left it empty
    hallucinated  the report does not state it, the model filled it in (the worst case)
    both_empty    correctly left empty

Outputs
    output/eval.csv          one row per field, plus an "ALL" row
    output/eval_details.csv  one row per (report, field) with both values
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from .config import Config
from .extract import load_schema
from .template import is_empty
from .utils import log, read_jsonl

CLOSE = 0.85
OUTCOMES = ["correct", "close", "wrong", "missed", "hallucinated", "both_empty"]


def norm(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:")


def to_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def score_scalar(gold: str, pred: Any) -> str:
    g, p = norm(gold), norm(pred)
    if g == p:
        return "correct"
    gn, pn = to_number(g), to_number(p)
    if gn is not None and pn is not None:
        return "correct" if abs(gn - pn) < 1e-9 else "wrong"
    if g in ("true", "yes") and p in ("true", "yes") or g in ("false", "no") and p in ("false", "no"):
        return "correct"
    return "close" if SequenceMatcher(None, g, p).ratio() >= CLOSE else "wrong"


def score_list(gold: str, pred: list) -> str:
    g = {norm(x) for x in gold.split(";") if norm(x)}
    p = {norm(x) for x in pred if norm(x)}
    if g == p:
        return "correct"
    # every gold item has a close match and nothing extra
    def has_match(item: str, pool: set[str]) -> bool:
        return any(SequenceMatcher(None, item, o).ratio() >= CLOSE for o in pool)

    if all(has_match(x, p) for x in g) and all(has_match(x, g) for x in p):
        return "close"
    return "wrong"


def score_cell(gold: Any, pred: Any, ftype: str) -> str:
    gold_empty = gold is None or (isinstance(gold, float) and pd.isna(gold)) or str(gold).strip() == ""
    pred_empty = is_empty(pred)
    if gold_empty and pred_empty:
        return "both_empty"
    if gold_empty:
        return "hallucinated"
    if pred_empty:
        return "missed"
    if ftype == "list_of_strings":
        return score_list(str(gold), pred if isinstance(pred, list) else [pred])
    return score_scalar(str(gold), pred)


def summarise(details: pd.DataFrame) -> pd.DataFrame:
    def row(name: str, part: pd.DataFrame) -> dict:
        counts = part["outcome"].value_counts()
        n = len(part)
        out = {"field": name, "n": n, **{o: int(counts.get(o, 0)) for o in OUTCOMES}}
        good = out["correct"] + out["both_empty"]
        out["accuracy"] = round(good / n, 3) if n else 0.0
        out["lenient_accuracy"] = round((good + out["close"]) / n, 3) if n else 0.0
        return out

    rows = [row(f, part) for f, part in details.groupby("field", sort=False)]
    rows.append(row("ALL", details))
    return pd.DataFrame(rows)


def run(cfg: Config, accept_draft: bool = False) -> pd.DataFrame:
    schema = load_schema(cfg, accept_draft)
    types = {f["name"]: f["type"] for f in schema["fields"]}
    if not cfg.gold_file.exists():
        raise RuntimeError(f"No gold file at {cfg.gold_file}. Fill in some reports by hand first.")
    gold = pd.read_csv(cfg.gold_file, dtype=str, keep_default_na=False)
    if "source_file" not in gold.columns:
        raise RuntimeError("gold.csv needs a 'source_file' column")
    preds = {r["source_file"]: r["data"] for r in read_jsonl(cfg.extractions_file)}

    scored_fields = [c for c in gold.columns if c in types and types[c] != "object_list"]
    ignored = [c for c in gold.columns if c != "source_file" and c not in scored_fields]
    if ignored:
        log.warning("Not scoring gold columns (unknown field or object_list): %s", ", ".join(ignored))

    details = []
    for _, g in gold.iterrows():
        src = g["source_file"]
        if src not in preds:
            log.warning("No extraction for gold report %s; skipped", src)
            continue
        for field in scored_fields:
            pred = preds[src].get(field)
            details.append(
                {
                    "source_file": src,
                    "field": field,
                    "gold": g[field],
                    "predicted": "; ".join(map(str, pred)) if isinstance(pred, list) else ("" if pred is None else pred),
                    "outcome": score_cell(g[field], pred, types[field]),
                }
            )
    if not details:
        raise RuntimeError("Nothing to score: no gold report has an extraction.")
    details_df = pd.DataFrame(details)
    summary = summarise(details_df)
    details_df.to_csv(cfg.eval_details_file, index=False)
    summary.to_csv(cfg.eval_file, index=False)
    overall = summary.iloc[-1]
    log.info(
        "Accuracy %.0f%% (lenient %.0f%%), hallucinated %d, missed %d -> %s",
        overall["accuracy"] * 100,
        overall["lenient_accuracy"] * 100,
        overall["hallucinated"],
        overall["missed"],
        cfg.eval_file,
    )
    return summary
